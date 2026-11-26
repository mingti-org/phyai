"""Native FLUX.2 autoencoder architecture for MOSS-Image inference."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from phyai.engine_config import get_engine_config
from phyai.layers.activation import SiLU
from phyai.layers.attention import Attention
from phyai.layers.conv import Conv2d
from phyai.layers.linear import ReplicatedLinear
from phyai.models.moss_image.configuration_vae import MOSSImageVAEConfig
from phyai.weights.shards import replicated


def moss_image_vae_weight_remap(name: str) -> str | None:
    return name.removeprefix("vae.")


def vae_group_norm(
    channels: int,
    groups: int,
    *,
    params_dtype: torch.dtype,
    device: torch.device | str,
    prefix: str,
) -> nn.GroupNorm:
    norm = nn.GroupNorm(groups, channels, eps=1e-6, dtype=params_dtype, device=device)
    for name, parameter in norm.named_parameters():
        parameter.requires_grad_(False)
        parameter.hf_keys = [(f"{prefix}.{name}", None)]
        parameter.weight_loader = replicated()
    return norm


class MOSSImageVAEResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        options = {"dtype": params_dtype, "device": device}
        norm_options = {"params_dtype": params_dtype, "device": device}
        self.norm1 = vae_group_norm(
            in_channels, groups, prefix=f"{prefix}.norm1", **norm_options
        )
        self.norm2 = vae_group_norm(
            out_channels, groups, prefix=f"{prefix}.norm2", **norm_options
        )
        self.conv1 = Conv2d(
            in_channels, out_channels, 3, padding=1, prefix=f"{prefix}.conv1", **options
        )
        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            prefix=f"{prefix}.conv2",
            **options,
        )
        self.conv_shortcut = (
            Conv2d(
                in_channels,
                out_channels,
                1,
                prefix=f"{prefix}.conv_shortcut",
                **options,
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.nonlinearity = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.nonlinearity(self.norm1(x)))
        h = self.conv2(self.nonlinearity(self.norm2(h)))
        return self.conv_shortcut(x) + h


class MOSSImageVAEAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        groups: int,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        options = {"params_dtype": params_dtype, "device": device}
        self.group_norm = vae_group_norm(
            channels, groups, prefix=f"{prefix}.group_norm", **options
        )
        self.to_q = ReplicatedLinear(
            channels, channels, prefix=f"{prefix}.to_q", **options
        )
        self.to_k = ReplicatedLinear(
            channels, channels, prefix=f"{prefix}.to_k", **options
        )
        self.to_v = ReplicatedLinear(
            channels, channels, prefix=f"{prefix}.to_v", **options
        )
        self.to_out = nn.ModuleList(
            [
                ReplicatedLinear(
                    channels, channels, prefix=f"{prefix}.to_out.0", **options
                )
            ]
        )
        # The FP32 VAE uses one head of width 384 or 512, requiring SDPA.
        self.attention = Attention(1, channels, causal=False, backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.group_norm(x).flatten(2).transpose(1, 2)
        q, _ = self.to_q(h)
        k, _ = self.to_k(h)
        v, _ = self.to_v(h)
        h = self.attention(q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)).squeeze(2)
        h, _ = self.to_out[0](h)
        return h.transpose(1, 2).reshape(batch, channels, height, width) + x


class MOSSImageVAEMidBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        config: MOSSImageVAEConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        options = {"params_dtype": params_dtype, "device": device}
        self.resnets = nn.ModuleList(
            [
                MOSSImageVAEResnetBlock(
                    channels,
                    channels,
                    config.norm_num_groups,
                    prefix=f"{prefix}.resnets.{index}",
                    **options,
                )
                for index in range(2)
            ]
        )
        self.attentions = nn.ModuleList(
            [
                MOSSImageVAEAttention(
                    channels,
                    config.norm_num_groups,
                    prefix=f"{prefix}.attentions.0",
                    **options,
                )
                if config.mid_block_add_attention
                else nn.Identity()
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class MOSSImageVAEResample(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        up: bool,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.up = up
        self.conv = Conv2d(
            channels,
            channels,
            3,
            stride=1 if up else 2,
            padding=1 if up else 0,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.conv",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (
            F.interpolate(x, scale_factor=2, mode="nearest")
            if self.up
            else F.pad(x, (0, 1, 0, 1))
        )
        return self.conv(x)


class MOSSImageVAEStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: MOSSImageVAEConfig,
        *,
        up: bool,
        resample: bool,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.up = up
        options = {"params_dtype": params_dtype, "device": device}
        self.resnets = nn.ModuleList(
            [
                MOSSImageVAEResnetBlock(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    config.norm_num_groups,
                    prefix=f"{prefix}.resnets.{index}",
                    **options,
                )
                for index in range(config.layers_per_block + int(up))
            ]
        )
        sampling_name = "upsamplers" if up else "downsamplers"
        setattr(
            self,
            sampling_name,
            nn.ModuleList(
                [
                    MOSSImageVAEResample(
                        out_channels,
                        up=up,
                        prefix=f"{prefix}.{sampling_name}.0",
                        **options,
                    )
                ]
            )
            if resample
            else nn.ModuleList(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resnets:
            x = block(x)
        for resample in self.upsamplers if self.up else self.downsamplers:
            x = resample(x)
        return x


class MOSSImageVAEEncoder(nn.Module):
    def __init__(
        self,
        config: MOSSImageVAEConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        channels = config.block_out_channels
        options = {"params_dtype": params_dtype, "device": device}
        conv_options = {"dtype": params_dtype, "device": device}
        self.conv_in = Conv2d(
            config.in_channels,
            channels[0],
            3,
            padding=1,
            prefix="encoder.conv_in",
            **conv_options,
        )
        self.down_blocks = nn.ModuleList(
            [
                MOSSImageVAEStage(
                    channels[max(index - 1, 0)],
                    width,
                    config,
                    up=False,
                    resample=index < len(channels) - 1,
                    prefix=f"encoder.down_blocks.{index}",
                    **options,
                )
                for index, width in enumerate(channels)
            ]
        )
        self.mid_block = MOSSImageVAEMidBlock(
            channels[-1], config, prefix="encoder.mid_block", **options
        )
        self.conv_norm_out = vae_group_norm(
            channels[-1],
            config.norm_num_groups,
            prefix="encoder.conv_norm_out",
            **options,
        )
        self.conv_act = SiLU()
        self.conv_out = Conv2d(
            channels[-1],
            2 * config.latent_channels,
            3,
            padding=1,
            prefix="encoder.conv_out",
            **conv_options,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class MOSSImageVAEDecoder(nn.Module):
    def __init__(
        self,
        config: MOSSImageVAEConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        channels = tuple(
            reversed(config.decoder_block_out_channels or config.block_out_channels)
        )
        options = {"params_dtype": params_dtype, "device": device}
        conv_options = {"dtype": params_dtype, "device": device}
        self.conv_in = Conv2d(
            config.latent_channels,
            channels[0],
            3,
            padding=1,
            prefix="decoder.conv_in",
            **conv_options,
        )
        self.mid_block = MOSSImageVAEMidBlock(
            channels[0], config, prefix="decoder.mid_block", **options
        )
        self.up_blocks = nn.ModuleList(
            [
                MOSSImageVAEStage(
                    channels[max(index - 1, 0)],
                    width,
                    config,
                    up=True,
                    resample=index < len(channels) - 1,
                    prefix=f"decoder.up_blocks.{index}",
                    **options,
                )
                for index, width in enumerate(channels)
            ]
        )
        self.conv_norm_out = vae_group_norm(
            channels[-1],
            config.norm_num_groups,
            prefix="decoder.conv_norm_out",
            **options,
        )
        self.conv_act = SiLU()
        self.conv_out = Conv2d(
            channels[-1],
            config.out_channels,
            3,
            padding=1,
            prefix="decoder.conv_out",
            **conv_options,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class MOSSImageVAE(nn.Module):
    def __init__(
        self,
        config: MOSSImageVAEConfig,
        *,
        params_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if device is None:
            device = get_engine_config().device.target
        options = {"params_dtype": params_dtype, "device": device}
        conv_options = {"dtype": params_dtype, "device": device}
        self.encoder = MOSSImageVAEEncoder(config, **options)
        self.decoder = MOSSImageVAEDecoder(config, **options)
        self.quant_conv = (
            Conv2d(
                2 * config.latent_channels,
                2 * config.latent_channels,
                1,
                prefix="quant_conv",
                **conv_options,
            )
            if config.use_quant_conv
            else nn.Identity()
        )
        self.post_quant_conv = (
            Conv2d(
                config.latent_channels,
                config.latent_channels,
                1,
                prefix="post_quant_conv",
                **conv_options,
            )
            if config.use_post_quant_conv
            else nn.Identity()
        )
        # The loader indexes parameters; keep fixed checkpoint statistics here.
        self.bn = nn.Module()
        for name, value in (
            (
                "running_mean",
                torch.zeros(config.packed_latent_channels, **conv_options),
            ),
            ("running_var", torch.ones(config.packed_latent_channels, **conv_options)),
            ("num_batches_tracked", torch.zeros((), dtype=torch.long, device=device)),
        ):
            parameter = nn.Parameter(value, requires_grad=False)
            parameter.hf_keys = [(f"bn.{name}", None)]
            parameter.weight_loader = replicated()
            self.bn.register_parameter(name, parameter)

    def encode_moments(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != self.config.in_channels:
            raise ValueError(
                "VAE pixels must have shape [batch, in_channels, height, width]"
            )
        return self.quant_conv(self.encoder(pixels))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4 or latents.shape[1] != self.config.latent_channels:
            raise ValueError(
                "VAE latents must have shape [batch, latent_channels, height, width]"
            )
        return self.decoder(self.post_quant_conv(latents))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)


__all__ = ["MOSSImageVAE", "moss_image_vae_weight_remap"]
