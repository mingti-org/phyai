# Copyright 2026 The Qwen Team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0
"""Native, stateless Qwen-Image-2.1 image VAE using PHYAI layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.engine_config import get_engine_config
from phyai.layers.attention import Attention
from phyai.layers.channel_l2_norm import ChannelL2Norm
from phyai.layers.conv import Conv2d

from phyai.models.qwen_image_21.configuration_vae import QwenImage21VAEConfig


class QwenImage21ImageConv(Conv2d):
    """Apply the checkpoint's image convolution to a single-frame tensor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dtype=dtype,
            device=device,
            prefix=prefix,
        )
        self.image_padding = (padding,) * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x.squeeze(2), self.image_padding)).unsqueeze(2)


class QwenImage21Upsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x.float(), scale_factor=2.0, mode="nearest-exact").to(
            x.dtype
        )


class QwenImage21Resample(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        up: bool,
        temporal: bool,
        out_channels: int | None = None,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        options = {"dtype": params_dtype, "device": device}
        if up:
            self.resample = nn.Sequential(
                QwenImage21Upsample(),
                Conv2d(
                    channels,
                    out_channels or channels // 2,
                    3,
                    padding=1,
                    prefix=f"{prefix}.resample.1",
                    **options,
                ),
            )
        else:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                Conv2d(
                    channels,
                    channels,
                    3,
                    stride=2,
                    prefix=f"{prefix}.resample.1",
                    **options,
                ),
            )
        if temporal:
            # The checkpoint retains these weights from the video architecture.
            # Single-frame image inference never executes the temporal branch.
            self.time_conv = QwenImage21ImageConv(
                channels,
                channels * 2 if up else channels,
                1,
                prefix=f"{prefix}.time_conv",
                **options,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x.squeeze(2)).unsqueeze(2)


class QwenImage21ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        options = {"dtype": params_dtype, "device": device}
        self.norm1 = ChannelL2Norm(
            in_channels, spatial_dims=3, prefix=f"{prefix}.norm1", **options
        )
        self.conv1 = QwenImage21ImageConv(
            in_channels, out_channels, 3, padding=1, prefix=f"{prefix}.conv1", **options
        )
        self.norm2 = ChannelL2Norm(
            out_channels, spatial_dims=3, prefix=f"{prefix}.norm2", **options
        )
        self.conv2 = QwenImage21ImageConv(
            out_channels,
            out_channels,
            3,
            padding=1,
            prefix=f"{prefix}.conv2",
            **options,
        )
        self.conv_shortcut = (
            QwenImage21ImageConv(
                in_channels,
                out_channels,
                1,
                prefix=f"{prefix}.conv_shortcut",
                **options,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv_shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        return self.conv2(F.silu(self.norm2(x))) + residual


class QwenImage21VAEAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        options = {"dtype": params_dtype, "device": device}
        self.norm = ChannelL2Norm(channels, prefix=f"{prefix}.norm", **options)
        self.to_qkv = Conv2d(
            channels, channels * 3, 1, prefix=f"{prefix}.to_qkv", **options
        )
        self.proj = Conv2d(channels, channels, 1, prefix=f"{prefix}.proj", **options)
        # The single attention head is 768/1152 channels in the released VAE.
        self.attention = Attention(1, channels, causal=False, backend="sdpa")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, height, width = x.shape
        qkv = self.to_qkv(self.norm(x.squeeze(2)))
        qkv = qkv.flatten(2).transpose(1, 2).unsqueeze(2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        attended = self.attention(q, k, v)
        attended = (
            attended.squeeze(2).transpose(1, 2).reshape(batch, channels, height, width)
        )
        return self.proj(attended).unsqueeze(2) + x


class QwenImage21MidBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        options = {"params_dtype": params_dtype, "device": device}
        self.resnets = nn.ModuleList(
            [
                QwenImage21ResidualBlock(
                    channels, channels, prefix=f"{prefix}.resnets.{i}", **options
                )
                for i in range(2)
            ]
        )
        self.attentions = nn.ModuleList(
            [
                QwenImage21VAEAttention(
                    channels, prefix=f"{prefix}.attentions.0", **options
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnets[1](self.attentions[0](self.resnets[0](x)))


class QwenImage21AverageShortcut(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, temporal: bool, spatial: bool
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = 2 if temporal else 1
        self.factor_s = 2 if spatial else 1
        self.factor = self.factor_t * self.factor_s**2
        if in_channels * self.factor % out_channels:
            raise ValueError(
                "Residual downsample channels must divide the grouped features"
            )
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor_t, factor_s = self.factor_t, self.factor_s
        x = F.pad(x, (0, 0, 0, 0, factor_t - 1, 0))
        batch, channels, _, height, width = x.shape
        x = x.reshape(
            batch,
            channels,
            1,
            factor_t,
            height // factor_s,
            factor_s,
            width // factor_s,
            factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.reshape(
            batch,
            self.out_channels,
            self.group_size,
            1,
            height // factor_s,
            width // factor_s,
        )
        return x.mean(dim=2)


class QwenImage21DuplicateShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temporal: bool) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = 2 if temporal else 1
        if out_channels * self.factor_t * 4 % in_channels:
            raise ValueError("Residual upsample channels must divide repeated features")
        self.repeats = out_channels * self.factor_t * 4 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, _, height, width = x.shape
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.reshape(batch, self.out_channels, self.factor_t, 2, 2, 1, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.reshape(batch, self.out_channels, self.factor_t, height * 2, width * 2)
        return x[:, :, self.factor_t - 1 :]


class QwenImage21DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        count: int,
        *,
        down: bool,
        temporal: bool,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        options = {"params_dtype": params_dtype, "device": device}
        self.avg_shortcut = QwenImage21AverageShortcut(
            in_channels, out_channels, temporal, down
        )
        self.resnets = nn.ModuleList(
            [
                QwenImage21ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    prefix=f"{prefix}.resnets.{i}",
                    **options,
                )
                for i in range(count)
            ]
        )
        self.downsampler = (
            QwenImage21Resample(
                out_channels,
                up=False,
                temporal=temporal,
                prefix=f"{prefix}.downsampler",
                **options,
            )
            if down
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(residual)


class QwenImage21UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        count: int,
        *,
        up: bool,
        temporal: bool,
        residual: bool,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        options = {"params_dtype": params_dtype, "device": device}
        self.resnets = nn.ModuleList(
            [
                QwenImage21ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    prefix=f"{prefix}.resnets.{i}",
                    **options,
                )
                for i in range(count + 1)
            ]
        )
        self.avg_shortcut = (
            QwenImage21DuplicateShortcut(in_channels, out_channels, temporal)
            if up and residual
            else None
        )
        self.upsampler = (
            QwenImage21Resample(
                out_channels,
                out_channels=out_channels,
                up=True,
                temporal=temporal,
                prefix=f"{prefix}.upsampler",
                **options,
            )
            if up and residual
            else None
        )
        self.upsamplers = (
            nn.ModuleList(
                [
                    QwenImage21Resample(
                        out_channels,
                        up=True,
                        temporal=temporal,
                        prefix=f"{prefix}.upsamplers.0",
                        **options,
                    )
                ]
            )
            if up and not residual
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x if self.avg_shortcut is None else x + self.avg_shortcut(residual)


class QwenImage21Encoder(nn.Module):
    def __init__(
        self,
        config: QwenImage21VAEConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        options = {"params_dtype": params_dtype, "device": device}
        conv_options = {"dtype": params_dtype, "device": device}
        dimensions = [
            config.base_dim * multiplier for multiplier in (1,) + config.dim_mult
        ]
        self.conv_in = QwenImage21ImageConv(
            config.in_channels,
            dimensions[0],
            3,
            padding=1,
            prefix="encoder.conv_in",
            **conv_options,
        )
        blocks = []
        scale = 1.0
        for i, (in_channels, out_channels) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            down = i < len(config.dim_mult) - 1
            temporal = down and config.temperal_downsample[i]
            if config.is_residual:
                blocks.append(
                    QwenImage21DownBlock(
                        in_channels,
                        out_channels,
                        config.num_res_blocks,
                        down=down,
                        temporal=temporal,
                        prefix=f"encoder.down_blocks.{len(blocks)}",
                        **options,
                    )
                )
            else:
                for _ in range(config.num_res_blocks):
                    blocks.append(
                        QwenImage21ResidualBlock(
                            in_channels,
                            out_channels,
                            prefix=f"encoder.down_blocks.{len(blocks)}",
                            **options,
                        )
                    )
                    if scale in config.attn_scales:
                        blocks.append(
                            QwenImage21VAEAttention(
                                out_channels,
                                prefix=f"encoder.down_blocks.{len(blocks)}",
                                **options,
                            )
                        )
                    in_channels = out_channels
                if down:
                    blocks.append(
                        QwenImage21Resample(
                            out_channels,
                            up=False,
                            temporal=temporal,
                            prefix=f"encoder.down_blocks.{len(blocks)}",
                            **options,
                        )
                    )
                    scale /= 2
        self.down_blocks = nn.ModuleList(blocks)
        self.mid_block = QwenImage21MidBlock(
            dimensions[-1], prefix="encoder.mid_block", **options
        )
        self.norm_out = ChannelL2Norm(
            dimensions[-1], spatial_dims=3, prefix="encoder.norm_out", **conv_options
        )
        self.conv_out = QwenImage21ImageConv(
            dimensions[-1],
            config.z_dim * 2,
            3,
            padding=1,
            prefix="encoder.conv_out",
            **conv_options,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.norm_out(self.mid_block(x))))


class QwenImage21Decoder(nn.Module):
    def __init__(
        self,
        config: QwenImage21VAEConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        options = {"params_dtype": params_dtype, "device": device}
        conv_options = {"dtype": params_dtype, "device": device}
        dimensions = [
            config.decoder_base_dim * multiplier
            for multiplier in (config.dim_mult[-1],) + config.dim_mult[::-1]
        ]
        self.conv_in = QwenImage21ImageConv(
            config.z_dim,
            dimensions[0],
            3,
            padding=1,
            prefix="decoder.conv_in",
            **conv_options,
        )
        self.mid_block = QwenImage21MidBlock(
            dimensions[0], prefix="decoder.mid_block", **options
        )
        blocks = []
        for i, (in_channels, out_channels) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            up = i < len(config.dim_mult) - 1
            if i and not config.is_residual:
                in_channels //= 2
            temporal = up and config.temperal_downsample[::-1][i]
            blocks.append(
                QwenImage21UpBlock(
                    in_channels,
                    out_channels,
                    config.num_res_blocks,
                    up=up,
                    temporal=temporal,
                    residual=config.is_residual,
                    prefix=f"decoder.up_blocks.{i}",
                    **options,
                )
            )
        self.up_blocks = nn.ModuleList(blocks)
        self.norm_out = ChannelL2Norm(
            dimensions[-1], spatial_dims=3, prefix="decoder.norm_out", **conv_options
        )
        self.conv_out = QwenImage21ImageConv(
            dimensions[-1],
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
        return self.conv_out(F.silu(self.norm_out(x)))


class QwenImage21VAE(nn.Module):
    """RGBA image encoder/decoder with raw, unnormalized latent outputs."""

    def __init__(
        self,
        config: QwenImage21VAEConfig,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if device is None:
            device = get_engine_config().device.target
        options = {"params_dtype": params_dtype, "device": device}
        self.encoder = QwenImage21Encoder(config, **options)
        self.quant_conv = QwenImage21ImageConv(
            config.z_dim * 2,
            config.z_dim * 2,
            1,
            dtype=params_dtype,
            device=device,
            prefix="quant_conv",
        )
        self.post_quant_conv = QwenImage21ImageConv(
            config.z_dim,
            config.z_dim,
            1,
            dtype=params_dtype,
            device=device,
            prefix="post_quant_conv",
        )
        self.decoder = QwenImage21Decoder(config, **options)

    def encode_moments(self, pixels: torch.Tensor) -> torch.Tensor:
        self.validate_input(
            pixels, self.config.in_channels // (self.config.patch_size or 1) ** 2
        )
        if (
            pixels.shape[-2] % self.config.scale_factor_spatial
            or pixels.shape[-1] % self.config.scale_factor_spatial
        ):
            raise ValueError(
                "VAE image dimensions must be divisible by the spatial compression ratio"
            )
        return self.quant_conv(
            self.encoder(patchify(pixels, self.config.patch_size or 1))
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode_moments(pixels).chunk(2, dim=1)[0]

    def decode(
        self, latents: torch.Tensor, *, clamp_output: bool = True
    ) -> torch.Tensor:
        self.validate_input(latents, self.config.z_dim)
        pixels = self.decoder(self.post_quant_conv(latents))
        pixels = unpatchify(pixels, self.config.patch_size or 1)
        return pixels.clamp(-1, 1) if clamp_output else pixels

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)

    @staticmethod
    def validate_input(value: torch.Tensor, channels: int) -> None:
        if value.ndim != 5 or value.shape[1] != channels or value.shape[2] != 1:
            raise ValueError(
                f"Qwen-Image-2.1 VAE expects [batch, {channels}, 1, height, width], got {tuple(value.shape)}"
            )


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    batch, channels, frames, height, width = x.shape
    x = x.reshape(
        batch,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    return x.permute(0, 1, 6, 4, 2, 3, 5).reshape(
        batch,
        channels * patch_size**2,
        frames,
        height // patch_size,
        width // patch_size,
    )


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    batch, channels, frames, height, width = x.shape
    x = x.reshape(
        batch, channels // patch_size**2, patch_size, patch_size, frames, height, width
    )
    return x.permute(0, 1, 4, 5, 3, 6, 2).reshape(
        batch,
        channels // patch_size**2,
        frames,
        height * patch_size,
        width * patch_size,
    )


def qwen_image_21_vae_weight_remap(name: str) -> str | None:
    return (
        name
        if name.startswith(("encoder.", "decoder.", "quant_conv.", "post_quant_conv."))
        else None
    )


__all__ = ["QwenImage21VAE", "qwen_image_21_vae_weight_remap"]
