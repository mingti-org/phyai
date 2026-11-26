"""DreamZero Wan2.1 VAE with causal frame caches and optional spatial tiling."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from phyai.engine_config import get_engine_config
from phyai.layers import SiLU
from phyai.layers.channel_l2_norm import ChannelL2Norm
from phyai.layers.conv import Conv2d, Conv3d
from phyai.layers.attention.nocache.layer import Attention
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroVAEConfig


@dataclass
class WanVAECache:
    """Per-call temporal history, visited in the same order for each chunk."""

    values: list[torch.Tensor | bool | None] = field(default_factory=list)
    index: int = 0

    def next(self) -> tuple[int, torch.Tensor | bool | None]:
        index = self.index
        self.index += 1
        if index == len(self.values):
            self.values.append(None)
        return index, self.values[index]


class CausalConv3d(Conv3d):
    """3-D convolution with left padding along the time dimension."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and padding[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(F.pad(x, padding))

    def forward_cached(self, x: torch.Tensor, cache: WanVAECache) -> torch.Tensor:
        index, previous = cache.next()
        history = x[:, :, -2:].clone()
        if history.shape[2] < 2 and previous is not None:
            history = torch.cat([previous[:, :, -1:], history], dim=2)
        output = self(x, previous)
        cache.values[index] = history
        return output


class Upsample(nn.Upsample):
    def forward(self, x):
        # Interpolate in FP32, then restore the input dtype as Wan does.
        return super().forward(x.float()).to(x.dtype)


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str, *, device, prefix: str):
        super().__init__()
        if mode not in {"upsample2d", "upsample3d", "downsample2d", "downsample3d"}:
            raise ValueError(f"unsupported resampling mode {mode!r}.")
        self.mode = mode
        if mode.startswith("upsample"):
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                Conv2d(
                    dim,
                    dim // 2,
                    3,
                    padding=1,
                    device=device,
                    prefix=f"{prefix}.resample.1",
                ),
            )
        else:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                Conv2d(
                    dim, dim, 3, stride=2, device=device, prefix=f"{prefix}.resample.1"
                ),
            )
        if mode == "upsample3d":
            self.time_conv = CausalConv3d(
                dim,
                dim * 2,
                (3, 1, 1),
                padding=(1, 0, 0),
                device=device,
                prefix=f"{prefix}.time_conv",
            )
        elif mode == "downsample3d":
            self.time_conv = CausalConv3d(
                dim,
                dim,
                (3, 1, 1),
                stride=(2, 1, 1),
                device=device,
                prefix=f"{prefix}.time_conv",
            )

    def forward(self, x: torch.Tensor, cache: WanVAECache) -> torch.Tensor:
        if self.mode == "upsample3d":
            index, previous = cache.next()
            if previous is None:
                # The first frame passes through without temporal upsampling.
                cache.values[index] = True
            else:
                history = x[:, :, -2:].clone()
                if history.shape[2] < 2:
                    previous_frame = (
                        torch.zeros_like(history)
                        if previous is True
                        else previous[:, :, -1:]
                    )
                    history = torch.cat([previous_frame, history], dim=2)
                x = self.time_conv(x, None if previous is True else previous)
                cache.values[index] = history
                x = rearrange(x, "b (r c) t h w -> b c (t r) h w", r=2)

        frames = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=frames)

        if self.mode == "downsample3d":
            index, previous = cache.next()
            if previous is None:
                cache.values[index] = x.clone()
            else:
                history = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([previous[:, :, -1:], x], dim=2))
                cache.values[index] = history
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, device, prefix: str):
        super().__init__()
        self.activation = SiLU()
        self.norm1 = ChannelL2Norm(
            in_dim,
            spatial_dims=3,
            dtype=torch.float32,
            compute_dtype=None,
            restore_input_dtype=False,
            device=device,
            prefix=f"{prefix}.residual.0",
        )
        self.conv1 = CausalConv3d(
            in_dim, out_dim, 3, padding=1, device=device, prefix=f"{prefix}.residual.2"
        )
        self.norm2 = ChannelL2Norm(
            out_dim,
            spatial_dims=3,
            dtype=torch.float32,
            compute_dtype=None,
            restore_input_dtype=False,
            device=device,
            prefix=f"{prefix}.residual.3",
        )
        # Checkpoint index 5 is a parameter-free training dropout.
        self.conv2 = CausalConv3d(
            out_dim, out_dim, 3, padding=1, device=device, prefix=f"{prefix}.residual.6"
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1, device=device, prefix=f"{prefix}.shortcut")
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cache: WanVAECache) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1.forward_cached(self.activation(self.norm1(x)), cache)
        x = self.conv2.forward_cached(self.activation(self.norm2(x)), cache)
        return x + residual


class AttentionBlock(nn.Module):
    """Spatial self-attention with a single head for each frame."""

    def __init__(self, dim: int, *, device, prefix: str):
        super().__init__()
        self.norm = ChannelL2Norm(
            dim,
            spatial_dims=2,
            dtype=torch.float32,
            compute_dtype=None,
            restore_input_dtype=False,
            device=device,
            prefix=f"{prefix}.norm",
        )
        self.to_qkv = Conv2d(dim, dim * 3, 1, device=device, prefix=f"{prefix}.to_qkv")
        self.proj = Conv2d(dim, dim, 1, device=device, prefix=f"{prefix}.proj")
        self.attention = Attention(
            num_heads=1,
            head_dim=dim,
            causal=False,
            kernel_role="dreamzero.vae_attention",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, channels, frames, height, width = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        q, k, v = (
            self.to_qkv(self.norm(x))
            .reshape(batch * frames, 1, channels * 3, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
            .chunk(3, dim=-1)
        )
        x = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = (
            x.squeeze(2)
            .transpose(1, 2)
            .reshape(batch * frames, channels, height, width)
        )
        x = self.proj(x)
        return rearrange(x, "(b t) c h w -> b c t h w", t=frames) + residual


class VAEMiddleBlock(nn.Module):
    def __init__(self, dim: int, *, device, prefix: str):
        super().__init__()
        self.residual1 = ResidualBlock(dim, dim, device=device, prefix=f"{prefix}.0")
        self.attention = AttentionBlock(dim, device=device, prefix=f"{prefix}.1")
        self.residual2 = ResidualBlock(dim, dim, device=device, prefix=f"{prefix}.2")

    def forward(self, x: torch.Tensor, cache: WanVAECache) -> torch.Tensor:
        x = self.residual1(x, cache)
        return self.residual2(self.attention(x), cache)


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim,
        z_dim,
        dim_mult,
        num_res_blocks,
        temporal_downsample,
        *,
        device,
        prefix,
    ):
        super().__init__()
        dims = [dim * value for value in (1, *dim_mult)]
        self.conv1 = CausalConv3d(
            3, dims[0], 3, padding=1, device=device, prefix=f"{prefix}.conv1"
        )
        layers = []
        for stage, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                layers.append(
                    ResidualBlock(
                        in_dim,
                        out_dim,
                        device=device,
                        prefix=f"{prefix}.downsamples.{len(layers)}",
                    )
                )
                in_dim = out_dim
            if stage < len(dim_mult) - 1:
                layers.append(
                    Resample(
                        out_dim,
                        "downsample3d"
                        if temporal_downsample[stage]
                        else "downsample2d",
                        device=device,
                        prefix=f"{prefix}.downsamples.{len(layers)}",
                    )
                )
        self.downsamples = nn.ModuleList(layers)
        self.middle = VAEMiddleBlock(out_dim, device=device, prefix=f"{prefix}.middle")
        self.norm_out = ChannelL2Norm(
            out_dim,
            spatial_dims=3,
            dtype=torch.float32,
            compute_dtype=None,
            restore_input_dtype=False,
            device=device,
            prefix=f"{prefix}.head.0",
        )
        self.activation = SiLU()
        self.conv_out = CausalConv3d(
            out_dim, z_dim, 3, padding=1, device=device, prefix=f"{prefix}.head.2"
        )

    def forward(self, x: torch.Tensor, cache: WanVAECache) -> torch.Tensor:
        x = self.conv1.forward_cached(x, cache)
        for layer in self.downsamples:
            x = layer(x, cache)
        x = self.middle(x, cache)
        return self.conv_out.forward_cached(self.activation(self.norm_out(x)), cache)


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim,
        z_dim,
        dim_mult,
        num_res_blocks,
        temporal_upsample,
        *,
        device,
        prefix,
    ):
        super().__init__()
        dims = [dim * value for value in (dim_mult[-1], *dim_mult[::-1])]
        self.conv1 = CausalConv3d(
            z_dim, dims[0], 3, padding=1, device=device, prefix=f"{prefix}.conv1"
        )
        self.middle = VAEMiddleBlock(dims[0], device=device, prefix=f"{prefix}.middle")
        layers = []
        for stage, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if stage > 0:
                in_dim //= 2
            for _ in range(num_res_blocks + 1):
                layers.append(
                    ResidualBlock(
                        in_dim,
                        out_dim,
                        device=device,
                        prefix=f"{prefix}.upsamples.{len(layers)}",
                    )
                )
                in_dim = out_dim
            if stage < len(dim_mult) - 1:
                layers.append(
                    Resample(
                        out_dim,
                        "upsample3d" if temporal_upsample[stage] else "upsample2d",
                        device=device,
                        prefix=f"{prefix}.upsamples.{len(layers)}",
                    )
                )
        self.upsamples = nn.ModuleList(layers)
        self.norm_out = ChannelL2Norm(
            out_dim,
            spatial_dims=3,
            dtype=torch.float32,
            compute_dtype=None,
            restore_input_dtype=False,
            device=device,
            prefix=f"{prefix}.head.0",
        )
        self.activation = SiLU()
        self.conv_out = CausalConv3d(
            out_dim, 3, 3, padding=1, device=device, prefix=f"{prefix}.head.2"
        )

    def forward(self, x: torch.Tensor, cache: WanVAECache) -> torch.Tensor:
        x = self.middle(self.conv1.forward_cached(x, cache), cache)
        for layer in self.upsamples:
            x = layer(x, cache)
        return self.conv_out.forward_cached(self.activation(self.norm_out(x)), cache)


class WanVAEModel(nn.Module):
    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        temporal_downsample=(False, True, True),
        *,
        device: torch.device | str | None = None,
        prefix: str = "model",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            temporal_downsample,
            device=device,
            prefix=f"{prefix}.encoder",
        )
        self.conv1 = CausalConv3d(
            z_dim * 2, z_dim * 2, 1, device=device, prefix=f"{prefix}.conv1"
        )
        self.conv2 = CausalConv3d(
            z_dim, z_dim, 1, device=device, prefix=f"{prefix}.conv2"
        )
        self.decoder = Decoder3d(
            dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            temporal_downsample[::-1],
            device=device,
            prefix=f"{prefix}.decoder",
        )

    def encode(
        self, x: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if x.shape[2] % 4 != 1:
            raise ValueError("Wan2.1 encoding requires 1 + 4n frames.")
        cache = WanVAECache()
        chunks = [self.encoder(x[:, :, :1], cache)]
        for start in range(1, x.shape[2], 4):
            cache.index = 0
            chunks.append(self.encoder(x[:, :, start : start + 4], cache))
        mu, _ = self.conv1(torch.cat(chunks, dim=2)).chunk(2, dim=1)
        mean, inv_std = (value.to(mu).view(1, self.z_dim, 1, 1, 1) for value in scale)
        return (mu - mean) * inv_std

    def decode(
        self, z: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        mean, inv_std = (value.to(z).view(1, self.z_dim, 1, 1, 1) for value in scale)
        x = self.conv2(z / inv_std + mean)
        cache = WanVAECache()
        chunks = []
        for index in range(x.shape[2]):
            cache.index = 0
            chunks.append(self.decoder(x[:, :, index : index + 1], cache))
        return torch.cat(chunks, dim=2)


def _tile_slices(height, width, tile_size, tile_stride):
    size_h, size_w = tile_size
    stride_h, stride_w = tile_stride
    if not (0 < stride_h <= size_h and 0 < stride_w <= size_w):
        raise ValueError("tile strides must be positive and no larger than tile sizes.")
    for h in range(0, height, stride_h):
        if h >= stride_h and h - stride_h + size_h >= height:
            continue
        for w in range(0, width, stride_w):
            if w >= stride_w and w - stride_w + size_w >= width:
                continue
            yield h, h + size_h, w, w + size_w


class DreamZeroWanVAE(nn.Module):
    """Wan2.1 latent normalization, causal encoding/decoding, and spatial tiling."""

    def __init__(
        self,
        config: DreamZeroVAEConfig | None = None,
        *,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.config = config or DreamZeroVAEConfig()
        device = get_engine_config().device.target if device is None else device
        mean = (
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        )
        std = (
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        )
        # Keep FP32 source constants even when parameter storage is cast.
        self.scale = (
            torch.tensor(mean, dtype=torch.float32, device=device),
            torch.tensor(std, dtype=torch.float32, device=device).reciprocal(),
        )
        self.model = WanVAEModel(
            dim=self.config.base_dim,
            z_dim=self.config.z_dim,
            dim_mult=self.config.dim_mult,
            num_res_blocks=self.config.num_res_blocks,
            temporal_downsample=self.config.temporal_downsample,
            device=device,
        )
        self.upsampling_factor = self.config.scale_factor_spatial
        self.z_dim = self.config.z_dim

    def build_1d_mask(self, length, left_bound, right_bound, border_width, device):
        x = torch.ones((length,), device=device)
        border = torch.arange(border_width, device=device) + 1
        if border_width == 0:
            return x
        if not left_bound:
            x[:border_width] = border / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(border / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(
            H, is_bound[0], is_bound[1], border_width[0], device=data.device
        )
        w = self.build_1d_mask(
            W, is_bound[2], is_bound[3], border_width[1], device=data.device
        )

        return torch.minimum(h[:, None], w[None, :])[None, None, None]

    def tiled_decode(self, hidden_states, tile_size, tile_stride):
        batch_size, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        tasks = _tile_slices(H, W, tile_size, tile_stride)

        out_T = T * 4 - 3
        weight = torch.zeros(
            (1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        values = torch.zeros(
            (
                batch_size,
                3,
                out_T,
                H * self.upsampling_factor,
                W * self.upsampling_factor,
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for h, h_, w, w_ in tasks:
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_]
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor,
                ),
            ).to(dtype=hidden_states.dtype)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        values = values.clamp_(-1, 1)
        return values

    def tiled_encode(self, video, tile_size, tile_stride):
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        tasks = _tile_slices(H, W, tile_size, tile_stride)

        out_T = (T + 3) // 4
        weight = torch.zeros(
            (1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=video.device,
        )
        values = torch.zeros(
            (
                1,
                self.z_dim,
                out_T,
                H // self.upsampling_factor,
                W // self.upsampling_factor,
            ),
            dtype=video.dtype,
            device=video.device,
        )

        for h, h_, w, w_ in tasks:
            hidden_states_batch = video[:, :, :, h:h_, w:w_]
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor,
                ),
            ).to(dtype=video.dtype)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / weight
        return values

    def encode(self, videos, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        if tiled:
            tile_size = tuple(value * self.upsampling_factor for value in tile_size)
            tile_stride = tuple(value * self.upsampling_factor for value in tile_stride)
        outputs = []
        for video in videos:
            video = video.unsqueeze(0)
            output = (
                self.tiled_encode(video, tile_size, tile_stride)
                if tiled
                else self.model.encode(video, self.scale)
            )
            outputs.append(output.squeeze(0))
        return torch.stack(outputs)

    def decode(self, latents, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        if tiled:
            return self.tiled_decode(latents, tile_size, tile_stride)
        return self.model.decode(latents, self.scale).clamp_(-1, 1)


def dreamzero_vae_weight_remap(key: str) -> str | None:
    """Map DreamZero checkpoint VAE keys to the local layer prefixes."""
    if key.startswith("action_head.vae."):
        return key.removeprefix("action_head.vae.")
    if key.startswith("model."):
        return key
    return None


__all__ = ["DreamZeroWanVAE", "dreamzero_vae_weight_remap"]
