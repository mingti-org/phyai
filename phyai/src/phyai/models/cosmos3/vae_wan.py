"""Native WAN VAE decode for Cosmos3."""

from __future__ import annotations

import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from phyai.utils.cuda import current_device
from phyai.layers.conv import Conv2d, Conv3d
from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3WanVAEConfig
from phyai.weights.shards import replicated


CACHE_T = 2


def _max_reduce_op() -> dist.ReduceOp:
    """The ``MAX`` reduce op, for reconciling per-rank tile shapes."""
    return dist.ReduceOp.MAX


@functools.lru_cache(maxsize=None)
def _feather_ramp(n: int, feather: int) -> tuple[float, ...]:
    """1-D linear ramp of length ``n``: rises over the first ``feather`` samples,
    falls over the last ``feather``, flat 1.0 in the middle.

    Used as the per-tile blend weight so overlapping tile edges cross-fade instead
    of hard-cutting (hiding WAN's spatial-conv receptive-field seams). Cached
    because every tile of the same size shares the ramp. ``feather <= 0`` -> all 1s.
    """
    if feather <= 0 or n <= 1:
        return tuple([1.0] * n)
    ramp = []
    for i in range(n):
        up = min(1.0, (i + 1) / (feather + 1))
        down = min(1.0, (n - i) / (feather + 1))
        ramp.append(min(up, down))
    return tuple(ramp)


def _feather_weight(
    h: int, w: int, feather: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Separable 2-D feather weight ``[1, 1, 1, h, w]`` = outer(ramp_h, ramp_w)."""
    rh = torch.tensor(_feather_ramp(h, feather), device=device, dtype=dtype)
    rw = torch.tensor(_feather_ramp(w, feather), device=device, dtype=dtype)
    return (rh[:, None] * rw[None, :]).view(1, 1, 1, h, w)


class WanCausalConv3d(nn.Module):
    """Causal 3-D conv: temporal left-pad only, with a feat_cache for chunked decode.

    Wraps a phyai ``Conv3d`` (constructed with ``padding=0``); the asymmetric
    causal pad (``2*pad_t`` left, 0 right; symmetric H/W) is applied in forward.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        self.conv = Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=True,
            prefix=f"{prefix}.conv" if prefix else "",
        )
        # (W_l, W_r, H_l, H_r, T_left=2*pad_t, T_right=0)
        self._padding = (
            padding[2],
            padding[2],
            padding[1],
            padding[1],
            2 * padding[0],
            0,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias

    def forward(
        self, x: torch.Tensor, cache_x: torch.Tensor | None = None
    ) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return self.conv(x)


class WanRMSNorm(nn.Module):
    """Channel dim L2 normalize then scale: no eps/variance, not normal RMSNorm.

    ``F.normalize(x, dim=1) * sqrt(C) * gamma (+ bias)`` over the channel axis of
    ``[B, C, T, H, W]`` (``images=False``) or ``[B*T, C, H, W]`` (``images=True``).

    TODO(wch): This kernel can be fused in the future, but not critical right now.
    """

    def __init__(
        self, dim: int, images: bool = True, bias: bool = False, prefix: str = ""
    ) -> None:
        super().__init__()
        broadcast = (1, 1) if images else (1, 1, 1)
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, *broadcast))
        self.bias = nn.Parameter(torch.zeros(dim, *broadcast)) if bias else 0.0
        if prefix:
            self.gamma.hf_keys = [(f"{prefix}.gamma", None)]
            self.gamma.weight_loader = replicated()
            if isinstance(self.bias, nn.Parameter):
                self.bias.hf_keys = [(f"{prefix}.bias", None)]
                self.bias.weight_loader = replicated()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class WanUpsample(nn.Upsample):
    """nearest exact upsample done in fp32, cast back to input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class DupUp3D(nn.Module):
    """Residual up shortcut: channel repeat, reshape to (t*ft, h*fs, w*fs)."""

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        factor = factor_t * factor_s * factor_s
        assert out_channels * factor % in_channels == 0
        self.repeats = out_channels * factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class AvgDown3D(nn.Module):
    """Residual down shortcut: group-average pool by (factor_t, factor_s, factor_s)."""

    def __init__(
        self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        x = x.view(
            b,
            c,
            t // self.factor_t,
            self.factor_t,
            h // self.factor_s,
            self.factor_s,
            w // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            b,
            c * self.factor,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        x = x.view(
            b,
            self.out_channels,
            self.group_size,
            t // self.factor_t,
            h // self.factor_s,
            w // self.factor_s,
        )
        return x.mean(dim=2)


class WanResample(nn.Module):
    """Spatial (optional temporal) resample. up/downsample 2d or 3d."""

    def __init__(
        self,
        dim: int,
        mode: str,
        upsample_out_dim: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        conv_prefix = f"{prefix}.resample.1" if prefix else ""
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                Conv2d(dim, upsample_out_dim, 3, padding=1, prefix=conv_prefix),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                Conv2d(dim, upsample_out_dim, 3, padding=1, prefix=conv_prefix),
            )
            self.time_conv = WanCausalConv3d(
                dim,
                dim * 2,
                (3, 1, 1),
                padding=(1, 0, 0),
                prefix=f"{prefix}.time_conv" if prefix else "",
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                Conv2d(dim, dim, 3, stride=(2, 2), prefix=conv_prefix),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                Conv2d(dim, dim, 3, stride=(2, 2), prefix=conv_prefix),
            )
            self.time_conv = WanCausalConv3d(
                dim,
                dim,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=(0, 0, 0),
                prefix=f"{prefix}.time_conv" if prefix else "",
            )
        else:
            raise ValueError(f"WanResample supports up/downsample 2d/3d, got {mode!r}.")

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if (
                    cache_x.shape[2] < 2
                    and feat_cache[idx] is not None
                    and feat_cache[idx] != "Rep"
                ):
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3)
                x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d" and feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(
                    torch.cat([feat_cache[idx][:, :, -1:, :, :], x], dim=2)
                )
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class WanResidualBlock(nn.Module):
    """norm silu conv x 2 + causal conv shortcut"""

    def __init__(
        self, in_dim: int, out_dim: int, dropout: float = 0.0, prefix: str = ""
    ) -> None:
        super().__init__()
        self.nonlinearity = nn.SiLU()
        self.norm1 = WanRMSNorm(
            in_dim, images=False, prefix=f"{prefix}.norm1" if prefix else ""
        )
        self.conv1 = WanCausalConv3d(
            in_dim, out_dim, 3, padding=1, prefix=f"{prefix}.conv1" if prefix else ""
        )
        self.norm2 = WanRMSNorm(
            out_dim, images=False, prefix=f"{prefix}.norm2" if prefix else ""
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(
            out_dim, out_dim, 3, padding=1, prefix=f"{prefix}.conv2" if prefix else ""
        )
        self.conv_shortcut = (
            WanCausalConv3d(
                in_dim, out_dim, 1, prefix=f"{prefix}.conv_shortcut" if prefix else ""
            )
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        h = self.conv_shortcut(x)
        x = self.nonlinearity(self.norm1(x))
        x = _cached_conv(self.conv1, x, feat_cache, feat_idx)
        x = self.nonlinearity(self.norm2(x))
        x = self.dropout(x)
        x = _cached_conv(self.conv2, x, feat_cache, feat_idx)
        return x + h


class WanAttentionBlock(nn.Module):
    """Single-head spatial self-attention over H * W tokens per frame."""

    def __init__(self, dim: int, prefix: str = "") -> None:
        super().__init__()
        # images=True -> (dim,1,1)
        self.norm = WanRMSNorm(dim, prefix=f"{prefix}.norm" if prefix else "")
        self.to_qkv = Conv2d(
            dim, dim * 3, 1, prefix=f"{prefix}.to_qkv" if prefix else ""
        )
        self.proj = Conv2d(dim, dim, 1, prefix=f"{prefix}.proj" if prefix else "")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        qkv = (
            self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous()
        )
        q, k, v = qkv.chunk(3, dim=-1)

        # Head dim is 1024, larger then normal FA libs can do.
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x + identity


class WanMidBlock(nn.Module):
    def __init__(
        self, dim: int, dropout: float = 0.0, num_layers: int = 1, prefix: str = ""
    ) -> None:
        super().__init__()
        resnets = [
            WanResidualBlock(
                dim, dim, dropout, prefix=f"{prefix}.resnets.0" if prefix else ""
            )
        ]
        attentions = []
        for i in range(num_layers):
            attentions.append(
                WanAttentionBlock(
                    dim, prefix=f"{prefix}.attentions.{i}" if prefix else ""
                )
            )
            resnets.append(
                WanResidualBlock(
                    dim,
                    dim,
                    dropout,
                    prefix=f"{prefix}.resnets.{i + 1}" if prefix else "",
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache, feat_idx)
        return x


class WanResidualUpBlock(nn.Module):
    """is_residual up block: resnets (+ optional upsampler) + DupUp3D shortcut."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2
            )
        else:
            self.avg_shortcut = None
        resnets = []
        current = in_dim
        for j in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(
                    current,
                    out_dim,
                    dropout,
                    prefix=f"{prefix}.resnets.{j}" if prefix else "",
                )
            )
            current = out_dim
        self.resnets = nn.ModuleList(resnets)
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(
                out_dim,
                mode=mode,
                upsample_out_dim=out_dim,
                prefix=f"{prefix}.upsampler" if prefix else "",
            )
        else:
            self.upsampler = None

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)
        return x


def _cached_conv(
    conv: nn.Module, x: torch.Tensor, feat_cache: list | None, feat_idx: list | None
) -> torch.Tensor:
    """Run a WanCausalConv3d with the chunked feat_cache protocol (or plain)."""
    if feat_cache is None:
        return conv(x)
    idx = feat_idx[0]
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
        cache_x = torch.cat(
            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
            dim=2,
        )
    out = conv(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return out


class WanResidualDownBlock(nn.Module):
    """is_residual down block: resnets (+ optional downsampler) + AvgDown3D shortcut."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_downsample: bool = False,
        down_flag: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        current = in_dim
        for j in range(num_res_blocks):
            resnets.append(
                WanResidualBlock(
                    current,
                    out_dim,
                    dropout,
                    prefix=f"{prefix}.resnets.{j}" if prefix else "",
                )
            )
            current = out_dim
        self.resnets = nn.ModuleList(resnets)
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(
                out_dim, mode=mode, prefix=f"{prefix}.downsampler" if prefix else ""
            )
        else:
            self.downsampler = None

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x, feat_cache, feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache, feat_idx)
        return x + self.avg_shortcut(x_copy)


class WanEncoder3d(nn.Module):
    """The WAN encoder (is_residual): pixels(patchified) -> (mean, logvar) channels."""

    def __init__(
        self,
        in_channels: int = 12,
        dim: int = 160,
        z_dim: int = 96,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: tuple[bool, ...] = (False, True, True),
        dropout: float = 0.0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.nonlinearity = nn.SiLU()
        dims = [dim * u for u in [1] + list(dim_mult)]
        self.conv_in = WanCausalConv3d(
            in_channels,
            dims[0],
            3,
            padding=1,
            prefix=f"{prefix}.conv_in" if prefix else "",
        )
        self.down_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            down_flag = i != len(dim_mult) - 1
            self.down_blocks.append(
                WanResidualDownBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    dropout=dropout,
                    temperal_downsample=temperal_downsample[i] if down_flag else False,
                    down_flag=down_flag,
                    prefix=f"{prefix}.down_blocks.{i}" if prefix else "",
                )
            )
        self.mid_block = WanMidBlock(
            out_dim,
            dropout,
            num_layers=1,
            prefix=f"{prefix}.mid_block" if prefix else "",
        )
        self.norm_out = WanRMSNorm(
            out_dim, images=False, prefix=f"{prefix}.norm_out" if prefix else ""
        )
        self.conv_out = WanCausalConv3d(
            out_dim, z_dim, 3, padding=1, prefix=f"{prefix}.conv_out" if prefix else ""
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
    ) -> torch.Tensor:
        x = _cached_conv(self.conv_in, x, feat_cache, feat_idx)
        for layer in self.down_blocks:
            x = layer(x, feat_cache, feat_idx)
        x = self.mid_block(x, feat_cache, feat_idx)
        x = self.nonlinearity(self.norm_out(x))
        x = _cached_conv(self.conv_out, x, feat_cache, feat_idx)
        return x


class WanDecoder3d(nn.Module):
    """The WAN decoder (is_residual)."""

    def __init__(
        self,
        dim: int = 256,
        z_dim: int = 48,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_upsample: tuple[bool, ...] = (True, True, False),
        dropout: float = 0.0,
        out_channels: int = 12,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.nonlinearity = nn.SiLU()
        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult[::-1])]
        self.conv_in = WanCausalConv3d(
            z_dim, dims[0], 3, padding=1, prefix=f"{prefix}.conv_in" if prefix else ""
        )
        self.mid_block = WanMidBlock(
            dims[0],
            dropout,
            num_layers=1,
            prefix=f"{prefix}.mid_block" if prefix else "",
        )
        self.up_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = i != len(dim_mult) - 1
            self.up_blocks.append(
                WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    prefix=f"{prefix}.up_blocks.{i}" if prefix else "",
                )
            )
        self.norm_out = WanRMSNorm(
            out_dim, images=False, prefix=f"{prefix}.norm_out" if prefix else ""
        )
        self.conv_out = WanCausalConv3d(
            out_dim,
            out_channels,
            3,
            padding=1,
            prefix=f"{prefix}.conv_out" if prefix else "",
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list | None = None,
        feat_idx: list | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        x = _cached_conv(self.conv_in, x, feat_cache, feat_idx)
        x = self.mid_block(x, feat_cache, feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx, first_chunk=first_chunk)
        x = self.nonlinearity(self.norm_out(x))
        x = _cached_conv(self.conv_out, x, feat_cache, feat_idx)
        return x


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, C, T, H, W] -> [B, C*p*p, T, H//p, W//p] (inverse of :func:`unpatchify`)."""
    if patch_size == 1:
        return x
    b, c, frames, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"H ({height}) and W ({width}) must be divisible by patch_size {patch_size}."
        )
    x = x.view(
        b, c, frames, height // patch_size, patch_size, width // patch_size, patch_size
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(
        b,
        c * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[B, C*p*p, T, H, W] -> [B, C, T, H*p, W*p]."""
    if patch_size == 1:
        return x
    b, c_patches, frames, height, width = x.shape
    channels = c_patches // (patch_size * patch_size)
    x = x.view(b, channels, patch_size, patch_size, frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return x.view(b, channels, frames, height * patch_size, width * patch_size)


class Cosmos3WanVAE(nn.Module):
    """WAN VAE decode wrapper: latents -> pixels in [-1, 1]."""

    def __init__(self, config: Cosmos3WanVAEConfig) -> None:
        if not isinstance(config, Cosmos3WanVAEConfig):
            raise TypeError(
                f"Expected Cosmos3WanVAEConfig, got {type(config).__name__}"
            )
        super().__init__()
        z_dim = config.z_dim
        base_dim = config.base_dim
        decoder_base_dim = config.decoder_base_dim
        dim_mult = config.dim_mult
        num_res_blocks = config.num_res_blocks
        temperal_downsample = config.temperal_downsample
        out_channels = config.out_channels

        self.config = config
        self.z_dim = z_dim
        self.patch_size = config.patch_size
        self.scale_factor_temporal = config.scale_factor_temporal
        self.scale_factor_spatial = config.scale_factor_spatial
        self.post_quant_conv = WanCausalConv3d(
            z_dim, z_dim, 1, prefix="post_quant_conv"
        )
        self.encoder = WanEncoder3d(
            in_channels=out_channels,
            dim=base_dim,
            z_dim=z_dim * 2,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal_downsample=temperal_downsample,
            dropout=0.0,
            prefix="encoder",
        )
        self.quant_conv = WanCausalConv3d(z_dim * 2, z_dim * 2, 1, prefix="quant_conv")
        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal_upsample=tuple(temperal_downsample[::-1]),
            dropout=0.0,
            out_channels=out_channels,
            prefix="decoder",
        )
        mean = torch.tensor(
            list(config.latents_mean)
            if config.latents_mean is not None
            else [0.0] * z_dim,
            device=current_device(),
        )
        std = torch.tensor(
            list(config.latents_std)
            if config.latents_std is not None
            else [1.0] * z_dim,
            device=current_device(),
        )
        self.register_buffer(
            "latents_mean", mean.view(1, z_dim, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "latents_std", std.view(1, z_dim, 1, 1, 1), persistent=False
        )
        self._conv_num = sum(
            isinstance(m, WanCausalConv3d) for m in self.decoder.modules()
        )
        self._enc_conv_num = sum(
            isinstance(m, WanCausalConv3d) for m in self.encoder.modules()
        )

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``[B, z_dim, t_lat, h_lat, w_lat]`` -> ``[B, 3, T, H, W]`` in [-1, 1]."""
        z = latents * self.latents_std.to(latents) + self.latents_mean.to(latents)
        feat_map: list = [None] * self._conv_num
        conv_idx = [0]
        x = self.post_quant_conv(z)
        num_frame = x.shape[2]
        out = None
        for i in range(num_frame):
            conv_idx[0] = 0
            frame = self.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=feat_map,
                feat_idx=conv_idx,
                first_chunk=(i == 0),
            )
            out = frame if out is None else torch.cat([out, frame], dim=2)
        if self.patch_size != 1:
            out = unpatchify(out, self.patch_size)
        return torch.clamp(out, -1.0, 1.0)

    def _tile_grid(self, tp_size: int) -> tuple[int, int]:
        """Factor ``tp_size`` into a near-square ``(rows, cols)`` tile grid.

        Picks the divisor pair closest to square (``tp=4 -> 2x2``, ``tp=8 ->
        2x4``, ``tp=2 -> 1x2``); a prime ``tp_size`` degenerates to ``1 x tp``
        (rows-only). Every rank computes the same grid, so the split is consistent.
        """
        best = (1, tp_size)
        for rows in range(1, int(math.isqrt(tp_size)) + 1):
            if tp_size % rows == 0:
                best = (rows, tp_size // rows)
        return best

    @staticmethod
    def _tile_bounds(
        length: int, n: int, idx: int, halo: int
    ) -> tuple[int, int, int, int]:
        """Return ``(core_start, core_end, halo_start, halo_end)`` for tile ``idx``.

        ``core`` is the seam-free region this tile owns; ``[halo_start, halo_end)``
        is the actually-decoded slice (core widened by ``halo`` on each interior
        side). Integer division spreads a non-divisible ``length`` across ``n``
        tiles without gaps.
        """
        core_start = (idx * length) // n
        core_end = ((idx + 1) * length) // n
        halo_start = max(0, core_start - halo)
        halo_end = min(length, core_end + halo)
        return core_start, core_end, halo_start, halo_end

    @torch.no_grad()
    def decode_parallel(self, latents: torch.Tensor, *, halo: int = 2) -> torch.Tensor:
        import phyai.parallel as P

        def _axis(name: str) -> tuple[int, int]:
            """``(size, local_rank)`` for ``name``; ``(1, 0)`` if the axis is
            absent from the mesh (or no mesh). Guards both a missing mesh and a
            mesh whose ``mesh_dim_names`` doesn't include ``name`` (e.g. a
            tp-only test mesh has no ``cfg`` axis)."""
            try:
                m = P.default_mesh()
                names = m.torch_mesh.mesh_dim_names or ()
                if name not in names:
                    return 1, 0
                return m.axis_size(name), m.axis_local_rank(name)
            except Exception:
                return 1, 0

        tp_size, tp_rank = _axis("tp")
        cfg_size, cfg_rank = _axis("cfg")
        n = cfg_size * tp_size
        if n <= 1:
            return self.decode(latents)
        # Only the axes actually present and >1 carry the gather/reduce. ``tp`` is
        # the inner axis (fastest-varying in global_idx), ``cfg`` the outer one.
        gather_axes = [ax for ax, sz in (("tp", tp_size), ("cfg", cfg_size)) if sz > 1]

        _, _, _, h_lat, w_lat = latents.shape
        scale = self.scale_factor_spatial
        gr, gc = self._tile_grid(n)
        n_tiles = gr * gc

        # Global tile index over the (cfg, tp) rank grid — cfg outer, tp inner, to
        # match the gather reshape order below and the row-major mesh layout. Ranks
        # beyond the tile grid idle but still join every collective.
        global_idx = cfg_rank * tp_size + tp_rank
        active = global_idx < n_tiles
        gi, gj = (global_idx // gc, global_idx % gc) if active else (0, 0)
        _, _, hs, he = self._tile_bounds(h_lat, gr, gi, halo)
        _, _, ws, we = self._tile_bounds(w_lat, gc, gj, halo)

        if active:
            tile = latents[:, :, :, hs:he, ws:we].contiguous()
        else:
            tile = latents[:, :, :, :1, :1].contiguous()
        pixels = self.decode(tile)  # [B, 3, T, (he-hs)*scale, (we-ws)*scale]

        # Per-rank pixel tiles differ in H/W; pad to a common max shape (reconciled
        # over every participating axis) so they can be stacked into one all_gather
        # per axis, with the real extent + placement carried in an int metadata tensor.
        b, c, t, ph, pw = pixels.shape
        dev = pixels.device
        meta = torch.tensor(
            [int(active), hs * scale, ws * scale, ph, pw], dtype=torch.long, device=dev
        )
        max_hw = torch.tensor([ph, pw], dtype=torch.long, device=dev)
        for ax in gather_axes:
            max_hw = P.all_reduce(max_hw, axis=ax, op=_max_reduce_op())
        max_h, max_w = int(max_hw[0].item()), int(max_hw[1].item())

        buf = pixels.new_zeros(b, c, t, max_h, max_w)
        buf[:, :, :, :ph, :pw] = pixels
        # Stage the gather over each participating axis, inner (tp) first then outer
        # (cfg), so the flattened leading dim is in cfg-outer/tp-inner order and index
        # r matches each tile's global_idx.
        gathered = buf.unsqueeze(0)
        meta_all = meta.unsqueeze(0)
        for ax in gather_axes:
            gathered = P.all_gather(gathered.unsqueeze(0), axis=ax, dim=0)
            gathered = gathered.reshape(-1, *buf.shape)
            meta_all = P.all_gather(meta_all.unsqueeze(0), axis=ax, dim=0)
            meta_all = meta_all.reshape(-1, 5)

        full_h, full_w = h_lat * scale, w_lat * scale
        acc = pixels.new_zeros(b, c, t, full_h, full_w)
        wsum = pixels.new_zeros(1, 1, 1, full_h, full_w)
        feather = halo * scale  # pixel-space ramp width over the overlap
        for r in range(n):
            m = meta_all[r]
            if int(m[0].item()) == 0:
                continue  # idle rank placeholder
            py, px = int(m[1].item()), int(m[2].item())
            th, tw = int(m[3].item()), int(m[4].item())
            tpix = gathered[r, :, :, :, :th, :tw]
            w = _feather_weight(th, tw, feather, dev, pixels.dtype)
            acc[:, :, :, py : py + th, px : px + tw] += tpix * w
            wsum[:, :, :, py : py + th, px : px + tw] += w
        out = acc / wsum.clamp_min(1e-6)
        return torch.clamp(out, -1.0, 1.0)

    # Back-compat alias: the tensor-parallel-only entry point. ``decode_parallel``
    # subsumes it (cfg_size defaults to 1 when there is no cfg axis).
    decode_tp = decode_parallel

    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """``[B, 3, T, H, W]`` in [-1, 1] -> normalized latent ``[B, z_dim, t, h, w]``.

        Chunked causal encode (mirror of :meth:`decode`): patchify, run the
        encoder over 1+4·k frame chunks with the feat_cache protocol, then
        ``quant_conv`` -> ``(mean, logvar)``. ``sample=False`` returns the mean.
        The returned latent is normalized ``(z - latents_mean) / latents_std`` —
        the inverse of :meth:`decode`'s denormalization — so it round-trips.
        """
        x = patchify(pixels, self.patch_size)
        num_frame = x.shape[2]
        feat_map: list = [None] * self._enc_conv_num
        conv_idx = [0]
        out = None
        iter_ = 1 + (num_frame - 1) // self.scale_factor_temporal
        for i in range(iter_):
            conv_idx[0] = 0
            if i == 0:
                chunk = x[:, :, :1, :, :]
            else:
                chunk = x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]
            enc = self.encoder(chunk, feat_cache=feat_map, feat_idx=conv_idx)
            out = enc if out is None else torch.cat([out, enc], dim=2)
        mean, logvar = self.quant_conv(out).chunk(2, dim=1)
        if sample:
            std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
            eps = torch.randn(
                mean.shape, generator=generator, device=mean.device, dtype=mean.dtype
            )
            z = mean + eps * std
        else:
            z = mean
        return (z - self.latents_mean.to(z)) / self.latents_std.to(z)


def _vae_key_to_phyai(key: str) -> str:
    """Rewrite diffusers WanCausalConv3d ``.weight``/``.bias`` to the phyai ``.conv.`` leaf.

    diffusers WanCausalConv3d subclasses nn.Conv3d (params at ``<prefix>.weight``);
    the phyai version holds an inner ``conv`` (params at ``<prefix>.conv.weight``).
    Norm gamma/bias, attention Conv2d, and resample Conv2d keep their names.
    """
    for conv_leaf in (
        ".conv_in",
        ".conv_out",
        ".conv1",
        ".conv2",
        ".conv_shortcut",
        ".time_conv",
        "post_quant_conv",
        "quant_conv",
    ):
        for suffix in (".weight", ".bias"):
            target = f"{conv_leaf}{suffix}"
            if key.endswith(target):
                return key[: -len(suffix)] + ".conv" + suffix
    return key


def cosmos3_vae_weight_remap(key: str) -> str | None:
    """Map a diffusers ``AutoencoderKLWan`` checkpoint key to a phyai VAE param name.

    Fed to :func:`phyai.weights.load_pretrained` as its ``remap``. Returns the
    remapped name, or ``None`` to drop the key. Keeps the full VAE —
    ``encoder.*`` / ``quant_conv.*`` (encode) and ``decoder.*`` /
    ``post_quant_conv.*`` (decode); everything else is dropped. The
    ``WanCausalConv3d`` ``.weight``/``.bias`` leaves are rewritten to the inner
    ``.conv.`` param via :func:`_vae_key_to_phyai`.
    """
    if not (
        key.startswith("decoder.")
        or key.startswith("post_quant_conv.")
        or key.startswith("encoder.")
        or key.startswith("quant_conv.")
    ):
        return None
    return _vae_key_to_phyai(key)


__all__ = [
    "Cosmos3WanVAE",
    "WanDecoder3d",
    "WanEncoder3d",
    "cosmos3_vae_weight_remap",
]
