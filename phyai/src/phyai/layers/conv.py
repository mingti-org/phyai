"""Conv{1,2,3}d wrappers tagged for the phyai loader system."""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.engine_config import get_engine_config
from phyai.weights.shards import replicated, weight_norm_fold

_size_1_t = Union[int, Tuple[int]]
_size_2_t = Union[int, Tuple[int, int]]
_size_3_t = Union[int, Tuple[int, int, int]]

_VALID_PADDING_MODES = ("zeros", "reflect", "replicate", "circular")
_VALID_PADDING_STRINGS = ("same", "valid")


def _ntuple(n: int, x: int | Tuple[int, ...]) -> Tuple[int, ...]:
    """Coerce an int or n-tuple of ints into a tuple of length ``n``."""
    if isinstance(x, int):
        return tuple([x] * n)
    t = tuple(x)
    if len(t) != n:
        raise ValueError(f"expected a length-{n} sequence, got {t!r}")
    return t


def _attach_conv_loaders(
    weight: nn.Parameter,
    bias: nn.Parameter | None,
    prefix: str,
    weight_norm: bool,
) -> None:
    """Tag a conv's ``weight``/``bias`` with loader metadata (no-op if no prefix).

    ``weight_norm=True`` expects a legacy ``weight_norm`` checkpoint — the weight
    arrives split as ``<prefix>.weight_g`` / ``<prefix>.weight_v`` and is folded into
    the single dense ``weight`` at load time via
    :func:`phyai.weights.shards.weight_norm_fold`. Otherwise the weight loads whole
    from ``<prefix>.weight``. Bias (if any) always loads whole from ``<prefix>.bias``.
    """
    if not prefix:
        return
    if weight_norm:
        weight.hf_keys = [(f"{prefix}.weight_g", "g"), (f"{prefix}.weight_v", "v")]
        weight.weight_loader = weight_norm_fold()
    else:
        weight.hf_keys = [(f"{prefix}.weight", None)]
        weight.weight_loader = replicated()
    if bias is not None:
        bias.hf_keys = [(f"{prefix}.bias", None)]
        bias.weight_loader = replicated()


class _ConvNd(nn.Module):
    """Shared state for Conv{1,2,3}d.

    Subclasses set :attr:`_ndim` and call :meth:`_conv` from their
    ``forward`` with the matching ``F.conv{1,2,3}d``. The weight has the
    canonical PyTorch layout
    ``(out_channels, in_channels // groups, *kernel_size)``, so HuggingFace
    / ``nn.Conv*`` checkpoints copy in straight through a replicated
    :func:`phyai.weights.shards.replicated` loader.
    """

    _ndim: int

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...] | str,
        dilation: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
        prefix: str = "",
        weight_norm: bool = False,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError(f"groups must be >= 1, got {groups}")
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels={in_channels} not divisible by groups={groups}"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels={out_channels} not divisible by groups={groups}"
            )
        if padding_mode not in _VALID_PADDING_MODES:
            raise ValueError(
                f"padding_mode={padding_mode!r} not in {_VALID_PADDING_MODES!r}"
            )
        if isinstance(padding, str):
            if padding not in _VALID_PADDING_STRINGS:
                raise ValueError(
                    f"padding={padding!r} not in {_VALID_PADDING_STRINGS!r}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is incompatible with strided convolutions"
                )
        if device is None:
            device = get_engine_config().device.target

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.prefix = prefix

        # F.pad takes pads in reverse axis order with each axis getting
        # (left, right). Only used when padding_mode != "zeros", but it's
        # also the only way to spell padding="same" out for non-zero modes,
        # so precompute it for strings too.
        self._reversed_padding_repeated_twice: tuple[int, ...] = (
            self._build_reversed_pad(padding, kernel_size, dilation)
        )

        weight_shape = (out_channels, in_channels // groups) + tuple(kernel_size)
        self.weight = nn.Parameter(
            torch.empty(weight_shape, dtype=dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        _attach_conv_loaders(self.weight, self.bias, prefix, weight_norm)

    @staticmethod
    def _build_reversed_pad(
        padding: tuple[int, ...] | str,
        kernel_size: tuple[int, ...],
        dilation: tuple[int, ...],
    ) -> tuple[int, ...]:
        n = len(kernel_size)
        out = [0] * (2 * n)
        if isinstance(padding, str):
            if padding == "same":
                for d, k, i in zip(dilation, kernel_size, range(n - 1, -1, -1)):
                    total = d * (k - 1)
                    left = total // 2
                    out[2 * i] = left
                    out[2 * i + 1] = total - left
            return tuple(out)
        for i in range(n):
            out[2 * (n - 1 - i)] = padding[i]
            out[2 * (n - 1 - i) + 1] = padding[i]
        return tuple(out)

    def _conv(self, fn, x: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return fn(
                x,
                self.weight,
                self.bias,
                self.stride,
                0,
                self.dilation,
                self.groups,
            )
        return fn(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != tuple([0] * self._ndim):
            s += f", padding={self.padding!r}"
        if self.dilation != tuple([1] * self._ndim):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class Conv1d(_ConvNd):
    """1-D convolution. Mirrors :class:`torch.nn.Conv1d` for inference."""

    _ndim = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t | str = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
        weight_norm: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            _ntuple(1, kernel_size),
            _ntuple(1, stride),
            padding if isinstance(padding, str) else _ntuple(1, padding),
            _ntuple(1, dilation),
            groups,
            bias,
            padding_mode,
            dtype,
            device,
            prefix=prefix,
            weight_norm=weight_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(F.conv1d, x)


class Conv2d(_ConvNd):
    """2-D convolution. Mirrors :class:`torch.nn.Conv2d` for inference."""

    _ndim = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
        weight_norm: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            _ntuple(2, kernel_size),
            _ntuple(2, stride),
            padding if isinstance(padding, str) else _ntuple(2, padding),
            _ntuple(2, dilation),
            groups,
            bias,
            padding_mode,
            dtype,
            device,
            prefix=prefix,
            weight_norm=weight_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(F.conv2d, x)


class Conv3d(_ConvNd):
    """3-D convolution. Mirrors :class:`torch.nn.Conv3d` for inference."""

    _ndim = 3

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t | str = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
        weight_norm: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            _ntuple(3, kernel_size),
            _ntuple(3, stride),
            padding if isinstance(padding, str) else _ntuple(3, padding),
            _ntuple(3, dilation),
            groups,
            bias,
            padding_mode,
            dtype,
            device,
            prefix=prefix,
            weight_norm=weight_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(F.conv3d, x)


class ConvTranspose1d(nn.Module):
    """1-D transposed convolution. Mirrors :class:`torch.nn.ConvTranspose1d` for inference.

    Weight layout is ``(in_channels, out_channels // groups, kernel_size)`` — the
    transposed-conv convention, which differs from :class:`_ConvNd` — so
    ``nn.ConvTranspose1d`` / HuggingFace checkpoints copy straight in. Like
    :class:`Conv1d` it tags each parameter with a :func:`replicated` loader (or a
    :func:`weight_norm_fold` loader when ``weight_norm=True``); no TP sharding, no
    kernel dispatch — just ``F.conv_transpose1d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
        weight_norm: bool = False,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError(f"groups must be >= 1, got {groups}")
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels={in_channels} not divisible by groups={groups}"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels={out_channels} not divisible by groups={groups}"
            )
        if device is None:
            device = get_engine_config().device.target

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _ntuple(1, kernel_size)
        self.stride = _ntuple(1, stride)
        self.padding = _ntuple(1, padding)
        self.output_padding = _ntuple(1, output_padding)
        self.dilation = _ntuple(1, dilation)
        self.groups = groups
        self.prefix = prefix

        weight_shape = (in_channels, out_channels // groups) + self.kernel_size
        self.weight = nn.Parameter(
            torch.empty(weight_shape, dtype=dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        _attach_conv_loaders(self.weight, self.bias, prefix, weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose1d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != (0,):
            s += f", padding={self.padding}"
        if self.output_padding != (0,):
            s += f", output_padding={self.output_padding}"
        if self.dilation != (1,):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        return s


__all__ = ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d"]
