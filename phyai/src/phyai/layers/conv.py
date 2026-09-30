"""Conv{1,2,3}d wrappers tagged for the phyai loader system."""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.engine_config import get_engine_config
from phyai.kernel.call import CallSite, backend_preference, token_shape
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
        compute_dtype: torch.dtype | None = None,
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
        self.compute_dtype = compute_dtype

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

        # Cache higher-precision compute copies after checkpoint loading.
        self.register_buffer("_compute_weight", None, persistent=False)
        self.register_buffer("_compute_bias", None, persistent=False)

    def post_load(self) -> None:
        """Build or refresh compute-dtype parameter copies after loading."""
        if self.compute_dtype is None:
            return
        if self._compute_weight is None:
            self._compute_weight = self.weight.detach().to(self.compute_dtype)
        else:
            self._compute_weight.copy_(self.weight.detach())

        if self.bias is None:
            self._compute_bias = None
        elif self._compute_bias is None:
            self._compute_bias = self.bias.detach().to(self.compute_dtype)
        else:
            self._compute_bias.copy_(self.bias.detach())

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
        weight = self.weight
        bias = self.bias
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
            weight = self._compute_weight
            bias = self._compute_bias
            if weight is None:
                weight = self.weight.to(self.compute_dtype)
                bias = (
                    self.bias.to(self.compute_dtype) if self.bias is not None else None
                )
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return fn(
                x,
                weight,
                bias,
                self.stride,
                0,
                self.dilation,
                self.groups,
            )
        return fn(
            x,
            weight,
            bias,
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
        if self.compute_dtype is not None:
            s += f", compute_dtype={self.compute_dtype}"
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
    """2-D convolution for inference.

    ``compute_dtype`` runs the convolution with the input and derived parameter
    copies in that dtype without changing parameter storage. The output keeps
    the compute dtype.
    """

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
        compute_dtype: torch.dtype | None = None,
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
            compute_dtype=compute_dtype,
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


class CausalConv1d(Conv1d):
    """Depthwise causal Conv1d fused with an activation and an output split.

    The mixer pattern of the Qwen3.5 / Qwen3-Next GDN family: a grouped
    causal convolution over token-major activations, the activation applied
    in the same pass, and the result split into (query, key, value) widths.
    Executed through the ``causal_conv`` catalog op, so the Triton fused
    kernel and the torch reference are selected the same way as every other
    layer — this class only owns the weight and describes the call.

    ``forward(x)`` takes ``(batch, seq, channels)`` token-major activations
    and returns one tensor per entry of ``split_sizes``, each
    ``(batch, seq, width)``. The convolution is causal: position ``t`` sees
    positions ``t - kernel_size + 1 .. t``.

    The weight is stored exactly like :class:`torch.nn.Conv1d` with
    ``groups=channels`` — shape ``(channels, 1, kernel_size)`` under
    ``{prefix}.weight`` — so checkpoints load unchanged.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        split_sizes: Tuple[int, ...],
        activation: str = "silu",
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
        backend: str | None = None,
        kernel_role: str = "causal_conv",
    ) -> None:
        kernel_size = int(kernel_size)
        super().__init__(
            channels,
            channels,
            kernel_size,
            padding=kernel_size - 1,
            groups=channels,
            bias=False,
            dtype=dtype,
            device=device,
            prefix=prefix,
        )
        self.split_sizes = tuple(int(size) for size in split_sizes)
        if sum(self.split_sizes) != channels:
            raise ValueError(
                f"CausalConv1d: split_sizes {self.split_sizes} must sum to "
                f"channels={channels}."
            )
        self.activation = activation
        self.kernel_role = kernel_role
        self._prefer = backend_preference("causal_conv", backend)
        self._call = CallSite(
            "causal_conv",
            role=kernel_role,
            prefer=self._prefer,
            dims={"channels": channels, "kernel": kernel_size},
            attrs={
                "activation": activation,
                "split": "qkv"
                if len(self.split_sizes) == 3
                else str(len(self.split_sizes)),
            },
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        handle = self._call.select(
            device=x.device,
            dtype={"input": x.dtype},
            dims=token_shape(x),
        )
        return handle.execute(
            x.contiguous(),
            self.weight.contiguous(),
            self.split_sizes,
        )


__all__ = ["CausalConv1d", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d"]
