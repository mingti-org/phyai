"""Numerical core for weight quantization — CPU-testable, convention-matched.

compressed-tensors is the authority for the quantization convention, so we delegate
scale/zero-point computation to its ``calculate_qparams``: RTN, GPTQ, and
serialization then share identical numerics — a fake-quant weight sits exactly on
the grid CT's packer expects, and re-quantizing it reproduces canonical codes.

Weight layout: 2-D ``(out_features, in_features)`` == ``(N, K)``; grouping is along K.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import calculate_qparams

_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0


@dataclass(frozen=True)
class WeightQuant:
    """Semantic weight-quant spec; the subset of a CT ``QuantizationArgs`` this
    toolkit emits."""

    num_bits: int
    is_float: bool = False
    symmetric: bool = True
    group_size: int = 0
    is_e5m2: bool = False

    @property
    def strategy(self) -> str:
        return "group" if self.group_size and self.group_size > 0 else "channel"

    def int_range(self) -> tuple[int, int]:
        return -(1 << (self.num_bits - 1)), (1 << (self.num_bits - 1)) - 1

    def __post_init__(self) -> None:
        if self.is_float and self.num_bits != 8:
            raise ValueError("float weight quant only supports num_bits=8 (fp8)")
        if self.is_float and self.group_size and self.group_size > 0:
            # note(chenghua): fp8 in compressed-tensors is per-tensor/channel; a grouped
            # fp8 scale is not the canonical layout phyai/humming read.
            raise ValueError("fp8 weight quant must be per-channel (group_size=0)")
        if self.num_bits < 2 or self.num_bits > 8:
            raise ValueError(f"num_bits must be in [2, 8], got {self.num_bits}")


def to_ct_args(q: WeightQuant) -> QuantizationArgs:
    strategy = (
        QuantizationStrategy.GROUP
        if q.strategy == "group"
        else QuantizationStrategy.CHANNEL
    )
    kwargs: dict = {
        "num_bits": q.num_bits,
        "type": QuantizationType.FLOAT if q.is_float else QuantizationType.INT,
        "symmetric": q.symmetric,
        "strategy": strategy,
    }
    if strategy == QuantizationStrategy.GROUP:
        kwargs["group_size"] = q.group_size
    return QuantizationArgs(**kwargs)


def _grouped_view(w: torch.Tensor, group_size: int) -> torch.Tensor:
    n, k = w.shape
    if group_size <= 0 or group_size == k:
        return w.view(n, 1, k)
    if k % group_size != 0:
        raise ValueError(f"in_features={k} not divisible by group_size={group_size}")
    return w.view(n, k // group_size, group_size)


def _minmax(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = _grouped_view(w.float(), group_size)
    return g.amin(dim=-1), g.amax(dim=-1)


def compute_scale_zp(
    w: torch.Tensor, q: WeightQuant
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return ``(scale, zero_point)`` of shape ``(N, num_groups)`` in CT's
    convention. ``zero_point`` is ``None`` for symmetric / fp8."""
    if q.is_float:
        fp8_max = _FP8_E5M2_MAX if q.is_e5m2 else _FP8_E4M3_MAX
        g = _grouped_view(w.float(), q.group_size)
        scale = (g.abs().amax(dim=-1) / fp8_max).clamp_min(1e-12)
        return scale, None
    wmin, wmax = _minmax(w, q.group_size)
    scale, zp = calculate_qparams(wmin, wmax, to_ct_args(q))
    if q.symmetric:
        return scale.float(), None
    return scale.float(), zp.float()


def _repeat_to_k(t: torch.Tensor, k: int) -> torch.Tensor:
    n, g = t.shape
    return t.unsqueeze(-1).expand(n, g, k // g).reshape(n, k)


def apply_fake_quant(
    w: torch.Tensor, q: WeightQuant, scale: torch.Tensor, zp: torch.Tensor | None
) -> torch.Tensor:
    """Elementwise fake-quant in CT convention (dequant of quant)."""
    wf = w.float()
    n, k = wf.shape
    s = _repeat_to_k(scale, k)
    if q.is_float:
        fp8_dtype = torch.float8_e5m2 if q.is_e5m2 else torch.float8_e4m3fn
        return (wf / s).to(fp8_dtype).float() * s
    qmin, qmax = q.int_range()
    if zp is None:
        codes = torch.clamp(torch.round(wf / s), qmin, qmax)
        return codes * s
    z = _repeat_to_k(zp, k)
    codes = torch.clamp(torch.round(wf / s) + z, qmin, qmax)
    return (codes - z) * s


def fake_quantize(
    w: torch.Tensor,
    q: WeightQuant,
    scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
) -> torch.Tensor:
    if scale is None:
        scale, zero_point = compute_scale_zp(w, q)
    return apply_fake_quant(w, q, scale, zero_point).to(w.dtype)


def quantize_to_codes(
    w: torch.Tensor,
    q: WeightQuant,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
) -> torch.Tensor:
    """Signed integer codes in CT convention (range ``[-2^(b-1), 2^(b-1)-1]``)."""
    if q.is_float:
        raise ValueError("quantize_to_codes is int-only")
    wf = w.float()
    n, k = wf.shape
    s = _repeat_to_k(scale, k)
    qmin, qmax = q.int_range()
    if zero_point is None:
        return torch.clamp(torch.round(wf / s), qmin, qmax).to(torch.int32)
    z = _repeat_to_k(zero_point, k)
    return torch.clamp(torch.round(wf / s) + z, qmin, qmax).to(torch.int32)


__all__ = [
    "WeightQuant",
    "to_ct_args",
    "compute_scale_zp",
    "fake_quantize",
    "apply_fake_quant",
    "quantize_to_codes",
]
