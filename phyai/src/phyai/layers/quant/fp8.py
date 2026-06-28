"""Fp8Spec — ``torch.float8_e4m3fn`` weight with configurable scale granularity.

Linear-only today: ``allocate`` reads ``weight_shape`` as ``(out_per_rank,
in_per_rank)`` and the per-tensor / per-channel / block scale shapes are
all derived under that 2-D assumption. Implements both
:class:`WeightSpec` (storage) and
:class:`phyai.layers.quant.linear.LinearActivationQuant` (activation
quant hook).

``granularity`` selects between:

* ``PER_TENSOR``  — one scalar per logical matrix (fanned out to
  per-channel by :meth:`process_after_loading`) + one static
  activation scale.
* ``PER_CHANNEL`` — per-output-row weight scale, per-token (rowwise)
  activation scale computed at runtime.
* ``BLOCK``       — ``(out_per_rank // block_n, in_per_rank // block_k)``
  weight scale, per-token activation scale at block-K granularity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from phyai.layers.quant.base import AllocationRequest
from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.linear import ActivationView


_FP8_E4M3_AMAX = 448.0


def _convert_to_channelwise(
    scale_per_logical: torch.Tensor,
    logical_widths: list[int],
) -> torch.Tensor:
    """Fan a ``[L]`` per-logical-matrix scale out to ``[sum(widths)]`` per-channel."""
    return torch.cat(
        [scale_per_logical[i].expand(w) for i, w in enumerate(logical_widths)]
    )


@dataclass
class Fp8Spec:
    granularity: Granularity = Granularity.PER_CHANNEL
    block_shape: tuple[int, int] | None = None
    weight_dtype: torch.dtype = torch.float8_e4m3fn
    needs_act_quant: bool = True

    def __post_init__(self) -> None:
        if self.granularity == Granularity.BLOCK and self.block_shape is None:
            raise ValueError("Fp8Spec(granularity=BLOCK) requires block_shape")

    @property
    def spec_id(self) -> str:
        if self.granularity == Granularity.BLOCK:
            assert self.block_shape is not None
            bn, bk = self.block_shape
            return f"fp8_block_{bn}_{bk}"
        return f"fp8_{self.granularity.value}"

    def allocate(self, layer: nn.Module, request: AllocationRequest) -> None:
        if len(request.weight_shape) != 2:
            raise ValueError(
                f"Fp8Spec.allocate expects a 2-D weight_shape (N, K), "
                f"got {request.weight_shape!r}"
            )
        out_per_rank, in_per_rank = request.weight_shape
        device = request.device

        layer.weight = nn.Parameter(
            torch.empty(
                out_per_rank, in_per_rank, dtype=self.weight_dtype, device=device
            ),
            requires_grad=False,
        )
        # TODO(quant-scales-from-disk): weight_scale / input_scale are
        # initialised to ones below and never loaded from disk. A future
        # fp8-quantised checkpoint will need extra placements (declared
        # by either this spec or the layer) to land their scales.

        if self.granularity == Granularity.PER_TENSOR:
            layer.weight_scale = nn.Parameter(
                torch.ones(
                    len(request.logical_widths),
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )
            layer.input_scale = nn.Parameter(
                torch.ones(1, dtype=torch.float32, device=device),
                requires_grad=False,
            )
        elif self.granularity == Granularity.PER_CHANNEL:
            layer.weight_scale = nn.Parameter(
                torch.ones(out_per_rank, dtype=torch.float32, device=device),
                requires_grad=False,
            )
        elif self.granularity == Granularity.BLOCK:
            assert self.block_shape is not None
            bn, bk = self.block_shape
            if out_per_rank % bn != 0:
                raise ValueError(
                    f"Fp8Spec(BLOCK): out_per_rank={out_per_rank} "
                    f"not divisible by block_n={bn}"
                )
            if in_per_rank % bk != 0:
                raise ValueError(
                    f"Fp8Spec(BLOCK): in_per_rank={in_per_rank} "
                    f"not divisible by block_k={bk}"
                )
            layer.weight_scale = nn.Parameter(
                torch.ones(
                    out_per_rank // bn,
                    in_per_rank // bk,
                    dtype=torch.float32,
                    device=device,
                ),
                requires_grad=False,
            )

        layer.logical_widths = list(request.logical_widths)

    def process_after_loading(self, layer: nn.Module) -> None:
        if self.granularity == Granularity.PER_TENSOR:
            # Fan one scalar per logical matrix out to a per-channel vector
            # so kernels can consume a uniform scale layout.
            layer.weight_scale = nn.Parameter(
                _convert_to_channelwise(layer.weight_scale, layer.logical_widths),
                requires_grad=False,
            )

    def quantize_activation(
        self,
        x: torch.Tensor,
        layer: nn.Module,
    ) -> ActivationView:
        if self.granularity == Granularity.PER_TENSOR:
            x_q = (
                (x.float() / layer.input_scale)
                .clamp(-_FP8_E4M3_AMAX, _FP8_E4M3_AMAX)
                .to(torch.float8_e4m3fn)
            )
            return ActivationView(x_q, layer.input_scale, Granularity.PER_TENSOR)

        if self.granularity in (Granularity.PER_CHANNEL, Granularity.BLOCK):
            # Per-token (rowwise) activation quant.
            x_amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            x_scale = (x_amax / _FP8_E4M3_AMAX).to(torch.float32)
            x_q = (x / x_scale).to(torch.float8_e4m3fn)
            return ActivationView(x_q, x_scale, self.granularity)

        raise RuntimeError(f"unhandled Fp8Spec granularity {self.granularity!r}")


__all__ = ["Fp8Spec"]
