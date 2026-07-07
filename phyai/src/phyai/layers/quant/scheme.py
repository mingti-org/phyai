"""Semantic quantization IR — what a tensor is quantized to, nothing else.

A :class:`QuantScheme` is pure semantics: it carries no scale layout,
kernel, or device decision. Those are lowered by
:func:`phyai.layers.quant.materialize.materialize` into a physical
:class:`phyai.layers.quant.base.WeightSpec`. Weight and activation are
described by the same :class:`TensorQuant`, so ``input=None`` cleanly
means weight-only (e.g. W4A16).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from phyai.layers.quant.granularity import Granularity


class QDType(Enum):
    """Quantized element type. ``BF16`` is the sentinel for 'not quantized'."""

    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8 = "int8"
    INT4 = "int4"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"


@dataclass(frozen=True)
class TensorQuant:
    """How one tensor (weight OR activation) is quantized.

    ``dynamic`` is meaningful only for activations (True = scale computed
    at runtime); it is always False for weights. ``micro_scaled`` marks
    block-microscaled formats (NVFP4/MXFP4: an in-block low-precision
    scale plus an outer global scale). ``block_shape`` is set for
    block-granularity weights. ``group_size`` is intentionally not
    modelled yet (no group/int4 path in this slice).
    """

    dtype: QDType
    granularity: Granularity
    symmetric: bool = True
    dynamic: bool = False
    micro_scaled: bool = False
    block_shape: tuple[int, int] | None = None


@dataclass(frozen=True)
class QuantScheme:
    """The complete quantization decision for one layer."""

    weight: TensorQuant
    input: TensorQuant | None = None
    online: bool = False

    @property
    def weight_only(self) -> bool:
        return self.input is None


__all__ = ["QDType", "TensorQuant", "QuantScheme"]
