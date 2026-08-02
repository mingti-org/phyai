"""Lower a semantic :class:`QuantScheme` to a physical ``WeightSpec``."""

from __future__ import annotations

from phyai.layers.quant.bf16 import Bf16Spec
from phyai.layers.quant.fp8 import Fp8Spec
from phyai.layers.quant.nvfp4 import Nvfp4Spec
from phyai.layers.quant.scheme import QDType, QuantScheme

_NVFP4_128X4_MIN_SM = 100


def materialize(scheme: QuantScheme, sm: int) -> object:
    """Return the physical ``WeightSpec`` for ``scheme`` on an SM-``sm`` device."""
    w = scheme.weight
    if w.dtype is QDType.BF16:
        return Bf16Spec()
    if w.dtype is QDType.FP8_E4M3:
        return Fp8Spec(granularity=w.granularity, block_shape=w.block_shape)
    if w.dtype is QDType.NVFP4:
        layout = "128x4" if sm >= _NVFP4_128X4_MIN_SM else "linear"
        return Nvfp4Spec(scale_layout=layout)
    raise NotImplementedError(f"materialize: unsupported weight dtype {w.dtype!r}")


__all__ = ["materialize"]
