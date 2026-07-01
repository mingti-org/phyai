"""phyai.layers.quant — op-agnostic weight-storage specs.

A :class:`WeightSpec` describes how a parameter is laid out on disk and
in memory: dtype, scales, granularity. Op-specific concerns (activation
quantisation for linear matmul, MoE grouping, …) live on separate
Protocols so a single spec class can cleanly serve multiple ops.

Public surface::

    from phyai.layers.quant import (
        AllocationRequest, WeightSpec,                   # base
        Granularity,                                     # scale layout enum
        Bf16Spec, Fp8Spec, Nvfp4Spec,                    # concrete specs
        ActivationView, LinearActivationQuant,           # linear-only mixin
    )
"""

from __future__ import annotations

from phyai.layers.quant.base import AllocationRequest, WeightSpec
from phyai.layers.quant.bf16 import Bf16Spec
from phyai.layers.quant.fp8 import Fp8Spec
from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.linear import ActivationView, LinearActivationQuant
from phyai.layers.quant.nvfp4 import Nvfp4Spec

__all__ = [
    "AllocationRequest",
    "WeightSpec",
    "Granularity",
    "Bf16Spec",
    "Fp8Spec",
    "Nvfp4Spec",
    "ActivationView",
    "LinearActivationQuant",
]
