"""Back-compat shim — the spec abstraction has moved to :mod:`phyai.layers.quant`.

Existing imports such as
``from phyai.layers.linear.spec import Bf16Spec, Fp8Spec, Nvfp4Spec, ActivationView``
keep working; new code should import from :mod:`phyai.layers.quant`.
"""

from __future__ import annotations

from phyai.layers.quant import (
    ActivationView,
    AllocationRequest,
    Bf16Spec,
    Fp8Spec,
    LinearActivationQuant,
    Nvfp4Spec,
    WeightSpec,
)
from phyai.layers.quant.fp8 import _convert_to_channelwise

__all__ = [
    "ActivationView",
    "AllocationRequest",
    "Bf16Spec",
    "Fp8Spec",
    "Nvfp4Spec",
    "LinearActivationQuant",
    "WeightSpec",
    "_convert_to_channelwise",
]
