"""Registered Gated Delta Net backends."""

from phyai.layers.attention.gdn.backends.fla import (
    FlaGatedDeltaNetBackend,
    FlaGatedDeltaNetPlan,
)
from phyai.layers.attention.gdn.backends.flashinfer import (
    FlashInferGatedDeltaNetBackend,
    FlashInferGatedDeltaNetPlan,
)


__all__ = [
    "FlaGatedDeltaNetBackend",
    "FlaGatedDeltaNetPlan",
    "FlashInferGatedDeltaNetBackend",
    "FlashInferGatedDeltaNetPlan",
]
