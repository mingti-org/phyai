"""Gated Delta Net with backend dispatch."""

from __future__ import annotations

from phyai.layers.attention.gdn.backends import (
    FlaGatedDeltaNetBackend,
    FlaGatedDeltaNetPlan,
    FlashInferGatedDeltaNetBackend,
    FlashInferGatedDeltaNetPlan,
)
from phyai.layers.attention.gdn.base import (
    GatedDeltaNetBackend,
    GatedDeltaNetCtx,
    GatedDeltaNetLayerProto,
    GatedDeltaNetMetadata,
    GatedDeltaNetPlanHandle,
)
from phyai.layers.attention.gdn.layer import GatedDeltaNet
from phyai.layers.attention.gdn.registry import (
    BackendFactory,
    get_backend_factory,
    list_backends,
    register_backend,
)


__all__ = [
    "BackendFactory",
    "FlaGatedDeltaNetBackend",
    "FlaGatedDeltaNetPlan",
    "FlashInferGatedDeltaNetBackend",
    "FlashInferGatedDeltaNetPlan",
    "GatedDeltaNet",
    "GatedDeltaNetBackend",
    "GatedDeltaNetCtx",
    "GatedDeltaNetLayerProto",
    "GatedDeltaNetMetadata",
    "GatedDeltaNetPlanHandle",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]
