"""`phyai.layers.attention.diffusion` — diffusion / action-expert paged attention."""

from __future__ import annotations

from phyai.layers.attention.diffusion.backends import (
    FlashInferDiffusionBackend,
    FlashInferDiffusionPlan,
)
from phyai.layers.attention.diffusion.base import (
    DiffusionAttentionBackend,
    DiffusionAttentionLayerProto,
    DiffusionAttnCtx,
    DiffusionAttnMetadata,
    DiffusionAttnPlanHandle,
)
from phyai.layers.attention.diffusion.layer import DiffusionAttention
from phyai.layers.attention.diffusion.registry import (
    BackendFactory,
    get_backend_factory,
    list_backends,
    register_backend,
)


__all__ = [
    "BackendFactory",
    "DiffusionAttention",
    "DiffusionAttentionBackend",
    "DiffusionAttentionLayerProto",
    "DiffusionAttnCtx",
    "DiffusionAttnMetadata",
    "DiffusionAttnPlanHandle",
    "FlashInferDiffusionBackend",
    "FlashInferDiffusionPlan",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]
