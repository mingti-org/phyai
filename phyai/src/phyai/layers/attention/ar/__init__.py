"""`phyai.layers.attention.ar` — autoregressive LM-side paged attention."""

from __future__ import annotations

from phyai.layers.attention.ar.backends import (
    FlashInferARBackend,
    FlashInferARPlan,
)
from phyai.layers.attention.ar.base import (
    ARAttentionBackend,
    ARAttentionLayerProto,
    ARAttnCtx,
    ARAttnMetadata,
    ARAttnPlanHandle,
)
from phyai.layers.attention.ar.layer import ARAttention
from phyai.layers.attention.ar.registry import (
    BackendFactory,
    get_backend_factory,
    list_backends,
    register_backend,
)


__all__ = [
    "ARAttention",
    "ARAttentionBackend",
    "ARAttentionLayerProto",
    "ARAttnCtx",
    "ARAttnMetadata",
    "ARAttnPlanHandle",
    "BackendFactory",
    "FlashInferARBackend",
    "FlashInferARPlan",
    "get_backend_factory",
    "list_backends",
    "register_backend",
]
