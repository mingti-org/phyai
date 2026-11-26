"""`phyai.layers.attention.paged` — unified paged-KV attention."""

from __future__ import annotations

from phyai.layers.attention.paged.backends import (
    FlashInferPagedBackend,
    FlashInferPagedPlan,
    SdpaPagedBackend,
    SdpaPagedPlan,
)
from phyai.layers.attention.paged.base import (
    PagedAttentionBackend,
    PagedAttentionLayerProto,
    PagedAttnCtx,
    PagedAttnMetadata,
    PagedAttnPlanHandle,
)
from phyai.layers.attention.paged.layer import PagedAttention


__all__ = [
    "FlashInferPagedBackend",
    "FlashInferPagedPlan",
    "SdpaPagedBackend",
    "SdpaPagedPlan",
    "PagedAttention",
    "PagedAttentionBackend",
    "PagedAttentionLayerProto",
    "PagedAttnCtx",
    "PagedAttnMetadata",
    "PagedAttnPlanHandle",
]
