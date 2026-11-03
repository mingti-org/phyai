"""phyai.layers.attention"""

from __future__ import annotations

from phyai.layers.attention.enums import AttnLayout, AttnMode
from phyai.layers.attention.mask import AttnMask
from phyai.layers.attention.gdn import (
    FlaGatedDeltaNetBackend,
    FlaGatedDeltaNetPlan,
    FlashInferGatedDeltaNetBackend,
    FlashInferGatedDeltaNetPlan,
    FlashQlaGatedDeltaNetBackend,
    FlashQlaGatedDeltaNetPlan,
    GatedDeltaNet,
    GatedDeltaNetBackend,
    GatedDeltaNetCtx,
    GatedDeltaNetLayerProto,
    GatedDeltaNetMetadata,
    GatedDeltaNetPlanHandle,
)
from phyai.layers.attention.nocache import (
    Attention,
    AttentionBackend,
    AttentionLayerProto,
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
    EagerAttentionBackend,
    EagerAttentionPlan,
    FlashInferAttentionBackend,
    FlashInferAttentionPlan,
    SdpaAttentionBackend,
    SdpaAttentionPlan,
)
from phyai.layers.attention.paged import (
    FlashInferPagedBackend,
    FlashInferPagedPlan,
    PagedAttention,
    PagedAttentionBackend,
    PagedAttentionLayerProto,
    PagedAttnCtx,
    PagedAttnMetadata,
    PagedAttnPlanHandle,
    SdpaPagedBackend,
    SdpaPagedPlan,
)
from phyai.layers.attention.utils import (
    get_global_fi_workspace,
    register_global_fi_workspace,
    resolve_workspace_bytes,
)


__all__ = [
    # === Layers ===
    "Attention",
    "PagedAttention",
    "GatedDeltaNet",
    # === Shared enums / mask ===
    "AttnLayout",
    "AttnMode",
    "AttnMask",
    # === nocache/ stack ===
    "AttentionBackend",
    "AttentionLayerProto",
    "AttnCtx",
    "AttnMetadata",
    "AttnPlanHandle",
    "EagerAttentionBackend",
    "EagerAttentionPlan",
    "FlashInferAttentionBackend",
    "FlashInferAttentionPlan",
    "SdpaAttentionBackend",
    "SdpaAttentionPlan",
    # === paged/ stack ===
    "PagedAttentionBackend",
    "PagedAttentionLayerProto",
    "PagedAttnCtx",
    "PagedAttnMetadata",
    "PagedAttnPlanHandle",
    "FlashInferPagedBackend",
    "FlashInferPagedPlan",
    "SdpaPagedBackend",
    "SdpaPagedPlan",
    # === gdn/ stack ===
    "GatedDeltaNetBackend",
    "GatedDeltaNetCtx",
    "GatedDeltaNetLayerProto",
    "GatedDeltaNetMetadata",
    "GatedDeltaNetPlanHandle",
    "FlaGatedDeltaNetBackend",
    "FlaGatedDeltaNetPlan",
    "FlashInferGatedDeltaNetBackend",
    "FlashInferGatedDeltaNetPlan",
    "FlashQlaGatedDeltaNetBackend",
    "FlashQlaGatedDeltaNetPlan",
    # === Workspace ===
    "get_global_fi_workspace",
    "register_global_fi_workspace",
    "resolve_workspace_bytes",
]
