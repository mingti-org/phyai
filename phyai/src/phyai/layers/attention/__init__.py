"""phyai.layers.attention"""

from __future__ import annotations

from phyai.layers.attention.ar import (
    ARAttention,
    ARAttentionBackend,
    ARAttentionLayerProto,
    ARAttnCtx,
    ARAttnMetadata,
    ARAttnPlanHandle,
    FlashInferARBackend,
    FlashInferARPlan,
)
from phyai.layers.attention.ar import (
    get_backend_factory as get_ar_backend_factory,
)
from phyai.layers.attention.ar import (
    list_backends as list_ar_backends,
)
from phyai.layers.attention.ar import (
    register_backend as register_ar_backend,
)
from phyai.layers.attention.attention import (
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
from phyai.layers.attention.attention import (
    get_backend_factory as get_attention_backend_factory,
)
from phyai.layers.attention.attention import (
    list_backends as list_attention_backends,
)
from phyai.layers.attention.attention import (
    register_backend as register_attention_backend,
)
from phyai.layers.attention.diffusion import (
    DiffusionAttention,
    DiffusionAttentionBackend,
    DiffusionAttentionLayerProto,
    DiffusionAttnCtx,
    DiffusionAttnMetadata,
    DiffusionAttnPlanHandle,
    FlashInferDiffusionBackend,
    FlashInferDiffusionPlan,
)
from phyai.layers.attention.diffusion import (
    get_backend_factory as get_diffusion_backend_factory,
)
from phyai.layers.attention.diffusion import (
    list_backends as list_diffusion_backends,
)
from phyai.layers.attention.diffusion import (
    register_backend as register_diffusion_backend,
)
from phyai.layers.attention.enums import AttnLayout, AttnMode
from phyai.layers.attention.gdn import (
    FlaGatedDeltaNetBackend,
    FlaGatedDeltaNetPlan,
    FlashInferGatedDeltaNetBackend,
    FlashInferGatedDeltaNetPlan,
    GatedDeltaNet,
    GatedDeltaNetBackend,
    GatedDeltaNetCtx,
    GatedDeltaNetLayerProto,
    GatedDeltaNetMetadata,
    GatedDeltaNetPlanHandle,
)
from phyai.layers.attention.gdn import (
    get_backend_factory as get_gdn_backend_factory,
)
from phyai.layers.attention.gdn import list_backends as list_gdn_backends
from phyai.layers.attention.gdn import register_backend as register_gdn_backend
from phyai.layers.attention.utils import (
    get_global_fi_workspace,
    register_global_fi_workspace,
    resolve_workspace_bytes,
)


__all__ = [
    # === Layers ===
    "Attention",
    "ARAttention",
    "DiffusionAttention",
    "GatedDeltaNet",
    # === Shared enums ===
    "AttnLayout",
    "AttnMode",
    # === attention/ stack ===
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
    "get_attention_backend_factory",
    "list_attention_backends",
    "register_attention_backend",
    # === ar/ stack ===
    "ARAttentionBackend",
    "ARAttentionLayerProto",
    "ARAttnCtx",
    "ARAttnMetadata",
    "ARAttnPlanHandle",
    "FlashInferARBackend",
    "FlashInferARPlan",
    "get_ar_backend_factory",
    "list_ar_backends",
    "register_ar_backend",
    # === diffusion/ stack ===
    "DiffusionAttentionBackend",
    "DiffusionAttentionLayerProto",
    "DiffusionAttnCtx",
    "DiffusionAttnMetadata",
    "DiffusionAttnPlanHandle",
    "FlashInferDiffusionBackend",
    "FlashInferDiffusionPlan",
    "get_diffusion_backend_factory",
    "list_diffusion_backends",
    "register_diffusion_backend",
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
    "get_gdn_backend_factory",
    "list_gdn_backends",
    "register_gdn_backend",
    # === Workspace ===
    "get_global_fi_workspace",
    "register_global_fi_workspace",
    "resolve_workspace_bytes",
]
