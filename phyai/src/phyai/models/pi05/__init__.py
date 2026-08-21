"""phyai.models.pi05 — pi0.5 inference and RL rollout support.

The package ships the full pi0.5 inference path:

* :mod:`configuration_pi05` — frozen dataclass configs.
* :mod:`modeling_pi05` — every ``nn.Module`` (vision tower, paligemma
  language model with :class:`~phyai.layers.attention.ARAttention`,
  gemma_300m action expert with
  :class:`~phyai.layers.attention.DiffusionAttention`, action/time
  heads, and the parameter-only :class:`PI05Model` container).
* :mod:`model_runner_pi05` — the three runners (vision / LLM / expert)
  that wrap captured CUDA graphs around the modeling code.
* :mod:`scheduler_ws1_pi05` — single-card (world_size=1) end-to-end
  inference orchestrator with multi-batch support. Owns the
  pi0.5-specific batch-layout helpers (cu_seqlens, write-indices,
  padded prefix layout, joint paged_kv_indices interleave).

RL actor training remains outside phyai. The rollout API provides stochastic
trajectory sampling and policy data for external RL frameworks.
"""

from __future__ import annotations

from phyai.models.pi05.configuration_pi05 import (
    GemmaExpertConfig,
    PaliGemmaTextConfig,
    PI05Config,
    SiglipVisionConfig,
)
from phyai.models.pi05.modeling_pi05 import (
    SIGLIP_NORM_HF_NAMES,
    ActionTimeHeads,
    ExpertLayerModulation,
    ExpertModulationTables,
    ExpertStepModulation,
    MultiModalProjector,
    PaliGemmaDecoderLayer,
    PaliGemmaEmbedTokens,
    PaliGemmaLanguageModel,
    PI05ExpertLayer,
    PI05ExpertStack,
    PI05Model,
    PI05ValueHead,
    PI05VisionTower,
    PositionEmbedding,
    SiglipVisionEmbeddings,
    SiglipVisionEncoder,
    SiglipVisionModel,
    VisionTowerWrapper,
    create_sinusoidal_pos_embedding,
)
from phyai.models.pi05.scheduler_ws1_pi05 import (
    PI05Request,
    PI05RolloutOutput,
    PI05RolloutRequest,
    PI05WS1Scheduler,
)

__all__ = [
    # Configuration
    "GemmaExpertConfig",
    "PaliGemmaTextConfig",
    "PI05Config",
    "SiglipVisionConfig",
    # Modeling
    "ActionTimeHeads",
    "ExpertLayerModulation",
    "ExpertModulationTables",
    "ExpertStepModulation",
    "MultiModalProjector",
    "PaliGemmaDecoderLayer",
    "PaliGemmaEmbedTokens",
    "PaliGemmaLanguageModel",
    "PI05ExpertLayer",
    "PI05ExpertStack",
    "PI05Model",
    "PI05Request",
    "PI05RolloutOutput",
    "PI05RolloutRequest",
    "PI05ValueHead",
    "PI05WS1Scheduler",
    "PI05VisionTower",
    "PositionEmbedding",
    "SIGLIP_NORM_HF_NAMES",
    "SiglipVisionEmbeddings",
    "SiglipVisionEncoder",
    "SiglipVisionModel",
    "VisionTowerWrapper",
    "create_sinusoidal_pos_embedding",
]
