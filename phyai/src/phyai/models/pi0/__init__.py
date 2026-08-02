"""phyai.models.pi0 -- pi0 inference configuration and modules.

The package ships the full pi0 inference path:

* :mod:`configuration_pi0` -- frozen dataclass configs.
* :mod:`modeling_pi0` -- every ``nn.Module`` (vision tower, paligemma
  language model, pi0 action expert, action/time heads, and the
  parameter-only :class:`PI0Model` container).
* :mod:`model_runner_pi0` -- the three runners (vision / LLM / expert)
  that wrap captured CUDA graphs around the modeling code.
* :mod:`scheduler_ws1_pi0` -- single-card (world_size=1) end-to-end
  inference orchestrator with pi0's prefix/state/action mask.

Training is not in scope here; this package is inference-only.
"""

from __future__ import annotations

from phyai.models.pi0.configuration_pi0 import (
    GemmaExpertConfig,
    PaliGemmaTextConfig,
    PI0Config,
    SiglipVisionConfig,
)
from phyai.models.pi0.modeling_pi0 import (
    SIGLIP_NORM_HF_NAMES,
    ActionTimeHeads,
    MultiModalProjector,
    PaliGemmaDecoderLayer,
    PaliGemmaEmbedTokens,
    PaliGemmaLanguageModel,
    PI0ExpertLayer,
    PI0ExpertStack,
    PI0Model,
    PI0VisionTower,
    PositionEmbedding,
    SiglipVisionEmbeddings,
    SiglipVisionEncoder,
    SiglipVisionModel,
    VisionTowerWrapper,
    create_sinusoidal_pos_embedding,
)


__all__ = [
    # Configuration
    "GemmaExpertConfig",
    "PaliGemmaTextConfig",
    "PI0Config",
    "SiglipVisionConfig",
    # Modeling
    "ActionTimeHeads",
    "MultiModalProjector",
    "PaliGemmaDecoderLayer",
    "PaliGemmaEmbedTokens",
    "PaliGemmaLanguageModel",
    "PI0ExpertLayer",
    "PI0ExpertStack",
    "PI0Model",
    "PI0VisionTower",
    "PositionEmbedding",
    "SIGLIP_NORM_HF_NAMES",
    "SiglipVisionEmbeddings",
    "SiglipVisionEncoder",
    "SiglipVisionModel",
    "VisionTowerWrapper",
    "create_sinusoidal_pos_embedding",
]
