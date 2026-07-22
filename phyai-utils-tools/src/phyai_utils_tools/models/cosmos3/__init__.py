"""Cosmos3 processors (T2V tokenization + action/policy)."""

from __future__ import annotations

from phyai_utils_tools.models.cosmos3.processor_cosmos3 import (
    COSMOS3_VISION_START_TOKEN,
    COSMOS3_ROBOLAB_CONCAT_VIEW_DESCRIPTION,
    Cosmos3GenerationOutput,
    Cosmos3GenerationPostProcessor,
    Cosmos3PolicyProcessedInputs,
    Cosmos3PolicyProcessor,
    Cosmos3RoboLabPolicyProcessor,
    Cosmos3Processor,
    Cosmos3TokenizedPrompt,
    EMBODIMENT_TO_DOMAIN_ID,
    EMBODIMENT_TO_RAW_ACTION_DIM,
    cosmos3_default_negative_prompt,
    cosmos3_generation_caption,
    resolve_domain_id,
    resolve_raw_action_dim,
)


__all__ = [
    "COSMOS3_VISION_START_TOKEN",
    "COSMOS3_ROBOLAB_CONCAT_VIEW_DESCRIPTION",
    "Cosmos3GenerationOutput",
    "Cosmos3GenerationPostProcessor",
    "Cosmos3PolicyProcessedInputs",
    "Cosmos3PolicyProcessor",
    "Cosmos3RoboLabPolicyProcessor",
    "Cosmos3Processor",
    "Cosmos3TokenizedPrompt",
    "EMBODIMENT_TO_DOMAIN_ID",
    "EMBODIMENT_TO_RAW_ACTION_DIM",
    "cosmos3_default_negative_prompt",
    "cosmos3_generation_caption",
    "resolve_domain_id",
    "resolve_raw_action_dim",
]
