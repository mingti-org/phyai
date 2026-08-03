"""LingBot-VLA 2.0 inference support."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .configuration_lingbotv2 import (
    LingBotV2DualQueryConfig,
    LingBotV2ExpertConfig,
    LingBotV2FlowMatchingConfig,
    LingBotV2MoEConfig,
    LingBotVLA2Config,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


_LAZY_EXPORTS = {
    "LingBotV2ActionTimeHeads": "modeling_lingbotv2",
    "LingBotV2AdaRMSNorm": "modeling_lingbotv2",
    "LingBotV2DualQuery": "modeling_lingbotv2",
    "LingBotV2ExpertLayer": "modeling_lingbotv2",
    "LingBotV2ExpertMLP": "modeling_lingbotv2",
    "LingBotV2ExpertStack": "modeling_lingbotv2",
    "LingBotV2Model": "modeling_lingbotv2",
    "LingBotV2TokenMoE": "modeling_lingbotv2",
    "Qwen3VLTextModel": "modeling_lingbotv2",
    "Qwen3VLVisionAttention": "modeling_lingbotv2",
    "Qwen3VLVisionBlock": "modeling_lingbotv2",
    "Qwen3VLVisionMLP": "modeling_lingbotv2",
    "Qwen3VLVisionModel": "modeling_lingbotv2",
    "Qwen3VLVisionPatchEmbed": "modeling_lingbotv2",
    "Qwen3VLVisionPatchMerger": "modeling_lingbotv2",
    "Qwen3VLVisionRotaryEmbedding": "modeling_lingbotv2",
    "apply_rotary_pos_emb_vision": "modeling_lingbotv2",
    "build_lingbot_v2_mrope_position_ids": "modeling_lingbotv2",
    "create_sinusoidal_pos_embedding": "modeling_lingbotv2",
    "get_vision_bilinear_indices_and_weights": "modeling_lingbotv2",
    "get_vision_cu_seqlens": "modeling_lingbotv2",
    "get_vision_position_ids": "modeling_lingbotv2",
    "lingbot_v2_weight_remap": "modeling_lingbotv2",
    "LingBotV2CacheAllocation": "model_runner_lingbotv2",
    "LingBotV2ExpertForwardBatch": "model_runner_lingbotv2",
    "LingBotV2ExpertRunner": "model_runner_lingbotv2",
    "LingBotV2PrefixEmbeddings": "model_runner_lingbotv2",
    "LingBotV2PrefixForwardBatch": "model_runner_lingbotv2",
    "LingBotV2PrefixRunner": "model_runner_lingbotv2",
    "LingBotV2VisionForwardBatch": "model_runner_lingbotv2",
    "LingBotV2VisionRunner": "model_runner_lingbotv2",
    "LingBotV2Request": "scheduler_ws1_lingbotv2",
    "LingBotV2WS1Scheduler": "scheduler_ws1_lingbotv2",
    "build_action_paged_kv_indices": "scheduler_ws1_lingbotv2",
    "build_prefix_padded_write_indices": "scheduler_ws1_lingbotv2",
    "build_state_paged_kv_indices": "scheduler_ws1_lingbotv2",
    "build_suffix_mrope_position_ids": "scheduler_ws1_lingbotv2",
    "LingBotV2Args": "main_lingbotv2",
    "LingBotV2Entry": "main_lingbotv2",
    "compose_lingbot_v2_weight_remap": "main_lingbotv2",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "LingBotV2ActionTimeHeads",
    "LingBotV2AdaRMSNorm",
    "LingBotV2Args",
    "LingBotV2CacheAllocation",
    "LingBotV2DualQuery",
    "LingBotV2DualQueryConfig",
    "LingBotV2Entry",
    "LingBotV2ExpertConfig",
    "LingBotV2ExpertForwardBatch",
    "LingBotV2ExpertLayer",
    "LingBotV2ExpertMLP",
    "LingBotV2ExpertRunner",
    "LingBotV2ExpertStack",
    "LingBotV2FlowMatchingConfig",
    "LingBotV2MoEConfig",
    "LingBotV2Model",
    "LingBotV2PrefixEmbeddings",
    "LingBotV2PrefixForwardBatch",
    "LingBotV2PrefixRunner",
    "LingBotV2Request",
    "LingBotV2TokenMoE",
    "LingBotV2VisionForwardBatch",
    "LingBotV2VisionRunner",
    "LingBotV2WS1Scheduler",
    "LingBotVLA2Config",
    "Qwen3VLTextConfig",
    "Qwen3VLTextModel",
    "Qwen3VLVisionAttention",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionConfig",
    "Qwen3VLVisionMLP",
    "Qwen3VLVisionModel",
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionPatchMerger",
    "Qwen3VLVisionRotaryEmbedding",
    "apply_rotary_pos_emb_vision",
    "build_action_paged_kv_indices",
    "build_lingbot_v2_mrope_position_ids",
    "build_prefix_padded_write_indices",
    "build_state_paged_kv_indices",
    "build_suffix_mrope_position_ids",
    "compose_lingbot_v2_weight_remap",
    "create_sinusoidal_pos_embedding",
    "get_vision_bilinear_indices_and_weights",
    "get_vision_cu_seqlens",
    "get_vision_position_ids",
    "lingbot_v2_weight_remap",
]
