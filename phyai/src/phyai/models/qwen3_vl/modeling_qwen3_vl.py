"""Compatibility exports for the shared Qwen3-VL backbone."""

from phyai.layers.backbones.qwen3_vl.modeling import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionMLP,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)

__all__ = [
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLTextModel",
    "Qwen3VLVisionAttention",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionMLP",
    "Qwen3VLVisionModel",
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionPatchMerger",
    "Qwen3VLVisionRotaryEmbedding",
    "apply_rotary_pos_emb_vision",
    "get_vision_bilinear_indices_and_weights",
    "get_vision_cu_seqlens",
    "get_vision_position_ids",
]
