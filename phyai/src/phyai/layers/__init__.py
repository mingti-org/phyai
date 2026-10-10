"""phyai.layers — model layer building blocks."""

from __future__ import annotations

from phyai.layers.activation import SiLU, Snake1d
from phyai.layers.modulation import AffineModulation
from phyai.layers.layer_norm import (
    AdaRMSNorm,
    GatedRMSNorm,
    GemmaRMSNorm,
    LayerNorm,
    RMSNorm,
)
from phyai.layers.rotary_embedding import (
    ROPE_INV_FREQ_FNS,
    InterleavedMRotaryEmbedding,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    compute_cos_sin_from_inv_freq,
    compute_qwen3vl_mrope_cos_sin_from_inv_freq,
    rotate_half,
)
from phyai.layers.transformer_block import TransformerBlock

__all__ = [
    "AdaRMSNorm",
    "AffineModulation",
    "GatedRMSNorm",
    "GemmaRMSNorm",
    "LayerNorm",
    "TransformerBlock",
    "RMSNorm",
    "SiLU",
    "Snake1d",
    "ROPE_INV_FREQ_FNS",
    "InterleavedMRotaryEmbedding",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "compute_cos_sin_from_inv_freq",
    "compute_qwen3vl_mrope_cos_sin_from_inv_freq",
    "rotate_half",
]
