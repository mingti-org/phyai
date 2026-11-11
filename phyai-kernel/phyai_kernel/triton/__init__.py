"""phyai-kernel Triton kernels (pure-Python, no tvm-ffi build)."""

from phyai_kernel.triton.ada_rms_norm import adarmsnorm
from phyai_kernel.triton.causal_conv1d import causal_conv1d_silu_split_qkv
from phyai_kernel.triton.channel_l2_norm import channel_l2_norm
from phyai_kernel.triton.layer_norm import layernorm
from phyai_kernel.triton.layer_norm_modulate import layernorm_modulate
from phyai_kernel.triton.layer_norm_residual import layernorm_gated_residual
from phyai_kernel.triton.masked_embedding import masked_embedding_lookup
from phyai_kernel.triton.modulate import modulate
from phyai_kernel.triton.paged_kv_indices import create_paged_kv_indices
from phyai_kernel.triton.qk_rmsnorm_rotary import qk_rmsnorm_rope
from phyai_kernel.triton.rms_norm import (
    fused_add_rmsnorm,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    rmsnorm_cast_affine,
    rmsnorm_hf,
)
from phyai_kernel.triton.rms_norm_silu_mul import rmsnorm_silu_mul
from phyai_kernel.triton.situ import situ_and_mul

__all__ = [
    "adarmsnorm",
    "causal_conv1d_silu_split_qkv",
    "channel_l2_norm",
    "create_paged_kv_indices",
    "fused_add_rmsnorm",
    "gemma_fused_add_rmsnorm",
    "gemma_rmsnorm",
    "layernorm",
    "layernorm_modulate",
    "layernorm_gated_residual",
    "masked_embedding_lookup",
    "modulate",
    "qk_rmsnorm_rope",
    "rmsnorm",
    "rmsnorm_cast_affine",
    "rmsnorm_hf",
    "rmsnorm_silu_mul",
    "situ_and_mul",
]
