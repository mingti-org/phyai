"""GoalWAM fusions; projections and attention retain PhyAI's public layers."""

import torch

from phyai.layers.layer_norm import RMSNorm
from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3CrossAttention,
    prepare_cosmos3_attention,
)


def add_goalwam_norm(
    residual: torch.Tensor, projected: torch.Tensor, norm: RMSNorm
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round the residual sum before computing its variance and affine."""
    if (
        residual.is_cuda
        and residual.dtype == projected.dtype == norm.weight.dtype == torch.bfloat16
        and residual.shape == projected.shape
        and residual.shape[-1] == 4096
        and residual.is_contiguous()
        and projected.is_contiguous()
        and norm.cast_before_affine
    ):
        from phyai_kernel.triton.goalwam import _add_rms_norm

        hidden, normalized = torch.empty_like(residual), torch.empty_like(residual)
        _add_rms_norm[(residual.numel() // 4096,)](
            residual,
            projected,
            norm.weight,
            hidden,
            normalized,
            norm.variance_epsilon,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return hidden, normalized
    hidden = residual + projected
    return hidden, norm(hidden)


def prepare_goalwam_attention(
    attention: Cosmos3CrossAttention,
    hidden: torch.Tensor,
    text_key: torch.Tensor,
    text_value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Q/K normalization, RoPE and text concatenation for B1 GQA4 D128."""
    batch, tokens, _ = hidden.shape
    qkv, _ = attention.qkv_proj(hidden)
    query, key, value = qkv.split(
        (attention.q_size, attention.k_size, attention.v_size), dim=-1
    )
    query = query.view(batch, tokens, attention.num_local_heads, attention.head_dim)
    key = key.view(batch, tokens, attention.num_local_kv_heads, attention.head_dim)
    value = value.view_as(key)
    tensors = (query, key, value, text_key, text_value, cos, sin)
    if (
        batch == 1
        and tokens > 0
        and attention.head_dim == 128
        and attention.num_local_heads in (8, 16, 32)
        and attention.num_local_heads == 4 * attention.num_local_kv_heads
        and attention.norm_q.cast_before_affine
        and attention.norm_k.cast_before_affine
        and attention.norm_q.weight.dtype
        == attention.norm_k.weight.dtype
        == torch.bfloat16
        and all(
            t.is_cuda and t.dtype == torch.bfloat16 and t.stride(-1) == 1
            for t in tensors
        )
    ):
        from phyai_kernel.triton.goalwam import _prepare_qkv

        query, key, value = (t.squeeze(0) for t in (query, key, value))
        text_key, text_value = text_key.squeeze(0), text_value.squeeze(0)
        cos, sin = cos[0, :, 0], sin[0, :, 0]
        heads, text = key.shape[1], text_key.shape[0]
        out_query = torch.empty_like(query, memory_format=torch.contiguous_format)
        out_key = key.new_empty((text + tokens, heads, 128))
        out_value = torch.empty_like(out_key)
        _prepare_qkv[(text + tokens, 5)](
            query,
            key,
            value,
            attention.norm_q.weight,
            attention.norm_k.weight,
            cos,
            sin,
            text_key,
            text_value,
            out_query,
            out_key,
            out_value,
            text,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            cos.stride(0),
            sin.stride(0),
            *text_key.stride(),
            *text_value.stride(),
            attention.norm_q.variance_epsilon,
            attention.norm_k.variance_epsilon,
            heads,
            num_warps=heads,
            enable_fp_fusion=False,
        )
        return out_query.unsqueeze(0), out_key.unsqueeze(0), out_value.unsqueeze(0)
    query = attention.norm_q(
        query.reshape(-1, attention.num_local_heads, attention.head_dim),
        preserve_3d_layout=True,
    ).view_as(query)
    key = attention.norm_k(
        key.reshape(-1, attention.num_local_kv_heads, attention.head_dim),
        preserve_3d_layout=True,
    ).view_as(key)
    return prepare_cosmos3_attention(query, key, value, cos, sin, text_key, text_value)
