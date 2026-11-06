"""Token-partitioned GEN blocks with full-reduction-dimension projections."""

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from phyai.layers.layer_norm import RMSNorm
from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Condition
from phyai.models.humanwam_goal.collective_humanwam_goal import HumanWAMGoalGather
from phyai.models.humanwam_goal.modeling_kernels_humanwam_goal import (
    add_humanwam_goal_norm,
    humanwam_goal_silu_mul,
    prepare_humanwam_goal_attention,
)


@contextmanager
def full_precision_gemm():
    """Use FP32 accumulation without split-K; restore the caller's BLAS policy."""
    blas = torch.backends.cuda.preferred_blas_library()
    matmul = torch.backends.cuda.matmul
    reduction = (
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction_split_k,
    )
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        matmul.allow_bf16_reduced_precision_reduction = (False, False)
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = reduction
        torch.backends.cuda.preferred_blas_library(blas)


def project_full_k(
    value: torch.Tensor, weight: torch.Tensor, collective: HumanWAMGoalGather
) -> torch.Tensor:
    """Shard output channels, rounding only after the complete dot product."""
    full = collective.gather(value)
    with full_precision_gemm():
        projected = F.linear(full, weight)
    return collective.gather(projected)


def forward_parallel(
    layers: nn.ModuleList,
    final_norm: RMSNorm,
    hidden: torch.Tensor,
    condition: Cosmos3Condition,
    output_rows: torch.Tensor,
    collective: HumanWAMGoalGather,
) -> torch.Tensor:
    """Keep all keys/values while distributing token work across the TP group."""
    total = hidden.shape[1]
    part = (total + collective.size - 1) // collective.size
    padding = part * collective.size - total
    start = collective.rank * part
    local = F.pad(hidden, (0, 0, 0, padding))[:, start : start + part].contiguous()
    normalized = layers[0].input_layernorm(local)
    cos, sin = condition.freqs_gen
    if padding:
        cos = torch.cat((cos, cos[:, -padding:]), dim=1)
        sin = torch.cat((sin, sin[:, -padding:]), dim=1)
    cos, sin = (value[:, start : start + part].contiguous() for value in (cos, sin))
    local_kv = collective.kv_workspace(part)
    with full_precision_gemm():
        for index, layer in enumerate(layers[:-1]):
            attention = layer.cross_attention
            text_key, text_value = condition.cached_kv[index]
            query, _, _ = prepare_humanwam_goal_attention(
                attention, normalized, text_key, text_value, cos, sin, local_kv=local_kv
            )
            key, value = collective.gather_kv(
                text_key, text_value, part=part, tokens=total
            )
            context = (
                None
                if condition.attention_contexts is None
                else condition.attention_contexts[index]
            )
            attended = attention.attn(query, key, value, context).flatten(-2)
            projected = F.linear(attended, attention.to_out.weight)
            local, normalized = add_humanwam_goal_norm(
                local, projected, layer.post_attention_layernorm
            )
            fused = F.linear(normalized, layer.mlp.gate_up_proj.weight)
            projected = F.linear(
                humanwam_goal_silu_mul(fused), layer.mlp.down_proj.weight
            )
            local, normalized = add_humanwam_goal_norm(
                local, projected, layers[index + 1].input_layernorm
            )

    hidden = collective.gather_rows(local)[:, :total].contiguous()
    normalized = collective.gather_rows(normalized)[:, :total].contiguous()
    layer = layers[-1]
    attention = layer.cross_attention
    key, value = condition.cached_kv[-1]
    query, key, value = prepare_humanwam_goal_attention(
        attention, normalized, key, value, *condition.freqs_gen
    )
    query = query.index_select(1, output_rows)
    hidden = hidden.index_select(1, output_rows)
    context = (
        None
        if condition.attention_contexts is None
        else condition.attention_contexts[-1]
    )
    attended = attention.attn(query, key, value, context).flatten(-2)
    projected = project_full_k(attended, attention.to_out.weight, collective)
    hidden, normalized = add_humanwam_goal_norm(
        hidden, projected, layer.post_attention_layernorm
    )
    fused, _ = layer.mlp.gate_up_proj(normalized)
    projected = project_full_k(
        humanwam_goal_silu_mul(fused), layer.mlp.down_proj.weight, collective
    )
    _, normalized = add_humanwam_goal_norm(hidden, projected, final_norm)
    return normalized
