"""Pure-PyTorch reference no-cache attention — slow but exact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from phyai.layers.attention.nocache.base import (
    AttentionBackend,
    AttentionLayerProto,
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
)
from phyai.layers.attention.common import eager_attn, repeat_kv


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class EagerAttentionPlan(AttnPlanHandle):
    """Empty plan handle — eager no-cache has no per-step state."""


class EagerAttentionBackend(AttentionBackend):
    """Reference no-cache attention backend."""

    name = "eager"

    def __init__(self, runner: "ModelRunner | None" = None) -> None:
        del runner

    def supports_capture(self) -> bool:
        return True

    def init_forward_metadata(self, meta: AttnMetadata) -> AttnPlanHandle:
        return EagerAttentionPlan()

    def forward(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        if ctx.mode.is_idle():
            return q.new_zeros(q.shape)
        if ctx.layout.is_padded():
            return self._forward_padded(layer, q, k, v, ctx)
        if ctx.attn_bias is not None:
            raise ValueError("additive attention bias requires the padded layout")
        return self._forward_ragged(layer, q, k, v, ctx)

    def _forward_padded(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        attn_mask = None
        if ctx.mask is not None:
            attn_mask = ctx.mask.dense(
                q.shape[1],
                k.shape[1],
                q.shape[0],
                q.device,
                causal=layer.causal,
                sliding_window=layer.sliding_window,
            )
        q_h = q.transpose(1, 2)
        k_h = repeat_kv(k.transpose(1, 2), layer.num_heads, layer.num_kv_heads)
        v_h = repeat_kv(v.transpose(1, 2), layer.num_heads, layer.num_kv_heads)
        out = eager_attn(
            q_h,
            k_h,
            v_h,
            scale=layer.scale,
            causal=layer.causal,
            sliding_window=layer.sliding_window,
            logits_soft_cap=layer.logits_soft_cap,
            attn_mask=attn_mask,
            attn_bias=ctx.attn_bias,
        )
        return out.transpose(1, 2).contiguous()

    def _forward_ragged(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        if ctx.cu_seqlens_q is None:
            raise ValueError(
                "EagerAttentionBackend ragged forward requires ctx.cu_seqlens_q."
            )
        cu_q = ctx.cu_seqlens_q
        cu_kv = ctx.cu_seqlens_kv if ctx.cu_seqlens_kv is not None else cu_q
        cu_q_list = cu_q.tolist()
        cu_kv_list = cu_kv.tolist()
        outs: list[torch.Tensor] = []
        for b in range(len(cu_q_list) - 1):
            qs, qe = cu_q_list[b], cu_q_list[b + 1]
            ks, ke = cu_kv_list[b], cu_kv_list[b + 1]
            qi = q[qs:qe].unsqueeze(0).transpose(1, 2)
            ki = repeat_kv(
                k[ks:ke].unsqueeze(0).transpose(1, 2),
                layer.num_heads,
                layer.num_kv_heads,
            )
            vi = repeat_kv(
                v[ks:ke].unsqueeze(0).transpose(1, 2),
                layer.num_heads,
                layer.num_kv_heads,
            )
            oi = eager_attn(
                qi,
                ki,
                vi,
                scale=layer.scale,
                causal=layer.causal,
                sliding_window=layer.sliding_window,
                logits_soft_cap=layer.logits_soft_cap,
            )
            outs.append(oi.transpose(1, 2).squeeze(0))
        return torch.cat(outs, dim=0)


__all__ = ["EagerAttentionBackend", "EagerAttentionPlan"]
