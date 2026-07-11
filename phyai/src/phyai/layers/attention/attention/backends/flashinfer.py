"""flashinfer no-cache attention backend (ragged prefill).

Routes batch=1 single-sequence calls through
``single_prefill_with_kv_cache`` (no plan needed) and multi-sequence
ragged calls through
:class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`. Plan
happens in :meth:`init_forward_metadata`, OUTSIDE any captured region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from phyai.layers.attention.attention.base import (
    AttentionBackend,
    AttentionLayerProto,
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
)
from phyai.layers.attention.attention.registry import register_backend
from phyai.layers.attention.utils import get_global_fi_workspace


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


def _fi_window_left(sliding_window: int | None) -> int:
    """flashinfer's ``window_left`` is "previous keys visible".

    Our ``sliding_window`` counts the current token, so a window of
    ``W`` tokens — current included — maps to ``W - 1``.
    """
    return -1 if sliding_window is None else sliding_window - 1


@dataclass(frozen=True)
class FlashInferAttentionPlan(AttnPlanHandle):
    """Plan handle for :class:`FlashInferAttentionBackend`.

    ``wrapper`` is ``None`` for the B=1 single-prefill fast path and a
    planned :class:`BatchPrefillWithRaggedKVCacheWrapper` otherwise.
    """

    wrapper: Any = None


@register_backend("flashinfer")
class FlashInferAttentionBackend(AttentionBackend):
    """flashinfer prefill kernels for :class:`Attention`.

    The single-sequence (B=1) path skips ``plan()`` entirely and is
    safe inside captured graphs. The batched ragged path calls
    ``wrapper.plan()`` from :meth:`init_forward_metadata` (called
    outside any captured region) and ``wrapper.run()`` from
    :meth:`forward`.
    """

    def __init__(
        self,
        runner: "ModelRunner | None" = None,
        *,
        fi_workspace: torch.Tensor | None = None,
        workspace_bytes: int | None = None,
    ) -> None:
        del runner
        try:
            import flashinfer.prefill  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='flashinfer' but flashinfer is not installed; "
                "either install flashinfer-python or pick "
                "backend='sdpa'/'eager'."
            ) from e
        self._fi_workspace: torch.Tensor | None = None
        self._fi_wrapper = None
        self._workspace_bytes = workspace_bytes
        if fi_workspace is not None:
            self._build_wrapper(fi_workspace)

    def supports_capture(self) -> bool:
        return True

    def _build_wrapper(self, workspace: torch.Tensor) -> None:
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        if workspace.dtype != torch.uint8 or workspace.ndim != 1:
            raise ValueError(
                f"fi_workspace must be a 1-D uint8 tensor, got "
                f"shape={tuple(workspace.shape)}, dtype={workspace.dtype}."
            )
        self._fi_workspace = workspace
        self._fi_wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD")

    def _ensure_wrapper(self, device: torch.device):
        if self._fi_wrapper is None:
            self._build_wrapper(get_global_fi_workspace(device))
        return self._fi_wrapper

    def init_forward_metadata(self, meta: AttnMetadata) -> AttnPlanHandle:
        from phyai.layers.attention.enums import AttnMode

        if meta.mode == AttnMode.IDLE or meta.batch_size <= 1:
            return FlashInferAttentionPlan(wrapper=None)
        if meta.cu_seqlens_q is None:
            raise ValueError(
                "FlashInferAttentionBackend.init_forward_metadata requires "
                "cu_seqlens_q on AttnMetadata for batch_size > 1."
            )
        cu_q = meta.cu_seqlens_q
        if cu_q.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer batched no-cache attention must be planned before "
                "CUDA Graph capture; pass a pre-planned AttnCtx to forward()."
            )
        cu_kv = meta.cu_seqlens_kv if meta.cu_seqlens_kv is not None else cu_q
        layer_proto = meta.extras.get("layer_proto")
        if layer_proto is None:
            raise ValueError(
                "FlashInferAttentionBackend.init_forward_metadata requires "
                "extras['layer_proto'] for B>1 plan; pass it via AttnMetadata."
            )
        wrapper = self._ensure_wrapper(cu_q.device)
        q_dtype = meta.extras.get("q_dtype")
        kv_dtype = meta.extras.get("kv_dtype", q_dtype)
        if q_dtype is None:
            raise ValueError(
                "FlashInferAttentionBackend.init_forward_metadata requires "
                "extras['q_dtype'] (and optional extras['kv_dtype'])."
            )
        wrapper.plan(
            cu_q.to(torch.int32),
            cu_kv.to(torch.int32),
            num_qo_heads=layer_proto.num_heads,
            num_kv_heads=layer_proto.num_kv_heads,
            head_dim_qk=layer_proto.head_dim,
            causal=layer_proto.causal,
            sm_scale=layer_proto.scale,
            window_left=_fi_window_left(layer_proto.sliding_window),
            logits_soft_cap=layer_proto.logits_soft_cap,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
        )
        return FlashInferAttentionPlan(wrapper=wrapper)

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
        return self._forward_ragged(layer, q, k, v, ctx)

    def _forward_padded(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        B, S_q, H, D = q.shape
        S_kv = k.shape[1]
        if B == 1:
            return self._single(layer, q[0], k[0], v[0]).unsqueeze(0)
        plan = ctx.plan
        if not isinstance(plan, FlashInferAttentionPlan) or plan.wrapper is None:
            raise ValueError(
                "FlashInferAttentionBackend padded forward with B>1 requires a "
                "planned FlashInferAttentionPlan; the runner / layer must call "
                "init_forward_metadata first."
            )
        out = plan.wrapper.run(
            q.reshape(B * S_q, H, D),
            k.reshape(B * S_kv, layer.num_kv_heads, D),
            v.reshape(B * S_kv, layer.num_kv_heads, D),
        )
        return out.reshape(B, S_q, H, D)

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
                "FlashInferAttentionBackend ragged forward requires ctx.cu_seqlens_q."
            )
        B = ctx.cu_seqlens_q.numel() - 1
        if B == 1:
            return self._single(layer, q, k, v)
        plan = ctx.plan
        if not isinstance(plan, FlashInferAttentionPlan) or plan.wrapper is None:
            raise ValueError(
                "FlashInferAttentionBackend ragged forward with B>1 requires a "
                "planned FlashInferAttentionPlan."
            )
        return plan.wrapper.run(q, k, v)

    def _single(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        from flashinfer.prefill import single_prefill_with_kv_cache

        return single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=layer.causal,
            kv_layout="NHD",
            sm_scale=layer.scale,
            window_left=_fi_window_left(layer.sliding_window),
            logits_soft_cap=layer.logits_soft_cap,
        )


__all__ = ["FlashInferAttentionBackend", "FlashInferAttentionPlan"]
