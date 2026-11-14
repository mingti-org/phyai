"""aiter paged-KV backend for the unified paged attention stack (ROCm)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from phyai.layers.attention.paged.base import (
    PagedAttentionBackend,
    PagedAttentionLayerProto,
    PagedAttnCtx,
    PagedAttnMetadata,
    PagedAttnPlanHandle,
)

if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class AiterPagedPlan(PagedAttnPlanHandle):
    """Views of the backend's static metadata buffers for one batch size."""

    cu_seqlens_q: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor


class AiterPagedBackend(PagedAttentionBackend):
    """Paged-KV attention through aiter's CK batch-prefill kernel."""

    name = "aiter"

    def __init__(self, runner: "ModelRunner | None" = None) -> None:
        del runner
        try:
            from aiter import mha_batch_prefill_func
        except ImportError as e:
            raise ImportError(
                "backend='aiter' (paged) but aiter is not installed; install amd-aiter."
            ) from e
        self._prefill = mha_batch_prefill_func
        self._cu_q_buf: torch.Tensor | None = None
        self._kv_indptr_buf: torch.Tensor | None = None
        self._kv_indices_buf: torch.Tensor | None = None
        self._last_page_len_buf: torch.Tensor | None = None
        # The kernel takes the longest query and KV lengths as host integers,
        # which a captured graph freezes. Runners capture with short seed
        # lengths and replay longer ones, so the static bounds are passed.
        self._max_seqlen_q = 0
        self._max_seqlen_kv = 0

    def supports_capture(self) -> bool:
        return True

    def init_cuda_graph_state(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        max_paged_kv_indices: int,
        device: torch.device,
        params_dtype: torch.dtype,
        layer_proto: PagedAttentionLayerProto,
    ) -> None:
        if self._cu_q_buf is not None:
            return
        self._max_seqlen_q = int(max_num_tokens)
        self._max_seqlen_kv = int(max_paged_kv_indices)
        self._cu_q_buf = torch.zeros(
            max_batch_size + 1, dtype=torch.int32, device=device
        )
        self._kv_indptr_buf = torch.zeros(
            max_batch_size + 1, dtype=torch.int32, device=device
        )
        self._kv_indices_buf = torch.zeros(
            max_paged_kv_indices, dtype=torch.int32, device=device
        )
        self._last_page_len_buf = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device
        )

    def _copy_into_static_buffers(self, meta: PagedAttnMetadata) -> int:
        if (
            meta.cu_seqlens_q is None
            or meta.paged_kv_indptr is None
            or meta.paged_kv_indices is None
            or meta.paged_kv_last_page_len is None
        ):
            raise ValueError(
                "AiterPagedBackend plan requires cu_seqlens_q, paged_kv_indptr, "
                "paged_kv_indices, and paged_kv_last_page_len on PagedAttnMetadata."
            )
        batch = meta.cu_seqlens_q.numel() - 1
        indices = meta.paged_kv_indices
        self._cu_q_buf[: batch + 1].copy_(meta.cu_seqlens_q)
        self._kv_indptr_buf[: batch + 1].copy_(meta.paged_kv_indptr)
        self._kv_indices_buf[: indices.numel()].copy_(indices)
        self._last_page_len_buf[:batch].copy_(meta.paged_kv_last_page_len)
        return batch

    def init_forward_metadata(self, meta: PagedAttnMetadata) -> AiterPagedPlan:
        if self._cu_q_buf is None:
            raise RuntimeError(
                "AiterPagedBackend not initialized — call "
                "init_cuda_graph_state(...) before planning."
            )
        batch = self._copy_into_static_buffers(meta)
        return AiterPagedPlan(
            cu_seqlens_q=self._cu_q_buf[: batch + 1],
            paged_kv_indptr=self._kv_indptr_buf[: batch + 1],
            paged_kv_indices=self._kv_indices_buf,
            paged_kv_last_page_len=self._last_page_len_buf[:batch],
        )

    def replay_metadata(
        self, plan: PagedAttnPlanHandle, replay_meta: PagedAttnMetadata
    ) -> None:
        if not isinstance(plan, AiterPagedPlan):
            raise TypeError("aiter paged replay requires an AiterPagedPlan")
        if replay_meta.cu_seqlens_q is not None and (
            replay_meta.cu_seqlens_q.numel() != plan.cu_seqlens_q.numel()
        ):
            raise ValueError(
                "aiter paged replay cannot change the batch size; capture a new graph"
            )
        self._copy_into_static_buffers(replay_meta)

    def forward(
        self,
        layer: PagedAttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: PagedAttnCtx,
    ) -> torch.Tensor:
        if ctx.mode.is_idle():
            return q.new_zeros(q.shape)
        plan = ctx.plan
        if not isinstance(plan, AiterPagedPlan):
            raise TypeError(
                f"AiterPagedBackend expected ctx.plan: AiterPagedPlan, "
                f"got {type(plan).__name__}."
            )
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
        k_cache, v_cache = ctx.kv_pool.kv_buffer(layer.layer_id)
        return self._prefill(
            q,
            k_cache,
            v_cache,
            plan.cu_seqlens_q,
            plan.paged_kv_indptr,
            plan.paged_kv_indices,
            self._max_seqlen_q,
            self._max_seqlen_kv,
            softmax_scale=layer.scale,
            causal=layer.causal,
            kv_last_page_lens=plan.paged_kv_last_page_len,
        )


__all__ = ["AiterPagedBackend", "AiterPagedPlan"]
