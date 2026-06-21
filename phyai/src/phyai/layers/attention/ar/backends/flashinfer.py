"""flashinfer paged-KV backend for AR (LM-side) attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from phyai.layers.attention.ar.base import (
    ARAttentionBackend,
    ARAttentionLayerProto,
    ARAttnCtx,
    ARAttnMetadata,
    ARAttnPlanHandle,
)
from phyai.layers.attention.ar.registry import register_backend
from phyai.layers.attention.utils import (
    get_global_fi_workspace,
    resolve_prefill_backend,
)


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class FlashInferARPlan(ARAttnPlanHandle):
    """Plan handle for :class:`FlashInferARBackend`.

    ``wrapper`` is the runner-scoped, pre-bound
    :class:`BatchPrefillWithPagedKVCacheWrapper`. Identity is stable
    across replays — :meth:`FlashInferARBackend.replay_metadata`
    re-plans the same wrapper rather than returning a fresh handle.
    """

    wrapper: Any


@register_backend("flashinfer")
class FlashInferARBackend(ARAttentionBackend):
    """Paged-KV flashinfer backend for :class:`ARAttention`."""

    def __init__(self, runner: "ModelRunner | None" = None) -> None:
        del runner
        try:
            import flashinfer.prefill  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "backend='flashinfer' (ar) but flashinfer is not installed; "
                "install flashinfer-python. The AR paged stack is "
                "flashinfer-only (GPU)."
            ) from e
        self._wrapper = None
        self._cu_q_buf: torch.Tensor | None = None
        self._paged_kv_indptr_buf: torch.Tensor | None = None
        self._paged_kv_indices_buf: torch.Tensor | None = None
        self._paged_kv_last_page_len_buf: torch.Tensor | None = None
        self._layer_proto: ARAttentionLayerProto | None = None
        self._params_dtype: torch.dtype | None = None
        self._max_batch_size: int | None = None
        self._max_paged_kv_indices: int | None = None

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
        layer_proto: ARAttentionLayerProto,
    ) -> None:
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        if self._wrapper is not None:
            return
        self._max_batch_size = int(max_batch_size)
        self._max_paged_kv_indices = int(max_paged_kv_indices)
        self._params_dtype = params_dtype
        self._layer_proto = layer_proto
        self._cu_q_buf = torch.zeros(
            max_batch_size + 1, dtype=torch.int32, device=device
        )
        self._paged_kv_indptr_buf = torch.zeros(
            max_batch_size + 1, dtype=torch.int32, device=device
        )
        self._paged_kv_indices_buf = torch.zeros(
            max_paged_kv_indices, dtype=torch.int32, device=device
        )
        self._paged_kv_last_page_len_buf = torch.zeros(
            max_batch_size, dtype=torch.int32, device=device
        )
        workspace = get_global_fi_workspace(device)
        # Kernel choice comes from the engine config
        # (``RuntimeConfig.flashinfer_prefill_backend``); ``None`` -> "auto".
        # Mirrors the diffusion backend — keep the two in lockstep.
        self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            backend=resolve_prefill_backend(),
            use_cuda_graph=True,
            qo_indptr_buf=self._cu_q_buf,
            paged_kv_indptr_buf=self._paged_kv_indptr_buf,
            paged_kv_indices_buf=self._paged_kv_indices_buf,
            paged_kv_last_page_len_buf=self._paged_kv_last_page_len_buf,
        )

    def init_forward_metadata(self, meta: ARAttnMetadata) -> ARAttnPlanHandle:
        self._plan_into_static_buffers(meta)
        return FlashInferARPlan(wrapper=self._wrapper)

    def init_capture_metadata(self, seed_meta: ARAttnMetadata) -> ARAttnPlanHandle:
        return self.init_forward_metadata(seed_meta)

    def replay_metadata(
        self,
        plan: ARAttnPlanHandle,
        replay_meta: ARAttnMetadata,
    ) -> None:
        self._plan_into_static_buffers(replay_meta)

    def _plan_into_static_buffers(self, meta: ARAttnMetadata) -> None:
        if self._wrapper is None or self._layer_proto is None:
            raise RuntimeError(
                "FlashInferARBackend not initialized — call "
                "init_cuda_graph_state(...) before planning."
            )
        if (
            meta.cu_seqlens_q is None
            or meta.paged_kv_indptr is None
            or meta.paged_kv_indices is None
            or meta.paged_kv_last_page_len is None
        ):
            raise ValueError(
                "FlashInferARBackend plan requires cu_seqlens_q, "
                "paged_kv_indptr, paged_kv_indices, and "
                "paged_kv_last_page_len on ARAttnMetadata."
            )
        proto = self._layer_proto
        self._wrapper.plan(
            meta.cu_seqlens_q.to(torch.int32),
            meta.paged_kv_indptr.to(torch.int32),
            meta.paged_kv_indices.to(torch.int32),
            meta.paged_kv_last_page_len.to(torch.int32),
            num_qo_heads=proto.num_heads,
            num_kv_heads=proto.num_kv_heads,
            head_dim_qk=proto.head_dim,
            page_size=1,
            causal=proto.causal,
            sm_scale=proto.scale,
            q_data_type=self._params_dtype,
            kv_data_type=self._params_dtype,
        )

    def forward(
        self,
        layer: ARAttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ARAttnCtx,
    ) -> torch.Tensor:
        if ctx.mode.is_idle():
            return q.new_zeros(q.shape)
        if not isinstance(ctx.plan, FlashInferARPlan):
            raise TypeError(
                f"FlashInferARBackend expected ctx.plan: FlashInferARPlan, "
                f"got {type(ctx.plan).__name__}."
            )
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
        k_cache, v_cache = ctx.kv_pool.kv_buffer(layer.layer_id)
        return ctx.plan.wrapper.run(q, (k_cache, v_cache))


__all__ = ["FlashInferARBackend", "FlashInferARPlan"]
