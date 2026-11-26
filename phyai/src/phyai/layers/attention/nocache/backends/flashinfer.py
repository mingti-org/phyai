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

from phyai.layers.attention.nocache.base import (
    AttentionBackend,
    AttentionLayerProto,
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
)
from phyai.layers.attention.mask import pack_rows
from phyai.layers.attention.utils import (
    get_global_fi_workspace,
)


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


class FlashInferAttentionBackend(AttentionBackend):
    """flashinfer prefill kernels for :class:`Attention`.

    The single-sequence (B=1) path skips ``plan()`` entirely and is
    safe inside captured graphs. The batched ragged path calls
    ``wrapper.plan()`` from :meth:`init_forward_metadata` (called
    outside any captured region) and ``wrapper.run()`` from
    :meth:`forward`.
    """

    name = "flashinfer"

    def __init__(
        self,
        runner: "ModelRunner | None" = None,
        *,
        fi_workspace: torch.Tensor | None = None,
        workspace_bytes: int | None = None,
        use_cuda_graph: bool = False,
        qo_indptr_buf: torch.Tensor | None = None,
        kv_indptr_buf: torch.Tensor | None = None,
        prefill_backend: str | None = None,
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
        self._use_cuda_graph = bool(use_cuda_graph)
        self._qo_indptr_buf = qo_indptr_buf
        self._kv_indptr_buf = kv_indptr_buf
        # The catalog row that selected this backend pins the kernel; ``None``
        # means the row was FlashInfer's general entry point, whose own
        # heuristic is spelled "auto".
        self._prefill_backend = prefill_backend or "auto"
        # The single-sequence API exposes only auto, FA2, and FA3. Other
        # selected kernels use the batch wrapper even for a single sequence.
        self._use_single_prefill = self._prefill_backend in {"auto", "fa2", "fa3"}
        if fi_workspace is not None:
            # In graph mode the fixed batch size may only be known when the
            # first metadata packet arrives, so defer wrapper construction
            # until ``_ensure_wrapper`` can allocate matching indptr buffers.
            self._fi_workspace = fi_workspace
            if not self._use_cuda_graph or (
                qo_indptr_buf is not None and kv_indptr_buf is not None
            ):
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
        kwargs: dict[str, object] = {}
        if self._use_cuda_graph:
            if self._qo_indptr_buf is None or self._kv_indptr_buf is None:
                raise ValueError(
                    "CUDA-graph FlashInfer attention needs qo_indptr_buf and "
                    "kv_indptr_buf."
                )
            kwargs.update(
                use_cuda_graph=True,
                qo_indptr_buf=self._qo_indptr_buf,
                kv_indptr_buf=self._kv_indptr_buf,
            )
        kwargs["backend"] = self._prefill_backend
        self._fi_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", **kwargs
        )

    def _ensure_wrapper(self, device: torch.device, *, batch_size: int):
        if self._fi_wrapper is None:
            if self._use_cuda_graph:
                if self._qo_indptr_buf is None:
                    self._qo_indptr_buf = torch.empty(
                        batch_size + 1, dtype=torch.int32, device=device
                    )
                if self._kv_indptr_buf is None:
                    self._kv_indptr_buf = torch.empty(
                        batch_size + 1, dtype=torch.int32, device=device
                    )
            workspace = (
                self._fi_workspace
                if self._fi_workspace is not None
                else get_global_fi_workspace(
                    device,
                    workspace_bytes=self._workspace_bytes,
                    prefill_backend=self._prefill_backend,
                )
            )
            self._build_wrapper(workspace)
        return self._fi_wrapper

    def init_forward_metadata(self, meta: AttnMetadata) -> AttnPlanHandle:
        from phyai.layers.attention.enums import AttnMode

        if meta.mask is not None:
            if meta.mask.segments is not None:
                # The catalog routes segments-carrying calls to the dense
                # backends; reaching here means a hand-built ctx skipped
                # selection.
                raise NotImplementedError(
                    "FlashInferAttentionBackend does not lower "
                    "AttnMask.segments; use the sdpa/eager backends."
                )
        elif meta.mode == AttnMode.IDLE or (
            meta.batch_size <= 1 and self._use_single_prefill
        ):
            return FlashInferAttentionPlan(wrapper=None)
        if meta.mode == AttnMode.IDLE:
            return FlashInferAttentionPlan(wrapper=None)
        if meta.cu_seqlens_q is None:
            raise ValueError(
                "FlashInferAttentionBackend.init_forward_metadata requires "
                "cu_seqlens_q on AttnMetadata for batch_size > 1."
            )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # The module docstring's contract: plan happens OUTSIDE any
            # captured region. Reaching here inside capture means the layer's
            # convenience path was first exercised during the capture pass —
            # the wrapper construction and plan would be baked into the graph.
            raise RuntimeError(
                "FlashInferAttentionBackend.plan called inside CUDA graph "
                "capture. Warm the call up once outside capture (same shapes "
                "and mode) or drive the layer with a pre-planned ctx."
            )
        cu_q = meta.cu_seqlens_q
        cu_kv = meta.cu_seqlens_kv if meta.cu_seqlens_kv is not None else cu_q
        layer_proto = meta.layer_proto
        if layer_proto is None:
            raise ValueError(
                "FlashInferAttentionBackend.init_forward_metadata requires "
                "meta.layer_proto for the B>1 plan; pass it via AttnMetadata."
            )
        wrapper = self._ensure_wrapper(cu_q.device, batch_size=meta.batch_size)
        q_dtype = meta.q_dtype
        kv_dtype = meta.kv_dtype if meta.kv_dtype is not None else q_dtype
        if q_dtype is None:
            raise ValueError(
                "FlashInferAttentionBackend.init_forward_metadata requires "
                "meta.q_dtype (and optional meta.kv_dtype)."
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
        if ctx.attn_bias is not None:
            raise ValueError(
                "FlashInfer no-cache attention does not support additive logit bias"
            )
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
        if ctx.mask is not None:
            return self._forward_padded_masked(layer, q, k, v, ctx)
        if B == 1 and self._use_single_prefill:
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

    def _forward_padded_masked(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        """Padded batch + declarative KV validity, lowered to varlen.

        Invalid K/V rows are gathered out (one batched pack, replacing the
        per-row Python loop this pattern used to be hand-rolled with) and
        the pre-planned ragged wrapper runs on the compacted rows.

        Causal layers additionally pack Q: FlashInfer aligns queries with
        the *trailing* keys, so leaving padded query rows in place would
        shift every real row's alignment. Packing both sides restores
        ``q_pos == kv_pos`` for right-padded self-attention; outputs
        scatter back into the padded layout (pad rows read as zeros).
        A scattered ``key_mask`` under a causal layer has no single
        position renumbering, so it is refused rather than guessed.
        """
        B, S_q, H, D = q.shape
        S_kv = k.shape[1]
        mask = ctx.mask
        valid_kv = mask.kv_valid(B, S_kv, q.device)
        plan = ctx.plan
        if not isinstance(plan, FlashInferAttentionPlan) or plan.wrapper is None:
            raise ValueError(
                "FlashInferAttentionBackend masked forward requires a planned "
                "FlashInferAttentionPlan; the runner / layer must call "
                "init_forward_metadata first."
            )
        if not layer.causal:
            _, _, (k_packed, v_packed) = pack_rows(valid_kv, k, v)
            out = plan.wrapper.run(q.reshape(B * S_q, H, D), k_packed, v_packed)
            return out.reshape(B, S_q, H, D)
        if mask.key_mask is not None:
            raise NotImplementedError(
                "causal attention over a scattered key_mask is ambiguous "
                "after packing; the sdpa/eager dense path serves it."
            )
        if S_q != S_kv:
            raise ValueError(
                "causal + seq_lens_kv expects self-attention shapes "
                f"(S_q == S_kv); got S_q={S_q}, S_kv={S_kv}."
            )
        _, index, (q_packed, k_packed, v_packed) = pack_rows(valid_kv, q, k, v)
        out_packed = plan.wrapper.run(q_packed, k_packed, v_packed)
        out = q.new_zeros(B * S_q, H, D)
        out.index_copy_(0, index, out_packed)
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
        if B == 1 and self._use_single_prefill:
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
            backend=self._prefill_backend,
        )


__all__ = ["FlashInferAttentionBackend", "FlashInferAttentionPlan"]
