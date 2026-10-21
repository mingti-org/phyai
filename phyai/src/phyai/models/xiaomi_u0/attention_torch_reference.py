"""UNIS-specific paged reference attention with eager BF16 boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from phyai.layers.attention.enums import AttnLayout
from phyai.layers.attention.paged.base import (
    PagedAttentionBackend,
    PagedAttentionLayerProto,
    PagedAttnCtx,
    PagedAttnMetadata,
    PagedAttnPlanHandle,
)


@dataclass(frozen=True)
class UNISTorchReferencePlan(PagedAttnPlanHandle):
    """One single-sequence plan; the runner owns its KV storage."""

    metadata: PagedAttnMetadata
    contiguous_kv_start: int | None = None


class UNISTorchReferenceBackend(PagedAttentionBackend):
    """Correctness reference for one UNIS sequence, without graph capture.

    This backend preserves the reference's BF16 score and probability
    materialization. It is selected explicitly by the U0 runner and is
    not registered as a general paged kernel or an optimization candidate.
    """

    name = "torch_reference"

    def supports_capture(self) -> bool:
        """Return False because this reference backend does not support graph capture."""
        return False

    def init_forward_metadata(self, meta: PagedAttnMetadata) -> PagedAttnPlanHandle:
        """Validate a single-sequence KV suffix and plan its visible cache slots.

        Cache writes must correspond to the absolute positions at the end of
        the visible sequence. Contiguous physical slots are recorded for direct
        buffer access during attention.
        """
        if meta.batch_size != 1 or meta.layout != AttnLayout.RAGGED_3D:
            raise ValueError("UNIS reference attention requires one ragged sequence.")
        if meta.num_query_tokens <= 0:
            raise ValueError("UNIS reference attention requires nonempty queries.")
        if meta.paged_kv_indices is None or meta.position_ids is None:
            raise ValueError(
                "Explicit cache indices and absolute positions are required."
            )
        if meta.write_indices is None:
            raise ValueError("Explicit cache write slots are required.")
        if meta.position_ids.shape != (meta.num_query_tokens,):
            raise ValueError("Positions must match the query count.")
        if meta.write_indices.shape != (meta.num_query_tokens,):
            raise ValueError("Write slots must match the query count.")
        if meta.paged_kv_indices.ndim != 1:
            raise ValueError(
                "Visible cache indices must be a one-dimensional sequence."
            )
        length = meta.paged_kv_indices.numel()
        if not 0 < meta.num_query_tokens <= length:
            raise ValueError("Visible KV length must cover all current query tokens.")
        positions = torch.arange(
            length - meta.num_query_tokens,
            length,
            device=meta.position_ids.device,
            dtype=meta.position_ids.dtype,
        )
        if not torch.equal(meta.position_ids, positions):
            raise ValueError("UNIS reference queries must be the suffix of visible KV.")
        visible_suffix = meta.paged_kv_indices[-meta.num_query_tokens :]
        if not torch.equal(visible_suffix.to(torch.int64), meta.write_indices):
            raise ValueError("Current write slots must match the visible KV suffix.")
        # Resolve the physical range once per plan, rather than per layer.
        slots = meta.paged_kv_indices.detach().cpu().tolist()
        contiguous = slots == list(range(slots[0], slots[0] + length))
        return UNISTorchReferencePlan(
            metadata=meta,
            contiguous_kv_start=slots[0] if contiguous else None,
        )

    def forward(
        self,
        layer: PagedAttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: PagedAttnCtx,
    ) -> torch.Tensor:
        """Write current K/V and return causal attention in query-head layout.

        Q/K/V must be BF16, with shape (tokens, heads, head_dim), and the KV
        pool must use one token per page. Score and probability casts preserve
        the reference implementation's rounding boundaries.
        """
        if not isinstance(ctx.plan, UNISTorchReferencePlan):
            raise TypeError("UNIS reference attention requires its own plan handle.")
        if ctx.kv_pool.page_size != 1 or not layer.causal:
            raise ValueError(
                "UNIS reference attention requires causal, page-size-one KV."
            )
        if not math.isclose(layer.scale, 1.0 / math.sqrt(layer.head_dim)):
            raise ValueError(
                "UNIS reference attention requires the default head scaling."
            )
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("The validated UNIS reference path requires BF16 Q/K/V.")
        if layer.num_heads % layer.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads.")
        meta = ctx.plan.metadata
        if q.shape != (meta.num_query_tokens, layer.num_heads, layer.head_dim):
            raise ValueError("Query shape does not match the planned heads and tokens.")
        kv_shape = (meta.num_query_tokens, layer.num_kv_heads, layer.head_dim)
        if k.shape != kv_shape or v.shape != kv_shape:
            raise ValueError(
                "Current K/V shape does not match the planned heads and tokens."
            )
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
        start = ctx.plan.contiguous_kv_start
        if start is None:
            cached_k, cached_v = ctx.kv_pool.gather_kv(
                layer.layer_id, meta.paged_kv_indices.to(torch.int64)
            )
        else:
            length = meta.paged_kv_indices.numel()
            if start < 0 or start + length > ctx.kv_pool.num_slots:
                raise ValueError("Visible KV range is outside the cache pool.")
            cached_k = ctx.kv_pool.k_buffer(layer.layer_id)[start : start + length, 0]
            cached_v = ctx.kv_pool.v_buffer(layer.layer_id)[start : start + length, 0]
        groups = layer.num_heads // layer.num_kv_heads
        query = q.transpose(0, 1).unsqueeze(0)
        keys = cached_k.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1)
        values = cached_v.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1)
        # Keep division and BF16 rounding boundaries identical to the reference.
        scores = query @ keys.transpose(-1, -2) / math.sqrt(layer.head_dim)
        blocked = (
            torch.arange(cached_k.shape[0], device=q.device)[None, :]
            > meta.position_ids[:, None]
        )
        mask = torch.zeros(blocked.shape, device=q.device, dtype=q.dtype)
        mask.masked_fill_(blocked, torch.finfo(q.dtype).min)
        scores = scores + mask[None, None]
        probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
        attended = probabilities @ values
        return attended[0].transpose(0, 1).contiguous()


__all__ = ["UNISTorchReferenceBackend", "UNISTorchReferencePlan"]
