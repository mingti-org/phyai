"""Paged-KV attention through PyTorch SDPA with token-sized cache pages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from phyai.layers.attention.enums import AttnLayout
from phyai.layers.attention.nocache.backends.sdpa import _sdpa_priority_ctx
from phyai.layers.attention.paged.base import (
    PagedAttentionBackend,
    PagedAttentionLayerProto,
    PagedAttnCtx,
    PagedAttnMetadata,
    PagedAttnPlanHandle,
)


@dataclass(frozen=True)
class SdpaPagedPlan(PagedAttnPlanHandle):
    q_offsets: tuple[int, ...]
    kv_offsets: tuple[int, ...]
    indices: torch.Tensor
    masks: tuple[torch.Tensor | None, ...]


class SdpaPagedBackend(PagedAttentionBackend):
    """Gather visible KV slots, then run one SDPA call per ragged sequence."""

    name = "sdpa"

    def __init__(self, runner=None, *, select_kernel: bool = False) -> None:
        del runner
        self.select_kernel = bool(select_kernel)
        self._layer_proto: PagedAttentionLayerProto | None = None

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
        self._layer_proto = layer_proto

    @staticmethod
    def _offsets(meta: PagedAttnMetadata) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if (
            meta.layout != AttnLayout.RAGGED_3D
            or meta.cu_seqlens_q is None
            or meta.paged_kv_indptr is None
            or meta.paged_kv_indices is None
        ):
            raise ValueError(
                "SDPA paged attention requires ragged query offsets and KV page indices"
            )
        q_offsets = tuple(meta.cu_seqlens_q.tolist())
        kv_offsets = tuple(meta.paged_kv_indptr.tolist())
        for offsets in (q_offsets, kv_offsets):
            if (
                len(offsets) != meta.batch_size + 1
                or offsets[0] != 0
                or any(a > b for a, b in zip(offsets, offsets[1:]))
            ):
                raise ValueError(
                    "paged offsets must start at zero and be nondecreasing for each sequence"
                )
        if (
            q_offsets[-1] != meta.num_query_tokens
            or kv_offsets[-1] != meta.paged_kv_indices.numel()
        ):
            raise ValueError("paged offsets must cover the query tokens and KV indices")
        return q_offsets, kv_offsets

    def init_forward_metadata(self, meta: PagedAttnMetadata) -> SdpaPagedPlan:
        if self._layer_proto is None:
            raise RuntimeError(
                "initialize the SDPA paged backend with a layer prototype before planning"
            )
        q_offsets, kv_offsets = self._offsets(meta)
        indices = meta.paged_kv_indices
        if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "paged KV indices must be a one-dimensional integer tensor"
            )
        masks = []
        for qs, qe, ks, ke in zip(q_offsets, q_offsets[1:], kv_offsets, kv_offsets[1:]):
            q_len, kv_len = qe - qs, ke - ks
            mask = None
            if self._layer_proto.causal and q_len < kv_len:
                mask = torch.ones(
                    (q_len, kv_len), dtype=torch.bool, device=indices.device
                ).tril(kv_len - q_len)
            masks.append(mask)
        return SdpaPagedPlan(q_offsets, kv_offsets, indices.clone(), tuple(masks))

    def init_capture_metadata(self, seed_meta: PagedAttnMetadata) -> SdpaPagedPlan:
        return self.init_forward_metadata(seed_meta)

    def replay_metadata(
        self, plan: PagedAttnPlanHandle, replay_meta: PagedAttnMetadata
    ) -> None:
        if not isinstance(plan, SdpaPagedPlan):
            raise TypeError("SDPA paged replay requires an SdpaPagedPlan")
        q_offsets, kv_offsets = self._offsets(replay_meta)
        if (q_offsets, kv_offsets) != (plan.q_offsets, plan.kv_offsets):
            raise ValueError(
                "SDPA paged replay cannot change sequence offsets; capture a new graph"
            )
        indices = replay_meta.paged_kv_indices
        if (
            indices.shape != plan.indices.shape
            or indices.device != plan.indices.device
            or indices.dtype != plan.indices.dtype
        ):
            raise ValueError(
                "SDPA paged replay indices must retain shape, dtype, and device"
            )
        plan.indices.copy_(indices)

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
        if not isinstance(plan, SdpaPagedPlan):
            raise TypeError("SDPA paged attention requires an SdpaPagedPlan")
        if q.shape[0] != plan.q_offsets[-1]:
            raise ValueError("query token count does not match the SDPA paged plan")
        if ctx.kv_pool.page_size != 1:
            raise NotImplementedError("SDPA paged attention supports page_size=1")
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
        if q.shape[0] == 0:
            return torch.empty_like(q)
        keys, values = ctx.kv_pool.gather_kv(layer.layer_id, plan.indices)
        outputs = []
        with (
            torch.autocast(q.device.type, enabled=False),
            _sdpa_priority_ctx(q.device, self.select_kernel),
        ):
            for qs, qe, ks, ke, mask in zip(
                plan.q_offsets,
                plan.q_offsets[1:],
                plan.kv_offsets,
                plan.kv_offsets[1:],
                plan.masks,
            ):
                query = q[qs:qe]
                if qe == qs or ke == ks:
                    outputs.append(torch.zeros_like(query))
                    continue
                if layer.causal and qe - qs > ke - ks:
                    # Fully masked rows are zero; SDPA only sees the visible suffix.
                    masked_rows = (qe - qs) - (ke - ks)
                    outputs.append(torch.zeros_like(query[:masked_rows]))
                    query = query[masked_rows:]
                result = F.scaled_dot_product_attention(
                    query.transpose(0, 1).unsqueeze(0),
                    keys[ks:ke].transpose(0, 1).unsqueeze(0),
                    values[ks:ke].transpose(0, 1).unsqueeze(0),
                    attn_mask=mask,
                    is_causal=layer.causal and mask is None,
                    scale=layer.scale,
                    enable_gqa=layer.num_heads != layer.num_kv_heads,
                )
                outputs.append(result.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)


__all__ = ["SdpaPagedBackend", "SdpaPagedPlan"]
