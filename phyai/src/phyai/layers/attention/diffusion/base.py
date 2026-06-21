"""ABC + per-call types for `phyai.layers.attention.diffusion` (action-expert paged attention)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import torch

from phyai.layers.attention.enums import AttnLayout, AttnMode


if TYPE_CHECKING:
    from phyai.cache import KVCachePool


@dataclass(frozen=True)
class DiffusionAttnMetadata:
    """Host-side description of the next diffusion attention step."""

    mode: AttnMode
    layout: AttnLayout
    batch_size: int
    num_query_tokens: int
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    seq_lens_kv: torch.Tensor | None = None
    paged_kv_indptr: torch.Tensor | None = None
    paged_kv_indices: torch.Tensor | None = None
    paged_kv_last_page_len: torch.Tensor | None = None
    write_indices: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 0 or self.num_query_tokens < 0:
            raise ValueError(
                f"DiffusionAttnMetadata: batch_size={self.batch_size}, "
                f"num_query_tokens={self.num_query_tokens} must be non-negative."
            )
        if self.mode == AttnMode.IDLE:
            return
        if self.layout == AttnLayout.RAGGED_3D and self.cu_seqlens_q is None:
            raise ValueError(
                "DiffusionAttnMetadata: layout=RAGGED_3D requires cu_seqlens_q."
            )


class DiffusionAttnPlanHandle:
    """Backend-private per-step state for diffusion attention."""


@dataclass(frozen=True)
class DiffusionAttnCtx:
    """Per-call context for diffusion attention layers.

    The runner builds one ctx per inference step. ``kv_pool`` and
    ``write_indices`` are mandatory — diffusion attention is paged-KV
    by definition.
    """

    backend: "DiffusionAttentionBackend"
    plan: DiffusionAttnPlanHandle
    mode: AttnMode
    layout: AttnLayout
    kv_pool: "KVCachePool"
    write_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None


@runtime_checkable
class DiffusionAttentionLayerProto(Protocol):
    """Static config a backend reads off the diffusion layer instance."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    causal: bool
    layer_id: int


class DiffusionAttentionBackend(ABC):
    """ABC for diffusion-side paged attention backends."""

    name: ClassVar[str]

    def supports_capture(self) -> bool:
        return False

    def init_cuda_graph_state(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        max_paged_kv_indices: int,
        device: torch.device,
        params_dtype: torch.dtype,
        layer_proto: DiffusionAttentionLayerProto,
    ) -> None:
        """Allocate every static buffer the backend touches inside a captured graph."""
        return None

    def init_capture_metadata(
        self, seed_meta: DiffusionAttnMetadata
    ) -> DiffusionAttnPlanHandle:
        return self.init_forward_metadata(seed_meta)

    def replay_metadata(
        self,
        plan: DiffusionAttnPlanHandle,
        replay_meta: DiffusionAttnMetadata,
    ) -> None:
        return None

    @abstractmethod
    def init_forward_metadata(
        self, meta: DiffusionAttnMetadata
    ) -> DiffusionAttnPlanHandle:
        """Eagerly plan one step. Returns a handle written to ``ctx.plan``."""

    @abstractmethod
    def forward(
        self,
        layer: DiffusionAttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: DiffusionAttnCtx,
    ) -> torch.Tensor:
        """Run diffusion attention.

        Backends are responsible for scattering ``k`` / ``v`` into
        ``ctx.kv_pool``. For ``ctx.mode == IDLE`` the backend MUST
        return zeros without any kernel launch.
        """


__all__ = [
    "DiffusionAttentionBackend",
    "DiffusionAttentionLayerProto",
    "DiffusionAttnCtx",
    "DiffusionAttnMetadata",
    "DiffusionAttnPlanHandle",
]
