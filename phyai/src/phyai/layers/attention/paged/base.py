"""Types for `phyai.layers.attention.paged` (paged-KV attention).

One stack serves every paged-KV consumer: the autoregressive LM prefix
(``kernel_role="prefix"``), the flow-matching action expert's joint
attention over prefix + suffix slots (``kernel_role="expert"``), and any
future AR decode loop. Causality is a per-layer trait, not a subsystem:
the two identical stacks this module replaces differed only in the
``causal`` default and their names.

The metadata lifecycle (plan outside capture, replay refreshes contents
in place) is the shared :mod:`phyai.layers.attention.contract`.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from phyai.layers.attention.contract import (
    AttentionBackendBase,
    AttnPlanHandleBase,
    BaseAttnMetadata,
)
from phyai.layers.attention.enums import AttnLayout, AttnMode


if TYPE_CHECKING:
    from phyai.cache import KVCachePool


@dataclass(frozen=True)
class PagedAttnMetadata(BaseAttnMetadata):
    """Host-side description of the next paged attention step.

    Built by the scheduler from per-batch tensors, handed to a
    :class:`PagedAttentionBackend` via :meth:`init_forward_metadata`
    (non-graph) or :meth:`replay_metadata` (graph replay).
    """

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    seq_lens_kv: torch.Tensor | None = None
    paged_kv_indptr: torch.Tensor | None = None
    paged_kv_indices: torch.Tensor | None = None
    paged_kv_last_page_len: torch.Tensor | None = None
    write_indices: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    # Host capacity per sample, including padded prefix and suffix slots.
    # Actual valid lengths remain in the paged metadata tensors.
    max_kv_tokens: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_kv_tokens is not None and (
            isinstance(self.max_kv_tokens, bool)
            or not isinstance(self.max_kv_tokens, int)
            or self.max_kv_tokens < 0
        ):
            raise ValueError("max_kv_tokens must be a nonnegative host integer or None")
        if self.mode == AttnMode.IDLE:
            return
        if self.layout == AttnLayout.RAGGED_3D and self.cu_seqlens_q is None:
            raise ValueError(
                "PagedAttnMetadata: layout=RAGGED_3D requires cu_seqlens_q."
            )


class PagedAttnPlanHandle(AttnPlanHandleBase):
    """Backend-private per-step state for paged attention.

    See :class:`~phyai.layers.attention.contract.AttnPlanHandleBase` for
    the identity-stability invariant graph capture relies on.
    """


@dataclass(frozen=True)
class PagedAttnCtx:
    """Per-call context for paged attention layers.

    The runner builds one ctx per inference step and threads it
    through every layer's forward. ``kv_pool`` and ``write_indices``
    are mandatory — the paged stack is paged-KV by definition.
    """

    backend: "PagedAttentionBackend"
    plan: PagedAttnPlanHandle
    mode: AttnMode
    layout: AttnLayout
    kv_pool: "KVCachePool"
    write_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None


@runtime_checkable
class PagedAttentionLayerProto(Protocol):
    """Static config a backend reads off the paged layer instance."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    causal: bool
    layer_id: int


class PagedAttentionBackend(
    AttentionBackendBase[PagedAttnMetadata, PagedAttnPlanHandle]
):
    """ABC for paged-KV attention backends."""

    def graph_capture_keys(self, *, rollout: bool = False) -> tuple[str | None, ...]:
        """Kernel variants that need separate graphs; ordinary backends use one."""
        return (None,)

    def set_graph_capture_key(self, key: str | None) -> None:
        if key is not None:
            raise ValueError(f"{type(self).__name__} has no graph variant {key!r}")

    def graph_replay_key(
        self, meta: PagedAttnMetadata, *, rollout: bool = False
    ) -> str | None:
        """Select from host metadata, without copying device values to the CPU."""
        return None

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
        """Paged variant of the contract hook.

        Extends the shared signature with ``max_paged_kv_indices`` — the
        static index buffer must cover the largest page-indices array any
        replayed step can carry.
        """
        return None

    @abstractmethod
    def forward(
        self,
        layer: PagedAttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: PagedAttnCtx,
    ) -> torch.Tensor:
        """Run paged attention.

        Backends are responsible for scattering ``k`` / ``v`` into
        ``ctx.kv_pool`` (the layer no longer does it). For
        ``ctx.mode == IDLE`` the backend MUST return zeros without
        any kernel launch.
        """


__all__ = [
    "PagedAttentionBackend",
    "PagedAttentionLayerProto",
    "PagedAttnCtx",
    "PagedAttnMetadata",
    "PagedAttnPlanHandle",
]
