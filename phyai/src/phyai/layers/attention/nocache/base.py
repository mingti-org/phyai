"""Types for `phyai.layers.attention.nocache` (the no-cache stack).

This subpackage is **stateless attention**: prefill-only, no KV cache,
no pool reads, no per-layer state across calls. Used today by the
SigLIP vision tower, cosmos3's DiT, and rectangular cross-attention.

Per-call lifecycle
------------------
The runner (or the layer's convenience-ctx builder) hands the layer an
:class:`AttnCtx`. Layers do not store backends; they route via
``ctx.backend.forward(layer, q, k, v, ctx)``. The runner-driven path
uses :meth:`AttentionBackend.init_forward_metadata` to build a plan
once per step; the convenience path (vision tower / unit tests) lazily
builds a degenerate ctx on the first ctx-less ``forward`` call.

The metadata lifecycle is the shared
:mod:`phyai.layers.attention.contract`; no-cache backends rarely carry
static buffers, so the graph hooks usually stay at their no-op defaults.

Sibling stack: ``phyai.layers.attention.paged`` (paged-KV attention).
The two are typed independently — :class:`AttnCtx` here deliberately
does NOT carry a ``kv_pool``.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from phyai.layers.attention.contract import (
    AttentionBackendBase,
    AttnPlanHandleBase,
    BaseAttnMetadata,
)
from phyai.layers.attention.enums import AttnLayout, AttnMode
from phyai.layers.attention.mask import AttnMask


@dataclass(frozen=True)
class AttnMetadata(BaseAttnMetadata):
    """Host-side description of the next attention step (no-cache).

    Built by the runner from per-batch tensors, handed to a
    :class:`AttentionBackend` via :meth:`init_forward_metadata`.
    Backends pick the fields they care about; unused fields stay
    ``None``.

    Fields
    ------
    cu_seqlens_q, cu_seqlens_kv:
        ``(B+1,)`` int32 cumulative offsets. Required when
        ``layout == RAGGED_3D``; backends that own a wrapper plan
        with these.
    seq_lens_kv:
        ``(B,)`` int32 — per-sample full KV length. Mostly informational.
    position_ids:
        ``(N,)`` int32 absolute positions. Some runners use these for
        rope or for staging into static buffers outside captured regions.
    layer_proto, q_dtype, kv_dtype:
        Planning inputs for backends that own a wrapper (the flashinfer
        B>1 path plans with the layer's head geometry and the tensor
        dtypes). Explicit fields — an untyped ``extras`` dict used to
        smuggle these through.
    mask:
        Declarative :class:`~phyai.layers.attention.mask.AttnMask` for
        this step (padded layout only); backends lower it their way.
    attn_bias:
        Additive logit bias broadcastable to ``(B, H, S_q, S_kv)``. Padded
        layout only; applied after logit scaling and before the structural mask.
    """

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    seq_lens_kv: torch.Tensor | None = None
    position_ids: torch.Tensor | None = None
    layer_proto: "AttentionLayerProto | None" = None
    q_dtype: torch.dtype | None = None
    kv_dtype: torch.dtype | None = None
    mask: AttnMask | None = None
    attn_bias: torch.Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode == AttnMode.IDLE:
            return
        if self.layout == AttnLayout.RAGGED_3D and self.cu_seqlens_q is None:
            raise ValueError("AttnMetadata: layout=RAGGED_3D requires cu_seqlens_q.")


class AttnPlanHandle(AttnPlanHandleBase):
    """Backend-private per-step state for the no-cache stack."""


@dataclass(frozen=True)
class AttnCtx:
    """Per-call context handed to the layer's ``forward``.

    The runner builds one ctx per inference step (when integrating
    with a runner-driven attention stack); the convenience path used
    by the vision tower / unit tests lazily builds a degenerate ctx
    on the first ctx-less forward call. Layers route via
    ``ctx.backend.forward(self, q, k, v, ctx)``.
    """

    backend: "AttentionBackend"
    plan: AttnPlanHandle
    mode: AttnMode
    layout: AttnLayout
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_kv: torch.Tensor | None = None
    mask: AttnMask | None = None
    attn_bias: torch.Tensor | None = None


@runtime_checkable
class AttentionLayerProto(Protocol):
    """Static config a backend reads off the layer instance.

    :class:`~phyai.layers.attention.nocache.layer.Attention` satisfies
    this Protocol. Backends type their ``forward(layer, ...)`` against
    it so they can read config without coupling to the concrete layer.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    causal: bool
    sliding_window: int | None
    logits_soft_cap: float | None


class AttentionBackend(AttentionBackendBase[AttnMetadata, AttnPlanHandle]):
    """ABC for every no-cache attention backend."""

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        """Run attention.

        Backend dispatches internally on ``ctx.mode`` and ``ctx.layout``.
        For ``mode == IDLE`` the backend MUST return zeros without any
        kernel launch.
        """


__all__ = [
    "AttentionBackend",
    "AttentionLayerProto",
    "AttnCtx",
    "AttnMetadata",
    "AttnPlanHandle",
]
