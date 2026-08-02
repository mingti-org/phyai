"""ABC and per-call types for FlashInfer-backed Gated Delta Net."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import torch

from phyai.layers.attention.enums import AttnLayout, AttnMode


@dataclass(frozen=True)
class GatedDeltaNetMetadata:
    """Host-side description of one GDN step."""

    mode: AttnMode
    layout: AttnLayout
    batch_size: int
    num_query_tokens: int
    cu_seqlens: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 0 or self.num_query_tokens < 0:
            raise ValueError(
                f"GatedDeltaNetMetadata: batch_size={self.batch_size}, "
                f"num_query_tokens={self.num_query_tokens} must be non-negative."
            )
        if self.mode == AttnMode.IDLE:
            return
        if self.mode == AttnMode.MIXED:
            raise NotImplementedError("GatedDeltaNet does not support MIXED mode.")
        if self.mode == AttnMode.PREFILL and self.cu_seqlens is None:
            raise ValueError("GatedDeltaNetMetadata: PREFILL requires cu_seqlens.")
        if self.mode == AttnMode.DECODE and self.layout != AttnLayout.PADDED_4D:
            raise ValueError("GatedDeltaNetMetadata: DECODE requires PADDED_4D layout.")


class GatedDeltaNetPlanHandle:
    """Backend-private per-step state."""


@dataclass(frozen=True)
class GatedDeltaNetCtx:
    """Per-call GDN context owned by the model runner.

    ``state`` is either a per-batch state tensor or a state pool. Decode
    selects pool mode when ``state_indices`` is present. Prefill accepts only
    per-batch initial state and writes its final state to ``output_state``.
    ``output`` optionally supplies a preallocated kernel output buffer.
    """

    backend: "GatedDeltaNetBackend"
    plan: GatedDeltaNetPlanHandle
    mode: AttnMode
    layout: AttnLayout
    cu_seqlens: torch.Tensor | None = None
    state: torch.Tensor | None = None
    output_state: torch.Tensor | None = None
    state_indices: torch.Tensor | None = None
    output_state_indices: torch.Tensor | None = None
    output: torch.Tensor | None = None


@runtime_checkable
class GatedDeltaNetLayerProto(Protocol):
    """Static configuration read by a GDN backend."""

    num_query_heads: int
    num_key_heads: int
    num_value_heads: int
    num_state_heads: int
    head_dim: int
    scale: float
    use_qk_l2norm: bool


class GatedDeltaNetBackend(ABC):
    """ABC for Gated Delta Net kernel backends."""

    name: ClassVar[str]

    def supports_capture(self) -> bool:
        return False

    def init_cuda_graph_state(
        self,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        device: torch.device,
        params_dtype: torch.dtype,
        layer_proto: GatedDeltaNetLayerProto,
    ) -> None:
        return None

    def init_capture_metadata(
        self, seed_meta: GatedDeltaNetMetadata
    ) -> GatedDeltaNetPlanHandle:
        return self.init_forward_metadata(seed_meta)

    def replay_metadata(
        self,
        plan: GatedDeltaNetPlanHandle,
        replay_meta: GatedDeltaNetMetadata,
    ) -> None:
        return None

    @abstractmethod
    def init_forward_metadata(
        self, meta: GatedDeltaNetMetadata
    ) -> GatedDeltaNetPlanHandle:
        """Prepare one eager step and return its opaque plan handle."""

    @abstractmethod
    def forward(
        self,
        layer: GatedDeltaNetLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ctx: GatedDeltaNetCtx,
    ) -> torch.Tensor:
        """Run one prefill or decode GDN step."""


__all__ = [
    "GatedDeltaNetBackend",
    "GatedDeltaNetCtx",
    "GatedDeltaNetLayerProto",
    "GatedDeltaNetMetadata",
    "GatedDeltaNetPlanHandle",
]
