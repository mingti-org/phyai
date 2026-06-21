"""Static-allocation cache-pool-aware varlen attention for diffusion / action-expert stacks."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from phyai.layers.attention.diffusion.base import DiffusionAttnCtx
from phyai.layers.attention.diffusion.registry import get_backend_factory


class DiffusionAttention(nn.Module):
    """Static-allocation cache-pool-aware varlen attention for diffusion / action-expert.

    Parameters
    ----------
    num_heads:
        Query head count.
    head_dim:
        Per-head dimension.
    layer_id:
        Index into the :class:`KVCachePool`'s per-layer K/V buffers.
        Required.
    num_kv_heads:
        K/V head count (defaults to ``num_heads`` for full MHA).
    scale:
        Softmax scale, defaults to ``1 / sqrt(head_dim)``.
    causal:
        Causal mask flag. Default ``False`` — diffusion / action-expert
        attention is typically bidirectional within the noise tokens.
        Override to ``True`` if your model needs causal masking.
    backend:
        Canonical name of the backend the runner will resolve for
        this stack. Validated against
        :func:`~phyai.layers.attention.diffusion.registry.get_backend_factory`
        at construction; the layer itself does not instantiate the
        backend.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        layer_id: int,
        num_kv_heads: int | None = None,
        scale: float | None = None,
        causal: bool = False,
        backend: str = "flashinfer",
        backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be a positive multiple of "
                f"num_kv_heads={num_kv_heads} for GQA."
            )
        if layer_id < 0:
            raise ValueError(f"layer_id must be non-negative, got {layer_id}.")
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.layer_id = int(layer_id)
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.causal = bool(causal)
        factory = get_backend_factory(backend)
        self.backend: str = getattr(factory, "name", str(backend))
        self.backend_kwargs: dict[str, Any] = dict(backend_kwargs or {})

    # ------------------------------------------------------------------ #

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: DiffusionAttnCtx,
    ) -> torch.Tensor:
        """Compute diffusion / action-expert attention via ``ctx.backend``.

        Returns ``(N_q, H_q, D)`` — same row count as ``q``. The backend
        scatters K/V into ``ctx.kv_pool``.

        Q and K/V row counts may differ (``S_q != S_kv``). The K/V rows
        passed here are the rows *written* into the pool this step, so
        ``k.shape[0]`` must match ``ctx.write_indices`` (the slots they
        scatter into) — NOT ``q.shape[0]``. This lets the stack express
        cross-attention and general extend; for plain self-attention the
        two counts coincide, as before.
        """
        if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
            raise ValueError(
                f"q/k/v must be 3-D (N, H, D); got q={tuple(q.shape)}, "
                f"k={tuple(k.shape)}, v={tuple(v.shape)}."
            )
        if q.shape[-2] != self.num_heads or q.shape[-1] != self.head_dim:
            raise ValueError(
                f"q heads/dim ({q.shape[-2]}, {q.shape[-1]}) does not match "
                f"module ({self.num_heads}, {self.head_dim})."
            )
        if (
            k.shape[-2] != self.num_kv_heads
            or k.shape[-1] != self.head_dim
            or k.shape != v.shape
        ):
            raise ValueError(
                f"k/v shape mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}, "
                f"expected (N_kv, {self.num_kv_heads}, {self.head_dim})."
            )
        # Invariant: the K/V rows handed in are exactly the rows written
        # into the pool this step, so they must pair 1:1 with the write
        # slots. Decoupling K/V count from q count is what enables
        # S_q != S_kv (cross-attention / extend).
        if k.shape[0] != ctx.write_indices.shape[0]:
            raise ValueError(
                f"k/v row count {k.shape[0]} != write_indices row count "
                f"{ctx.write_indices.shape[0]}; K/V rows must pair 1:1 with "
                f"the cache slots they are scattered into."
            )

        return ctx.backend.forward(self, q, k, v, ctx)

    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, layer_id={self.layer_id}, "
            f"causal={self.causal}, backend={self.backend!r}"
        )


__all__ = ["DiffusionAttention"]
