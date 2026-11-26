"""Static-allocation cache-pool-aware varlen attention over paged KV slots."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from phyai.layers.attention.paged.base import PagedAttnCtx
from phyai.kernel.call import backend_preference


class PagedAttention(nn.Module):
    """Static-allocation cache-pool-aware varlen attention over paged KV.

    "Static" means the KV slots come from a one-shot
    :class:`~phyai.cache.static_cache.StaticCache` allocator (contiguous
    range, reset between requests, no eviction). The attention module
    itself is allocator-agnostic — it just routes ``q``/``k``/``v`` to
    the runner-supplied backend through ``ctx``.

    One class serves every paged consumer; what used to be two identical
    subsystems ("ar" and "diffusion") is expressed per layer:

    * the LM prefix of a VLA constructs
      ``PagedAttention(..., causal=False, kernel_role="prefix")``
      (PaliGemma-style bidirectional prefix) or ``causal=True`` for a
      genuine AR decoder;
    * the action expert's joint attention over prefix + suffix slots
      constructs ``PagedAttention(..., causal=False, kernel_role="expert")``
      — the block structure comes from which KV slots the metadata makes
      visible, not from a mask.

    Parameters
    ----------
    num_heads:
        Query head count.
    head_dim:
        Per-head dimension.
    layer_id:
        Index into the :class:`KVCachePool`'s per-layer K/V buffers.
        Required.
    causal:
        Causal mask flag. Required — the merged stack serves both
        causal decoders and bidirectional prefix/expert layers, so the
        caller states which one this layer is.
    num_kv_heads:
        K/V head count (defaults to ``num_heads`` for full MHA).
    scale:
        Softmax scale, defaults to ``1 / sqrt(head_dim)``.
    backend:
        Canonical name of the backend the runner will resolve for
        this stack (``"flashinfer"`` or ``"sdpa"``).
        Validated against the kernel catalog at construction; the
        layer itself does not instantiate the backend.
    kernel_role:
        Policy-visible role for this call site (``"prefix"``,
        ``"expert"``, or a caller-chosen name).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        layer_id: int,
        causal: bool,
        num_kv_heads: int | None = None,
        scale: float | None = None,
        backend: str | None = None,
        kernel_role: str = "paged",
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
        self.backend = backend
        # Validated against the catalog. The runner owns selection for the paged
        # stack, because those backends hold runner-scoped buffers; this layer
        # only records the preference for the runner to pass along.
        self.prefer = backend_preference("attention_paged", backend)
        self.kernel_role = kernel_role

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: PagedAttnCtx,
    ) -> torch.Tensor:
        """Compute paged attention via ``ctx.backend``.

        Returns ``(N_q, H_q, D)`` — same row count as ``q``. The backend
        is responsible for scattering K/V into ``ctx.kv_pool``.

        Q and K/V row counts may differ (``S_q != S_kv``). The K/V rows
        passed here are the rows *written* into the pool this step, so
        ``k.shape[0]`` must match ``ctx.write_indices`` (the slots they
        scatter into) — NOT ``q.shape[0]``. This is what lets the paged
        stack express cross-attention (Q from one sequence, fresh K/V
        from another) and general extend (a short query chunk appended
        to a longer cached KV span). For plain self-attention the two
        counts coincide.
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
        if k.shape[0] != ctx.write_indices.shape[0]:
            raise ValueError(
                f"k/v row count {k.shape[0]} != write_indices row count "
                f"{ctx.write_indices.shape[0]}; K/V rows must pair 1:1 with "
                f"the cache slots they are scattered into."
            )

        return ctx.backend.forward(self, q, k, v, ctx)

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, layer_id={self.layer_id}, "
            f"causal={self.causal}, backend={self.backend!r}"
        )


__all__ = ["PagedAttention"]
