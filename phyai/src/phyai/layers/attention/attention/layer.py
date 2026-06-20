"""Stateless prefill attention — no KV cache, no radix.

The math is ``softmax(Q K^T / sqrt(D) + mask) V``; nothing is cached
between calls. Three kernel backends share one forward contract,
selected at construction time via ``backend=``:

* ``"flashinfer"`` (default) — :mod:`flashinfer.prefill`. Uses
  ``single_prefill_with_kv_cache`` when the batch is one sequence and
  :class:`BatchPrefillWithRaggedKVCacheWrapper` otherwise.
* ``"sdpa"`` — :func:`torch.nn.functional.scaled_dot_product_attention`.
  Wrapped with :func:`torch.compile` (``dynamic=True``) by default so
  the mask build / soft-cap fallback / SDPA call fuse on first use;
  toggle via ``backend_kwargs={"compile": False}``.
* ``"eager"`` — pure PyTorch matmul + masked softmax. Reference path,
  slow but exact.

Optional left sliding window: each query attends to at most
``sliding_window`` keys, with the current position counted toward the
window (HF / Mistral / Qwen / Gemma2 convention). Sliding window is
causal-only.

This module is the attention *op* — Q/K/V projection and RoPE are the
caller's responsibility. Q/K/V come in already-projected and the
attention output goes back out in the same layout.

Per-call ctx
------------
``forward(q, k, v, ctx=None, *, cu_seqlens_q=None, cu_seqlens_kv=None)``.
Two usage modes:

* **Convenience (vision tower / unit tests)** — pass ``ctx=None``.
  The layer infers the layout from ``q.ndim``, lazily constructs a
  backend instance the first call (cached on ``self``), and builds a
  degenerate :class:`AttnCtx` in-place.
* **Runner-coordinated** — pass an explicit :class:`AttnCtx`. Used by
  callers that share one runner-scoped backend across many layers and
  pre-stage metadata via the four-hook contract.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from phyai.layers.attention.attention.base import (
    AttentionBackend,
    AttnCtx,
    AttnMetadata,
)
from phyai.layers.attention.attention.registry import get_backend_factory
from phyai.layers.attention.enums import AttnLayout, AttnMode


class Attention(nn.Module):
    """Prefill-only attention with selectable kernel backend.

    Parameters
    ----------
    num_heads:
        Number of query heads.
    head_dim:
        Per-head dimension. ``Q @ K^T`` is divided by ``sqrt(head_dim)``
        unless ``scale`` overrides it.
    num_kv_heads:
        Number of K/V heads. Defaults to ``num_heads`` (MHA). For GQA,
        must divide ``num_heads``.
    scale:
        Softmax scale. Defaults to ``1 / sqrt(head_dim)``.
    causal:
        Apply a (lower-triangular) causal mask. Required when
        ``sliding_window`` is set.
    sliding_window:
        Window size in tokens — current position counted in the window.
        Query at offset ``q_pos`` attends to keys
        ``[max(0, q_pos - W + 1), q_pos]``. ``None`` means full prefix.
    logits_soft_cap:
        If set, apply ``cap * tanh(logits / cap)`` to attention logits
        before softmax (Gemma2 / Grok / Gemini style).
    backend:
        ``"flashinfer"`` (default), ``"sdpa"``, or ``"eager"``. Resolved
        through :func:`~phyai.layers.attention.attention.registry.get_backend_factory`.
    backend_kwargs:
        Optional dict forwarded to the backend factory after ``runner``.
        ``"sdpa"`` accepts ``{"compile": bool}``; ``"flashinfer"``
        accepts ``{"fi_workspace": Tensor, "workspace_bytes": int}``.

    Forward shape conventions
    -------------------------
    Two layouts are auto-detected by ``q.ndim`` when ``ctx=None``:

    * **Padded batch (4-D)** — ``q: (B, S_q, H, D)``,
      ``k/v: (B, S_kv, H_kv, D)``. -> ``out: (B, S_q, H, D)``.
    * **Ragged / varlen (3-D)** — packed buffers plus indptrs.
      ``q: (N_q, H, D)``, ``k/v: (N_kv, H_kv, D)``. ``cu_seqlens_q``
      required (``cu_seqlens_kv`` defaults to ``cu_seqlens_q``).
      -> ``out: (N_q, H, D)``.

    For "append" prefill where K/V is longer than Q, queries are aligned
    with the *trailing* keys (``q_pos[i] = i + (S_kv - S_q)``).
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        num_kv_heads: int | None = None,
        scale: float | None = None,
        causal: bool = True,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
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
        if sliding_window is not None:
            if sliding_window <= 0:
                raise ValueError(
                    f"sliding_window must be positive, got {sliding_window}."
                )
            if not causal:
                raise ValueError("sliding_window requires causal=True.")
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.causal = bool(causal)
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        factory = get_backend_factory(backend)
        self._backend_factory = factory
        self._backend_kwargs: dict[str, Any] = dict(backend_kwargs or {})
        self.backend: str = getattr(factory, "name", str(backend))
        self._lazy_backend: AttentionBackend | None = None

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx | None = None,
        *,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.ndim == 4:
            self._check_padded(q, k, v)
            layout = AttnLayout.PADDED_4D
        elif q.ndim == 3:
            if ctx is None and cu_seqlens_q is None:
                raise ValueError(
                    "ragged forward requires cu_seqlens_q (q has shape "
                    f"{tuple(q.shape)})."
                )
            self._check_ragged(q, k, v)
            layout = AttnLayout.RAGGED_3D
        else:
            raise ValueError(
                f"q must be 3-D (ragged) or 4-D (padded batch); got shape "
                f"{tuple(q.shape)}."
            )

        if ctx is None:
            ctx = self._build_default_ctx(q, k, v, layout, cu_seqlens_q, cu_seqlens_kv)
        return ctx.backend.forward(self, q, k, v, ctx)

    # ------------------------------------------------------------------ #
    # Default ctx construction (vision tower / unit tests)               #
    # ------------------------------------------------------------------ #

    def _build_default_ctx(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layout: AttnLayout,
        cu_seqlens_q: torch.Tensor | None,
        cu_seqlens_kv: torch.Tensor | None,
    ) -> AttnCtx:
        backend = self._ensure_backend()
        if layout.is_padded():
            B, S_q = q.shape[0], q.shape[1]
            S_kv = k.shape[1]
            num_query_tokens = B * S_q
            # The flashinfer B>1 padded path plans a ragged-KV wrapper, which needs
            # per-row q/kv offsets. Synthesize uniform cu_seqlens from the padded
            # shapes so a 4-D batch works without the caller hand-packing — S_q may
            # differ from S_kv (rectangular cross-attention). B==1 ignores these
            # (single_prefill); sdpa/eager build their mask from shapes, not these.
            if cu_seqlens_q is None:
                cu_seqlens_q = torch.arange(
                    0, (B + 1) * S_q, S_q, dtype=torch.int32, device=q.device
                )
            if cu_seqlens_kv is None:
                cu_seqlens_kv = torch.arange(
                    0, (B + 1) * S_kv, S_kv, dtype=torch.int32, device=q.device
                )
        else:
            B = (cu_seqlens_q.numel() - 1) if cu_seqlens_q is not None else 1
            num_query_tokens = q.shape[0]
        meta = AttnMetadata(
            mode=AttnMode.PREFILL,
            layout=layout,
            batch_size=int(B),
            num_query_tokens=int(num_query_tokens),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            extras={"layer_proto": self, "q_dtype": q.dtype, "kv_dtype": k.dtype},
        )
        plan = backend.init_forward_metadata(meta)
        return AttnCtx(
            backend=backend,
            plan=plan,
            mode=AttnMode.PREFILL,
            layout=layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
        )

    def _ensure_backend(self) -> AttentionBackend:
        if self._lazy_backend is None:
            self._lazy_backend = self._backend_factory(None, **self._backend_kwargs)
        return self._lazy_backend

    # ------------------------------------------------------------------ #
    # Shape validation                                                   #
    # ------------------------------------------------------------------ #

    def _check_padded(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape:
            raise ValueError(
                f"k.shape={tuple(k.shape)} must equal v.shape={tuple(v.shape)}."
            )
        B, _, H_q, D = q.shape
        if H_q != self.num_heads or D != self.head_dim:
            raise ValueError(
                f"q heads/dim ({H_q}, {D}) does not match module "
                f"({self.num_heads}, {self.head_dim})."
            )
        if k.shape[0] != B or k.shape[2] != self.num_kv_heads or k.shape[3] != D:
            raise ValueError(
                f"k.shape={tuple(k.shape)} not compatible with q="
                f"{tuple(q.shape)} and num_kv_heads={self.num_kv_heads}."
            )

    def _check_ragged(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape:
            raise ValueError(
                f"k.shape={tuple(k.shape)} must equal v.shape={tuple(v.shape)}."
            )
        _, H_q, D = q.shape
        _, H_kv, _ = k.shape
        if H_q != self.num_heads or D != self.head_dim or H_kv != self.num_kv_heads:
            raise ValueError(
                f"ragged input head/dim mismatch (q: {H_q}, {D}; k: {H_kv}); "
                f"expected ({self.num_heads}, {self.head_dim}) and "
                f"num_kv_heads={self.num_kv_heads}."
            )

    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        s = (
            f"num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, causal={self.causal}, "
            f"backend={self.backend!r}"
        )
        if self.sliding_window is not None:
            s += f", sliding_window={self.sliding_window}"
        if self.logits_soft_cap is not None:
            s += f", logits_soft_cap={self.logits_soft_cap}"
        return s


__all__ = ["Attention"]
