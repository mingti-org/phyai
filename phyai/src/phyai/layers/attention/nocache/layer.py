"""Stateless prefill attention — no KV cache, no radix."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from phyai.layers.attention.nocache.base import (
    AttentionBackend,
    AttnCtx,
    AttnMetadata,
)
from phyai.layers.attention.enums import AttnLayout, AttnMode
from phyai.layers.attention.mask import AttnMask
from phyai.kernel.call import CallSite, backend_preference
from phyai.layers.attention.shared import attention_dtypes, attention_shape
from phyai.kernel.types import KernelMode


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
        ``"flashinfer"`` (default), ``"sdpa"``, or ``"eager"``.
        A preference, not a decision: the kernel selector filters by what each
        backend declares it can execute, then orders by policy. Validated
        against the catalog, so a typo raises.
    backend_kwargs:
        Optional dict forwarded to the backend factory after ``runner``.
        ``"sdpa"`` accepts ``{"compile": bool, "select_kernel": bool}``;
        ``"flashinfer"`` accepts
        ``{"fi_workspace": Tensor, "workspace_bytes": int}``.

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
        backend: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        kernel_role: str = "attention",
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
        self.kernel_role = kernel_role
        self.backend: str | None = backend
        # Validated against the catalog; a soft ordering hint, not a decision.
        self.prefer = backend_preference("attention", backend)
        self._backend_kwargs: dict[str, Any] = dict(backend_kwargs or {})
        # One CallSite per layer: selection facts that never change live on
        # the site (pay-per-use memo keys; a facts-identical call is a ~2-4us
        # memo hit instead of a ~50us KernelQuery construction — the same
        # lesson dense_mlp's activation binding records).
        self._call_site = CallSite(
            "attention",
            role=self.kernel_role,
            prefer=self.prefer,
            dims={
                "heads": self.num_heads,
                "kv_heads": self.num_kv_heads,
                "head_dim": self.head_dim,
            },
            attrs={
                "causal": bool(self.causal),
                "sliding_window": self.sliding_window,
                "logits_soft_cap": self.logits_soft_cap,
            },
        )
        # Constructed backend instances are memoized here, keyed on the kernel
        # + mode + rule params that produced them. This is *not* redundant with
        # the selector's cache: the selector caches the factory, while a
        # FlashInfer backend instance owns a wrapper and a workspace tensor
        # that must not be rebuilt per forward — and must not be one-per-shape
        # either (the wrapper re-plans per metadata; keying on exact token
        # counts used to leak one wrapper per distinct shape). It stays on the
        # layer rather than on the process selector so two engines in one
        # process cannot share device buffers.
        self._resolved_backends: dict[tuple[object, ...], AttentionBackend] = {}

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
        mask: AttnMask | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_bias is not None and ctx is not None:
            raise ValueError(
                "pass attn_bias through AttnCtx when providing an explicit ctx"
            )
        bias = attn_bias if ctx is None else ctx.attn_bias
        if bias is not None:
            self._check_attn_bias(q, k, bias)
        if mask is not None:
            if ctx is not None:
                raise ValueError(
                    "pass the mask through the ctx's AttnMetadata when driving "
                    "the layer with an explicit ctx; the mask= keyword is the "
                    "convenience path's entry."
                )
            if mask.segments is not None and self.causal:
                raise ValueError(
                    "AttnMask.segments describes the block structure itself; "
                    "construct the layer with causal=False."
                )
        if q.ndim == 4:
            self._check_padded(q, k, v)
            layout = AttnLayout.PADDED_4D
        elif q.ndim == 3:
            if mask is not None:
                raise ValueError(
                    "mask= applies to padded (4-D) input; ragged callers "
                    "already express per-row lengths via cu_seqlens."
                )
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
            ctx = self._build_default_ctx(
                q,
                k,
                v,
                layout,
                cu_seqlens_q,
                cu_seqlens_kv,
                mask=mask,
                attn_bias=attn_bias,
            )
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
        *,
        mask: AttnMask | None = None,
        mode: KernelMode | str | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> AttnCtx:
        backend = self._ensure_backend(
            q,
            k,
            layout,
            dtypes={"input": q.dtype, "key": k.dtype, "value": v.dtype},
            mask=mask,
            mode=mode,
            attn_bias=attn_bias,
        )
        if layout.is_padded():
            B, S_q = q.shape[0], q.shape[1]
            S_kv = k.shape[1]
            num_query_tokens = B * S_q
            # The flashinfer padded path plans a ragged-KV wrapper, which needs
            # per-row q/kv offsets. Synthesize uniform cu_seqlens from the padded
            # shapes so a 4-D batch works without the caller hand-packing — S_q may
            # differ from S_kv (rectangular cross-attention). B==1 without a mask
            # ignores these (single_prefill); sdpa/eager build their mask from
            # shapes, not these.
            if mask is not None:
                # Plan with the offsets the packed forward will produce: the
                # KV side compacts to the valid rows; causal layers pack Q the
                # same way (see the backend's masked path for why).
                valid = mask.kv_valid(B, S_kv, q.device)
                if valid is not None:
                    counts = valid.sum(dim=1, dtype=torch.int32)
                    packed = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
                    torch.cumsum(counts, dim=0, out=packed[1:])
                    cu_seqlens_kv = packed
                    if self.causal:
                        cu_seqlens_q = packed
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
            layer_proto=self,
            q_dtype=q.dtype,
            kv_dtype=k.dtype,
            mask=mask,
            attn_bias=attn_bias,
        )
        plan = backend.init_forward_metadata(meta)
        return AttnCtx(
            backend=backend,
            plan=plan,
            mode=AttnMode.PREFILL,
            layout=layout,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            mask=mask,
            attn_bias=attn_bias,
        )

    def _ensure_backend(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        layout: AttnLayout | None = None,
        dtypes: dict[str, object] | None = None,
        mask: AttnMask | None = None,
        mode: KernelMode | str | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> AttentionBackend:
        # Callers that only need the backend's metadata/lifecycle methods may
        # ask before any tensor exists. Selection then runs on the layer's
        # static facts alone, with the default activation dtype.
        if q is not None:
            call_dims = {
                name: value
                for name, value in attention_shape(self, q, k).items()
                if name in ("tokens", "kv_tokens")
            }
            call_dtypes = attention_dtypes(q, k, dtypes)
        else:
            call_dims = {}
            call_dtypes = {"input": "bf16", "key": "bf16", "value": "bf16"}
        selection = self._call_site.select(
            device=None if q is None else q.device,
            dtype=call_dtypes,
            dims=call_dims,
            attrs={
                "layout": (
                    "padded" if layout is None or layout.is_padded() else "ragged"
                ),
                # None means "no declarative mask" — a known-absent fact the
                # catalog rows can require; "segments" routes away from
                # backends that cannot lower it.
                "mask_kind": None if mask is None else mask.kind,
                "has_attn_bias": attn_bias is not None,
            },
            mode=mode,
        )
        key = (
            selection.kernel_id,
            selection.query.mode.value,
            tuple(sorted((str(k_), str(v)) for k_, v in selection.params.items())),
        )
        cached = self._resolved_backends.get(key)
        if cached is not None:
            return cached
        backend = selection.implementation(None, **self._backend_kwargs)
        self._resolved_backends[key] = backend
        return backend

    # ------------------------------------------------------------------ #
    # Shape validation                                                   #
    # ------------------------------------------------------------------ #

    def _check_attn_bias(
        self, q: torch.Tensor, k: torch.Tensor, bias: torch.Tensor
    ) -> None:
        if q.ndim != 4 or bias.ndim not in (2, 3, 4):
            raise ValueError(
                "attn_bias requires padded attention and a 2-D, 3-D, or 4-D tensor"
            )
        if bias.device != q.device or bias.dtype not in (q.dtype, torch.float32):
            raise ValueError(
                "attn_bias must share q's device and use q's dtype or float32"
            )
        shape = (1,) * (4 - bias.ndim) + tuple(bias.shape)
        target = (q.shape[0], self.num_heads, q.shape[1], k.shape[1])
        if any(
            size not in (1, expected)
            for size, expected in zip(shape, target, strict=True)
        ):
            raise ValueError(
                f"attn_bias shape {tuple(bias.shape)} cannot broadcast to {target}"
            )

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
