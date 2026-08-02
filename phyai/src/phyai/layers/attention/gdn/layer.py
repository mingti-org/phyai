"""Gated Delta Net layer with selectable kernel backend."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from phyai.layers.attention.enums import AttnLayout, AttnMode
from phyai.layers.attention.gdn.base import (
    GatedDeltaNetBackend,
    GatedDeltaNetCtx,
    GatedDeltaNetMetadata,
)
from phyai.layers.attention.gdn.registry import get_backend_factory


class GatedDeltaNet(nn.Module):
    """Gated Delta Net core over already-projected Q/K/V and gate logits.

    The caller owns projections, causal convolution, gated normalization, and
    recurrent-state allocation. This layer mirrors :class:`Attention`: it
    validates tensor layouts and routes the numerical core through a backend.

    ``a`` and ``b`` are the input-dependent decay and update logits.
    ``a_log`` and ``dt_bias`` are per-state-head parameters. The FlashInfer
    backend forms ``g`` and ``beta`` for prefill and passes the raw values to
    the fused decode kernel.
    """

    def __init__(
        self,
        num_query_heads: int,
        head_dim: int,
        *,
        num_key_heads: int | None = None,
        num_value_heads: int | None = None,
        scale: float | None = None,
        use_qk_l2norm: bool = True,
        backend: str = "flashinfer",
        backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if num_key_heads is None:
            num_key_heads = num_query_heads
        if num_value_heads is None:
            num_value_heads = num_query_heads
        if min(num_query_heads, num_key_heads, num_value_heads, head_dim) <= 0:
            raise ValueError(
                "num_query_heads, num_key_heads, num_value_heads, and "
                "head_dim must all be positive."
            )
        if num_query_heads != num_key_heads:
            raise ValueError(
                f"GatedDeltaNet requires num_query_heads == num_key_heads; got "
                f"{num_query_heads} and {num_key_heads}."
            )
        if num_value_heads % num_query_heads != 0:
            raise ValueError(
                f"num_value_heads={num_value_heads} must be a multiple of "
                f"num_query_heads={num_query_heads} for GVA."
            )

        self.num_query_heads = int(num_query_heads)
        self.num_key_heads = int(num_key_heads)
        self.num_value_heads = int(num_value_heads)
        self.num_state_heads = self.num_value_heads
        self.head_dim = int(head_dim)
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.use_qk_l2norm = bool(use_qk_l2norm)

        factory = get_backend_factory(backend)
        self._backend_factory = factory
        self._backend_kwargs = dict(backend_kwargs or {})
        self.backend = getattr(factory, "name", str(backend))
        self._lazy_backend: GatedDeltaNetBackend | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ctx: GatedDeltaNetCtx | None = None,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = self._check_inputs(q, k, v, a, b, a_log, dt_bias)
        if ctx is None:
            ctx = self._build_default_ctx(q, layout, cu_seqlens)
        elif ctx.layout != layout:
            raise ValueError(
                f"ctx.layout={ctx.layout.name} does not match q layout {layout.name}."
            )
        if ctx.mode == AttnMode.DECODE and (q.ndim != 4 or q.shape[1] != 1):
            raise ValueError(
                f"GatedDeltaNet decode requires q shape (B, 1, H, D), got {tuple(q.shape)}."
            )
        self._check_ctx(q, ctx)
        return ctx.backend.forward(self, q, k, v, a, b, a_log, dt_bias, ctx)

    def _build_default_ctx(
        self,
        q: torch.Tensor,
        layout: AttnLayout,
        cu_seqlens: torch.Tensor | None,
    ) -> GatedDeltaNetCtx:
        backend = self._ensure_backend()
        if layout.is_padded():
            batch_size, seq_len = q.shape[:2]
            if cu_seqlens is None:
                cu_seqlens = torch.arange(
                    0,
                    (batch_size + 1) * seq_len,
                    seq_len,
                    dtype=torch.int32,
                    device=q.device,
                )
            num_tokens = batch_size * seq_len
        else:
            if cu_seqlens is None:
                raise ValueError("ragged GatedDeltaNet prefill requires cu_seqlens.")
            batch_size = cu_seqlens.numel() - 1
            num_tokens = q.shape[0]

        meta = GatedDeltaNetMetadata(
            mode=AttnMode.PREFILL,
            layout=layout,
            batch_size=int(batch_size),
            num_query_tokens=int(num_tokens),
            cu_seqlens=cu_seqlens,
        )
        plan = backend.init_forward_metadata(meta)
        return GatedDeltaNetCtx(
            backend=backend,
            plan=plan,
            mode=AttnMode.PREFILL,
            layout=layout,
            cu_seqlens=cu_seqlens,
        )

    def _ensure_backend(self) -> GatedDeltaNetBackend:
        if self._lazy_backend is None:
            self._lazy_backend = self._backend_factory(None, **self._backend_kwargs)
        return self._lazy_backend

    def _check_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> AttnLayout:
        if q.ndim not in (3, 4):
            raise ValueError(
                f"q must be 3-D (ragged) or 4-D (padded), got {tuple(q.shape)}."
            )
        if k.ndim != q.ndim or v.ndim != q.ndim:
            raise ValueError("q, k, and v must have the same rank.")
        if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
            raise ValueError("q, k, and v must have matching batch/token dimensions.")
        expected = (
            ("q", q, self.num_query_heads),
            ("k", k, self.num_key_heads),
            ("v", v, self.num_value_heads),
        )
        for name, tensor, heads in expected:
            if tensor.shape[-2:] != (heads, self.head_dim):
                raise ValueError(
                    f"{name} trailing shape {tuple(tensor.shape[-2:])} != "
                    f"({heads}, {self.head_dim})."
                )

        gate_shape = (*q.shape[:-2], self.num_state_heads)
        if a.shape != gate_shape or b.shape != gate_shape:
            raise ValueError(
                f"a and b must have shape {gate_shape}; got "
                f"a={tuple(a.shape)}, b={tuple(b.shape)}."
            )
        state_param_shape = (self.num_state_heads,)
        if a_log.shape != state_param_shape or dt_bias.shape != state_param_shape:
            raise ValueError(
                f"a_log and dt_bias must have shape {state_param_shape}; got "
                f"a_log={tuple(a_log.shape)}, dt_bias={tuple(dt_bias.shape)}."
            )
        if a_log.dtype != torch.float32:
            raise ValueError(f"a_log must be float32, got {a_log.dtype}.")
        return AttnLayout.RAGGED_3D if q.ndim == 3 else AttnLayout.PADDED_4D

    def _check_ctx(self, q: torch.Tensor, ctx: GatedDeltaNetCtx) -> None:
        output_shape = (*q.shape[:-2], self.num_state_heads, self.head_dim)
        if ctx.output is not None and ctx.output.shape != output_shape:
            raise ValueError(
                f"ctx.output shape {tuple(ctx.output.shape)} != {output_shape}."
            )
        if ctx.mode == AttnMode.IDLE:
            return

        if ctx.mode == AttnMode.PREFILL:
            if ctx.cu_seqlens is None:
                raise ValueError("GatedDeltaNet prefill requires ctx.cu_seqlens.")
            batch_size = ctx.cu_seqlens.numel() - 1
            state_shape = (
                batch_size,
                self.num_state_heads,
                self.head_dim,
                self.head_dim,
            )
            for name, state in (
                ("state", ctx.state),
                ("output_state", ctx.output_state),
            ):
                if state is not None and state.shape != state_shape:
                    raise ValueError(
                        f"ctx.{name} shape {tuple(state.shape)} != {state_shape}."
                    )
            return

        if ctx.mode == AttnMode.DECODE:
            if ctx.state is None:
                raise ValueError("GatedDeltaNet decode requires ctx.state.")
            batch_size = q.shape[0]
            state_tail = (
                self.num_state_heads,
                self.head_dim,
                self.head_dim,
            )
            if ctx.state_indices is None:
                expected_state_shape = (batch_size, *state_tail)
                if ctx.state.shape != expected_state_shape:
                    raise ValueError(
                        f"ctx.state shape {tuple(ctx.state.shape)} != "
                        f"{expected_state_shape}."
                    )
                if ctx.output_state_indices is not None:
                    raise ValueError(
                        "ctx.output_state_indices requires ctx.state_indices."
                    )
            else:
                if ctx.state.shape[1:] != state_tail:
                    raise ValueError(
                        f"ctx.state pool tail {tuple(ctx.state.shape[1:])} != "
                        f"{state_tail}."
                    )
                if ctx.state_indices.shape != (batch_size,):
                    raise ValueError(
                        f"ctx.state_indices shape {tuple(ctx.state_indices.shape)} "
                        f"!= ({batch_size},)."
                    )
                if ctx.state_indices.dtype not in (torch.int32, torch.int64):
                    raise ValueError("ctx.state_indices must be int32 or int64.")
                if ctx.output_state_indices is not None:
                    if ctx.output_state_indices.shape != (batch_size,):
                        raise ValueError(
                            "ctx.output_state_indices must match batch size."
                        )
                    if ctx.output_state_indices.dtype not in (
                        torch.int32,
                        torch.int64,
                    ):
                        raise ValueError(
                            "ctx.output_state_indices must be int32 or int64."
                        )
            if ctx.output_state is not None:
                raise ValueError(
                    "GatedDeltaNet decode updates ctx.state in place; "
                    "ctx.output_state must be None."
                )
            return

        raise NotImplementedError(
            f"GatedDeltaNet does not support mode={ctx.mode.name}."
        )

    def extra_repr(self) -> str:
        return (
            f"num_query_heads={self.num_query_heads}, "
            f"num_key_heads={self.num_key_heads}, "
            f"num_value_heads={self.num_value_heads}, "
            f"head_dim={self.head_dim}, use_qk_l2norm={self.use_qk_l2norm}, "
            f"backend={self.backend!r}"
        )


__all__ = ["GatedDeltaNet"]
