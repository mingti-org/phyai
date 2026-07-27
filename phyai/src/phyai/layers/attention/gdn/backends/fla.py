"""Flash Linear Attention Gated Delta Net backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from phyai.layers.attention.enums import AttnMode
from phyai.layers.attention.gdn.base import (
    GatedDeltaNetBackend,
    GatedDeltaNetCtx,
    GatedDeltaNetLayerProto,
    GatedDeltaNetMetadata,
    GatedDeltaNetPlanHandle,
)
from phyai.layers.attention.gdn.registry import register_backend


def _load_chunk_op() -> Callable[..., Any]:
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        raise ImportError(
            "backend='fla' requires flash-linear-attention>=0.5.1."
        ) from exc
    return chunk_gated_delta_rule


def _load_recurrent_op() -> Callable[..., Any]:
    try:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
    except ImportError as exc:
        raise ImportError(
            "backend='fla' requires flash-linear-attention>=0.5.1."
        ) from exc
    return fused_recurrent_gated_delta_rule


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> None:
    if q.device.type != "cuda":
        raise RuntimeError("FLA GDN requires CUDA tensors.")
    activation_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("a", a), ("b", b)):
        if tensor.dtype not in activation_dtypes:
            raise ValueError(
                f"FLA GDN {name} must be float16, bfloat16, or float32, "
                f"got {tensor.dtype}."
            )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("FLA GDN q, k, and v must have the same dtype.")
    if a_log.dtype != torch.float32:
        raise ValueError(f"FLA GDN a_log must be float32, got {a_log.dtype}.")
    if dt_bias.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"FLA GDN dt_bias must be bfloat16 or float32, got {dt_bias.dtype}."
        )


def _check_state(name: str, state: torch.Tensor | None, device: torch.device) -> None:
    if state is None:
        return
    if state.dtype != torch.float32:
        raise ValueError(f"FLA GDN {name} must be float32, got {state.dtype}.")
    if state.device != device:
        raise ValueError(f"FLA GDN {name} must be on {device}, got {state.device}.")


@dataclass(frozen=True)
class FlaGatedDeltaNetPlan(GatedDeltaNetPlanHandle):
    """FLA GDN needs no separate planning object."""


@register_backend("fla")
class FlaGatedDeltaNetBackend(GatedDeltaNetBackend):
    """Route GDN prefill and decode through Flash Linear Attention."""

    def __init__(self, runner=None, *, chunk_size: int = 64) -> None:
        del runner
        if chunk_size not in (16, 32, 64):
            raise ValueError("FLA GDN chunk_size must be 16, 32, or 64.")
        self.chunk_size = int(chunk_size)

    def init_forward_metadata(
        self, meta: GatedDeltaNetMetadata
    ) -> GatedDeltaNetPlanHandle:
        return FlaGatedDeltaNetPlan()

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
        if ctx.mode == AttnMode.IDLE:
            shape = (*q.shape[:-2], layer.num_state_heads, layer.head_dim)
            return q.new_zeros(shape)
        _check_inputs(q, k, v, a, b, a_log, dt_bias)
        _check_state("state", ctx.state, q.device)
        _check_state("output_state", ctx.output_state, q.device)
        if ctx.mode == AttnMode.PREFILL:
            return self._forward_prefill(layer, q, k, v, a, b, a_log, dt_bias, ctx)
        if ctx.mode == AttnMode.DECODE:
            return self._forward_decode(layer, q, k, v, a, b, a_log, dt_bias, ctx)
        raise NotImplementedError(f"FLA GDN does not support mode={ctx.mode.name}.")

    def _forward_prefill(
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
        if ctx.state_indices is not None or ctx.output_state_indices is not None:
            raise ValueError(
                "FLA GDN prefill does not accept state-pool indices; "
                "pass per-batch state/output_state tensors."
            )

        ragged = ctx.layout.is_ragged()
        q_input = q.unsqueeze(0) if ragged else q
        k_input = k.unsqueeze(0) if ragged else k
        v_input = v.unsqueeze(0) if ragged else v
        a_input = a.unsqueeze(0) if ragged else a
        b_input = b.unsqueeze(0) if ragged else b
        cu_seqlens = None
        if ragged:
            if ctx.cu_seqlens is None:
                raise ValueError("FLA GDN ragged prefill requires ctx.cu_seqlens.")
            cu_seqlens = ctx.cu_seqlens.to(device=q.device, dtype=torch.int64)

        out, final_state = _load_chunk_op()(
            q=q_input,
            k=k_input,
            v=v_input,
            g=a_input,
            beta=b_input,
            scale=layer.scale,
            initial_state=ctx.state,
            output_final_state=ctx.output_state is not None,
            use_qk_l2norm_in_kernel=layer.use_qk_l2norm,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            chunk_size=self.chunk_size,
        )
        if ctx.output_state is not None:
            ctx.output_state.copy_(final_state)
        if ragged:
            out = out.squeeze(0)
        if ctx.output is not None:
            ctx.output.copy_(out)
            return ctx.output
        return out

    def _forward_decode(
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
        if ctx.state is None:
            raise ValueError("FLA GDN decode requires ctx.state.")

        if ctx.state_indices is None:
            initial_state = ctx.state
        else:
            safe_indices = ctx.state_indices.clamp_min(0).to(torch.int64)
            initial_state = ctx.state.index_select(0, safe_indices).contiguous()

        out, final_state = _load_recurrent_op()(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=b,
            scale=layer.scale,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=layer.use_qk_l2norm,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid_in_kernel=True,
            state_v_first=True,
        )

        if ctx.state_indices is None:
            ctx.state.copy_(final_state)
        else:
            output_indices = (
                ctx.state_indices
                if ctx.output_state_indices is None
                else ctx.output_state_indices
            )
            active = (ctx.state_indices >= 0) & (output_indices >= 0)
            ctx.state.index_copy_(
                0,
                output_indices[active].to(torch.int64),
                final_state[active],
            )
            out.masked_fill_(~(ctx.state_indices >= 0).view(-1, 1, 1, 1), 0)

        if ctx.output is not None:
            ctx.output.copy_(out)
            return ctx.output
        return out


__all__ = ["FlaGatedDeltaNetBackend", "FlaGatedDeltaNetPlan"]
