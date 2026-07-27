"""FlashInfer Gated Delta Net backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F

from phyai.layers.attention.enums import AttnMode
from phyai.layers.attention.gdn.base import (
    GatedDeltaNetBackend,
    GatedDeltaNetCtx,
    GatedDeltaNetLayerProto,
    GatedDeltaNetMetadata,
    GatedDeltaNetPlanHandle,
)
from phyai.layers.attention.gdn.registry import register_backend


def _load_prefill_op() -> Callable[..., Any]:
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    return chunk_gated_delta_rule


def _load_decode_op() -> Callable[..., Any]:
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    return gated_delta_rule_decode_pretranspose


def _check_supported_device(x: torch.Tensor) -> None:
    if x.device.type != "cuda":
        raise RuntimeError("FlashInfer GDN requires CUDA tensors.")
    major, minor = torch.cuda.get_device_capability(x.device)
    if major not in (9, 10):
        raise RuntimeError(
            "FlashInfer 0.6.12 GDN supports SM90/SM100 devices; "
            f"got compute capability {major}.{minor}."
        )


def _check_input_dtypes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> None:
    activation_dtypes = (torch.float16, torch.bfloat16)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("a", a), ("b", b)):
        if tensor.dtype not in activation_dtypes:
            raise ValueError(
                f"FlashInfer GDN {name} must be float16 or bfloat16, "
                f"got {tensor.dtype}."
            )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError("FlashInfer GDN q, k, and v must have the same dtype.")
    if a_log.dtype != torch.float32:
        raise ValueError(f"FlashInfer GDN a_log must be float32, got {a_log.dtype}.")
    if dt_bias.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"FlashInfer GDN dt_bias must be bfloat16 or float32, got {dt_bias.dtype}."
        )


@dataclass(frozen=True)
class FlashInferGatedDeltaNetPlan(GatedDeltaNetPlanHandle):
    """FlashInfer GDN needs no separate planning object."""


@register_backend("flashinfer")
class FlashInferGatedDeltaNetBackend(GatedDeltaNetBackend):
    """Route GDN prefill and decode through FlashInfer."""

    def __init__(self, runner=None) -> None:
        del runner

    def init_forward_metadata(
        self, meta: GatedDeltaNetMetadata
    ) -> GatedDeltaNetPlanHandle:
        return FlashInferGatedDeltaNetPlan()

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
        _check_input_dtypes(q, k, v, a, b, a_log, dt_bias)
        _check_supported_device(q)
        if ctx.mode == AttnMode.PREFILL:
            return self._forward_prefill(layer, q, k, v, a, b, a_log, dt_bias, ctx)
        if ctx.mode == AttnMode.DECODE:
            return self._forward_decode(layer, q, k, v, a, b, a_log, dt_bias, ctx)
        raise NotImplementedError(
            f"FlashInfer GDN does not support mode={ctx.mode.name}."
        )

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
        if ctx.cu_seqlens is None:
            raise ValueError("FlashInfer GDN prefill requires ctx.cu_seqlens.")
        if ctx.state_indices is not None or ctx.output_state_indices is not None:
            raise ValueError(
                "FlashInfer GDN prefill does not accept state-pool indices; "
                "pass per-batch state/output_state tensors."
            )

        padded_shape = q.shape[:-2] if ctx.layout.is_padded() else None
        q_flat = q.reshape(-1, layer.num_query_heads, layer.head_dim).contiguous()
        k_flat = k.reshape(-1, layer.num_key_heads, layer.head_dim).contiguous()
        v_flat = v.reshape(-1, layer.num_value_heads, layer.head_dim).contiguous()
        a_flat = a.reshape(-1, layer.num_state_heads)
        b_flat = b.reshape(-1, layer.num_state_heads)

        g = -a_log.exp().unsqueeze(0) * F.softplus(a_flat.float() + dt_bias.float())
        beta = b_flat.float().sigmoid()
        output = None
        if ctx.output is not None:
            output = ctx.output.reshape(-1, layer.num_state_heads, layer.head_dim)

        result = _load_prefill_op()(
            q_flat,
            k_flat,
            v_flat,
            g=g,
            beta=beta,
            scale=layer.scale,
            initial_state=ctx.state,
            output_final_state=ctx.output_state is not None,
            cu_seqlens=ctx.cu_seqlens,
            use_qk_l2norm_in_kernel=layer.use_qk_l2norm,
            output=output,
            output_state=ctx.output_state,
        )
        out = result[0] if isinstance(result, tuple) else result
        if padded_shape is not None:
            out = out.reshape(*padded_shape, layer.num_state_heads, layer.head_dim)
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
            raise ValueError("FlashInfer GDN decode requires ctx.state.")
        if ctx.output_state is not None:
            raise ValueError(
                "FlashInfer GDN decode updates ctx.state in place; "
                "ctx.output_state must be None."
            )

        pool_mode = ctx.state_indices is not None
        out, _ = _load_decode_op()(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            state=None if pool_mode else ctx.state,
            A_log=a_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            scale=layer.scale,
            output=ctx.output,
            use_qk_l2norm=layer.use_qk_l2norm,
            initial_state=ctx.state if pool_mode else None,
            initial_state_indices=ctx.state_indices,
            output_state_indices=ctx.output_state_indices,
        )
        return out


__all__ = [
    "FlashInferGatedDeltaNetBackend",
    "FlashInferGatedDeltaNetPlan",
]
