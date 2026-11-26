"""SDPA backend — :func:`F.scaled_dot_product_attention`, padded prefill only."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from phyai.layers.attention.nocache.base import (
    AttentionBackend,
    AttentionLayerProto,
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
)
from phyai.layers.attention.common import eager_attn, repeat_kv
from phyai.layers.attention.mask import causal_block_mask
from phyai.utils.logging import get_logger


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


_logger = get_logger(__name__)


# cuDNN first: it keeps masked + GQA attention on a fused kernel where Flash
# falls back to the mem-efficient (cutlassF) path; MATH last so the priority
# list can never raise "no available kernel".
_CUDA_SDPA_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _sdpa_priority_ctx(
    device: torch.device, enabled: bool, *, mixed_bias_dtype: bool = False
):
    """Kernel-priority context for CUDA SDPA; ``nullcontext`` elsewhere."""
    if mixed_bias_dtype and device.type == "cuda":
        # PyTorch 2.11 cuDNN SDPA can return invalid output for FP32 bias
        # with half Q/K/V; efficient SDPA rejects that dtype combination.
        # Math SDPA keeps the bias precision and still executes on CUDA.
        return sdpa_kernel(SDPBackend.MATH)
    if enabled and device.type == "cuda":
        return sdpa_kernel(_CUDA_SDPA_PRIORITY, set_priority=True)
    return nullcontext()


@dataclass(frozen=True)
class SdpaAttentionPlan(AttnPlanHandle):
    """Empty plan handle — sdpa no-cache has no per-step state."""


def _sdpa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    num_heads: int,
    num_kv_heads: int,
    attn_mask: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inner SDPA call with mask + soft-cap dispatch.

    ``q`` is ``(*, num_heads, S_q, D)``; ``k``/``v`` are
    ``(*, num_kv_heads, S_kv, D)`` — unexpanded for GQA. ``num_heads`` /
    ``num_kv_heads`` are static Python ints so ``enable_gqa`` constant-folds
    under ``torch.compile``. ``attn_mask`` is a pre-lowered
    :class:`~phyai.layers.attention.mask.AttnMask` (True = attend) that
    already includes the causal / sliding-window structure, so it replaces
    the flag-derived mask entirely when given.
    """
    enable_gqa = num_heads != num_kv_heads
    if logits_soft_cap is not None:
        # SDPA has no soft-cap; the eager reference does, but it needs K/V
        # expanded to num_heads first.
        k = repeat_kv(k, num_heads, num_kv_heads)
        v = repeat_kv(v, num_heads, num_kv_heads)
        return eager_attn(
            q,
            k,
            v,
            scale=scale,
            causal=causal,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            attn_mask=attn_mask,
            attn_bias=attn_bias,
        )
    if attn_bias is not None:
        mask = attn_mask
        if mask is None:
            mask = causal_block_mask(
                q.shape[-2],
                k.shape[-2],
                q.device,
                causal=causal,
                sliding_window=sliding_window,
            )
        combined = (
            attn_bias if mask is None else attn_bias.masked_fill(~mask, float("-inf"))
        )
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=combined, scale=scale, enable_gqa=enable_gqa
        )
    if attn_mask is not None:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale, enable_gqa=enable_gqa
        )
    S_q = q.shape[-2]
    S_kv = k.shape[-2]
    if not causal and sliding_window is None:
        return F.scaled_dot_product_attention(
            q, k, v, scale=scale, enable_gqa=enable_gqa
        )
    # is_causal=True is top-left aligned, but phyai aligns queries with the
    # trailing keys (q_pos[i] = i + (S_kv - S_q)); the two agree only when
    # S_q == S_kv, so rectangular causal must go through the explicit mask.
    if causal and sliding_window is None and S_q == S_kv:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale, enable_gqa=enable_gqa
        )
    mask = causal_block_mask(
        S_q, S_kv, q.device, causal=causal, sliding_window=sliding_window
    )
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=enable_gqa
    )


def _wrap_with_compile_log(
    fn: Callable[..., torch.Tensor],
    *,
    label: str,
    log_kwargs: dict | None = None,
) -> Callable[..., torch.Tensor]:
    """Compile ``fn`` with ``dynamic=True`` and log the first call on every rank."""
    compiled = torch.compile(fn, dynamic=True)
    state = {"logged": False}
    log_kwargs = log_kwargs or {}

    def _logged(*args, **kwargs):
        if state["logged"]:
            return compiled(*args, **kwargs)
        state["logged"] = True
        _logger.info(
            "%s: torch.compile tracing sdpa kernel (dynamic=True%s)",
            label,
            "".join((f", {k}={v!r}" for k, v in log_kwargs.items())),
        )
        t0 = time.perf_counter()
        out = compiled(*args, **kwargs)
        dt = time.perf_counter() - t0
        _logger.info("%s: sdpa torch.compile first-call took %.2fs", label, dt)
        return out

    return _logged


class SdpaAttentionBackend(AttentionBackend):
    """SDPA no-cache backend — padded prefill only."""

    name = "sdpa"

    def __init__(
        self,
        runner: "ModelRunner | None" = None,
        *,
        compile: bool = False,
        select_kernel: bool = False,
    ) -> None:
        del runner
        self.compile = bool(compile)
        self.select_kernel = bool(select_kernel)
        if self.compile:
            self._sdpa_call = _wrap_with_compile_log(
                _sdpa_attn,
                label="Attention[sdpa]",
            )
        else:
            self._sdpa_call = _sdpa_attn

    def supports_capture(self) -> bool:
        return True

    def init_forward_metadata(self, meta: AttnMetadata) -> AttnPlanHandle:
        return SdpaAttentionPlan()

    def forward(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        if ctx.mode.is_idle():
            return q.new_zeros(q.shape)
        if not ctx.layout.is_padded():
            raise NotImplementedError(
                "SdpaAttentionBackend supports only the padded (4-D) layout; "
                "SDPA has no ragged/varlen API. Use backend='flashinfer' for "
                "ragged sequences."
            )
        return self._forward_padded(layer, q, k, v, ctx)

    def _forward_padded(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        attn_mask = None
        if ctx.mask is not None:
            attn_mask = ctx.mask.dense(
                q.shape[1],
                k.shape[1],
                q.shape[0],
                q.device,
                causal=layer.causal,
                sliding_window=layer.sliding_window,
            )
        # K/V stay at num_kv_heads; enable_gqa inside _sdpa_attn broadcasts.
        q_h = q.transpose(1, 2)
        k_h = k.transpose(1, 2)
        v_h = v.transpose(1, 2)
        with _sdpa_priority_ctx(
            q.device,
            self.select_kernel,
            mixed_bias_dtype=ctx.attn_bias is not None
            and ctx.attn_bias.dtype != q.dtype,
        ):
            out = self._sdpa_call(
                q_h,
                k_h,
                v_h,
                scale=layer.scale,
                causal=layer.causal,
                sliding_window=layer.sliding_window,
                logits_soft_cap=layer.logits_soft_cap,
                num_heads=layer.num_heads,
                num_kv_heads=layer.num_kv_heads,
                attn_mask=attn_mask,
                attn_bias=ctx.attn_bias,
            )
        return out.transpose(1, 2).contiguous()


__all__ = ["SdpaAttentionBackend", "SdpaAttentionPlan"]
