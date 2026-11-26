"""Shared attention kernels used by every backend.

The KV-head expansion and reference matmul-softmax kernel are pure
torch primitives. Lifting them out of any single backend lets sdpa fall
back to ``eager_attn`` for soft-cap, eager use them as the reference
path, and any future backend reuse them without crossing module privacy.
Mask construction lives in :mod:`phyai.layers.attention.mask`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phyai.layers.attention.mask import causal_block_mask


def repeat_kv(x: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """``(B, H_kv, S, D) -> (B, H_q, S, D)``; identity for MHA."""
    if num_heads == num_kv_heads:
        return x
    rep = num_heads // num_kv_heads
    return x.repeat_interleave(rep, dim=1)


def eager_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    attn_mask: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference matmul + masked softmax. Inputs are ``(*, H, S, D)``.

    ``attn_mask`` (True = attend, broadcastable against the logits) is a
    pre-lowered :class:`~phyai.layers.attention.mask.AttnMask` and already
    includes the causal / sliding-window structure, so it replaces the
    flag-derived mask entirely when given.
    """
    S_q = q.shape[-2]
    S_kv = k.shape[-2]
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    if logits_soft_cap is not None:
        cap = logits_soft_cap
        attn = cap * torch.tanh(attn / cap)
    if attn_bias is not None:
        attn = attn + attn_bias
    mask = (
        attn_mask
        if attn_mask is not None
        else causal_block_mask(
            S_q, S_kv, q.device, causal=causal, sliding_window=sliding_window
        )
    )
    if mask is not None:
        attn = attn.masked_fill(~mask, float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    # Fully masked rows produce NaN after softmax(-inf); zero is the
    # subsystem-wide convention for rows with nothing to attend.
    if mask is not None or attn_bias is not None:
        attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v)


__all__ = ["eager_attn", "repeat_kv"]
