"""aiter RMSNorm over 2-D ``(tokens, hidden)`` input.

aiter requires the weight in the input dtype; an fp32 weight with bf16 input
returns inf.
"""

from __future__ import annotations

import torch


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from aiter import rms_norm

    return rms_norm(x, weight, eps)


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm, ``x * (1 + weight)``; aiter has no Gemma variant."""

    from aiter import rms_norm

    # Derived on every call: a copy cached during CUDA graph capture is filled
    # only when that one graph replays.
    return rms_norm(x, weight + 1, eps)


__all__ = ["gemma_rmsnorm", "rmsnorm"]
