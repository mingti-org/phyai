"""aiter gated activations: ``act(x[..., :d]) * x[..., d:]``."""

from __future__ import annotations

import torch


def _gated(function_name: str, x: torch.Tensor) -> torch.Tensor:
    import aiter

    out = x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
    getattr(aiter, function_name)(out, x)
    return out


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _gated("silu_and_mul", x)


def gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _gated("gelu_and_mul", x)


def gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _gated("gelu_tanh_and_mul", x)


__all__ = ["gelu_and_mul", "gelu_tanh_and_mul", "silu_and_mul"]
