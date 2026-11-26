"""Shared paired Q/K normalization and precomputed rotary embedding."""

from __future__ import annotations

import math

import torch
from torch import nn

from phyai.kernel.call import CallSite, backend_preference, token_shape


class QKRMSNormRotary(nn.Module):
    """Normalize full projections before rotating a selectable range of heads.

    Parameters stay in the caller's Q/K normalization modules. This stateless
    operation accepts their weights and preserves rounding after RMSNorm and
    gamma multiplication, followed by separate FP32 rotary products and sums.
    ``head_start`` and ``local_heads`` select TP output heads without changing
    the full-width normalization statistics.
    """

    def __init__(
        self,
        head_dim: int,
        eps: float = 1e-6,
        *,
        interleave: bool = True,
        backend: str | None = None,
        kernel_role: str = "qk_norm_rope",
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be positive and even")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.head_dim = head_dim
        self.eps = eps
        self.interleave = interleave
        self.call_site = CallSite(
            "qk_rmsnorm_rope",
            role=kernel_role,
            prefer=backend_preference("qk_rmsnorm_rope", backend),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        head_start: int = 0,
        local_heads: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim not in (2, 3) or q.shape != k.shape:
            raise ValueError("Q/K must have matching rank-2 or rank-3 shapes")
        width = q.shape[-1]
        if width <= 0 or width % self.head_dim:
            raise ValueError("projection width must be a positive multiple of head_dim")
        if local_heads is None:
            local_heads = width // self.head_dim - head_start
        if (
            head_start < 0
            or local_heads <= 0
            or (head_start + local_heads) * self.head_dim > width
        ):
            raise ValueError("requested head slice is outside the full projection")
        supported = (torch.float16, torch.bfloat16, torch.float32)
        if q.dtype not in supported or k.dtype != q.dtype:
            raise ValueError("Q/K must share an fp16, bf16, or fp32 dtype")
        if any(t.device != q.device for t in (k, q_weight, k_weight, cos, sin)):
            raise ValueError("all tensors must share a device")
        if cos.dtype not in supported or sin.dtype not in supported:
            raise ValueError("cosine and sine must use fp16, bf16, or fp32")
        for weight in (q_weight, k_weight):
            if (
                weight.shape != (width,)
                or not weight.is_contiguous()
                or weight.dtype != q.dtype
            ):
                raise ValueError(
                    "weights must be contiguous vectors matching projection width and dtype"
                )
        cos = torch.broadcast_to(cos, (*q.shape[:-1], self.head_dim))
        sin = torch.broadcast_to(sin, (*q.shape[:-1], self.head_dim))
        selected = self.call_site.select(
            device=q.device,
            dtype={
                "input": q.dtype,
                "weight": q_weight.dtype,
                "cos": cos.dtype,
                "sin": sin.dtype,
            },
            dims=token_shape(
                q,
                batch=q.shape[0] if q.ndim == 3 else 1,
                hidden=width,
                head_dim=self.head_dim,
                local_heads=local_heads,
            ),
            attrs={"interleave": self.interleave, "head_start": head_start},
        )
        return selected.execute(
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            self.eps,
            self.head_dim,
            self.interleave,
            head_start,
            local_heads,
        )


__all__ = ["QKRMSNormRotary"]
