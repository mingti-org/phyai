"""Conditional affine modulation for token sequences."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from phyai.kernel.call import CallSite, backend_preference, token_shape


class AffineModulation(nn.Module):
    """Apply ``x * (1 + scale[:, None]) + shift[:, None]``.

    Inputs have shape ``(batch, tokens, hidden)``; shift and scale broadcast
    from ``(batch, hidden)``. Each operation rounds to the input dtype, matching
    the unfused expression. CUDA uses a fused kernel through the selector.
    """

    def __init__(self, *, backend: str | None = None) -> None:
        super().__init__()
        self.call_site = CallSite(
            "modulate",
            role="modulation",
            prefer=backend_preference("modulate", backend),
        )

    def forward(
        self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 3 or x.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("modulation requires a 3-D fp16, bf16, or fp32 input")
        for parameter in (shift, scale):
            if (
                parameter.ndim != 2
                or parameter.shape[0] not in (1, x.shape[0])
                or parameter.shape[1] not in (1, x.shape[2])
            ):
                raise ValueError("shift and scale must broadcast to (batch, hidden)")
            if parameter.device != x.device or parameter.dtype != x.dtype:
                raise ValueError(
                    "shift and scale must match the input device and dtype"
                )
        selected = self.call_site.select(
            device=x.device,
            dtype={"input": x.dtype, "shift": shift.dtype, "scale": scale.dtype},
            dims=token_shape(x, batch=x.shape[0], hidden=x.shape[-1]),
        )
        return selected.execute(x, shift, scale)


class LayerNormModulation(nn.Module):
    """Fuse non-affine LayerNorm and conditional affine modulation.

    ``forward(x, shift, scale, residual=None)`` returns ``(output, updated)``.
    With a residual, ``updated`` is a separately rounded ``x + residual``;
    otherwise it is ``x`` itself. LayerNorm computes in fp32 and rounds to the
    input dtype before modulation. Modulation rounds every arithmetic step,
    matching :class:`AffineModulation`. Inputs are never modified.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        *,
        backend: str | None = None,
        kernel_role: str = "norm_modulation",
    ) -> None:
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.eps = eps
        self.call_site = CallSite(
            "layernorm_modulate",
            role=kernel_role,
            prefer=backend_preference("layernorm_modulate", backend),
        )

    def forward(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            x.ndim != 3
            or x.shape[-1] == 0
            or x.dtype
            not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            )
        ):
            raise ValueError(
                "normalization modulation requires a 3-D fp16, bf16, or fp32 "
                "input with nonempty hidden dimension"
            )
        for parameter in (shift, scale):
            if (
                parameter.ndim != 2
                or parameter.shape[0] not in (1, x.shape[0])
                or parameter.shape[1] not in (1, x.shape[2])
            ):
                raise ValueError("shift and scale must broadcast to (batch, hidden)")
            if parameter.device != x.device or parameter.dtype != x.dtype:
                raise ValueError(
                    "shift and scale must match the input device and dtype"
                )
        if residual is not None and (
            residual.shape != x.shape
            or residual.dtype != x.dtype
            or residual.device != x.device
        ):
            raise ValueError("residual must match the input shape, device and dtype")
        selected = self.call_site.select(
            device=x.device,
            dtype={"input": x.dtype, "shift": shift.dtype, "scale": scale.dtype},
            dims=token_shape(x, batch=x.shape[0], hidden=x.shape[-1]),
            attrs={"residual": residual is not None},
        )
        return selected.execute(x, shift, scale, self.eps, residual)


__all__ = ["AffineModulation", "LayerNormModulation"]
