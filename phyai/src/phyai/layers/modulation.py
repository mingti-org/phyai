"""Conditional affine modulation for token sequences."""

from __future__ import annotations

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


__all__ = ["AffineModulation"]
