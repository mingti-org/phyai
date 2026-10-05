"""Independent rotary frequency bands for spatial and temporal coordinates."""

from __future__ import annotations

import torch
from torch import nn
from phyai_kernel.triton.rotary_embedding import apply_rope_precomputed

from phyai.layers.rotary_embedding import (
    ROPE_INV_FREQ_FNS,
    RotaryEmbedding,
    apply_rope,
    compute_cos_sin_from_inv_freq,
)


class AxialRotaryEmbedding(nn.Module):
    """Concatenate one interleaved RoPE band per coordinate axis.

    Positions have shape ``(..., axes)`` and may be negative or fractional.
    Rotation accumulates in float32 and returns the input dtype.
    """

    def __init__(
        self,
        axes_dims: tuple[int, ...],
        *,
        theta: float = 10000.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not axes_dims or any(dim <= 0 or dim % 2 for dim in axes_dims):
            raise ValueError("axes_dims must contain positive even dimensions")
        self.axes_dims = tuple(axes_dims)
        self.axes = nn.ModuleList(
            RotaryEmbedding(
                dim,
                max_position_embeddings=1,
                rope_theta=theta,
                interleave=True,
                device=device,
            )
            for dim in axes_dims
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for axis in self.axes:
            # Recompute rather than promote already-rounded low-precision data.
            inv_freq, _ = ROPE_INV_FREQ_FNS["default"](
                axis.rotary_dim, axis.rope_theta, device=axis.inv_freq.device
            )
            axis.inv_freq = inv_freq
            axis.cos_sin_cache = axis.cos_sin_cache.float()
        return self

    def get_cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.shape[-1] != len(self.axes):
            raise ValueError("positions must have one coordinate per rotary axis")
        bands = [
            compute_cos_sin_from_inv_freq(
                positions[..., i], axis.inv_freq, interleave=True
            )
            for i, axis in enumerate(self.axes)
        ]
        return (
            torch.cat([band[0] for band in bands], dim=-1),
            torch.cat([band[1] for band in bands], dim=-1),
        )

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape[-1] != sum(self.axes_dims) or k.shape[-1] != sum(self.axes_dims):
            raise ValueError("head dimension must equal the sum of axes_dims")
        if q.is_cuda and all(
            value.dtype in (torch.float16, torch.bfloat16, torch.float32)
            for value in (q, k, cos, sin)
        ):
            return apply_rope_precomputed(q, k, cos, sin, interleave=True)
        rotated_q, rotated_k = apply_rope(
            q.float(), k.float(), cos, sin, interleave=True
        )
        return rotated_q.to(q.dtype), rotated_k.to(k.dtype)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply(q, k, *self.get_cos_sin(positions))
