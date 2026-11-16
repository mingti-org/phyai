"""aiter RoPE with a packed ``(max_pos, rotary_dim)`` cos / sin cache."""

from __future__ import annotations

import torch


def rope_cached(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    interleave: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate flat ``q (tokens, Hq * D)`` and ``k (tokens, Hk * D)`` at ``positions``.

    Returns new tensors shaped like ``q`` and ``k``.
    """

    from aiter import rope_cached_2c_fwd

    tokens = positions.shape[0]
    # aiter reads cos and sin as contiguous ``(tokens, 1, 1, rotary_dim // 2)``
    # and silently misreads strided views of the packed cache.
    cos, sin = (
        cos_sin_cache.index_select(0, positions).view(tokens, 1, 1, -1).chunk(2, dim=-1)
    )
    q_out, k_out = rope_cached_2c_fwd(
        q.contiguous().view(tokens, 1, -1, head_dim),
        k.contiguous().view(tokens, 1, -1, head_dim),
        cos.contiguous(),
        sin.contiguous(),
        # rotate_style: 0 rotates halves (NeoX), 1 rotates pairs (GPT-J).
        1 if interleave else 0,
        True,
        False,
    )
    return q_out.view(q.shape), k_out.view(k.shape)


__all__ = ["rope_cached"]
