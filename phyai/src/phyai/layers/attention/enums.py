"""Shared attention enums.

`AttnMode` and `AttnLayout` describe the per-step staging and Q/K/V
tensor layout. They are shared verbatim across the four attention
stacks (`attention/`, `ar/`, `diffusion/`, `gdn/`) — the per-stack
`AttnMetadata` / `AttnCtx` reference them directly so a backend can
key its kernel selection off `mode` / `layout` without coupling to a
sibling stack.
"""

from __future__ import annotations

from enum import IntEnum


class AttnMode(IntEnum):
    """Business stage the backend may special-case.

    The mode is **business-stage**, not tensor-shape: it tells the
    backend which kernel route is profitable. ``PREFILL`` and ``DECODE``
    are the bread-and-butter values; ``MIXED`` and ``IDLE`` are reserved
    for chunked prefill and DP-padding empty steps respectively.
    """

    PREFILL = 0
    DECODE = 1
    MIXED = 2
    IDLE = 3

    def is_prefill(self) -> bool:
        return self == AttnMode.PREFILL

    def is_decode(self) -> bool:
        return self == AttnMode.DECODE

    def is_mixed(self) -> bool:
        return self == AttnMode.MIXED

    def is_idle(self) -> bool:
        return self == AttnMode.IDLE


class AttnLayout(IntEnum):
    """Q/K/V tensor layout — padded 4-D batch vs packed 3-D varlen."""

    PADDED_4D = 0  # (B, S, H, D) — same length per batch row
    RAGGED_3D = 1  # (N, H, D) + cu_seqlens_q / cu_seqlens_kv

    def is_padded(self) -> bool:
        return self == AttnLayout.PADDED_4D

    def is_ragged(self) -> bool:
        return self == AttnLayout.RAGGED_3D


__all__ = ["AttnLayout", "AttnMode"]
