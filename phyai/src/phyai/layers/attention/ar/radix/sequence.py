"""Per-request planning state for radix-cache-backed AR attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RadixSequence:
    """Mutable per-request state for one planned AR attention step.

    Carries the request's pre-encoded, page-aligned ``atoms`` in; the
    :class:`~phyai.layers.attention.ar.radix.planner.RadixAttentionPlanner`
    fills the prefix/suffix slot split and the radix-cache handles (the
    prefix lock and the suffix units) that must outlive the forward. Treat
    every field other than ``atoms`` as planner-owned.
    """

    atoms: bytes
    prefix_len: int = 0
    prefix_slots: torch.Tensor | None = None
    suffix_slots: torch.Tensor | None = None
    node_ref: Any | None = None  # phyai_ext.radix_cache.NodeRef | None
    suffix_units: Any | None = None  # phyai_ext.radix_cache.OwnedUnits | None
    committed: bool = False
    released: bool = False

    @property
    def suffix_len(self) -> int:
        """Newly allocated (written) slot count; 0 until plan()."""
        return 0 if self.suffix_slots is None else int(self.suffix_slots.numel())

    @property
    def total_len(self) -> int:
        """prefix + suffix slot count; valid after plan()."""
        return self.prefix_len + self.suffix_len


__all__ = ["RadixSequence"]
