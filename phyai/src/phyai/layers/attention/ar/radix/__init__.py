"""Radix-cache → AR attention bridge.

:class:`RadixAttentionPlanner` turns per-request :class:`RadixSequence`
objects (pre-encoded ``atoms``) into prefix-reusing
:class:`~phyai.layers.attention.ar.base.ARAttnMetadata` on top of
:class:`phyai_ext.radix_cache.PrefixCache`. Model- and encoding-agnostic;
the foundation a radix-enabled AR runner (e.g. cosmos) builds on.
"""

from __future__ import annotations

from phyai.layers.attention.ar.radix.planner import RadixAttentionPlanner
from phyai.layers.attention.ar.radix.sequence import RadixSequence


__all__ = ["RadixAttentionPlanner", "RadixSequence"]
