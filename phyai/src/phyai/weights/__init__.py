"""Param-attached weight loaders.

Every parameter-bearing layer attaches two attributes to its
``nn.Parameter``s during ``__init__``:

* ``param.hf_keys: list[(hf_key, shard_id)]`` — the HF tensor(s) that
  fill this param. ``shard_id`` is ``None`` for non-fused params, a
  string/int leg tag for fused params (e.g. ``"q"``/``"k"``/``"v"``).
* ``param.weight_loader: WeightLoader`` — a callable
  ``(param, loaded, shard_id) -> None`` that performs the copy. Built by
  one of the shared factories in :mod:`phyai.weights.shards`
  (:func:`replicated`, :func:`sharded`, :func:`fused`, :func:`vocab`).
* ``param.optional: bool`` (optional, defaults False) — if True, the
  loader does not raise when the HF source is absent. Used for quant
  scales that are only present in quantised checkpoints.

The whole loader chain is :func:`load_pretrained` — select safetensors or
PyTorch checkpoint files, walk their keys, dispatch via the per-param loader,
and finally walk modules calling ``post_load(self)`` for any spec-driven
fixups (fp8 scale fanning, etc.).
"""

from __future__ import annotations

from phyai.weights.loader import (
    LoadReport,
    checkpoint_format,
    iter_checkpoint_tensors,
    load_pretrained,
)
from phyai.weights.shards import (
    WeightLoader,
    _Leg,
    fused,
    replicated,
    sharded,
    vocab,
)


__all__ = [
    "LoadReport",
    "WeightLoader",
    "_Leg",
    "checkpoint_format",
    "fused",
    "iter_checkpoint_tensors",
    "load_pretrained",
    "replicated",
    "sharded",
    "vocab",
]
