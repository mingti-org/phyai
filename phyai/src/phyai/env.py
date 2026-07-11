"""Typed env-var registry for ``PHYAI_*`` overrides.

Every env var phyai consults goes through one :class:`EnvField`
descriptor in :class:`envs` below — instead of scattered
``os.environ.get(...)`` calls in random modules. Two upsides:

* one place to document and tune the env-var contract;
* typed parsing (``int`` / ``bool`` / ``torch.dtype``) lives next to
  the declaration, so callers always get the right shape back without
  rewriting parsing logic per-site.

Usage
-----
::

    from phyai.env import envs

    if (raw := envs.PHYAI_ATTN_BACKEND.get()) is not None:
        cfg = cfg.replace(backends=cfg.backends.replace(attn=raw))

The module is intentionally tiny and dependency-free (no torch, no
phyai imports) so it can be consulted from low-level modules during
their own bootstrap without import cycles.
"""

from __future__ import annotations

import os
from typing import Callable, Generic, TypeVar


T = TypeVar("T")


class EnvField(Generic[T]):
    """One typed env-var slot with a parser and an optional default.

    ``get()`` returns ``self.default`` when the variable is unset, and
    a parsed value otherwise. Parsing errors raise :class:`ValueError`
    with the env-var name attached so the failure points back at the
    user's environment rather than the consuming module.
    """

    __slots__ = ("name", "default", "parser")

    def __init__(
        self,
        name: str,
        default: T | None,
        parser: Callable[[str], T],
    ) -> None:
        self.name = name
        self.default = default
        self.parser = parser

    def is_set(self) -> bool:
        """``True`` if the env var is present (even if empty)."""
        return self.name in os.environ

    def get(self) -> T | None:
        raw = os.environ.get(self.name)
        if raw is None:
            return self.default
        try:
            return self.parser(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{self.name}={raw!r}: {e}") from e


def _parse_bool(s: str) -> bool:
    """Accept ``1/true/yes/on`` (case-insensitive) -> True; ``0/false/no/off`` -> False."""
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean (1/0/true/false/yes/no/on/off), got {s!r}")


def _parse_dtype(s: str):
    """Map a name like ``"bf16"`` / ``"bfloat16"`` to ``torch.dtype``.

    Imports torch lazily so importing :mod:`phyai.env` stays cheap
    and dependency-free at module-import time.
    """
    import torch

    table: dict[str, "torch.dtype"] = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
        "double": torch.float64,
    }
    key = s.strip().lower()
    if key not in table:
        raise ValueError(
            f"expected one of {sorted(table)} (case-insensitive), got {s!r}"
        )
    return table[key]


def _parse_regex_list(s: str) -> tuple[str, ...]:
    """Parse a JSON array of regex strings, e.g. ``'["o_proj$", "\\\\.heads\\\\."]'``.

    JSON (not a comma-split) because regex patterns routinely contain
    ``,`` / ``|`` / ``.`` that a naive split would mangle. A single bare
    string (not valid JSON, or JSON that isn't a list) is treated as a
    one-element list so ``PHYAI_DEBUG_TENSOR_DUMP_FILTER='o_proj$'`` also
    works for the common single-pattern case. An empty / whitespace string
    yields the empty tuple. Used for the tensor-dump operator filter.
    """
    import json

    raw = s.strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return (raw,)
    if isinstance(parsed, str):
        return (parsed,)
    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return tuple(parsed)
    raise ValueError(
        f"expected a JSON array of regex strings (or a single pattern), got {s!r}"
    )


class envs:
    """Process-level typed env-var registry.

    Read each via ``envs.PHYAI_FOO.get()`` and check ``.is_set()`` when
    you need to distinguish "unset" from "set to default". Adding a new
    env var means a single new line here and a one-line consumer change.
    """

    # ---------- backend / kernel selection ---------- #
    PHYAI_ATTN_BACKEND = EnvField("PHYAI_ATTN_BACKEND", None, str)
    PHYAI_NORM_BACKEND = EnvField("PHYAI_NORM_BACKEND", None, str)
    PHYAI_VGPU_BACKEND = EnvField("PHYAI_VGPU_BACKEND", None, str)

    # ---------- device / dtype ---------- #
    PHYAI_DEVICE = EnvField("PHYAI_DEVICE", None, str)
    PHYAI_PARAMS_DTYPE = EnvField("PHYAI_PARAMS_DTYPE", None, _parse_dtype)

    # ---------- runtime ---------- #
    PHYAI_USE_CUDA_GRAPH = EnvField("PHYAI_USE_CUDA_GRAPH", None, _parse_bool)

    # ---------- parallel ---------- #
    PHYAI_WORLD_SIZE = EnvField("PHYAI_WORLD_SIZE", None, int)
    PHYAI_DP_SIZE = EnvField("PHYAI_DP_SIZE", None, int)
    PHYAI_CFG_SIZE = EnvField("PHYAI_CFG_SIZE", None, int)
    PHYAI_EP_SIZE = EnvField("PHYAI_EP_SIZE", None, int)
    PHYAI_SP_SIZE = EnvField("PHYAI_SP_SIZE", None, int)
    PHYAI_CP_SIZE = EnvField("PHYAI_CP_SIZE", None, int)
    PHYAI_TP_SIZE = EnvField("PHYAI_TP_SIZE", None, int)

    # ---------- low-level tuning ---------- #
    PHYAI_FLASHINFER_WORKSPACE_BYTES = EnvField(
        "PHYAI_FLASHINFER_WORKSPACE_BYTES", None, int
    )
    PHYAI_FLASHINFER_PREFILL_BACKEND = EnvField(
        "PHYAI_FLASHINFER_PREFILL_BACKEND", None, str
    )
    PHYAI_FLASHINFER_BF16_BACKEND = EnvField("PHYAI_FLASHINFER_BF16_BACKEND", None, str)
    PHYAI_FLASHINFER_AUTOTUNE = EnvField("PHYAI_FLASHINFER_AUTOTUNE", None, _parse_bool)
    PHYAI_FLASHINFER_AUTOTUNE_CACHE = EnvField(
        "PHYAI_FLASHINFER_AUTOTUNE_CACHE", None, str
    )
    PHYAI_FORCE_LINEAR_KERNEL = EnvField("PHYAI_FORCE_LINEAR_KERNEL", None, str)
    # Backend preference for quantized dense Linear: "auto" | "humming" |
    # "flashinfer" | "torch". Resolved at materialize time (weight layout is
    # fixed at load, so the backend can't be a per-forward dispatch choice).
    # "auto" leans humming for every format humming can run; nvfp4 always uses
    # flashinfer and bf16 is unquantized regardless.
    PHYAI_LINEAR_QUANT_BACKEND = EnvField("PHYAI_LINEAR_QUANT_BACKEND", "auto", str)

    # ---------- debug / tensor dump ---------- #
    # When the dump dir is set the engine runs eager (cuda graph forced
    # off) and records every selected leaf operator's output, one .pt per
    # step. FILTER is a JSON array of regexes matched against operator
    # names (or a single bare pattern); FILTER_FN is a "pkg.mod:func" /
    # "/path.py:func" predicate. The two filters are mutually exclusive.
    PHYAI_DEBUG_TENSOR_DUMP_DIR = EnvField("PHYAI_DEBUG_TENSOR_DUMP_DIR", None, str)
    PHYAI_DEBUG_TENSOR_DUMP_FILTER = EnvField(
        "PHYAI_DEBUG_TENSOR_DUMP_FILTER", None, _parse_regex_list
    )
    PHYAI_DEBUG_TENSOR_DUMP_FILTER_FN = EnvField(
        "PHYAI_DEBUG_TENSOR_DUMP_FILTER_FN", None, str
    )


__all__ = ["EnvField", "envs"]
