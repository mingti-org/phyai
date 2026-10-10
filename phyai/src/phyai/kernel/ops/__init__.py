"""Register kernel operation definitions."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phyai.kernel.registry import Catalog


#: Operation modules, in import order.
OP_MODULES: tuple[str, ...] = (
    "gemm",
    "norm",
    "activation",
    "modulation",
    "embedding",
    "rope",
    "attention",
    "conv",
)


def populate(catalog: "Catalog") -> "Catalog":
    """Import each operation module and let it register itself."""

    for name in OP_MODULES:
        module = importlib.import_module(f"phyai.kernel.ops.{name}")
        module.register(catalog)
    return catalog


__all__ = ["OP_MODULES", "populate"]
