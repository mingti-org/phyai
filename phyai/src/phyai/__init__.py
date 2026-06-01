"""phyai — Physical AI. Main library.

Top-level convenience surface. The engine entry points are re-exported
here so callers can write ``from phyai import Engine, EngineConfig``
without remembering which submodule each lives in.

The re-exports are **lazy** (PEP 562 ``__getattr__``). ``phyai.engine``
and ``phyai.engine_config`` transitively import torch, flashinfer, and
every registered model plugin — several seconds and hundreds of MiB.
Binding them eagerly here would make even a bare ``import phyai`` (e.g.
to read :data:`__version__`) pay that whole cost. Deferring to first
attribute access keeps ``import phyai`` near-instant while
``from phyai import Engine`` still works.

The functional subpackages (:mod:`phyai.parallel`, :mod:`phyai.layers`,
:mod:`phyai.vgpu`, :mod:`phyai.cache`, :mod:`phyai.runtime`,
:mod:`phyai.weights`, :mod:`phyai.payload`, :mod:`phyai.tokenizer`,
:mod:`phyai.utils`) are addressed by their full path
(``import phyai.parallel as P``) and are intentionally not hoisted here.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING

try:
    __version__ = _pkg_version("phyai")
except PackageNotFoundError:  # raw source tree, not installed
    __version__ = "0.0.0+unknown"


# Re-export name -> defining submodule. Resolved lazily on first access.
_LAZY: dict[str, str] = {
    # phyai.engine
    "Engine": "phyai.engine",
    "EngineArgs": "phyai.engine",
    "Entry": "phyai.engine",
    "EntryArgs": "phyai.engine",
    # phyai.engine_config
    "BackendConfig": "phyai.engine_config",
    "DeviceConfig": "phyai.engine_config",
    "EngineConfig": "phyai.engine_config",
    "ParallelConfig": "phyai.engine_config",
    "RuntimeConfig": "phyai.engine_config",
    "get_engine_config": "phyai.engine_config",
    "init_engine_config": "phyai.engine_config",
    "set_engine_config": "phyai.engine_config",
}

# Type-checkers and IDEs don't run __getattr__; declare the names statically
# so ``from phyai import Engine`` resolves under mypy / pyright / autocomplete.
if TYPE_CHECKING:
    from phyai.engine import Engine, EngineArgs, Entry, EntryArgs
    from phyai.engine_config import (
        BackendConfig,
        DeviceConfig,
        EngineConfig,
        ParallelConfig,
        RuntimeConfig,
        get_engine_config,
        init_engine_config,
        set_engine_config,
    )


def __getattr__(name: str) -> object:
    """Lazily import a re-exported engine symbol (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    obj = getattr(importlib.import_module(module), name)
    globals()[name] = obj  # cache: subsequent access skips __getattr__
    return obj


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "__version__",
    # engine
    "Engine",
    "EngineArgs",
    "Entry",
    "EntryArgs",
    # engine config
    "EngineConfig",
    "BackendConfig",
    "DeviceConfig",
    "ParallelConfig",
    "RuntimeConfig",
    "get_engine_config",
    "set_engine_config",
    "init_engine_config",
]
