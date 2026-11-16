"""Probe importable libraries for ``lib.*`` facts."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version

from phyai.utils.logging import get_logger

logger = get_logger(__name__)

#: Import name -> distribution name, where they differ.
_DISTRIBUTIONS = {
    "flashinfer": "flashinfer-python",
    "phyai_kernel": "phyai-kernel",
    "fla": "flash-linear-attention",
    "flash_qla": "flash-qla",
    "humming": "humming-kernels",
}

#: Import name -> module whose ``available()`` probes it, for libraries that
#: phyai reaches only through phyai_kernel and must not import itself.
_DELEGATED_PROBES = {"aiter": "phyai_kernel.aiter.availability"}

_probes: dict[str, bool] = {}


def library_available(name: str) -> bool:
    """Return whether a library imports, memoized for the process lifetime."""

    cached = _probes.get(name)
    if cached is not None:
        return cached

    try:
        probe = _DELEGATED_PROBES.get(name)
        if probe is None:
            importlib.import_module(name)
        elif not importlib.import_module(probe).available():
            raise ImportError(f"{probe}.available() returned False")
    except BaseException as exc:
        # Any import-time failure makes the library unavailable.
        logger.debug_rank0(
            "kernel: library %r is unavailable (%s: %s)", name, type(exc).__name__, exc
        )
        available = False
    else:
        logger.debug_rank0(
            "kernel: library %r is available (%s)",
            name,
            _distribution_version(name) or "unknown version",
        )
        available = True

    _probes[name] = available
    return available


def _distribution_version(import_name: str) -> str | None:
    try:
        return version(_DISTRIBUTIONS.get(import_name, import_name))
    except PackageNotFoundError:
        return None


def library_facts(
    names: frozenset[str] | set[str] | tuple[str, ...],
) -> dict[str, bool]:
    """Build ``lib.<name>`` facts for the requested import names."""

    return {f"lib.{name}": library_available(name) for name in names}


def reset_library_probes() -> None:
    """Clear cached library probes."""

    _probes.clear()


__all__ = [
    "library_available",
    "library_facts",
    "reset_library_probes",
]
