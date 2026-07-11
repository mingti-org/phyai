"""Guarded access to the optional ``humming-kernels`` library."""

from __future__ import annotations

try:  # pragma: no cover - depends on optional install + CUDA toolchain
    import humming  # noqa: F401
    from humming import dtypes as humming_dtypes
    from humming.layer import HummingMethod
    from humming.schema import BaseInputSchema, BaseWeightSchema
    from humming.schema.humming import HummingInputSchema, HummingWeightSchema

    _HAS_HUMMING = True
except Exception:  # pragma: no cover - the common CPU/dev path
    humming = None  # type: ignore[assignment]
    humming_dtypes = None  # type: ignore[assignment]
    HummingMethod = None  # type: ignore[assignment]
    BaseInputSchema = None  # type: ignore[assignment]
    BaseWeightSchema = None  # type: ignore[assignment]
    HummingInputSchema = None  # type: ignore[assignment]
    HummingWeightSchema = None  # type: ignore[assignment]
    _HAS_HUMMING = False


def has_humming() -> bool:
    """True when ``humming-kernels`` is importable in this process."""
    return _HAS_HUMMING


def require_humming() -> None:
    """Raise a clear error if humming is needed but unavailable."""
    if not _HAS_HUMMING:
        raise RuntimeError(
            "humming-kernels is not installed. Install the optional 'humming' "
            "extra on a CUDA host (e.g. `uv sync --extra humming`) to use "
            "humming quantization specs."
        )


def humming_supports_sm(sm: int) -> bool:
    """True when the installed humming has tile heuristics for compute capability
    ``sm`` (``major*10 + minor``, e.g. 90, 100, 120).

    humming selects heuristics via a hardcoded ``heuristics_map`` with no fallback,
    so any SM it does not list (e.g. Jetson Thor ``sm_110``, or ``sm_88``) would
    raise ``KeyError`` deep inside its tuning. Probing that table directly makes this
    auto-adapt when humming adds an arch — no version pin needed. Touches no CUDA.
    """
    if not _HAS_HUMMING:
        return False
    try:
        from humming.tune import heuristics_map
    except Exception:
        return False
    return sm in heuristics_map


def require_humming_supports_sm(sm: int) -> None:
    """Raise if humming is installed but has no kernels for compute capability ``sm``.

    No-op when humming is absent — that case is reported by :func:`require_humming`
    at schema-build time.
    """
    if _HAS_HUMMING and not humming_supports_sm(sm):
        raise RuntimeError(
            f"the installed humming-kernels build has no kernels for sm_{sm} "
            f"(e.g. Jetson Thor sm_110 is unsupported) and would KeyError inside "
            f"humming's tuning. Set PHYAI_LINEAR_QUANT_BACKEND=flashinfer or torch "
            f"where the format allows, or install a humming build supporting sm_{sm}."
        )


__all__ = [
    "has_humming",
    "require_humming",
    "humming_supports_sm",
    "require_humming_supports_sm",
    "humming_dtypes",
    "HummingMethod",
    "BaseInputSchema",
    "BaseWeightSchema",
    "HummingInputSchema",
    "HummingWeightSchema",
]
