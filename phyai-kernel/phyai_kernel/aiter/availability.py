"""Probe whether aiter can be used in this process."""

from __future__ import annotations


def available() -> bool:
    """Return whether aiter imports; raise the import error when it fails.

    Non-ROCm Torch builds return False without importing aiter.
    """

    import torch

    if torch.version.hip is None:
        return False
    import aiter  # noqa: F401

    return True


__all__ = ["available"]
