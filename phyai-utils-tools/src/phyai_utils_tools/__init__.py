"""phyai-utils-tools — Shared utilities and tools for phyai."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("phyai-utils-tools")
except PackageNotFoundError:  # raw source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
