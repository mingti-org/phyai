"""phyai-model-optimizer — Model optimization library for phyai."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("phyai-model-optimizer")
except PackageNotFoundError:  # raw source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
