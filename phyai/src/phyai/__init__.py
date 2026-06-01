"""phyai — Physical Large Model main library."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("phyai")
except PackageNotFoundError:  # raw source tree, not installed
    __version__ = "0.0.0+unknown"


def hello() -> str:
    return "Hello from phyai!"


__all__ = ["__version__", "hello"]
