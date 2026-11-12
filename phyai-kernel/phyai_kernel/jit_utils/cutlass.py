"""Resolve the same pinned CUTLASS headers in editable and packaged installs."""

from functools import cache
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CUTLASS_REVISION = (
    (PACKAGE_ROOT / "third_party/cutlass_revision.txt").read_text().strip()
)


@cache
def cutlass_include_dirs() -> tuple[str, str]:
    bundled = PACKAGE_ROOT / "third_party/cutlass"
    checkout = PACKAGE_ROOT.parents[1] / "third_party/cutlass"
    root = bundled if bundled.is_dir() else checkout
    if not (root / "include/cutlass/cutlass.h").is_file():
        raise RuntimeError(
            "Pinned CUTLASS headers are unavailable. In a source checkout, run "
            "'git submodule update --init third_party/cutlass'. "
            "A phyai-kernel wheel or source distribution must bundle these headers."
        )
    directories = (root / "include", root / "tools/util/include")
    if not all(path.is_dir() for path in directories):
        raise RuntimeError("The installed CUTLASS header directories are incomplete.")
    return tuple(str(path) for path in directories)
