#!/usr/bin/env python3
"""Generate a PEP 503 "simple" wheel index for the phyai workspace packages.

Given a directory of built distributions (wheels + sdists), this emits a static
site that GitHub Pages can serve and ``uv pip install`` / ``pip install`` can
consume via ``--extra-index-url``::

    <out-dir>/.nojekyll                         # tell Pages not to run Jekyll
    <out-dir>/packages/<dist files...>          # the actual wheels + sdists
    <out-dir>/simple/index.html                 # root: one link per project
    <out-dir>/simple/<project>/index.html       # PEP 503 page per project

Each artifact link carries a ``#sha256=<digest>`` fragment so installers can
verify integrity. Links are relative, so the generated tree works unchanged at
whatever URL Pages serves it from — no base URL needs to be baked in.

Stdlib only, no third-party imports: this runs in CI before anything is
installed, and it must not depend on the very packages it is indexing.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import shutil

# A distribution file is either a wheel or an sdist; nothing else is indexed.
WHEEL_SUFFIX = ".whl"
SDIST_SUFFIX = ".tar.gz"


def normalize_project(name: str) -> str:
    """Normalize a project name per PEP 503 (runs of -_. -> single -, lowercased)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def project_of(filename: str) -> str:
    """Extract the normalized project name from a wheel or sdist filename.

    Wheel names are ``{name}-{version}-...whl`` and sdist names are
    ``{name}-{version}.tar.gz``; in both, ``{name}`` uses ``_`` where the
    project name has ``-``. Splitting on the first ``-`` (wheel) or stripping
    the trailing ``-{version}.tar.gz`` (sdist) recovers it, and
    ``normalize_project`` collapses the ``_`` back.
    """
    if filename.endswith(WHEEL_SUFFIX):
        raw = filename.split("-")[0]
    else:  # sdist: strip the trailing -<version>.tar.gz
        raw = re.sub(r"-\d.*$", "", filename[: -len(SDIST_SUFFIX)])
    return normalize_project(raw)


def sha256_of(path: pathlib.Path) -> str:
    """Return the hex sha256 of a file, read in chunks to bound memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def build_index(dist_dir: pathlib.Path, out_dir: pathlib.Path) -> dict[str, list[str]]:
    """Copy dists into ``out_dir/packages`` and write the PEP 503 tree.

    Returns the mapping of project name -> list of filenames that was indexed.
    """
    packages_dir = out_dir / "packages"
    simple_dir = out_dir / "simple"
    packages_dir.mkdir(parents=True, exist_ok=True)
    simple_dir.mkdir(parents=True, exist_ok=True)

    # Group every dist file under its project, copying it into packages/.
    projects: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(dist_dir.iterdir()):
        if not (path.name.endswith(WHEEL_SUFFIX) or path.name.endswith(SDIST_SUFFIX)):
            continue
        shutil.copy2(path, packages_dir / path.name)
        projects.setdefault(project_of(path.name), []).append(
            (path.name, sha256_of(path))
        )

    if not projects:
        raise SystemExit(
            f"no wheels or sdists found in {dist_dir} — refusing to write an empty index"
        )

    # Per-project page: one verifiable link per artifact, relative to the page.
    for project, files in projects.items():
        page_dir = simple_dir / project
        page_dir.mkdir(exist_ok=True)
        links = "".join(
            f'<a href="../../packages/{name}#sha256={digest}">{name}</a><br>\n'
            for name, digest in sorted(files)
        )
        (page_dir / "index.html").write_text(
            f"<!DOCTYPE html>\n<html><body>\n{links}</body></html>\n",
            encoding="utf-8",
        )

    # Root page: one link per project directory.
    root_links = "".join(
        f'<a href="{project}/">{project}</a><br>\n' for project in sorted(projects)
    )
    (simple_dir / "index.html").write_text(
        f"<!DOCTYPE html>\n<html><body>\n{root_links}</body></html>\n",
        encoding="utf-8",
    )

    # GitHub Pages runs Jekyll by default, which would skip files it dislikes;
    # .nojekyll serves the tree verbatim.
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    return {project: [n for n, _ in files] for project, files in projects.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=pathlib.Path,
        required=True,
        help="directory containing the built wheels and sdists",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="directory to write the static index site into",
    )
    args = parser.parse_args()

    indexed = build_index(args.dist_dir, args.out_dir)
    total = sum(len(files) for files in indexed.values())
    print(f"indexed {total} files across {len(indexed)} projects -> {args.out_dir}")
    for project in sorted(indexed):
        print(f"  {project}: {len(indexed[project])} files")


if __name__ == "__main__":
    main()
