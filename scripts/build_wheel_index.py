#!/usr/bin/env python3
"""Generate a PEP 503 index from local builds and GitHub Release assets.

Local wheels and sdists are copied into the site. Release assets stay on GitHub;
the index links to every published release in the supplied API response::

    <out-dir>/.nojekyll                         # tell Pages not to run Jekyll
    <out-dir>/packages/<dist files...>          # local wheels + sdists only
    <out-dir>/simple/index.html                 # root: one link per project
    <out-dir>/simple/<project>/index.html       # PEP 503 page per project

Local links include a SHA-256 hash. Release links use the digest from GitHub
when available. Installers consume the site via ``--extra-index-url``.
The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import html
import json
import pathlib
import re
import shutil
from urllib.parse import quote, urlsplit

# A distribution file is either a wheel or an sdist; nothing else is indexed.
WHEEL_SUFFIX = ".whl"
SDIST_SUFFIX = ".tar.gz"


@dataclass(frozen=True)
class Artifact:
    url: str
    digest: str | None
    local_path: pathlib.Path | None = None


def normalize_project(name: str) -> str:
    """Normalize a project name per PEP 503 (runs of -_. -> single -, lowercased)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def project_of(filename: str) -> str:
    """Extract a safe, normalized project directory from a distribution name."""
    if any(c in filename for c in "/\\") or any(ord(c) < 32 for c in filename):
        raise ValueError(f"invalid distribution filename: {filename!r}")
    if filename.endswith(WHEEL_SUFFIX):
        raw, separator, version = filename[: -len(WHEEL_SUFFIX)].partition("-")
    elif filename.endswith(SDIST_SUFFIX):
        raw, separator, version = filename[: -len(SDIST_SUFFIX)].rpartition("-")
    else:
        raise ValueError(f"unsupported distribution filename: {filename!r}")
    project = normalize_project(raw)
    if (
        not separator
        or not version
        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", project)
    ):
        raise ValueError(f"invalid distribution filename: {filename!r}")
    return project


def sha256_of(path: pathlib.Path) -> str:
    """Return the hex sha256 of a file, read in chunks to bound memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def release_artifacts(releases_json: pathlib.Path) -> dict[str, Artifact]:
    """Read a release list or the pages emitted by ``gh api --paginate --slurp``."""
    releases = json.loads(releases_json.read_text(encoding="utf-8"))
    if not isinstance(releases, list):
        raise ValueError("release JSON must contain a list of releases or pages")
    pages = releases if releases and isinstance(releases[0], list) else [releases]
    artifacts: dict[str, Artifact] = {}
    for page in pages:
        if not isinstance(page, list):
            raise ValueError("each release API page must be a list")
        for release in page:
            if not isinstance(release, dict) or type(release.get("draft")) is not bool:
                raise ValueError("each release must contain a boolean draft field")
            if release["draft"]:
                continue
            assets = release.get("assets")
            if not isinstance(assets, list):
                raise ValueError("each published release must contain an assets list")
            for asset in assets:
                if not isinstance(asset, dict) or not isinstance(
                    asset.get("name"), str
                ):
                    raise ValueError("each release asset must contain a name")
                name = asset["name"]
                if not name.endswith((WHEEL_SUFFIX, SDIST_SUFFIX)):
                    continue
                project_of(name)
                url = asset.get("browser_download_url")
                if not isinstance(url, str) or any(c.isspace() for c in url):
                    raise ValueError(f"invalid download URL for {name!r}")
                parsed_url = urlsplit(url)
                if (
                    parsed_url.scheme != "https"
                    or not parsed_url.netloc
                    or parsed_url.username is not None
                    or parsed_url.password is not None
                    or parsed_url.fragment
                ):
                    raise ValueError(f"invalid download URL for {name!r}")
                digest = asset.get("digest")
                if digest is not None:
                    if not isinstance(digest, str) or not re.fullmatch(
                        r"sha256:[a-fA-F0-9]{64}", digest
                    ):
                        raise ValueError(f"invalid SHA-256 digest for {name!r}")
                    digest = digest.removeprefix("sha256:").lower()
                previous = artifacts.get(name)
                if previous is not None:
                    if previous.url != url or (
                        previous.digest and digest and previous.digest != digest
                    ):
                        raise ValueError(f"conflicting release assets named {name!r}")
                    digest = previous.digest or digest
                artifacts[name] = Artifact(url, digest)
    return artifacts


def build_index(
    dist_dir: pathlib.Path | None,
    out_dir: pathlib.Path,
    releases_json: pathlib.Path | None = None,
) -> dict[str, list[str]]:
    """Validate inputs, copy local builds, and write the PEP 503 tree.

    Returns the mapping of project name -> list of filenames that was indexed.
    """
    artifacts = release_artifacts(releases_json) if releases_json is not None else {}
    if dist_dir is not None:
        for path in sorted(dist_dir.iterdir()):
            if not path.name.endswith((WHEEL_SUFFIX, SDIST_SUFFIX)):
                continue
            project_of(path.name)
            if not path.is_file():
                raise ValueError(f"distribution is not a file: {path}")
            digest = sha256_of(path)
            if path.name in artifacts:
                released = artifacts[path.name]
                if released.digest is not None and released.digest != digest:
                    raise ValueError(
                        f"local distribution conflicts with release asset {path.name!r}"
                    )
                continue
            artifacts[path.name] = Artifact(
                f"../../packages/{quote(path.name, safe='')}", digest, path
            )

    if not artifacts:
        raise ValueError("no wheels or sdists found; refusing to write an empty index")

    projects: dict[str, list[str]] = {}
    for name in sorted(artifacts):
        projects.setdefault(project_of(name), []).append(name)

    packages_dir = out_dir / "packages"
    simple_dir = out_dir / "simple"
    simple_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in artifacts.items():
        if artifact.local_path is not None:
            packages_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact.local_path, packages_dir / name)

    for project, files in projects.items():
        page_dir = simple_dir / project
        page_dir.mkdir(exist_ok=True)
        links = []
        for name in files:
            artifact = artifacts[name]
            url = artifact.url
            if artifact.digest is not None:
                url += f"#sha256={artifact.digest}"
            links.append(
                f'<a href="{html.escape(url, quote=True)}">{html.escape(name)}</a><br>\n'
            )
        (page_dir / "index.html").write_text(
            f"<!DOCTYPE html>\n<html><body>\n{''.join(links)}</body></html>\n",
            encoding="utf-8",
        )

    # Root page: one link per project directory.
    root_links = "".join(
        f'<a href="{html.escape(quote(project, safe=""), quote=True)}/">{html.escape(project)}</a><br>\n'
        for project in sorted(projects)
    )
    (simple_dir / "index.html").write_text(
        f"<!DOCTYPE html>\n<html><body>\n{root_links}</body></html>\n",
        encoding="utf-8",
    )

    # GitHub Pages runs Jekyll by default, which would skip files it dislikes;
    # .nojekyll serves the tree verbatim.
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    return projects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist-dir",
        type=pathlib.Path,
        help="directory containing the built wheels and sdists",
    )
    parser.add_argument(
        "--releases-json",
        type=pathlib.Path,
        help="GitHub releases JSON from gh api --paginate --slurp",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="directory to write the static index site into",
    )
    args = parser.parse_args()
    if args.dist_dir is None and args.releases_json is None:
        parser.error("provide --dist-dir or --releases-json (or both)")
    try:
        indexed = build_index(args.dist_dir, args.out_dir, args.releases_json)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    total = sum(len(files) for files in indexed.values())
    print(f"indexed {total} files across {len(indexed)} projects -> {args.out_dir}")
    for project in sorted(indexed):
        print(f"  {project}: {len(indexed[project])} files")


if __name__ == "__main__":
    main()
