"""Build a reviewable, source-identified robot wheel and matching examples.

Run with the existing model environment's Python. Only phyai-robot is built;
no workspace sync, native extension build, or dependency upgrades are performed.
"""

from __future__ import annotations

import os
import json
import hashlib
import tarfile
import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    files = sorted(
        p
        for directory in (root / "phyai-robot", root / "examples/deployment")
        for p in directory.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".toml", ".yaml", ".md", ".sh", ".in", ".lock")
        and not any(
            part.startswith(".") or part in {"__pycache__", "build", "dist"}
            for part in p.relative_to(root).parts[:-1]
        )
    )
    # Include matching examples/configs in the source identity even though only
    # the framework goes into the wheel. A dirty review tree is identified by
    # these contents; base_commit alone cannot identify uncommitted changes.
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    source_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    version = f"0.0.0+source.{source_sha[:12]}"
    out = root / ".cache/robot-isolation/dist" / source_sha[:12]
    out.mkdir(parents=True, exist_ok=True)
    # Build only this pure-Python package. Do not sync the root workspace or
    # build PhyAI native extensions on the robot host as a packaging side effect.
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out), str(root / "phyai-robot")],
        cwd=root,
        check=True,
        env={**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": version},
    )
    wheel = out / f"phyai_robot-{version}-py3-none-any.whl"
    if not wheel.is_file():
        raise RuntimeError("Expected a pure Python py3-none-any wheel")
    archive = out / "deployment-sources.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(root)))
    manifest = dict(
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        source_sha256=source_sha,
        source_files=hashes,
        artifacts={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel, archive)
        },
        wheel_version=version,
        profiles=["agilex-humble-py310-x86_64", "tianji-jazzy-py312-aarch64"],
    )
    (out / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
