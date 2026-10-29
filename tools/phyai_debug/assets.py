"""Snapshot only referenced URDF geometry, never a robot control installation."""

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def snapshot_robot(source, trace):
    source = Path(source).expanduser().resolve()
    destination = Path(trace) / "robot"
    urdf = source / "robot.urdf"
    tree = ET.parse(urdf)
    files = {Path("robot.urdf")}
    for mesh in tree.findall(".//mesh"):
        uri = mesh.get("filename", "")
        relative = Path(uri.removeprefix("package://"))
        if (
            not uri.startswith("package://")
            or relative.is_absolute()
            or ".." in relative.parts
        ):
            raise ValueError(f"Unsupported mesh URI: {uri}")
        files.add(relative)
    for relative in files:
        path = source / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(source)
            or not path.is_file()
        ):
            raise ValueError(f"Unsafe or missing robot asset: {relative}")
    destination.mkdir(exist_ok=False)
    manifest = {}
    for relative in sorted(files):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
