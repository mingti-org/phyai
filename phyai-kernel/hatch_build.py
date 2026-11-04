"""Bundle pinned CUTLASS headers needed for source-based runtime JIT."""

from pathlib import Path
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        package = root / "phyai_kernel"
        revision = (package / "third_party/cutlass_revision.txt").read_text().strip()
        bundled = package / "third_party/cutlass"
        checkout = root.parent / "third_party/cutlass"
        source = bundled if bundled.is_dir() else checkout
        if not (source / "include/cutlass/cutlass.h").is_file():
            raise RuntimeError(
                "CUTLASS headers are missing. From a Git checkout, run "
                "'git submodule update --init third_party/cutlass' before building. "
                "Source distributions must already contain these headers."
            )
        if source == checkout:
            actual = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != revision:
                raise RuntimeError(
                    f"CUTLASS submodule is at {actual}; expected {revision}. "
                    "Run 'git submodule update --init third_party/cutlass'."
                )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "diff",
                    "--quiet",
                    "HEAD",
                    "--",
                    "include",
                    "tools/util/include",
                    "LICENSE.txt",
                ],
                check=True,
            )
        for relative in ("include", "tools/util/include", "LICENSE.txt"):
            path = source / relative
            if not path.exists():
                raise RuntimeError(
                    f"Required CUTLASS package content is missing: {relative}"
                )
            if source == checkout:
                build_data.setdefault("force_include", {})[str(path)] = (
                    f"phyai_kernel/third_party/cutlass/{relative}"
                )
