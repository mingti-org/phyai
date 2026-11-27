"""Bundle inputs must come from the tracked source tree."""

import subprocess

import pytest

from examples.deployment.environments.build_bundle import source_files


def test_bundle_excludes_untracked_and_ignored_files(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    configs = tmp_path / "examples/deployment/configs"
    configs.mkdir(parents=True)
    tracked = configs / "default.yaml"
    tracked.write_text("robot: {}\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    (configs / "local.yaml").write_text("private: true\n")
    (configs / "local.py").write_text("private = True\n")
    (tmp_path / ".gitignore").write_text("ignored.yaml\n")
    (configs / "ignored.yaml").write_text("secret: true\n")
    assert source_files(tmp_path) == [tracked]


def test_bundle_requires_git_checkout(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        source_files(tmp_path)
