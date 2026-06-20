"""Unit tests for phyai.utils.checkpoint folder helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from phyai.models.configuration import PretrainedConfig
from phyai.utils.checkpoint import (
    find_safetensors,
    load_config,
    resolve_checkpoint,
)


@dataclass(frozen=True)
class _TinyConfig(PretrainedConfig):
    hidden_size: int = 16
    name: str = "tiny"


# --------------------------------------------------------------------------- #
# find_safetensors                                                            #
# --------------------------------------------------------------------------- #


def test_find_safetensors_single_file(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"")  # empty placeholder
    out = find_safetensors(tmp_path)
    assert len(out) == 1
    assert out[0].name == "model.safetensors"
    assert out[0].is_absolute()


def test_find_safetensors_index_two_shards(tmp_path: Path):
    save_file({"a": torch.zeros(2)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"b": torch.zeros(2)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            }
        )
    )
    out = find_safetensors(tmp_path)
    assert [p.name for p in out] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    for p in out:
        assert p.is_absolute()


def test_find_safetensors_index_deduplicates_shared_shards(tmp_path: Path):
    """Multiple keys pointing at the same shard collapse to one entry."""
    save_file(
        {"a": torch.zeros(2), "b": torch.zeros(2)},
        str(tmp_path / "shared.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {
                    "a": "shared.safetensors",
                    "b": "shared.safetensors",
                },
            }
        )
    )
    out = find_safetensors(tmp_path)
    assert [p.name for p in out] == ["shared.safetensors"]


def test_find_safetensors_index_prefers_over_single(tmp_path: Path):
    """index.json wins even when model.safetensors is also present."""
    save_file({"x": torch.zeros(2)}, str(tmp_path / "shard-00001.safetensors"))
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "shard-00001.safetensors"}})
    )
    out = find_safetensors(tmp_path)
    assert [p.name for p in out] == ["shard-00001.safetensors"]


def test_find_safetensors_index_missing_shard_raises(tmp_path: Path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"k": "ghost.safetensors"}})
    )
    with pytest.raises(FileNotFoundError, match="ghost.safetensors"):
        find_safetensors(tmp_path)


def test_find_safetensors_index_empty_weight_map_raises(tmp_path: Path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}})
    )
    with pytest.raises(ValueError, match="weight_map"):
        find_safetensors(tmp_path)


def test_find_safetensors_index_no_weight_map_key_raises(tmp_path: Path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"meta": {}}))
    with pytest.raises(ValueError, match="weight_map"):
        find_safetensors(tmp_path)


def test_find_safetensors_glob_fallback(tmp_path: Path):
    """No index, no canonical name -> glob picks up *.safetensors files."""
    (tmp_path / "weights-a.safetensors").write_bytes(b"")
    (tmp_path / "weights-b.safetensors").write_bytes(b"")
    out = find_safetensors(tmp_path)
    assert [p.name for p in out] == ["weights-a.safetensors", "weights-b.safetensors"]


def test_find_safetensors_no_shards_raises(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no safetensors shards"):
        find_safetensors(tmp_path)


def test_find_safetensors_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_safetensors(tmp_path / "nope")


def test_find_safetensors_file_path_raises(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(NotADirectoryError, match="folder"):
        find_safetensors(tmp_path / "model.safetensors")


def test_find_safetensors_accepts_str(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    out = find_safetensors(str(tmp_path))
    assert len(out) == 1


# --------------------------------------------------------------------------- #
# load_config                                                                 #
# --------------------------------------------------------------------------- #


def test_load_config_basic(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 32, "name": "x"}))
    cfg = load_config(tmp_path, _TinyConfig)
    assert cfg.hidden_size == 32
    assert cfg.name == "x"


def test_load_config_unknown_keys_dropped(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"hidden_size": 8, "totally_unrelated": [1, 2, 3]})
    )
    cfg = load_config(tmp_path, _TinyConfig)
    assert cfg.hidden_size == 8
    assert cfg.name == "tiny"  # default — silent drop, fall back


def test_load_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_config(tmp_path, _TinyConfig)


def test_load_config_custom_filename(tmp_path: Path):
    (tmp_path / "geometry.json").write_text(json.dumps({"hidden_size": 7}))
    cfg = load_config(tmp_path, _TinyConfig, filename="geometry.json")
    assert cfg.hidden_size == 7


def test_load_config_dir_validation(tmp_path: Path):
    # Nonexistent path is resolved first: it is neither local nor a valid repo
    # id, so resolve_checkpoint raises before _ensure_dir is reached.
    with pytest.raises(FileNotFoundError, match="not a valid HuggingFace repo id"):
        load_config(tmp_path / "ghost", _TinyConfig)


# --------------------------------------------------------------------------- #
# resolve_checkpoint                                                          #
# --------------------------------------------------------------------------- #


def test_resolve_checkpoint_local_dir_returns_path(tmp_path: Path):
    out = resolve_checkpoint(tmp_path)
    assert out == tmp_path
    assert out.is_dir()


def test_resolve_checkpoint_local_file_returns_path(tmp_path: Path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"")
    out = resolve_checkpoint(f)
    assert out == f
    assert out.is_file()


def test_resolve_checkpoint_local_str_path(tmp_path: Path):
    out = resolve_checkpoint(str(tmp_path))
    assert isinstance(out, Path)
    assert out == tmp_path


def test_resolve_checkpoint_typo_path_raises_filenotfound(tmp_path: Path):
    """A nonexistent path containing '/' is rejected offline (no network)."""
    with pytest.raises(FileNotFoundError, match="not a valid HuggingFace repo id"):
        resolve_checkpoint(tmp_path / "ghost" / "sub")


def test_resolve_checkpoint_repo_id_forwarded(tmp_path: Path, monkeypatch):
    """A non-local source is treated as a repo id and downloaded."""
    seen: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    out = resolve_checkpoint("nvidia/Cosmos3-Nano", revision="abc123")
    assert out == tmp_path
    assert seen["repo_id"] == "nvidia/Cosmos3-Nano"
    assert seen["repo_type"] == "model"
    assert seen["revision"] == "abc123"


def test_load_config_from_repo_id(tmp_path: Path, monkeypatch):
    """load_config resolves a repo id, then parses config.json from the cache."""
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 64}))
    seen: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    cfg = load_config("org/model", _TinyConfig, revision="r1")
    assert cfg.hidden_size == 64
    assert seen["repo_id"] == "org/model"
    assert seen["revision"] == "r1"
