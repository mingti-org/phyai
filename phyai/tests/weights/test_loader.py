"""End-to-end tests for phyai.weights.load_pretrained."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import phyai.layers.linear as L
from phyai.weights import (
    LoadReport,
    WeightLoadSession,
    checkpoint_format,
    iter_checkpoint_tensors,
    load_pretrained,
)
from phyai.weights import loader as loader_mod


def test_load_replicated_linear_end_to_end(tmp_path: Path, fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.ReplicatedLinear(
        in_features=4,
        out_features=8,
        bias=True,
        params_dtype=torch.float32,
        prefix="mod.fc",
    )

    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    save_file(
        {"mod.fc.weight": src_w, "mod.fc.bias": src_b},
        str(tmp_path / "shard.safetensors"),
    )

    report = load_pretrained(layer, [tmp_path / "shard.safetensors"])
    assert isinstance(report, LoadReport)
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]
    assert not report.missing
    assert not report.unexpected
    torch.testing.assert_close(layer.weight.data.cpu(), src_w)
    torch.testing.assert_close(layer.bias.data.cpu(), src_b)


def test_load_qkv_fused(tmp_path: Path, fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.QKVParallelLinear(
        hidden_size=8,
        head_dim=4,
        num_heads=2,
        num_kv_heads=2,
        bias=False,
        params_dtype=torch.float32,
        prefix="model.layers.0.self_attn.qkv_proj",
    )
    # q_size = 8, kv_size = 8 -> fused = 24.
    q = torch.full((8, 8), 1.0, dtype=torch.float32)
    k = torch.full((8, 8), 2.0, dtype=torch.float32)
    v = torch.full((8, 8), 3.0, dtype=torch.float32)
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": q,
            "model.layers.0.self_attn.k_proj.weight": k,
            "model.layers.0.self_attn.v_proj.weight": v,
        },
        str(tmp_path / "qkv.safetensors"),
    )

    report = load_pretrained(layer, [tmp_path / "qkv.safetensors"])
    assert len(report.loaded) == 3
    assert torch.all(layer.weight.data[0:8] == 1.0)
    assert torch.all(layer.weight.data[8:16] == 2.0)
    assert torch.all(layer.weight.data[16:24] == 3.0)


def test_incremental_weight_session_loads_fused_legs(fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.QKVParallelLinear(
        hidden_size=8,
        head_dim=4,
        num_heads=2,
        num_kv_heads=2,
        bias=False,
        params_dtype=torch.float32,
        prefix="model.layers.0.self_attn.qkv_proj",
    )
    session = WeightLoadSession(layer, source_label="test update")
    session.load({"model.layers.0.self_attn.q_proj.weight": torch.full((8, 8), 1.0)})
    session.load(
        {
            "model.layers.0.self_attn.k_proj.weight": torch.full((8, 8), 2.0),
            "model.layers.0.self_attn.v_proj.weight": torch.full((8, 8), 3.0),
        }
    )

    report = session.finish()

    assert len(report.loaded) == 3
    assert not report.missing
    assert torch.all(layer.weight.data[0:8] == 1.0)
    assert torch.all(layer.weight.data[8:16] == 2.0)
    assert torch.all(layer.weight.data[16:24] == 3.0)


def test_incremental_weight_session_allows_partial_hot_update(fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.ReplicatedLinear(
        in_features=4,
        out_features=8,
        bias=True,
        params_dtype=torch.float32,
        prefix="mod.fc",
    )
    original_bias = layer.bias.detach().clone()
    updated_weight = torch.randn(8, 4)
    session = WeightLoadSession(layer)
    session.load({"mod.fc.weight": updated_weight})

    report = session.finish(require_all=False)

    assert report.loaded == ["mod.fc.weight"]
    assert not report.missing
    torch.testing.assert_close(layer.weight.cpu(), updated_weight)
    torch.testing.assert_close(layer.bias, original_bias)


def test_incremental_weight_session_rejects_duplicate_remap_before_write(fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.ReplicatedLinear(
        in_features=4,
        out_features=8,
        bias=False,
        params_dtype=torch.float32,
        prefix="mod.fc",
    )
    first = torch.randn(8, 4)
    second = torch.randn(8, 4)
    session = WeightLoadSession(layer, remap=lambda _name: "mod.fc.weight")

    session.load({"upstream.a": first})
    with pytest.raises(RuntimeError, match="appears more than once after remap"):
        session.load({"upstream.b": second})

    torch.testing.assert_close(layer.weight.cpu(), first)


def test_missing_keys_raise_when_strict_and_are_reported_otherwise(
    tmp_path: Path, fake_mesh
):
    fake_mesh(tp_size=1)
    layer = L.ReplicatedLinear(
        in_features=2,
        out_features=2,
        bias=True,
        params_dtype=torch.float32,
        prefix="x",
    )
    # Save only the weight; bias is missing.
    save_file({"x.weight": torch.zeros(2, 2)}, str(tmp_path / "incomplete.safetensors"))
    with pytest.raises(RuntimeError, match="strict failure"):
        load_pretrained(layer, [tmp_path / "incomplete.safetensors"])
    report = load_pretrained(layer, [tmp_path / "incomplete.safetensors"], strict=False)
    assert "x.bias" in report.missing


def test_unexpected_key_recorded(tmp_path: Path, fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.ReplicatedLinear(
        in_features=2,
        out_features=2,
        bias=False,
        params_dtype=torch.float32,
        prefix="y",
    )
    save_file(
        {"y.weight": torch.zeros(2, 2), "totally_unrelated.tensor": torch.zeros(3)},
        str(tmp_path / "extra.safetensors"),
    )
    report = load_pretrained(layer, [tmp_path / "extra.safetensors"], strict=False)
    assert "totally_unrelated.tensor" in report.unexpected
    assert "y.weight" in report.loaded


def test_remap_accepts_callables_and_substring_dicts_and_drops_none(
    tmp_path: Path, fake_mesh
):
    fake_mesh(tp_size=1)
    layer = L.ReplicatedLinear(
        in_features=2,
        out_features=2,
        bias=False,
        params_dtype=torch.float32,
        prefix="model.fc",
    )
    src = torch.randn(2, 2)
    save_file(
        {"transformer.fc.weight": src, "junk.weight": torch.zeros(3)},
        str(tmp_path / "t.safetensors"),
    )
    # Callable: rewrite the prefix, drop anything mapped to None.
    report = load_pretrained(
        layer,
        [tmp_path / "t.safetensors"],
        remap=lambda k: None if "junk" in k else k.replace("transformer.", "model."),
    )
    assert report.loaded == ["model.fc.weight"]
    assert "junk.weight" not in report.unexpected
    torch.testing.assert_close(layer.weight.data.cpu(), src)
    # Dict: substring rewrite; the un-remapped junk key is then unexpected.
    report = load_pretrained(
        layer,
        [tmp_path / "t.safetensors"],
        remap={"transformer.": "model."},
        strict=False,
    )
    assert report.loaded == ["model.fc.weight"]
    assert report.unexpected == ["junk.weight"]


def test_dtype_cast_recorded(tmp_path: Path, fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.ReplicatedLinear(
        in_features=2,
        out_features=2,
        bias=False,
        params_dtype=torch.bfloat16,
        prefix="z.fc",
    )
    src_fp32 = torch.randn(2, 2, dtype=torch.float32)
    save_file({"z.fc.weight": src_fp32}, str(tmp_path / "cast.safetensors"))
    report = load_pretrained(layer, [tmp_path / "cast.safetensors"])
    assert len(report.casts) == 1
    cast_key, src_dt, dst_dt = report.casts[0]
    assert cast_key == "z.fc.weight"
    assert src_dt == torch.float32
    assert dst_dt == torch.bfloat16


def test_post_load_runs_for_modules_with_hook(tmp_path: Path, fake_mesh):
    """Verify post_load() is called on every module that defines it."""
    fake_mesh(tp_size=1)

    class HookedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.touched = False

        def post_load(self):
            self.touched = True

    layer = HookedModule()
    save_file({}, str(tmp_path / "empty.safetensors"))
    load_pretrained(layer, [tmp_path / "empty.safetensors"], strict=False)
    assert layer.touched is True


def test_optional_param_absent_does_not_raise(tmp_path: Path, fake_mesh):
    fake_mesh(tp_size=1)

    class WithOptional(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
            self.w.hf_keys = [("w.weight", None)]
            self.scale = nn.Parameter(torch.ones(1), requires_grad=False)
            self.scale.hf_keys = [("w.weight_scale", None)]
            self.scale.optional = True

    layer = WithOptional()
    src = torch.randn(2, 2)
    save_file({"w.weight": src}, str(tmp_path / "no_scale.safetensors"))
    report = load_pretrained(layer, [tmp_path / "no_scale.safetensors"], strict=True)
    assert "w.weight_scale" in report.optional_missing
    assert "w.weight_scale" not in report.missing
    torch.testing.assert_close(layer.w.data, src)


def test_double_claim_raises(tmp_path: Path, fake_mesh):
    fake_mesh(tp_size=1)

    class TwoOwners(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.zeros(2), requires_grad=False)
            self.a.hf_keys = [("shared.weight", None)]
            self.b = nn.Parameter(torch.zeros(2), requires_grad=False)
            self.b.hf_keys = [("shared.weight", None)]

    layer = TwoOwners()
    save_file({}, str(tmp_path / "x.safetensors"))
    with pytest.raises(RuntimeError, match="claimed by two params"):
        load_pretrained(layer, [tmp_path / "x.safetensors"], strict=False)


# --------------------------------------------------------------------------- #
# Source resolution: folder / single file / iterable forms accepted.          #
# --------------------------------------------------------------------------- #


def _make_replicated(prefix: str = "mod.fc") -> "L.ReplicatedLinear":
    return L.ReplicatedLinear(
        in_features=4,
        out_features=8,
        bias=True,
        params_dtype=torch.float32,
        prefix=prefix,
    )


def test_every_source_form_resolves_to_the_same_load(tmp_path: Path, fake_mesh):
    """Folder (Path or str), single file (Path or str), iterable of str."""
    fake_mesh(tp_size=1)
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    folder = tmp_path / "ckpt"
    folder.mkdir()
    save_file(
        {"mod.fc.weight": src_w, "mod.fc.bias": src_b},
        str(folder / "model.safetensors"),
    )
    split_a, split_b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file({"mod.fc.weight": src_w}, str(split_a))
    save_file({"mod.fc.bias": src_b}, str(split_b))

    for source in (
        folder,
        str(folder),
        folder / "model.safetensors",
        str(folder / "model.safetensors"),
        [str(split_a), str(split_b)],
    ):
        layer = _make_replicated()
        report = load_pretrained(layer, source)
        assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"], source
        torch.testing.assert_close(layer.weight.data.cpu(), src_w)
        torch.testing.assert_close(layer.bias.data.cpu(), src_b)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no supported model weight files"):
        load_pretrained(_make_replicated(), empty)


def test_load_from_folder_with_index(tmp_path: Path, fake_mesh):
    """source = folder using model.safetensors.index.json across two shards."""
    fake_mesh(tp_size=1)
    layer = _make_replicated()
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    save_file(
        {"mod.fc.weight": src_w},
        str(tmp_path / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"mod.fc.bias": src_b},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 4 * (8 * 4 + 8)},
                "weight_map": {
                    "mod.fc.weight": "model-00001-of-00002.safetensors",
                    "mod.fc.bias": "model-00002-of-00002.safetensors",
                },
            }
        )
    )
    report = load_pretrained(layer, tmp_path)
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]
    torch.testing.assert_close(layer.weight.data.cpu(), src_w)
    torch.testing.assert_close(layer.bias.data.cpu(), src_b)


def test_load_from_single_file_path(tmp_path: Path, fake_mesh):
    """source = a single file path (str or Path), not a folder."""
    fake_mesh(tp_size=1)
    layer = _make_replicated()
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    shard = tmp_path / "shard.safetensors"
    save_file({"mod.fc.weight": src_w, "mod.fc.bias": src_b}, str(shard))

    # Path
    report = load_pretrained(layer, shard)
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]

    # str
    layer2 = _make_replicated(prefix="mod.fc")
    report = load_pretrained(layer2, str(shard))
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]


@pytest.mark.parametrize(
    ("suffix", "wrapper_key"),
    [(".bin", None), (".pt", "model_state_dict"), (".pth", "state_dict")],
)
def test_load_from_pytorch_checkpoint(
    tmp_path: Path,
    fake_mesh,
    suffix: str,
    wrapper_key: str | None,
):
    """PyTorch formats use the same dispatch and report as safetensors."""

    fake_mesh(tp_size=1)
    layer = _make_replicated()
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    state = {"mod.fc.weight": src_w, "mod.fc.bias": src_b}
    checkpoint = (
        state if wrapper_key is None else {wrapper_key: state, "current_iter": 42}
    )
    path = tmp_path / f"checkpoint{suffix}"
    torch.save(checkpoint, path)

    report = load_pretrained(layer, path)

    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]
    assert not report.missing
    assert not report.unexpected
    torch.testing.assert_close(layer.weight.data.cpu(), src_w)
    torch.testing.assert_close(layer.bias.data.cpu(), src_b)


def test_pytorch_folder_and_legacy_serialization_load(tmp_path: Path, fake_mesh):
    fake_mesh(tp_size=1)
    assert checkpoint_format("model.safetensors") == "safetensors"
    assert checkpoint_format("model.bin") == "pytorch"
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)

    # A folder whose only weights file is a .pth with a wrapper key.
    torch.save(
        {"model_state_dict": {"mod.fc.weight": src_w, "mod.fc.bias": src_b}},
        tmp_path / "model.pth",
    )
    layer = _make_replicated()
    report = load_pretrained(layer, tmp_path)
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]
    torch.testing.assert_close(layer.weight.data.cpu(), src_w)

    # Legacy (non-zipfile) serialization.
    torch.save(
        {"mod.fc.weight": src_w, "mod.fc.bias": src_b},
        tmp_path / "legacy.pth",
        _use_new_zipfile_serialization=False,
    )
    layer = _make_replicated()
    report = load_pretrained(layer, tmp_path / "legacy.pth")
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]
    torch.testing.assert_close(layer.bias.data.cpu(), src_b)


def test_duplicate_key_after_remap_raises(tmp_path: Path, fake_mesh):
    fake_mesh(tp_size=1)
    layer = _make_replicated()
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    save_file({"upstream.a": torch.randn(8, 4)}, str(first))
    save_file({"upstream.b": torch.randn(8, 4)}, str(second))

    with pytest.raises(RuntimeError, match="appears more than once after remap"):
        load_pretrained(
            layer,
            [first, second],
            remap=lambda _key: "mod.fc.weight",
            strict=False,
        )


def test_safetensors_dispatches_keys_before_materializing_tensors(
    tmp_path: Path,
    fake_mesh,
    monkeypatch,
):
    fake_mesh(tp_size=1)
    layer = _make_replicated()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"placeholder")
    tensors = {
        "drop.this.weight": torch.zeros(2),
        "totally.unexpected": torch.zeros(2),
        "mod.fc.weight": torch.randn(8, 4),
        "mod.fc.bias": torch.randn(8),
    }
    materialized: list[str] = []

    class TrackingSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def keys(self):
            return list(tensors)

        def get_tensor(self, name):
            materialized.append(name)
            return tensors[name]

    monkeypatch.setattr(
        loader_mod, "safe_open", lambda *_args, **_kwargs: TrackingSafeOpen()
    )

    report = load_pretrained(
        layer,
        checkpoint,
        remap=lambda key: None if key.startswith("drop.") else key,
        strict=False,
        progress=False,
    )

    assert materialized == ["mod.fc.weight", "mod.fc.bias"]
    assert report.unexpected == ["totally.unexpected"]
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]


# --------------------------------------------------------------------------- #
# Progress bar.                                                               #
# --------------------------------------------------------------------------- #


class _FakeBar:
    """Records tqdm interactions so tests can assert bar behaviour."""

    def __init__(self, *, total=None, disable=None, unit=None, **_kwargs):
        self.total = total
        self.disable = disable
        self.unit = unit
        self.updates = 0
        self.closed = False
        self.postfixes: list[str] = []

    def update(self, n=1):
        self.updates += n

    def set_postfix_str(self, s, refresh=True):
        self.postfixes.append(s)

    def close(self):
        self.closed = True


@pytest.fixture
def spy_bar(monkeypatch):
    """Swap the loader's tqdm for a recording fake; yield the captured bars."""
    bars: list[_FakeBar] = []

    def factory(*args, **kwargs):
        bar = _FakeBar(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(loader_mod, "tqdm", factory)
    return bars


def test_count_progress_units_sums_keys_across_checkpoint_formats(tmp_path: Path):
    a, b, c = tmp_path / "a.safetensors", tmp_path / "b.safetensors", tmp_path / "c.pth"
    save_file({"x": torch.zeros(2), "y": torch.zeros(2)}, str(a))
    save_file({"z": torch.zeros(2)}, str(b))
    torch.save({"model_state_dict": {"u": torch.zeros(2)}, "current_iter": 42}, c)
    assert loader_mod._count_progress_units([a, b]) == 3
    assert loader_mod._count_progress_units([a, c]) == 3


def test_progress_disable_resolution(fake_mesh):
    """Non-distributed rank-0: False->disabled, True->on, None->auto."""
    fake_mesh(tp_size=1)
    assert loader_mod._progress_disable(False) is True
    assert loader_mod._progress_disable(True) is False
    assert loader_mod._progress_disable(None) is None


def test_progress_bar_ticks_per_key_and_honours_the_progress_flag(
    tmp_path: Path, fake_mesh, spy_bar
):
    fake_mesh(tp_size=1)
    save_file(
        {
            "mod.fc.weight": torch.randn(8, 4),
            "mod.fc.bias": torch.randn(8),
            "drop.this.weight": torch.zeros(2),  # remapped to None
            "totally.unexpected": torch.zeros(2),  # no owning param
        },
        str(tmp_path / "model.safetensors"),
    )
    drop = lambda k: None if k.startswith("drop.") else k  # noqa: E731

    # progress=True: bar total == key count, one tick per key, dropped included.
    load_pretrained(
        _make_replicated(), tmp_path, progress=True, strict=False, remap=drop
    )
    bar = spy_bar[-1]
    assert (bar.total, bar.updates, bar.disable, bar.unit) == (4, 4, False, "tensor")
    assert bar.closed and bar.postfixes == ["model.safetensors"]

    # progress=False: disabled bar, key pre-count skipped, updates are no-ops.
    load_pretrained(
        _make_replicated(), tmp_path, progress=False, strict=False, remap=drop
    )
    assert (spy_bar[-1].disable, spy_bar[-1].total, spy_bar[-1].updates) == (
        True,
        None,
        4,
    )

    # Default defers to tqdm's own TTY detection.
    load_pretrained(_make_replicated(), tmp_path, strict=False, remap=drop)
    assert spy_bar[-1].disable is None


def test_pytorch_progress_counts_files_and_loads_each_once(
    tmp_path: Path,
    fake_mesh,
    spy_bar,
    monkeypatch,
):
    fake_mesh(tp_size=1)
    layer = _make_replicated()
    weight_path = tmp_path / "weight.bin"
    bias_path = tmp_path / "bias.bin"
    torch.save({"mod.fc.weight": torch.randn(8, 4)}, weight_path)
    torch.save({"mod.fc.bias": torch.randn(8)}, bias_path)
    original_load = loader_mod.torch.load
    calls: list[dict[str, object]] = []

    def counted_load(*args, **kwargs):
        calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(loader_mod.torch, "load", counted_load)

    load_pretrained(layer, [weight_path, bias_path], progress=True)

    assert len(calls) == 2
    assert all(call["map_location"] == "cpu" for call in calls)
    assert all(call["weights_only"] is True for call in calls)
    bar = spy_bar[0]
    assert bar.total == 2
    assert bar.updates == 2
    assert bar.unit == "file"
    assert bar.closed is True


def test_pytorch_load_retries_without_weights_only_for_legacy_tar_only(
    tmp_path: Path, monkeypatch, caplog
):
    checkpoint = tmp_path / "legacy.pth"
    checkpoint.write_bytes(b"placeholder")
    tensor = torch.ones(2)
    calls: list[bool] = []

    def fake_load(*_args, **kwargs):
        calls.append(kwargs["weights_only"])
        if kwargs["weights_only"] is True:
            raise RuntimeError("Cannot load weights in legacy .tar format")
        return {"weight": tensor}

    monkeypatch.setattr(loader_mod.torch, "load", fake_load)
    with caplog.at_level("WARNING", logger=loader_mod.__name__):
        loaded = list(iter_checkpoint_tensors(checkpoint))
    assert calls == [True, False]
    assert loaded[0][0] == "weight"
    torch.testing.assert_close(loaded[0][1], tensor)
    assert "weights_only=False" in caplog.text

    # Any other error is not retried.
    calls.clear()

    def broken_load(*_args, **kwargs):
        calls.append(kwargs["weights_only"])
        raise RuntimeError("corrupted checkpoint")

    monkeypatch.setattr(loader_mod.torch, "load", broken_load)
    with pytest.raises(RuntimeError, match="corrupted checkpoint"):
        list(iter_checkpoint_tensors(tmp_path / "broken.pth"))
    assert calls == [True]


# --------------------------------------------------------------------------- #
# HuggingFace repo-id source (offline; snapshot_download monkeypatched).      #
# --------------------------------------------------------------------------- #


def test_load_pretrained_repo_id_forwarded(tmp_path: Path, fake_mesh, monkeypatch):
    """A repo-id source downloads (faked) then loads from the cached dir.

    Proves ``revision`` threads load_pretrained -> _resolve_source ->
    resolve_checkpoint, and that the returned snapshot dir flows into
    find_safetensors. No network: snapshot_download is monkeypatched.
    """
    fake_mesh(tp_size=1)
    layer = _make_replicated()
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    save_file(
        {"mod.fc.weight": src_w, "mod.fc.bias": src_b},
        str(tmp_path / "model.safetensors"),
    )

    seen: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    report = load_pretrained(layer, "org/model", revision="v1")
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]
    assert seen["repo_id"] == "org/model"
    assert seen["revision"] == "v1"
    torch.testing.assert_close(layer.weight.data.cpu(), src_w)
