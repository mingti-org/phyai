"""End-to-end tests for phyai.weights.load_pretrained."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import phyai.layers.linear as L
from phyai.layers.layer_norm import RMSNorm
from phyai.weights import LoadReport, load_pretrained
from phyai.weights import loader as loader_mod


def _init_dispatcher():
    return L.init(register_flashinfer=False, validate=False)


def test_load_replicated_linear_end_to_end(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
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
    torch.testing.assert_close(layer.weight.data, src_w)
    torch.testing.assert_close(layer.bias.data, src_b)


def test_load_qkv_fused(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
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


def test_load_norm(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    norm = RMSNorm(8, backend="phyai-kernel", prefix="ln")
    src = torch.randn(8)
    save_file({"ln.weight": src}, str(tmp_path / "ln.safetensors"))
    report = load_pretrained(norm, [tmp_path / "ln.safetensors"])
    assert report.loaded == ["ln.weight"]
    torch.testing.assert_close(norm.weight.data, src)


def test_load_strict_missing_raises(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=2,
        out_features=2,
        bias=True,
        params_dtype=torch.float32,
        prefix="x",
    )
    # Save only the weight; bias is missing.
    save_file(
        {"x.weight": torch.zeros(2, 2)},
        str(tmp_path / "incomplete.safetensors"),
    )
    with pytest.raises(RuntimeError, match="strict failure"):
        load_pretrained(layer, [tmp_path / "incomplete.safetensors"])


def test_load_strict_missing_non_strict_returns_report(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=2,
        out_features=2,
        bias=True,
        params_dtype=torch.float32,
        prefix="x",
    )
    save_file(
        {"x.weight": torch.zeros(2, 2)},
        str(tmp_path / "incomplete.safetensors"),
    )
    report = load_pretrained(layer, [tmp_path / "incomplete.safetensors"], strict=False)
    assert "x.bias" in report.missing


def test_unexpected_key_recorded(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
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


def test_remap_callable_rewrites_keys(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=2,
        out_features=2,
        bias=False,
        params_dtype=torch.float32,
        prefix="model.fc",
    )
    src = torch.randn(2, 2)
    save_file(
        {"transformer.fc.weight": src},
        str(tmp_path / "t.safetensors"),
    )
    # Rewrite "transformer." -> "model." at load time.
    report = load_pretrained(
        layer,
        [tmp_path / "t.safetensors"],
        remap=lambda k: k.replace("transformer.", "model."),
    )
    assert report.loaded == ["model.fc.weight"]
    torch.testing.assert_close(layer.weight.data, src)


def test_remap_dict_substring_rewrites(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=2,
        out_features=2,
        bias=False,
        params_dtype=torch.float32,
        prefix="model.fc",
    )
    src = torch.randn(2, 2)
    save_file({"transformer.fc.weight": src}, str(tmp_path / "t.safetensors"))
    report = load_pretrained(
        layer,
        [tmp_path / "t.safetensors"],
        remap={"transformer.": "model."},
    )
    assert report.loaded == ["model.fc.weight"]


def test_remap_returns_none_drops_key(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=2,
        out_features=2,
        bias=False,
        params_dtype=torch.float32,
        prefix="m.fc",
    )
    save_file(
        {"m.fc.weight": torch.zeros(2, 2), "junk.weight": torch.zeros(3)},
        str(tmp_path / "drop.safetensors"),
    )
    # Drop anything matching "junk" — those keys never appear in any list.
    report = load_pretrained(
        layer,
        [tmp_path / "drop.safetensors"],
        remap=lambda k: None if "junk" in k else k,
    )
    assert "junk.weight" not in report.unexpected
    assert "m.fc.weight" in report.loaded


def test_dtype_cast_recorded(tmp_path: Path, fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
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
    fake_mesh(sizes={"tp": 1})

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
    fake_mesh(sizes={"tp": 1})

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
    fake_mesh(sizes={"tp": 1})

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


def test_load_from_folder_single_safetensors(tmp_path: Path, fake_mesh):
    """source = checkpoint folder containing model.safetensors."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = _make_replicated()
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    save_file(
        {"mod.fc.weight": src_w, "mod.fc.bias": src_b},
        str(tmp_path / "model.safetensors"),
    )
    report = load_pretrained(layer, tmp_path)
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]
    torch.testing.assert_close(layer.weight.data, src_w)
    torch.testing.assert_close(layer.bias.data, src_b)


def test_load_from_folder_str_path(tmp_path: Path, fake_mesh):
    """source = str path to a folder."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = _make_replicated()
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    save_file(
        {"mod.fc.weight": src_w, "mod.fc.bias": src_b},
        str(tmp_path / "model.safetensors"),
    )
    report = load_pretrained(layer, str(tmp_path))
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]


def test_load_from_folder_with_index(tmp_path: Path, fake_mesh):
    """source = folder using model.safetensors.index.json across two shards."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
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
    torch.testing.assert_close(layer.weight.data, src_w)
    torch.testing.assert_close(layer.bias.data, src_b)


def test_load_from_single_file_path(tmp_path: Path, fake_mesh):
    """source = a single file path (str or Path), not a folder."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
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


def test_load_from_iterable_of_str(tmp_path: Path, fake_mesh):
    """source = iterable of str (existing-iterable contract preserved)."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = _make_replicated()
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    save_file({"mod.fc.weight": src_w}, str(a))
    save_file({"mod.fc.bias": src_b}, str(b))
    report = load_pretrained(layer, [str(a), str(b)])
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]


def test_load_from_empty_folder_raises(tmp_path: Path, fake_mesh):
    """source = folder with no safetensors files -> FileNotFoundError."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = _make_replicated()
    with pytest.raises(FileNotFoundError, match="no safetensors shards"):
        load_pretrained(layer, tmp_path)


def test_load_unexpected_keys_with_dropping_remap_via_folder(tmp_path: Path, fake_mesh):
    """End-to-end: folder source + remap dropping a known-unwanted key.

    Mirrors the pi05 ``_compose_remap`` use case where the upstream
    checkpoint carries an ``lm_head`` tensor that has no phyai param to
    land in.
    """
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = _make_replicated()
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    save_file(
        {
            "mod.fc.weight": src_w,
            "mod.fc.bias": src_b,
            "drop.this.weight": torch.zeros(2),
        },
        str(tmp_path / "model.safetensors"),
    )
    report = load_pretrained(
        layer,
        tmp_path,
        remap=lambda k: None if k.startswith("drop.") else k,
    )
    assert "drop.this.weight" not in report.unexpected
    assert sorted(report.loaded) == ["mod.fc.bias", "mod.fc.weight"]


# --------------------------------------------------------------------------- #
# Progress bar.                                                               #
# --------------------------------------------------------------------------- #


class _FakeBar:
    """Records tqdm interactions so tests can assert bar behaviour."""

    def __init__(self, *, total=None, disable=None, **_kwargs):
        self.total = total
        self.disable = disable
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


def test_count_keys_sums_across_shards(tmp_path: Path):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    save_file({"x": torch.zeros(2), "y": torch.zeros(2)}, str(a))
    save_file({"z": torch.zeros(2)}, str(b))
    assert loader_mod._count_keys([a, b]) == 3


def test_progress_disable_resolution(fake_mesh):
    """Non-distributed rank-0: False->disabled, True->on, None->auto."""
    fake_mesh(sizes={"tp": 1})
    assert loader_mod._progress_disable(False) is True
    assert loader_mod._progress_disable(True) is False
    assert loader_mod._progress_disable(None) is None


def test_progress_bar_advances_once_per_key(tmp_path: Path, fake_mesh, spy_bar):
    """Bar total == key count and it ticks for every key, dropped ones included."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = _make_replicated()
    src_w = torch.randn(8, 4, dtype=torch.float32)
    src_b = torch.randn(8, dtype=torch.float32)
    save_file(
        {
            "mod.fc.weight": src_w,
            "mod.fc.bias": src_b,
            "drop.this.weight": torch.zeros(2),  # remapped to None
            "totally.unexpected": torch.zeros(2),  # no owning param
        },
        str(tmp_path / "model.safetensors"),
    )
    load_pretrained(
        layer,
        tmp_path,
        progress=True,
        strict=False,  # the unexpected key would otherwise raise before the bar closes
        remap=lambda k: None if k.startswith("drop.") else k,
    )
    assert len(spy_bar) == 1
    bar = spy_bar[0]
    assert bar.total == 4  # every key counted
    assert bar.updates == 4  # advanced once per key
    assert bar.disable is False  # progress=True forces it on
    assert bar.closed is True
    assert bar.postfixes == ["model.safetensors"]


def test_progress_false_disables_bar(tmp_path: Path, fake_mesh, spy_bar):
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = _make_replicated()
    save_file(
        {"mod.fc.weight": torch.randn(8, 4), "mod.fc.bias": torch.randn(8)},
        str(tmp_path / "model.safetensors"),
    )
    load_pretrained(layer, tmp_path, progress=False)
    bar = spy_bar[0]
    assert bar.disable is True
    assert bar.total is None  # key pre-count skipped when disabled
    assert bar.updates == 2  # still iterates; updates are no-ops on a disabled bar


def test_progress_default_is_auto(tmp_path: Path, fake_mesh, spy_bar):
    """Default (no progress kwarg) defers to tqdm's own TTY detection."""
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
    layer = _make_replicated()
    save_file(
        {"mod.fc.weight": torch.randn(8, 4), "mod.fc.bias": torch.randn(8)},
        str(tmp_path / "model.safetensors"),
    )
    load_pretrained(layer, tmp_path)
    assert spy_bar[0].disable is None


# --------------------------------------------------------------------------- #
# HuggingFace repo-id source (offline; snapshot_download monkeypatched).      #
# --------------------------------------------------------------------------- #


def test_load_pretrained_repo_id_forwarded(tmp_path: Path, fake_mesh, monkeypatch):
    """A repo-id source downloads (faked) then loads from the cached dir.

    Proves ``revision`` threads load_pretrained -> _resolve_source ->
    resolve_checkpoint, and that the returned snapshot dir flows into
    find_safetensors. No network: snapshot_download is monkeypatched.
    """
    fake_mesh(sizes={"tp": 1})
    _init_dispatcher()
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
    torch.testing.assert_close(layer.weight.data, src_w)
