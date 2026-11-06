"""CPU tests for selection and state ownership; no model or CUDA execution."""

from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from phyai.kernel.impls.thor_gemm import PackedWeightCache, ThorDenseGEMM
from phyai.kernel.impls.thor_attention import CompactAttentionBackend, unsupported
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery


@pytest.fixture
def selector(monkeypatch):
    import phyai.kernel.selector as module

    monkeypatch.setattr(
        module, "library_facts", lambda names: {f"lib.{n}": True for n in names}
    )
    return Selector(build_catalog())


def context(enabled):
    return {
        "family": "cosmos3_policy",
        "tags": ["thor_gemm", "thor_compact_attention"] if enabled else [],
    }


def gemm_query(
    arch="sm110", m=3542, n=4096, k=4096, dtype="bf16", mode="eager", enabled=True
):
    return KernelQuery.build(
        "gemm",
        device=f"nvidia:{arch}",
        model=context(enabled),
        dtype={"input": dtype, "output": dtype, "weight": dtype},
        quant={"format": dtype},
        shape={"M": m, "N": n, "K": k},
        mode=mode,
    )


def attention_query(
    arch="sm110",
    heads=32,
    kv_heads=8,
    dim=128,
    causal=False,
    mask=None,
    bias=False,
    mode="eager",
    enabled=True,
):
    return KernelQuery.build(
        "attention",
        device=f"nvidia:{arch}",
        mode=mode,
        model=context(enabled),
        dtype={"input": "bf16", "key": "bf16", "value": "bf16"},
        shape={
            "tokens": 1472,
            "kv_tokens": 1513,
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": dim,
        },
        attrs={
            "layout": "padded",
            "causal": causal,
            "mask_kind": mask,
            "has_attn_bias": bias,
            "sliding_window": None,
            "logits_soft_cap": None,
        },
    )


def chosen(selector, query, prefer):
    return selector.explain(query, prefer=(prefer,)).selected


@pytest.mark.parametrize(
    "m,n,k",
    [(1472, 6144, 4096), (1472, 4096, 12288), (3542, 4096, 4096), (3542, 24576, 4096)],
)
def test_gemm_enabled_supported_geometry(selector, m, n, k):
    assert (
        chosen(selector, gemm_query(m=m, n=n, k=k), "thor.gemm.bf16")
        == "thor.gemm.bf16"
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"arch": "sm90"},
        {"arch": "sm100"},
        {"arch": "sm120"},
        {"m": 816},
        {"m": 1982},
        {"m": 3526},
        {"k": 4095},
        {"dtype": "fp16"},
        {"mode": "capture"},
        {"enabled": False},
    ],
)
def test_gemm_unsupported_uses_original_catalog(selector, kw):
    assert chosen(selector, gemm_query(**kw), "thor.gemm.bf16") != "thor.gemm.bf16"


def test_default_selection_remains_original(selector):
    assert selector.explain(gemm_query(enabled=False)).selected != "thor.gemm.bf16"
    assert (
        selector.explain(attention_query(enabled=False)).selected
        != "thor.attention.compact"
    )
    assert (
        chosen(selector, attention_query(), "thor.attention.compact")
        == "thor.attention.compact"
    )


@pytest.mark.parametrize(
    "kw",
    [
        {"arch": "sm90"},
        {"arch": "sm100"},
        {"heads": 16},
        {"kv_heads": 4},
        {"dim": 64},
        {"causal": True},
        {"mask": "keys"},
        {"bias": True},
        {"mode": "capture"},
        {"enabled": False},
    ],
)
def test_attention_unsupported_uses_original_catalog(selector, kw):
    assert (
        chosen(selector, attention_query(**kw), "thor.attention.compact")
        != "thor.attention.compact"
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_pack_preserves_input_and_contents(transpose):
    base = torch.arange(35, dtype=torch.float32).view(5, 7)
    weight = base.t() if transpose else base
    original = weight.clone()
    for row in [False, True]:
        entry = PackedWeightCache(10000).get(weight, row)
        assert torch.equal(entry.value, original.t()) and torch.equal(weight, original)
        expected = (entry.value.shape[1], 1) if row else (1, entry.value.shape[0])
        assert entry.value.stride() == expected


def test_padded_outer_strides_are_materialized():
    weight = torch.arange(80, dtype=torch.float32).view(8, 10)[:5, :7]
    for row in [False, True]:
        entry = PackedWeightCache(10000).get(weight, row)
        assert torch.equal(entry.value, weight.t())
        expected = (entry.value.shape[1], 1) if row else (1, entry.value.shape[0])
        assert entry.value.stride() == expected


def test_inplace_weight_update_rebuilds_pack():
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    cache = PackedWeightCache(10000)
    first = cache.get(weight, True)
    assert cache.get(weight, True) is first
    weight.add_(11)
    second = cache.get(weight, True)
    assert first.value.data_ptr() != second.value.data_ptr()
    assert torch.equal(second.value, weight.t()) and not torch.equal(
        first.value, second.value
    )


def test_identity_dtype_and_invalidation_are_respected():
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    cache = PackedWeightCache(10000)
    cache.get(weight, True)
    replacement = weight.clone().to(torch.bfloat16)
    second = cache.get(replacement, True)
    assert second.value.dtype == torch.bfloat16 and second.owner() is replacement
    cache.invalidate([weight])
    assert all(key[0] != id(weight) for key in cache.entries)
    assert any(key[0] == id(replacement) for key in cache.entries)


def test_budget_and_lru_and_clear():
    a = torch.randn(4, 6)
    b = torch.randn(4, 6)
    cache = PackedWeightCache(a.numel() * a.element_size())
    cache.get(a, True)
    cache.get(b, True)
    assert len(cache.entries) == 1 and cache.bytes <= cache.max_bytes
    assert PackedWeightCache(0).get(a, True) is None
    cache.invalidate()
    assert not cache.entries and cache.bytes == 0


def test_unversioned_inference_weight_declines_cache():
    with torch.inference_mode():
        weight = torch.ones(4, 6)
    assert PackedWeightCache(1000).get(weight, True) is None


@pytest.mark.parametrize("bias", [False, True])
def test_cpu_gemm_fallback_matches_real_flinear(bias):
    layer = SimpleNamespace(
        prefix="gen_layers.0.cross_attention.to_out", weight=torch.randn(7, 9)
    )
    x = torch.randn(2, 3, 9)
    value = torch.randn(7) if bias else None
    route = ThorDenseGEMM()
    assert torch.equal(route(layer, x, value), F.linear(x, layer.weight, value))
    assert not route.plans and not route.workspaces


def test_attention_private_abi_requires_four_int32_vectors():
    valid = [torch.empty(3, dtype=torch.int32) for _ in range(4)]
    assert CompactAttentionBackend.plan_info(
        SimpleNamespace(_plan_info=valid), torch.device("cpu")
    ) == tuple(valid)
    for invalid in [
        valid[:3],
        [torch.ones(1) for _ in range(4)],
        [torch.ones(2, 2, dtype=torch.int32) for _ in range(4)],
        None,
    ]:
        assert (
            CompactAttentionBackend.plan_info(
                SimpleNamespace(_plan_info=invalid), torch.device("cpu")
            )
            is None
        )


def test_compact_declines_cpu_before_launch():
    q = torch.zeros(1, 3, 32, 128, dtype=torch.bfloat16)
    kv = torch.zeros(1, 7, 8, 128, dtype=torch.bfloat16)
    assert unsupported(None, q, kv, kv, None) == "device_or_dtype"


def test_compact_declines_additive_bias_before_launch():
    ctx = SimpleNamespace(attn_bias=torch.zeros(1))
    assert unsupported(None, None, None, None, ctx) == "attention_bias"


@pytest.mark.parametrize("scale", [None, 128**-0.5, 0.125])
def test_compact_runtime_default_scale_guard(monkeypatch, scale):
    from phyai.layers.attention.nocache import Attention
    from phyai.layers.attention.enums import AttnLayout, AttnMode

    layer = Attention(32, 128, num_kv_heads=8, causal=False, scale=scale)
    q = torch.zeros(1, 3, 32, 128, dtype=torch.bfloat16)
    kv = torch.zeros(1, 7, 8, 128, dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        attn_bias=None, mask=None, mode=AttnMode.PREFILL, layout=AttnLayout.PADDED_4D
    )
    # CPU guard regression only; CUDA arithmetic is validated on Thor.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    expected = "semantics" if scale == 0.125 else None
    assert unsupported(layer, q, kv, kv, ctx) == expected
