"""aiter rows on AMD: selection gating anywhere, numerics against the reference rows on a ROCm GPU."""

from __future__ import annotations

import pytest
import torch

import phyai.kernel.selector as selector_module
from phyai.kernel.library import library_facts
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery

QUERIES = {
    "aiter.attention_paged": KernelQuery.build(
        "attention_paged",
        role="prefix",
        dtype={"input": "bf16"},
        shape={"tokens": 968, "heads": 8, "kv_heads": 1, "head_dim": 256},
        attrs={"layout": "paged", "causal": False},
    ),
    "aiter.rmsnorm": KernelQuery.build(
        "rmsnorm",
        dtype={"input": "bf16", "weight": "bf16"},
        shape={"tokens": 968, "hidden": 2048},
        attrs={"variant": "gemma"},
    ),
    "aiter.rope": KernelQuery.build(
        "rope",
        dtype={"input": "bf16"},
        shape={
            "tokens": 968,
            "head_dim": 256,
            "rotary_dim": 256,
            "heads_q": 8,
            "heads_k": 1,
        },
        attrs={"rope_type": "default", "interleave": False},
    ),
    "aiter.activation.gelu_and_mul": KernelQuery.build(
        "activation",
        dtype={"input": "bf16"},
        shape={"tokens": 968, "hidden": 16384},
        attrs={"activation": "gelu", "gated": True},
    ),
}


@pytest.mark.parametrize("kernel_id", QUERIES)
def test_aiter_rows_are_selected_on_amd_only(kernel_id, monkeypatch):
    monkeypatch.setattr(
        selector_module,
        "library_facts",
        lambda names: {**library_facts(names), "lib.aiter": "aiter" in names},
    )
    amd = Selector(build_catalog(), device="amd:gfx942").explain(QUERIES[kernel_id])
    nvidia = Selector(build_catalog(), device="nvidia:sm90").explain(QUERIES[kernel_id])
    assert amd.selected == kernel_id
    assert not nvidia.selected.startswith("aiter.")


def test_aiter_rmsnorm_rejects_fp32_weight(monkeypatch):
    monkeypatch.setattr(
        selector_module,
        "library_facts",
        lambda names: {**library_facts(names), "lib.aiter": "aiter" in names},
    )
    query = KernelQuery.build(
        "rmsnorm",
        dtype={"input": "bf16", "weight": "fp32"},
        shape={"tokens": 8, "hidden": 64},
        attrs={"variant": "gemma"},
    )
    assert (
        Selector(build_catalog(), device="amd:gfx942").explain(query).selected
        != "aiter.rmsnorm"
    )


ADARMSNORM_FIELDS = dict(
    dtype={"input": "bf16", "modulation": "bf16"},
    shape={"tokens": 50, "hidden": 1024, "cond_dim": 1024},
)


def test_triton_adarmsnorm_is_selected_on_amd(monkeypatch):
    monkeypatch.setattr(
        selector_module,
        "library_facts",
        lambda names: {
            **library_facts(names),
            "lib.phyai_kernel": "phyai_kernel" in names,
        },
    )
    amd = Selector(build_catalog(), device="amd:gfx942").explain(
        KernelQuery.build("adarmsnorm", **ADARMSNORM_FIELDS)
    )
    assert amd.selected == "phyai_kernel.adarmsnorm"


rocm = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="aiter kernels need a ROCm GPU",
)


def _select(op: str, reference: bool = False, **fields):
    selector = Selector(build_catalog(), device="amd:gfx942")
    query = KernelQuery.build(op, **fields)
    if not reference:
        return selector.select(query)
    eligible = [c.kernel_id for c in selector.explain(query).candidates if c.eligible]
    return selector.select(
        query, prefer=[k for k in eligible if k.startswith(("torch.", "eager."))][:1]
    )


@rocm
@pytest.mark.parametrize(
    "x_shape, modulation_shape",
    [
        ((1, 50, 1024), (1, 1, 3072)),
        ((2, 25, 1024), (2, 1, 3072)),
        ((50, 1024), (1, 3072)),
    ],
)
def test_triton_adarmsnorm_matches_torch(x_shape, modulation_shape):
    pytest.importorskip("phyai_kernel")
    triton_row = _select("adarmsnorm", **ADARMSNORM_FIELDS)
    torch_row = _select("adarmsnorm", reference=True, **ADARMSNORM_FIELDS)
    assert (triton_row.kernel_id, torch_row.kernel_id) == (
        "phyai_kernel.adarmsnorm",
        "torch.adarmsnorm",
    )
    torch.manual_seed(0)
    x = torch.randn(x_shape, device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn(modulation_shape, device="cuda", dtype=torch.bfloat16)
    for actual, expected in zip(
        triton_row.execute(x, modulation, 1e-6), torch_row.execute(x, modulation, 1e-6)
    ):
        torch.testing.assert_close(
            actual, expected.expand_as(actual), atol=3e-2, rtol=2e-2
        )


@rocm
@pytest.mark.parametrize("variant", ["gemma", "default"])
def test_aiter_rmsnorm_matches_torch(variant):
    pytest.importorskip("aiter")
    fields = dict(
        dtype={"input": "bf16", "weight": "bf16"},
        shape={"tokens": 50, "hidden": 1024},
        attrs={"variant": variant},
    )
    aiter_row, torch_row = (
        _select("rmsnorm", **fields),
        _select("rmsnorm", reference=True, **fields),
    )
    assert (aiter_row.kernel_id, torch_row.kernel_id) == (
        "aiter.rmsnorm",
        "torch.rmsnorm",
    )
    torch.manual_seed(0)
    x = torch.randn(2, 25, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1
    for _ in range(2):
        torch.testing.assert_close(
            aiter_row.execute(x, weight, 1e-6),
            torch_row.execute(x, weight, 1e-6),
            atol=3e-2,
            rtol=2e-2,
        )
        # An in-place weight update must not be served from the cached Gemma multiplier.
        weight.add_(0.5)


@rocm
@pytest.mark.parametrize("interleave", [False, True])
def test_aiter_rope_matches_eager(interleave):
    pytest.importorskip("aiter")
    fields = dict(
        dtype={"input": "bf16"},
        shape={
            "tokens": 50,
            "head_dim": 256,
            "rotary_dim": 256,
            "heads_q": 8,
            "heads_k": 1,
        },
        attrs={"rope_type": "default", "interleave": interleave},
    )
    aiter_row, eager_row = (
        _select("rope", **fields),
        _select("rope", reference=True, **fields),
    )
    assert (aiter_row.kernel_id, eager_row.kernel_id) == ("aiter.rope", "eager.rope")
    torch.manual_seed(0)
    inv = 1.0 / (
        10000 ** (torch.arange(0, 256, 2, device="cuda", dtype=torch.float32) / 256)
    )
    freqs = torch.outer(torch.arange(2048, device="cuda", dtype=torch.float32), inv)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    q = torch.randn(50, 8, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(50, 1, 256, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(968, 1018, device="cuda", dtype=torch.int32)
    for actual, expected in zip(
        aiter_row.execute(positions, q, k, cache),
        eager_row.execute(positions, q, k, cache),
    ):
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)


@rocm
def test_aiter_rows_replay_in_a_graph_other_than_the_first():
    # Runners capture one graph per bucket; a row first run inside the capture
    # of one graph must still be correct when only another graph replays.
    pytest.importorskip("aiter")
    rope_fields = dict(
        dtype={"input": "bf16"},
        shape={
            "tokens": 50,
            "head_dim": 256,
            "rotary_dim": 256,
            "heads_q": 8,
            "heads_k": 1,
        },
        attrs={"rope_type": "default", "interleave": False},
    )
    norm_fields = dict(
        dtype={"input": "bf16", "weight": "bf16"},
        shape={"tokens": 50, "hidden": 1024},
        attrs={"variant": "gemma"},
    )
    rope, eager_rope = (
        _select("rope", **rope_fields),
        _select("rope", reference=True, **rope_fields),
    )
    norm, torch_norm = (
        _select("rmsnorm", **norm_fields),
        _select("rmsnorm", reference=True, **norm_fields),
    )
    inv = 1.0 / (
        10000 ** (torch.arange(0, 256, 2, device="cuda", dtype=torch.float32) / 256)
    )
    freqs = torch.outer(torch.arange(2048, device="cuda", dtype=torch.float32), inv)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    q = torch.randn(50, 8, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(50, 1, 256, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(50, 1024, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16) * 0.1
    positions = torch.arange(968, 1018, device="cuda", dtype=torch.int32)

    def run(rope=rope, norm=norm):
        return (*rope.execute(positions, q, k, cache), norm.execute(x, weight, 1e-6))

    # Load the aiter modules through separately prepared rows, outside capture.
    run(_select("rope", **rope_fields), _select("rmsnorm", **norm_fields))
    torch.cuda.synchronize()
    first, second = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(first):
        run()
    with torch.cuda.graph(second):
        outputs = run()
    second.replay()
    expected = (
        *eager_rope.execute(positions, q, k, cache),
        torch_norm.execute(x, weight, 1e-6),
    )
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, atol=3e-2, rtol=2e-2)


@rocm
@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
def test_aiter_gated_activation_matches_torch(activation):
    pytest.importorskip("aiter")
    fields = dict(
        dtype={"input": "bf16"},
        shape={"tokens": 50, "hidden": 4096},
        attrs={"activation": activation, "gated": True},
    )
    aiter_row, torch_row = (
        _select("activation", **fields),
        _select("activation", reference=True, **fields),
    )
    assert aiter_row.kernel_id.startswith("aiter.") and torch_row.kernel_id.startswith(
        "torch."
    )
    x = torch.randn(50, 8192, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        aiter_row.execute(x), torch_row.execute(x), atol=3e-2, rtol=2e-2
    )


def _paged_setup(causal: bool, q_lens, kv_lens):
    from phyai.cache import KVCachePool
    from phyai.layers.attention import (
        AttnLayout,
        AttnMode,
        PagedAttention,
        PagedAttnMetadata,
    )

    H, H_kv, D = 8, 1, 256
    pool = KVCachePool(
        num_layers=1,
        num_slots=4096,
        num_kv_heads=H_kv,
        head_dim=D,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    layer = PagedAttention(
        num_heads=H, head_dim=D, layer_id=0, num_kv_heads=H_kv, causal=causal
    )

    def meta(seed: int):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        slots = torch.randperm(4096, device="cuda", generator=generator)[: sum(kv_lens)]
        new_tokens = [min(q, kv) for q, kv in zip(q_lens, kv_lens)]
        writes = torch.cat(
            [
                slots[end - n : end]
                for end, n in zip(torch.tensor(kv_lens).cumsum(0).tolist(), new_tokens)
            ]
        )
        return PagedAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=len(q_lens),
            num_query_tokens=sum(q_lens),
            cu_seqlens_q=torch.tensor([0, *q_lens], device="cuda")
            .cumsum(0)
            .to(torch.int32),
            paged_kv_indptr=torch.tensor([0, *kv_lens], device="cuda")
            .cumsum(0)
            .to(torch.int32),
            paged_kv_indices=slots.to(torch.int32),
            paged_kv_last_page_len=torch.ones(
                len(q_lens), device="cuda", dtype=torch.int32
            ),
            write_indices=writes.to(torch.int64),
        ), sum(new_tokens)

    return pool, layer, meta, (H, H_kv, D)


def _run(backend, plan, pool, layer, meta, q, k, v):
    from phyai.layers.attention import AttnLayout, AttnMode, PagedAttnCtx

    ctx = PagedAttnCtx(
        backend=backend,
        plan=plan,
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        kv_pool=pool,
        write_indices=meta.write_indices,
    )
    return layer(q, k, v, ctx)


def _backends(layer, q_lens, kv_lens):
    from phyai.layers.attention.paged.backends.aiter import AiterPagedBackend
    from phyai.layers.attention.paged.backends.sdpa import SdpaPagedBackend

    backends = (AiterPagedBackend(None), SdpaPagedBackend(None))
    for backend in backends:
        backend.init_cuda_graph_state(
            max_batch_size=len(q_lens),
            max_num_tokens=max(sum(q_lens), sum(kv_lens)),
            max_paged_kv_indices=sum(kv_lens),
            device=torch.device("cuda"),
            params_dtype=torch.bfloat16,
            layer_proto=layer,
        )
    return backends


@rocm
@pytest.mark.parametrize(
    "causal,q_lens,kv_lens",
    [
        (False, [968], [968]),
        (False, [50], [1018]),
        (True, [50], [1018]),
        (True, [7, 33], [7, 40]),
    ],
)
def test_aiter_paged_matches_sdpa(causal, q_lens, kv_lens):
    pytest.importorskip("aiter")
    pool, layer, make_meta, (H, H_kv, D) = _paged_setup(causal, q_lens, kv_lens)
    meta, writes = make_meta(0)
    aiter_backend, sdpa_backend = _backends(layer, q_lens, kv_lens)
    torch.manual_seed(0)
    q = torch.randn(sum(q_lens), H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(writes, H_kv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(writes, H_kv, D, device="cuda", dtype=torch.bfloat16)
    pool.k_buffers[0].normal_()
    pool.v_buffers[0].normal_()
    actual = _run(
        aiter_backend,
        aiter_backend.init_forward_metadata(meta),
        pool,
        layer,
        meta,
        q,
        k,
        v,
    )
    expected = _run(
        sdpa_backend,
        sdpa_backend.init_forward_metadata(meta),
        pool,
        layer,
        meta,
        q,
        k,
        v,
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@rocm
def test_aiter_paged_replays_in_a_cuda_graph():
    pytest.importorskip("aiter")
    # Runners capture with a short seed KV length and replay the real one.
    q_lens, kv_lens = [50], [1018]
    pool, layer, make_meta, (H, H_kv, D) = _paged_setup(False, q_lens, kv_lens)
    seed_meta, writes = _paged_setup(False, q_lens, [200])[2](0)
    aiter_backend, sdpa_backend = _backends(layer, q_lens, kv_lens)
    pool.k_buffers[0].normal_()
    pool.v_buffers[0].normal_()
    q = torch.randn(50, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(writes, H_kv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(writes, H_kv, D, device="cuda", dtype=torch.bfloat16)
    write_indices = seed_meta.write_indices.clone()
    plan = aiter_backend.init_capture_metadata(seed_meta)

    from phyai.layers.attention import AttnLayout, AttnMode, PagedAttnCtx

    ctx = PagedAttnCtx(
        backend=aiter_backend,
        plan=plan,
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        kv_pool=pool,
        write_indices=write_indices,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer(q, k, v, ctx)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer(q, k, v, ctx)

    replay_meta, replay_writes = make_meta(1)
    assert replay_writes == writes
    aiter_backend.replay_metadata(plan, replay_meta)
    write_indices.copy_(replay_meta.write_indices)
    q.normal_()
    graph.replay()
    expected = _run(
        sdpa_backend,
        sdpa_backend.init_forward_metadata(replay_meta),
        pool,
        layer,
        replay_meta,
        q,
        k,
        v,
    )
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)

    two_sequences, _ = _paged_setup(False, [25, 25], [500, 518])[2](2)
    with pytest.raises(ValueError, match="batch size"):
        aiter_backend.replay_metadata(plan, two_sequences)
