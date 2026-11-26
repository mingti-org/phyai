"""Paged SDPA numerical, policy, and CUDA graph contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from phyai.cache import KVCachePool
from phyai.kernel.policy import policy_from_mapping
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery
from phyai.layers.attention import (
    AttnLayout,
    AttnMode,
    PagedAttention,
    PagedAttnCtx,
    PagedAttnMetadata,
    SdpaPagedBackend,
)


def _metadata(q_offsets, kv_offsets, indices, write_indices):
    return PagedAttnMetadata(
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        batch_size=len(q_offsets) - 1,
        num_query_tokens=q_offsets[-1],
        cu_seqlens_q=torch.tensor(q_offsets, dtype=torch.int32),
        paged_kv_indptr=torch.tensor(kv_offsets, dtype=torch.int32),
        paged_kv_indices=indices,
        paged_kv_last_page_len=torch.ones(len(q_offsets) - 1, dtype=torch.int32),
        write_indices=write_indices,
    )


def _case(
    dtype,
    device,
    *,
    causal=True,
    q_offsets=(0, 3, 6),
    kv_offsets=(0, 5, 7),
    strided=False,
    select_kernel=False,
):
    torch.manual_seed(43)
    layer = PagedAttention(
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        layer_id=0,
        causal=causal,
        backend="sdpa",
    )
    pool = KVCachePool(
        num_layers=1,
        num_slots=24,
        num_kv_heads=2,
        head_dim=32,
        dtype=dtype,
        device=device,
    )
    pool.k_buffer(0).normal_()
    pool.v_buffer(0).normal_()
    q = torch.randn(q_offsets[-1], 4, 64 if strided else 32, dtype=dtype, device=device)
    if strided:
        q = q[..., ::2]
    k = torch.randn(2, 2, 32, dtype=dtype, device=device)
    v = torch.randn_like(k)
    write_indices = torch.tensor([4, 6], dtype=torch.int64, device=device)
    indices = torch.tensor([7, 2, 9, 1, 4, 12, 6], dtype=torch.int32, device=device)[
        : kv_offsets[-1]
    ]
    meta = _metadata(q_offsets, kv_offsets, indices, write_indices)
    backend = SdpaPagedBackend(select_kernel=select_kernel)
    backend.init_cuda_graph_state(
        max_batch_size=2,
        max_num_tokens=q_offsets[-1],
        max_paged_kv_indices=7,
        device=torch.device(device),
        params_dtype=dtype,
        layer_proto=layer,
    )
    plan = backend.init_forward_metadata(meta)
    ctx = PagedAttnCtx(
        backend=backend,
        plan=plan,
        mode=meta.mode,
        layout=meta.layout,
        kv_pool=pool,
        write_indices=write_indices,
    )
    return layer, q, k, v, ctx, meta


def _reference(layer, q, ctx):
    """Independent CPU float64 oracle, including fully masked query rows."""
    plan = ctx.plan
    keys = (
        ctx.kv_pool.k_buffer(layer.layer_id)
        .cpu()
        .double()[plan.indices.cpu().long(), 0]
    )
    values = (
        ctx.kv_pool.v_buffer(layer.layer_id)
        .cpu()
        .double()[plan.indices.cpu().long(), 0]
    )
    keys = keys.repeat_interleave(layer.num_heads // layer.num_kv_heads, dim=1)
    values = values.repeat_interleave(layer.num_heads // layer.num_kv_heads, dim=1)
    results = []
    for qs, qe, ks, ke in zip(
        plan.q_offsets, plan.q_offsets[1:], plan.kv_offsets, plan.kv_offsets[1:]
    ):
        query = q[qs:qe].cpu().double()
        if qs == qe or ks == ke:
            results.append(torch.zeros_like(query))
            continue
        logits = torch.einsum("qhd,khd->hqk", query, keys[ks:ke]) * layer.scale
        if layer.causal:
            visible = torch.arange(ke - ks)[None, :] <= torch.arange(qe - qs)[
                :, None
            ] + (ke - ks) - (qe - qs)
            logits.masked_fill_(~visible, float("-inf"))
        probs = logits.softmax(-1).nan_to_num()
        results.append(torch.einsum("hqk,khd->qhd", probs, values[ks:ke]))
    return (
        torch.cat(results)
        if results
        else torch.empty_like(q, device="cpu", dtype=torch.float64)
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_sdpa_paged_ragged_gqa_matches_reference(device, dtype, causal, strided):
    layer, q, k, v, ctx, _ = _case(dtype, device, causal=causal, strided=strided)
    actual = layer(q, k, v, ctx)
    assert actual.dtype == dtype
    tol = 8e-3 if dtype == torch.bfloat16 else 1e-3 if dtype == torch.float16 else 4e-6
    torch.testing.assert_close(
        actual.cpu().double(), _reference(layer, q, ctx), atol=tol, rtol=tol
    )
    torch.testing.assert_close(ctx.kv_pool.k_buffer(0)[ctx.write_indices, 0], k)
    torch.testing.assert_close(ctx.kv_pool.v_buffer(0)[ctx.write_indices, 0], v)


@pytest.mark.parametrize("select_kernel", [False, True])
def test_sdpa_paged_fp32_survives_autocast(select_kernel):
    layer, q, k, v, ctx, _ = _case(torch.float32, "cuda", select_kernel=select_kernel)
    expected = layer(q, k, v, ctx)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = layer(q, k, v, ctx)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "q_offsets,kv_offsets", [((0, 0, 0), (0, 0, 0)), ((0, 0, 3), (0, 2, 2))]
)
def test_sdpa_paged_empty_queries_and_keys(q_offsets, kv_offsets):
    layer, q, k, v, ctx, _ = _case(
        torch.float32, "cuda", q_offsets=q_offsets, kv_offsets=kv_offsets
    )
    actual = layer(q, k, v, ctx)
    assert actual.shape == q.shape
    assert torch.count_nonzero(actual) == 0


def test_sdpa_paged_idle_does_not_write_cache():
    layer, q, k, v, ctx, _ = _case(torch.float32, "cpu")
    before = ctx.kv_pool.k_buffer(0).clone()
    actual = layer(q, k, v, replace(ctx, mode=AttnMode.IDLE))
    assert torch.count_nonzero(actual) == 0
    torch.testing.assert_close(ctx.kv_pool.k_buffer(0), before, atol=0, rtol=0)


@pytest.mark.parametrize(
    "field,offsets", [("cu_seqlens_q", (0, 1, 6)), ("paged_kv_indptr", (0, 4, 7))]
)
def test_sdpa_paged_replay_rejects_changed_sequence_lengths(field, offsets):
    _, _, _, _, ctx, meta = _case(torch.float32, "cpu")
    changed = replace(meta, **{field: torch.tensor(offsets, dtype=torch.int32)})
    with pytest.raises(ValueError, match="cannot change sequence offsets"):
        ctx.backend.replay_metadata(ctx.plan, changed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sdpa_paged_graph_replay_reads_updated_kv_and_slot_indices(dtype):
    layer, q, k, v, ctx, meta = _case(dtype, "cuda")
    plan = ctx.backend.init_capture_metadata(meta)
    assert plan.indices.data_ptr() != meta.paged_kv_indices.data_ptr()
    ctx = replace(ctx, plan=plan)
    pointer = plan.indices.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(q, k, v, ctx)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(q, k, v, ctx)
    graph.replay()
    before = output.clone()

    q.normal_()
    k.normal_()
    v.normal_()
    ctx.write_indices.copy_(torch.tensor([17, 19], dtype=torch.int64, device="cuda"))
    changed_indices = torch.tensor(
        [13, 14, 15, 16, 17, 18, 19], dtype=torch.int32, device="cuda"
    )
    ctx.backend.replay_metadata(plan, replace(meta, paged_kv_indices=changed_indices))
    changed_indices.zero_()
    meta.paged_kv_indices.zero_()
    assert plan.indices.data_ptr() == pointer
    graph.replay()
    tol = 8e-3 if dtype == torch.bfloat16 else 4e-6
    torch.testing.assert_close(
        output.cpu().double(), _reference(layer, q, ctx), atol=tol, rtol=tol
    )
    assert not torch.equal(output, before)
    torch.testing.assert_close(ctx.kv_pool.k_buffer(0)[ctx.write_indices, 0], k)


def _query(dtype):
    return KernelQuery.build(
        "attention_paged",
        dtype={"input": dtype, "output": dtype},
        shape={"tokens": 6, "heads": 4, "kv_heads": 2, "head_dim": 64},
        attrs={"layout": "paged", "causal": True},
        mode="capture",
    )


@pytest.mark.parametrize(
    "device,dtype,expected",
    [
        ("cpu", "fp32", "sdpa.attention_paged"),
        ("nvidia:SM90", "fp32", "sdpa.attention_paged"),
        ("nvidia:SM90", "bf16", "flashinfer.attention_paged"),
    ],
)
def test_sdpa_paged_selection_preserves_flashinfer_default(device, dtype, expected):
    assert (
        Selector(build_catalog(), device=device).select(_query(dtype)).kernel_id
        == expected
    )


def test_sdpa_paged_policy_selects_real_backend_instance():
    catalog = build_catalog()
    policy = policy_from_mapping(
        {
            "schema": 1,
            "rules": [
                {
                    "id": "sdpa-paged",
                    "match": {"op": "attention_paged"},
                    "restrict_to": "sdpa.attention_paged",
                    "params": {"select_kernel": True},
                }
            ],
        },
        catalog,
        source="test",
    )
    selected = Selector(catalog, policy, device="nvidia:SM90").select(_query("bf16"))
    assert selected.kernel_id == "sdpa.attention_paged"
    backend = selected.implementation(None)
    assert isinstance(backend, SdpaPagedBackend)
    assert backend.select_kernel
    assert backend.supports_capture()
