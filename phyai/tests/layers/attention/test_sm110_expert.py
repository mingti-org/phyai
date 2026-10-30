"""Paged metadata, fallback and captured Expert backend lifecycle."""

from dataclasses import replace

import pytest
import torch

from phyai.cache import KVCachePool
from phyai.layers.attention import (
    AttnLayout,
    AttnMode,
    PagedAttention,
    PagedAttnCtx,
    PagedAttnMetadata,
)
from phyai.layers.attention.paged.backends.flashinfer import FlashInferPagedBackend
from phyai.layers.attention.paged.backends.sm110 import SM110ExpertBackend
from phyai.runtime.cuda_graph_manager import CudaGraph

pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (11, 0), reason="requires SM110"
)


@pytest.mark.parametrize(
    "capacity,variant",
    (
        (578, "c27"),
        (610, "flashinfer"),
        (674, "c27"),
        (762, "c27"),
        (580, "flashinfer"),
        (None, "flashinfer"),
    ),
)
def test_host_capacity_selects_graph_without_device_read(capacity, variant):
    meta = PagedAttnMetadata(
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        batch_size=1,
        num_query_tokens=50,
        cu_seqlens_q=torch.tensor([0, 50]),
        max_kv_tokens=capacity,
    )
    backend = SM110ExpertBackend()
    assert backend.graph_replay_key(meta) == variant
    assert backend.graph_replay_key(meta, rollout=True) == "flashinfer"
    assert backend.graph_capture_keys(rollout=True) == ("flashinfer",)
    assert backend.graph_replay_key(replace(meta, batch_size=2)) == "flashinfer"
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="max_kv_tokens"):
            replace(meta, max_kv_tokens=bad)


@torch.no_grad()
def test_backend_graph_variants_replay_changed_lengths_and_page_order():
    torch.manual_seed(1021)
    pool = KVCachePool(
        num_layers=1,
        num_slots=800,
        num_kv_heads=1,
        head_dim=256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    layer = PagedAttention(
        num_heads=8, num_kv_heads=1, head_dim=256, layer_id=0, causal=False
    )
    q = torch.randn(50, 8, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(50, 1, 256, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    writes = torch.arange(750, 800, device="cuda", dtype=torch.int64)
    pool.k_buffer(0).normal_()
    pool.v_buffer(0).normal_()

    def meta(capacity, valid):
        return PagedAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=1,
            num_query_tokens=50,
            cu_seqlens_q=torch.tensor([0, 50], device="cuda", dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, valid], device="cuda", dtype=torch.int32),
            paged_kv_indices=torch.randperm(750, device="cuda", dtype=torch.int32)[
                :valid
            ],
            paged_kv_last_page_len=torch.ones(1, device="cuda", dtype=torch.int32),
            write_indices=writes,
            max_kv_tokens=capacity,
        )

    backend = SM110ExpertBackend()
    reference = FlashInferPagedBackend()
    for b in (backend, reference):
        b.init_cuda_graph_state(
            max_batch_size=1,
            max_num_tokens=50,
            max_paged_kv_indices=800,
            device=torch.device("cuda"),
            params_dtype=q.dtype,
            layer_proto=layer,
        )
    seed = meta(578, 173)
    plan = backend.init_capture_metadata(seed)
    ctx = PagedAttnCtx(
        backend=backend,
        plan=plan,
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        kv_pool=pool,
        write_indices=writes,
    )
    graphs = {}
    for key in backend.graph_capture_keys():
        backend.set_graph_capture_key(key)
        graph = CudaGraph()
        graph.capture(lambda q, k, v: layer(q, k, v, ctx), {"q": q, "k": k, "v": v})
        graphs[key] = graph
    stable = tuple(
        t.data_ptr()
        for t in (backend._paged_kv_indices_buf, backend._paged_kv_indptr_buf)
    )
    for capacity, valid in ((610, 73), (762, 711), (578, 257), (580, 201), (674, 529)):
        current = meta(capacity, valid)
        backend.replay_metadata(plan, current)
        refplan = reference.init_forward_metadata(current)
        refctx = replace(ctx, backend=reference, plan=refplan)
        q.add_(0.01)
        actual = (
            graphs[backend.graph_replay_key(current)]
            .replay({"q": q, "k": k, "v": v})
            .clone()
        )
        expected = layer(q, k, v, refctx)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        assert stable == tuple(
            t.data_ptr()
            for t in (backend._paged_kv_indices_buf, backend._paged_kv_indptr_buf)
        )
    for graph in graphs.values():
        graph.reset()


@torch.no_grad()
def test_same_layout_after_hot_update_replans_seeded_attention():
    from types import SimpleNamespace
    from phyai.models.pi05.scheduler_pi05 import PI05Scheduler

    scheduler = object.__new__(PI05Scheduler)
    scheduler.cfg = SimpleNamespace(num_inference_steps=10, chunk_size=50)
    scheduler.device = torch.device("cuda")
    scheduler.params_dtype = torch.bfloat16
    scheduler.vision_input_dtype = torch.float32
    scheduler.max_batch_size = 1
    scheduler.time_emb_table = torch.zeros(10, device="cuda")
    key = (1, (4,))
    layout = SimpleNamespace(
        n_per_sample=4,
        n_real_total=4,
        prefix_meta=object(),
        joint_meta=object(),
        position_ids=torch.arange(4, device="cuda"),
        write_indices=torch.arange(4, device="cuda"),
    )
    scheduler._layout_cache = {key: layout}
    scheduler._last_layout_key = key
    scheduler._bucket_n_per_sample = lambda length: 4
    hidden = torch.zeros(1, 4, 4, device="cuda", dtype=torch.bfloat16)
    scheduler._pack_prefix = lambda images, language, current: hidden[0]
    state = {"expert_metadata": layout.joint_meta, "replans": 0}

    def recapture():
        state["expert_metadata"] = "capture seed"
        return True

    def plan(metadata):
        state["expert_metadata"] = metadata
        state["replans"] += 1

    scheduler.model = SimpleNamespace(
        heads=SimpleNamespace(embed_time=lambda times: times),
        paligemma_lm=SimpleNamespace(embed_lang=lambda ids: hidden),
    )
    scheduler.expert_runner = SimpleNamespace(
        bind_euler_schedule=lambda *args, **kwargs: None,
        recapture_after_weight_update=recapture,
        plan_inference=plan,
    )
    scheduler.vision_runner = SimpleNamespace(forward=lambda batch: hidden)
    scheduler.llm_runner = SimpleNamespace(
        plan_inference=lambda metadata: None,
        forward=lambda batch, **kwargs: hidden[0],
    )
    allocator = SimpleNamespace(reset=lambda: None, allocate=lambda rows: None)
    scheduler.prefix_static = scheduler.suffix_static = allocator
    request = SimpleNamespace(
        lang_lens=torch.tensor([4]),
        pixel_values=torch.zeros(1, 1, 3, 4, 4, device="cuda"),
        input_ids=torch.zeros(1, 4, device="cuda", dtype=torch.int64),
    )
    scheduler.refresh_weight_dependent_state()
    scheduler._prepare_prefix(request)
    assert state["expert_metadata"] is layout.joint_meta
    assert state["replans"] == 1
    scheduler._prepare_prefix(request)
    assert state["replans"] == 1
