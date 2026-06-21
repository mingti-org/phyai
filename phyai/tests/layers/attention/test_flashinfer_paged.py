"""Tests for the flashinfer paged-KV attention backends (AR + Diffusion).

Gated on CUDA + flashinfer-python. Exercises the 4-hook lifecycle
(:meth:`init_cuda_graph_state` -> :meth:`init_capture_metadata` ->
:meth:`replay_metadata` -> :meth:`forward`) and numerical agreement
with a reference path. The paged stacks are flashinfer-only, so the
reference is the no-cache :class:`Attention` layer's eager backend on
CPU fp32 (a single full-sample contiguous-slot paged read reduces to
the same softmax as a ragged no-cache prefill). Parametrized over both
AR and Diffusion stacks since their flashinfer paged backends are
byte-identical implementations today.
"""

from __future__ import annotations

import pytest
import torch

from phyai.cache import KVCachePool
from phyai.layers.attention import (
    ARAttention,
    ARAttnCtx,
    ARAttnMetadata,
    Attention,
    AttnLayout,
    AttnMode,
    DiffusionAttention,
    DiffusionAttnCtx,
    DiffusionAttnMetadata,
    get_ar_backend_factory,
    get_diffusion_backend_factory,
)


def _has_flashinfer() -> bool:
    try:
        import flashinfer.prefill  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _has_flashinfer()),
    reason="flashinfer paged backends require CUDA + flashinfer-python.",
)


_PAGED_FLAVORS = ("ar", "diffusion")


def _layer_cls(flavor: str):
    return ARAttention if flavor == "ar" else DiffusionAttention


def _ctx_cls(flavor: str):
    return ARAttnCtx if flavor == "ar" else DiffusionAttnCtx


def _meta_cls(flavor: str):
    return ARAttnMetadata if flavor == "ar" else DiffusionAttnMetadata


def _factory_for(flavor: str, name: str):
    if flavor == "ar":
        return get_ar_backend_factory(name)
    return get_diffusion_backend_factory(name)


def _make_pool_and_layer(flavor: str):
    """Build a small KV pool + paged layer on CUDA."""
    H, H_kv, D = 4, 2, 64
    pool = KVCachePool(
        num_layers=1,
        num_slots=16,
        num_kv_heads=H_kv,
        head_dim=D,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )
    layer = _layer_cls(flavor)(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H_kv,
        causal=False,
        backend="flashinfer",
    )
    return pool, layer, (H, H_kv, D)


def _meta_for(flavor: str, *, B: int, N: int, kv_indices, last_page, write_indices):
    cu_q = torch.tensor([0, N], dtype=torch.int32, device="cuda") if B == 1 else None
    return _meta_cls(flavor)(
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        batch_size=B,
        num_query_tokens=N,
        cu_seqlens_q=cu_q,
        paged_kv_indptr=torch.tensor([0, N], dtype=torch.int32, device="cuda"),
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=last_page,
        write_indices=write_indices,
    )


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_flashinfer_paged_4_hook_lifecycle(flavor: str):
    """init_cuda_graph_state -> init_capture_metadata -> replay_metadata.

    Plan handle wrapper identity is stable across init_capture_metadata
    + replay_metadata calls (graph capture invariant).
    """
    pool, layer, (H, H_kv, D) = _make_pool_and_layer(flavor)
    backend = _factory_for(flavor, "flashinfer")(None)

    backend.init_cuda_graph_state(
        max_batch_size=1,
        max_num_tokens=8,
        max_paged_kv_indices=8,
        device=torch.device("cuda"),
        params_dtype=torch.float16,
        layer_proto=layer,
    )

    seed_meta = _meta_for(
        flavor,
        B=1,
        N=8,
        kv_indices=torch.arange(8, dtype=torch.int32, device="cuda"),
        last_page=torch.tensor([1], dtype=torch.int32, device="cuda"),
        write_indices=torch.arange(8, dtype=torch.int64, device="cuda"),
    )
    plan = backend.init_capture_metadata(seed_meta)
    wrapper_after_capture = plan.wrapper

    new_meta = _meta_for(
        flavor,
        B=1,
        N=4,
        kv_indices=torch.arange(4, dtype=torch.int32, device="cuda"),
        last_page=torch.tensor([1], dtype=torch.int32, device="cuda"),
        write_indices=torch.arange(4, dtype=torch.int64, device="cuda"),
    )
    backend.replay_metadata(plan, new_meta)
    assert plan.wrapper is wrapper_after_capture, (
        "Plan.wrapper identity must be stable across "
        "replay_metadata for graph capture to work."
    )


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_flashinfer_paged_matches_eager_single_sample(flavor: str):
    """flashinfer paged output ~= no-cache eager output for the same Q/K/V.

    For a single full-sample batch with contiguous slots, the paged read
    over slots ``0..N-1`` reduces to the same ``softmax(QK^T/sqrt(D)) V``
    as a ragged no-cache prefill with ``cu_seqlens_q = cu_seqlens_kv =
    [0, N]`` and ``causal=False``.
    """
    pool, layer, (H, H_kv, D) = _make_pool_and_layer(flavor)
    N = 4
    torch.manual_seed(0)
    q = torch.randn(N, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(N, H_kv, D, dtype=torch.float16, device="cuda")
    v = torch.randn(N, H_kv, D, dtype=torch.float16, device="cuda")
    write_indices = torch.arange(N, dtype=torch.int64, device="cuda")

    meta = _meta_for(
        flavor,
        B=1,
        N=N,
        kv_indices=torch.arange(N, dtype=torch.int32, device="cuda"),
        last_page=torch.tensor([1], dtype=torch.int32, device="cuda"),
        write_indices=write_indices,
    )

    fi = _factory_for(flavor, "flashinfer")(None)
    fi.init_cuda_graph_state(
        max_batch_size=1,
        max_num_tokens=N,
        max_paged_kv_indices=N,
        device=torch.device("cuda"),
        params_dtype=torch.float16,
        layer_proto=layer,
    )
    fi_plan = fi.init_forward_metadata(meta)
    ctx_fi = _ctx_cls(flavor)(
        backend=fi,
        plan=fi_plan,
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        kv_pool=pool,
        write_indices=write_indices,
    )
    out_fi = layer(q, k, v, ctx_fi)

    # Reference: the no-cache Attention layer on CPU fp32 (eager backend
    # is kept only in the no-cache stack). Same softmax over the same N
    # keys with causal=False.
    ref = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        causal=False,
        backend="eager",
    )
    cu = torch.tensor([0, N], dtype=torch.int32)
    out_e = ref(
        q.cpu().float(),
        k.cpu().float(),
        v.cpu().float(),
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
    )

    assert torch.allclose(out_fi.cpu().float(), out_e, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
@pytest.mark.parametrize("n_q,n_kv", [(4, 8), (8, 3)])
def test_flashinfer_paged_cross_attention_sq_ne_skv(flavor: str, n_q: int, n_kv: int):
    """Paged read with S_q != S_kv (cross-attention / extend).

    Q has ``n_q`` rows; K/V have ``n_kv`` rows written to ``n_kv`` slots.
    The query offsets (``cu_seqlens_q = [0, n_q]``) are decoupled from the
    KV page offsets (``paged_kv_indptr = [0, n_kv]``), so each of the
    ``n_q`` query rows attends over all ``n_kv`` keys. The reference is the
    no-cache eager Attention with a rectangular ``cu_seqlens_q != cu_seqlens_kv``
    and ``causal=False``.
    """
    pool, layer, (H, H_kv, D) = _make_pool_and_layer(flavor)
    torch.manual_seed(0)
    q = torch.randn(n_q, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(n_kv, H_kv, D, dtype=torch.float16, device="cuda")
    v = torch.randn(n_kv, H_kv, D, dtype=torch.float16, device="cuda")
    # K/V rows pair 1:1 with the slots they are scattered into.
    write_indices = torch.arange(n_kv, dtype=torch.int64, device="cuda")

    meta = _meta_cls(flavor)(
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        batch_size=1,
        num_query_tokens=n_q,
        cu_seqlens_q=torch.tensor([0, n_q], dtype=torch.int32, device="cuda"),
        paged_kv_indptr=torch.tensor([0, n_kv], dtype=torch.int32, device="cuda"),
        paged_kv_indices=torch.arange(n_kv, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
        write_indices=write_indices,
    )

    fi = _factory_for(flavor, "flashinfer")(None)
    fi.init_cuda_graph_state(
        max_batch_size=1,
        max_num_tokens=max(n_q, n_kv),
        max_paged_kv_indices=n_kv,
        device=torch.device("cuda"),
        params_dtype=torch.float16,
        layer_proto=layer,
    )
    fi_plan = fi.init_forward_metadata(meta)
    ctx_fi = _ctx_cls(flavor)(
        backend=fi,
        plan=fi_plan,
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        kv_pool=pool,
        write_indices=write_indices,
    )
    out_fi = layer(q, k, v, ctx_fi)
    assert out_fi.shape[0] == n_q

    ref = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        causal=False,
        backend="eager",
    )
    out_e = ref(
        q.cpu().float(),
        k.cpu().float(),
        v.cpu().float(),
        cu_seqlens_q=torch.tensor([0, n_q], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, n_kv], dtype=torch.int32),
    )

    assert torch.allclose(out_fi.cpu().float(), out_e, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_flashinfer_paged_rejects_kv_write_mismatch(flavor: str):
    """The layer enforces K/V row count == write_indices length, not == q.

    A K/V batch whose row count disagrees with ``ctx.write_indices`` is a
    caller bug (the scattered rows would not line up with their slots) and
    must raise — independently of how many query rows there are.
    """
    pool, layer, (H, H_kv, D) = _make_pool_and_layer(flavor)
    q = torch.randn(4, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(8, H_kv, D, dtype=torch.float16, device="cuda")
    v = torch.randn(8, H_kv, D, dtype=torch.float16, device="cuda")
    # write_indices has 6 slots but K/V have 8 rows -> mismatch.
    write_indices = torch.arange(6, dtype=torch.int64, device="cuda")

    fi = _factory_for(flavor, "flashinfer")(None)
    fi.init_cuda_graph_state(
        max_batch_size=1,
        max_num_tokens=8,
        max_paged_kv_indices=8,
        device=torch.device("cuda"),
        params_dtype=torch.float16,
        layer_proto=layer,
    )
    meta = _meta_cls(flavor)(
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        batch_size=1,
        num_query_tokens=4,
        cu_seqlens_q=torch.tensor([0, 4], dtype=torch.int32, device="cuda"),
        paged_kv_indptr=torch.tensor([0, 6], dtype=torch.int32, device="cuda"),
        paged_kv_indices=torch.arange(6, dtype=torch.int32, device="cuda"),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
        write_indices=write_indices,
    )
    ctx_fi = _ctx_cls(flavor)(
        backend=fi,
        plan=fi.init_forward_metadata(meta),
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        kv_pool=pool,
        write_indices=write_indices,
    )
    with pytest.raises(ValueError, match="write_indices row count"):
        layer(q, k, v, ctx_fi)
