"""CUDA end-to-end: radix prefix reuse through ARAttention(flashinfer)
matches a full eager recompute. Gated on CUDA + flashinfer-python."""

from __future__ import annotations

import struct

import pytest
import torch

from phyai_ext.radix_cache import PrefixCache

from phyai.cache import KVCachePool
from phyai.layers.attention import (
    ARAttention,
    ARAttnCtx,
    AttnLayout,
    AttnMode,
)
from phyai.layers.attention.ar import FlashInferARBackend
from phyai.layers.attention.ar.radix import RadixAttentionPlanner, RadixSequence
from phyai.layers.attention.common import eager_attn, repeat_kv


def _has_flashinfer() -> bool:
    # flashinfer initialises CUDA eagerly on import; on a broken / misconfigured
    # CUDA setup that raises a RuntimeError (not just ImportError). Treat ANY
    # failure as "unavailable" so this test skips instead of crashing collection.
    try:
        import flashinfer.prefill  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _has_flashinfer()),
    reason="radix flashinfer e2e requires CUDA + flashinfer-python.",
)


def _atoms(tokens):
    return struct.pack(f"{len(tokens)}i", *tokens)


def _eager_full_causal(q, k, v, H, H_kv, scale):
    qi = q.transpose(0, 1).unsqueeze(0)
    ki = repeat_kv(k.transpose(0, 1).unsqueeze(0), H, H_kv)
    vi = repeat_kv(v.transpose(0, 1).unsqueeze(0), H, H_kv)
    o = eager_attn(
        qi, ki, vi, scale=scale, causal=True, sliding_window=None, logits_soft_cap=None
    )
    return o.squeeze(0).transpose(0, 1)


def test_radix_flashinfer_e2e_prefix_reuse_matches_eager():
    torch.manual_seed(0)
    H, H_kv, D = 4, 2, 64
    N, P = 6, 3
    dev = torch.device("cuda")
    q = torch.randn(N, H, D, dtype=torch.float16, device=dev)
    k = torch.randn(N, H_kv, D, dtype=torch.float16, device=dev)
    v = torch.randn(N, H_kv, D, dtype=torch.float16, device=dev)
    scale = 1.0 / D**0.5

    # Reference: fp32 eager full causal recompute, suffix rows.
    ref_suffix = _eager_full_causal(q.float(), k.float(), v.float(), H, H_kv, scale)[P:]

    cache = PrefixCache(atom_bytes=4, atoms_per_unit=1, device_total_units=64)
    pool = KVCachePool(
        num_layers=1,
        num_slots=64,
        num_kv_heads=H_kv,
        head_dim=D,
        dtype=torch.float16,
        device=dev,
    )
    planner = RadixAttentionPlanner(cache, pool)
    layer = ARAttention(
        num_heads=H,
        head_dim=D,
        layer_id=0,
        num_kv_heads=H_kv,
        causal=True,
        backend="flashinfer",
    )
    backend = FlashInferARBackend()
    backend.init_cuda_graph_state(
        max_batch_size=1,
        max_num_tokens=N,
        max_paged_kv_indices=N,
        device=dev,
        params_dtype=torch.float16,
        layer_proto=layer,
    )

    def run(meta, qsub, ksub, vsub):
        plan = backend.init_forward_metadata(meta)
        ctx = ARAttnCtx(
            backend=backend,
            plan=plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=pool,
            write_indices=meta.write_indices,
        )
        return layer(qsub, ksub, vsub, ctx)

    toks = [10, 11, 12, 20, 21, 22]
    a = RadixSequence(_atoms(toks[:P]))
    run(planner.plan([a]), q[:P], k[:P], v[:P])
    planner.commit([a])
    c = RadixSequence(_atoms(toks))
    meta_c = planner.plan([c])
    assert c.prefix_len == P
    out = run(meta_c, q[P:], k[P:], v[P:])

    assert torch.allclose(out.float().cpu(), ref_suffix.cpu(), atol=2e-2, rtol=2e-2)
