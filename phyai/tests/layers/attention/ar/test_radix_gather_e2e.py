"""CPU end-to-end: radix prefix reuse reassembles the exact logical KV.

The AR stack is flashinfer-only (GPU) after upstream removed the eager
backend, so full attention numerics are covered by the CUDA-gated
``test_radix_flashinfer.py``. This CPU test guards the part the planner
actually owns and that CI (CPU) can exercise without a kernel: after a
shared prefix is committed and a decoy pushes the reuse suffix off the
prefix's contiguous slot range, the gathered ``[prefix ++ suffix]`` KV
must equal a full recompute's KV — i.e. the planner's ``paged_kv_indices``
and ``write_indices`` round-trip through the pool correctly.
"""

from __future__ import annotations

import struct

import torch

from phyai_ext.radix_cache import PrefixCache

from phyai.cache import KVCachePool
from phyai.layers.attention.ar.radix import RadixAttentionPlanner, RadixSequence


def _atoms(tokens):
    return struct.pack(f"{len(tokens)}i", *tokens)


def test_radix_reuse_gathers_exact_prefix_plus_suffix():
    torch.manual_seed(0)
    H_kv, D = 2, 8
    N, P = 6, 3
    k = torch.randn(N, H_kv, D)
    v = torch.randn(N, H_kv, D)

    cache = PrefixCache(atom_bytes=4, atoms_per_unit=1, device_total_units=64)
    pool = KVCachePool(
        num_layers=1,
        num_slots=64,
        num_kv_heads=H_kv,
        head_dim=D,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    planner = RadixAttentionPlanner(cache, pool)

    toks = [10, 11, 12, 20, 21, 22]

    # Seed the shared prefix [10,11,12], writing K/V[0:P] at its slots.
    a = RadixSequence(_atoms(toks[:P]))
    meta_a = planner.plan([a])
    pool.write_kv(0, meta_a.write_indices, k[:P], v[:P])
    planner.commit([a])

    # Decoy: commit an unrelated sequence so its slots sit between the prefix's
    # range and the reuse suffix's range, forcing non-contiguous indices.
    d = RadixSequence(_atoms([55, 66]))
    meta_d = planner.plan([d])
    pool.write_kv(
        0, meta_d.write_indices, torch.randn(2, H_kv, D), torch.randn(2, H_kv, D)
    )
    planner.commit([d])

    # Reuse: query = suffix only; KV = cached prefix + freshly written suffix.
    c = RadixSequence(_atoms(toks))
    meta_c = planner.plan([c])
    assert c.prefix_len == P
    assert meta_c.paged_kv_indices.tolist() == [1, 2, 3, 6, 7, 8]  # non-contiguous
    pool.write_kv(0, meta_c.write_indices, k[P:], v[P:])

    gathered_k, gathered_v = pool.gather_kv(0, meta_c.paged_kv_indices)

    # The reuse gather must reproduce the whole sequence's KV bit-for-bit:
    # prefix rows come from the seed call's cache, suffix rows from this call.
    assert torch.equal(gathered_k, k)
    assert torch.equal(gathered_v, v)
