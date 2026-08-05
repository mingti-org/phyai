"""CPU planner-logic tests for the radix → AR attention bridge."""

from __future__ import annotations

import struct

import pytest
import torch

from phyai_ext.radix_cache import CacheCapacityError, PrefixCache, Tier

from phyai.cache import KVCachePool
from phyai.layers.attention.ar.radix import RadixAttentionPlanner, RadixSequence


def _atoms(tokens):
    return struct.pack(f"{len(tokens)}i", *tokens)


def _build(num_slots: int = 64, total_units: int = 64):
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=1, device_total_units=total_units)
    pool = KVCachePool(
        num_layers=1,
        num_slots=num_slots,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return cache, pool, RadixAttentionPlanner(cache, pool)


def test_sequence_defaults():
    s = RadixSequence(atoms=_atoms([7]))
    assert s.prefix_len == 0
    assert s.suffix_len == 0
    assert s.total_len == 0
    assert not s.committed and not s.released


def test_construction_rejects_unit_size_mismatch():
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=2, device_total_units=64)
    pool = KVCachePool(
        num_layers=1,
        num_slots=64,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )  # page_size=1 != atoms_per_unit=2
    with pytest.raises(ValueError, match="atoms_per_unit"):
        RadixAttentionPlanner(cache, pool)


def test_construction_rejects_non_device_tier():
    # Even with the host tier enabled, the device-slot-only planner must reject
    # it: host unit ids are not device-pool slots.
    cache = PrefixCache(
        atom_bytes=4, atoms_per_unit=1, device_total_units=64, host_total_units=64
    )
    pool = KVCachePool(
        num_layers=1,
        num_slots=64,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="Tier.DEVICE"):
        RadixAttentionPlanner(cache, pool, tier=Tier.HOST)


def test_construction_rejects_multi_token_page():
    # Multi-token pages aren't supported end-to-end (write_kv / flashinfer are
    # page_size=1 only; the metadata counts radix units).
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=2, device_total_units=64)
    pool = KVCachePool(
        num_layers=1,
        num_slots=64,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
        page_size=2,
    )
    with pytest.raises(ValueError, match="page_size == 1"):
        RadixAttentionPlanner(cache, pool)


def test_construction_rejects_cache_larger_than_pool():
    # A cache whose device tier can hand out more unit ids than the pool has
    # slots would produce out-of-bounds prefix/suffix slot indices.
    cache = PrefixCache(atom_bytes=4, atoms_per_unit=1, device_total_units=128)
    pool = KVCachePool(
        num_layers=1,
        num_slots=64,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="num_slots"):
        RadixAttentionPlanner(cache, pool)


def test_plan_fresh_sequence_allocates_full_suffix():
    cache, pool, planner = _build()
    seq = RadixSequence(_atoms([10, 11, 12]))
    meta = planner.plan([seq])
    assert seq.prefix_len == 0
    assert seq.suffix_slots.tolist() == [1, 2, 3]
    assert seq.node_ref is None
    assert meta.batch_size == 1
    assert meta.num_query_tokens == 3
    assert meta.cu_seqlens_q.tolist() == [0, 3]
    assert meta.paged_kv_indptr.tolist() == [0, 3]
    assert meta.paged_kv_indices.tolist() == [1, 2, 3]
    assert meta.paged_kv_indices.dtype == torch.int32
    assert meta.write_indices.tolist() == [1, 2, 3]
    assert meta.write_indices.dtype == torch.int64
    assert meta.paged_kv_last_page_len.tolist() == [1]
    assert meta.position_ids.tolist() == [0, 1, 2]


def test_plan_rejects_unaligned_atoms():
    cache, pool, planner = _build()
    with pytest.raises(ValueError, match="page_bytes"):
        planner.plan([RadixSequence(b"abcde")])  # 5 bytes, page_bytes=4


def test_plan_reuse_shares_prefix_slots_non_contiguous():
    cache, pool, planner = _build()
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])  # tree: [10,11,12] -> [1,2,3]
    d = RadixSequence(_atoms([55, 66]))  # decoy consumes [4,5]
    planner.plan([d])
    planner.commit([d])
    c = RadixSequence(_atoms([10, 11, 12, 77, 88]))
    meta = planner.plan([c])
    assert c.prefix_len == 3
    assert c.prefix_slots.tolist() == [1, 2, 3]
    assert c.suffix_slots.tolist() == [6, 7]
    assert meta.paged_kv_indices.tolist() == [1, 2, 3, 6, 7]  # gap 3->6
    assert meta.cu_seqlens_q.tolist() == [0, 2]
    assert meta.paged_kv_indptr.tolist() == [0, 5]
    assert meta.write_indices.tolist() == [6, 7]
    assert meta.position_ids.tolist() == [3, 4]
    assert c.node_ref is not None


def test_plan_fully_cached_sequence_has_no_query_rows():
    cache, pool, planner = _build()
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])
    b = RadixSequence(_atoms([10, 11, 12]))  # identical -> fully cached
    meta = planner.plan([b])
    assert b.prefix_len == 3
    assert b.suffix_len == 0
    assert meta.num_query_tokens == 0
    assert meta.cu_seqlens_q.tolist() == [0, 0]
    assert meta.paged_kv_indices.tolist() == [1, 2, 3]
    assert meta.write_indices.numel() == 0


def test_commit_seeds_fresh_sequence():
    cache, pool, planner = _build()
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])
    assert a.committed and a.suffix_units is None
    mr = cache.match(_atoms([10, 11, 12]))
    assert int(mr.matched_atoms[int(Tier.DEVICE)]) == 3


def test_commit_skips_reused_sequence_no_growth_no_alloc():
    """commit() seeds only fully-new sequences. A reused (prefix>0) sequence is
    a no-op — no transient overlap allocation (which could evict/fail) and no
    tree growth; release() frees its written suffix."""
    cache, pool, planner = _build()
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])
    c = RadixSequence(_atoms([10, 11, 12, 77, 88]))
    planner.plan([c])  # prefix [1,2,3], suffix [4,5]
    avail_after_plan = cache.available(Tier.DEVICE)
    planner.commit([c])  # reuse -> no-op
    assert not c.committed
    assert cache.available(Tier.DEVICE) == avail_after_plan  # no transient alloc
    mr = cache.match(_atoms([10, 11, 12, 77, 88]))
    assert int(mr.matched_atoms[int(Tier.DEVICE)]) == 3  # tree not grown
    planner.release([c])
    assert c.suffix_units is None


def test_commit_seeds_fresh_overlapping_sequence_regardless_of_order():
    """Two fresh sequences where one is a prefix of the other: committing the
    shorter first must NOT cause the longer (planned fresh) one to be skipped.
    The seed decision uses plan-time prefix_len, not a commit-time re-match."""
    cache, pool, planner = _build()
    ab = RadixSequence(_atoms([1, 2]))
    abc = RadixSequence(_atoms([1, 2, 3]))
    planner.plan([ab, abc])  # cache empty at plan -> both prefix_len == 0
    assert ab.prefix_len == 0 and abc.prefix_len == 0
    planner.commit([ab, abc])  # commit ab first, then abc
    assert ab.committed and abc.committed
    assert int(cache.match(_atoms([1, 2, 3])).matched_atoms[int(Tier.DEVICE)]) == 3
    planner.release([ab, abc])


def test_commit_does_not_pin_seeded_node():
    """commit() must NOT pin the seeded node — pinning leaks units when a later
    shorter-prefix match splits it. A committed sequence is an evictable cache
    entry: a competing request under capacity pressure reuses its slots."""
    cache, pool, planner = _build(num_slots=8, total_units=4)  # ids 1..3 usable
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])
    assert a.committed and a.node_ref is None  # not pinned
    # Cache is full (3/3) but a is unpinned -> a competitor evicts it and fits.
    b = RadixSequence(_atoms([20, 21, 22]))
    meta_b = planner.plan([b])
    assert meta_b.write_indices.numel() == 3
    planner.release([a])
    planner.release([b])


def test_release_frees_lock_and_uncommitted_units():
    cache, pool, planner = _build()
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])
    avail_before = cache.available(Tier.DEVICE)
    c = RadixSequence(_atoms([10, 11, 12, 77, 88]))
    planner.plan([c])  # locks prefix, allocates 2 suffix
    assert c.node_ref is not None
    assert cache.available(Tier.DEVICE) == avail_before - 2
    planner.release([c])  # not committed -> free suffix + drop lock
    assert c.node_ref is None
    assert c.suffix_units is None
    assert c.released
    assert cache.available(Tier.DEVICE) == avail_before


def test_plan_raises_capacity_error_when_locked_and_full():
    cache, pool, planner = _build(num_slots=8, total_units=4)  # ids 1..3 usable
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])  # avail 0, tree=[10,11,12]->[1,2,3]
    c = RadixSequence(_atoms([10, 11, 12, 20]))  # reuse + 1 new, but no free slot
    with pytest.raises(CacheCapacityError):
        planner.plan([c])


def test_plan_rolls_back_locks_and_units_on_failure():
    """A mid-batch plan() failure must release every lock and free every unit
    acquired so far, leaving touched sequences re-plannable (atomic plan)."""
    cache, pool, planner = _build(num_slots=8, total_units=6)  # ids 1..5 usable
    a = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([a])
    planner.commit([a])  # seed [1,2,3], avail 2
    avail0 = cache.available(Tier.DEVICE)
    seq_c = RadixSequence(_atoms([10, 11, 12, 77]))  # reuse: locks [1,2,3], allocs 1
    seq_d = RadixSequence(_atoms([20, 21, 22, 23]))  # fresh: needs 4 > free -> fails
    with pytest.raises(CacheCapacityError):
        planner.plan([seq_c, seq_d])
    assert seq_c.node_ref is None  # lock rolled back
    assert seq_c.suffix_units is None  # units freed
    assert not seq_c.released  # still re-plannable
    assert cache.available(Tier.DEVICE) == avail0  # capacity restored


def test_bridge_reexported_from_ar_package():
    import phyai.layers.attention.ar as ar

    assert ar.RadixAttentionPlanner is RadixAttentionPlanner
    assert ar.RadixSequence is RadixSequence


def test_ar_import_does_not_pull_radix_extension():
    """phyai-ext is an optional extra; importing the base AR package must not
    import phyai_ext, so non-[ext] installs can still use the attention
    layers/backends. The radix bridge is re-exported lazily."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import phyai.layers.attention.ar\n"
        "assert 'phyai_ext' not in sys.modules, 'phyai_ext imported eagerly'\n"
        "assert 'phyai.layers.attention.ar.radix' not in sys.modules, 'radix eager'\n"
        "from phyai.layers.attention.ar import RadixAttentionPlanner\n"
        "assert 'phyai_ext' in sys.modules, 'lazy access should load phyai_ext'\n"
        "print('LAZY_OK')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "LAZY_OK" in res.stdout


def test_star_import_does_not_pull_radix_extension():
    """`from ...ar import *` must not eagerly resolve the lazy radix names —
    that would call __getattr__, import phyai_ext, and break a base install
    without the optional [ext] extra. So the radix names stay out of __all__."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from phyai.layers.attention.ar import *  # noqa: F401,F403\n"
        "assert 'phyai_ext' not in sys.modules, 'star-import pulled phyai_ext'\n"
        "assert 'phyai.layers.attention.ar.radix' not in sys.modules\n"
        "print('STAR_OK')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "STAR_OK" in res.stdout


def test_no_leak_when_batch_reuses_nested_prefixes():
    """A batch reusing a committed prefix to different depths must not leak
    units. Locking a node that a sibling's shorter match later splits would
    orphan the split-off child's duplicated ref_count; plan() pre-splits the
    tree before taking any lock to avoid that."""
    cache, pool, planner = _build(num_slots=16, total_units=16)
    avail_empty = cache.available(Tier.DEVICE)
    seed = RadixSequence(_atoms([1, 2, 3, 4, 5]))
    planner.plan([seed])
    planner.commit([seed])
    planner.release([seed])
    # Batch: x reuses the full [1..5]; y reuses only [1,2,3] (a shorter prefix).
    x = RadixSequence(_atoms([1, 2, 3, 4, 5, 6]))
    y = RadixSequence(_atoms([1, 2, 3, 9]))
    planner.plan([x, y])
    assert x.prefix_len == 5 and y.prefix_len == 3
    planner.release([x, y])
    # Drain fully: a true leak (ref_count > 0) can never be reclaimed.
    prev = -1
    while cache.available(Tier.DEVICE) != prev:
        prev = cache.available(Tier.DEVICE)
        try:
            cache.ensure_capacity(Tier.DEVICE, cache.total(Tier.DEVICE))
        except CacheCapacityError:
            break
    assert cache.available(Tier.DEVICE) == avail_empty


def test_failed_plan_does_not_evict_committed_entries():
    """An invalid sequence later in a batch makes plan() raise WITHOUT having
    evicted committed entries: validation runs before any ensure_capacity."""
    cache, pool, planner = _build(num_slots=4, total_units=4)  # ids 1..3 usable
    c = RadixSequence(_atoms([10, 11, 12]))
    planner.plan([c])
    planner.commit([c])
    planner.release([c])  # C is a committed, evictable cache entry (avail 0)
    assert int(cache.match(_atoms([10, 11, 12])).matched_atoms[int(Tier.DEVICE)]) == 3
    good = RadixSequence(_atoms([20, 21, 22]))  # valid; would need to evict C
    bad = RadixSequence(b"abcde")  # unaligned -> raises
    with pytest.raises(ValueError, match="page_bytes"):
        planner.plan([good, bad])
    # C must survive: the failed plan did no eviction / allocation.
    assert int(cache.match(_atoms([10, 11, 12])).matched_atoms[int(Tier.DEVICE)]) == 3
    assert not good.committed and good.suffix_units is None
