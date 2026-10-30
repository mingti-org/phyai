"""Canonical K/V vector writes, host slot admission and Graph consumers."""

import pytest
import torch

from phyai_kernel.cuda.jit.kv_vector_scatter import (
    prepare,
    run,
    slot_span_supported,
    span_supported,
    validate,
    validate_slots,
)


def test_actual_source_span():
    nbytes = 2 * (2304 + 49 * 2560 + 256)
    assert span_supported((50, 1, 256), (2560, 256, 1), 4096, 2304, nbytes)
    assert not span_supported((50, 1, 256), (2560, 256, 1), 4096, 2304, nbytes - 1)
    assert not span_supported((50, 1, 256), (2560, 256, 1), 4098, 2305, nbytes)
    assert not span_supported((50, 1, 256), (2561, 256, 1), 4096, 0, nbytes)
    rows, stride = 4096, 2**24
    assert span_supported(
        (rows, 1, 256), (stride, 256, 1), 4096, 0, 2 * ((rows - 1) * stride + 256)
    )


def test_actual_cache_span():
    shape, stride = (1019, 1, 1, 256), (256, 256, 256, 1)
    assert span_supported(shape, stride, 4096, 0, 1019 * 512, cache=True)
    assert not span_supported(shape, stride, 4096, 0, 1019 * 512 - 1, cache=True)
    assert not span_supported(
        (1020, 1, 1, 256), stride, 4096, 0, 1020 * 512, cache=True
    )


def test_slot_terminal_span():
    assert slot_span_supported((50,), (1,), 4104, 1, 51 * 8)
    assert not slot_span_supported((50,), (1,), 4104, 1, 51 * 8 - 1)
    assert not slot_span_supported((50,), (1,), 4105, 1, 51 * 8)
    assert not slot_span_supported((50,), (2,), 4096, 0, 100 * 8)
    assert not slot_span_supported((0,), (1,), 4096, 0, 8)


def test_host_slots_allow_padding_and_capacity_boundary():
    validate_slots(torch.tensor([0, 1, 1018, 0], dtype=torch.int64))


@pytest.mark.parametrize("values", [[-1], [1019], [1, 1], [1018, 1018]])
def test_invalid_host_slot_values(values):
    with pytest.raises(ValueError):
        validate_slots(torch.tensor(values, dtype=torch.int64))


def test_host_slot_metadata():
    for slots in (
        torch.ones(1, dtype=torch.int32),
        torch.ones(1, 1, dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
        torch.arange(8)[::2],
        torch.arange(4097),
    ):
        with pytest.raises(ValueError):
            validate_slots(slots)


def test_cpu_sources_rejected_without_launch():
    k = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
    kc, vc = [torch.zeros(1019, 1, 1, 256, dtype=torch.bfloat16) for _ in range(2)]
    with pytest.raises(ValueError, match="K/V vector view"):
        validate(k, k.clone(), kc, vc, torch.ones(1, dtype=torch.int64))


@pytest.fixture(scope="module")
def kernel():
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("the kernel targets SM110")
    return prepare()


def make_sources(rows, generator, *, strided):
    if strided:
        fused = torch.randn(
            rows, 2560, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        k = fused[:, 2048:2304].view(rows, 1, 256)
        v = fused[:, 2304:2560].view(rows, 1, 256)
        return k, v
    return tuple(
        torch.randn(
            rows, 1, 256, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        for _ in range(2)
    )


def reference(k, v, kc, vc, host_slots):
    expected_k, expected_v = kc.clone(), vc.clone()
    positive = host_slots > 0
    slots = host_slots[positive].to(k.device)
    source_rows = positive.nonzero().flatten().to(k.device)
    expected_k[slots, 0] = k[source_rows]
    expected_v[slots, 0] = v[source_rows]
    return expected_k, expected_v


@pytest.mark.parametrize("rows", [1, 2, 49, 50, 112, 1020])
@pytest.mark.parametrize("strided", [False, True])
def test_matches_valid_indexed_writes_and_preserves_padding(kernel, rows, strided):
    generator = torch.Generator(device="cuda").manual_seed(100 + rows)
    k, v = make_sources(rows, generator, strided=strided)
    caches = [
        torch.full((1019 * 256 + 16,), 42, device="cuda", dtype=k.dtype)
        for _ in range(2)
    ]
    kc, vc = [storage[8:-8].view(1019, 1, 1, 256) for storage in caches]
    host_slots = torch.zeros(rows, dtype=torch.int64)
    count = min(rows, 1018)
    host_slots[:count] = torch.arange(1, count + 1)
    if rows > 1:
        host_slots[0] = 0
    host_slots[count - 1] = 1018
    validate_slots(host_slots)
    slots = host_slots.to("cuda")
    expected_k, expected_v = reference(k, v, kc, vc, host_slots)
    snapshots = k.clone(), v.clone()
    run(k, v, kc, vc, slots, kernel)
    torch.testing.assert_close(kc, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(vc, expected_v, rtol=0, atol=0)
    for tensor, snapshot in zip((k, v), snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    for storage in caches:
        torch.testing.assert_close(
            storage[:8], torch.full((8,), 42, device="cuda", dtype=k.dtype)
        )
        torch.testing.assert_close(
            storage[-8:], torch.full((8,), 42, device="cuda", dtype=k.dtype)
        )


def test_cache_alias_rejected_before_launch(kernel):
    generator = torch.Generator(device="cuda").manual_seed(16)
    k, v = make_sources(50, generator, strided=True)
    kc, vc = [
        torch.zeros(1019, 1, 1, 256, device="cuda", dtype=k.dtype) for _ in range(2)
    ]
    slots = torch.arange(1, 51, device="cuda", dtype=torch.int64)
    with pytest.raises(ValueError, match="aliases"):
        validate(k, v, kc, kc, slots)
    with pytest.raises(ValueError, match="aliases"):
        validate(kc[:50, 0], v, kc, vc, slots)


def test_nondefault_stream_graph_replay_and_cache_consumer(kernel):
    generator = torch.Generator(device="cuda").manual_seed(48)
    k, v = make_sources(50, generator, strided=True)
    kc, vc = [
        torch.zeros(1019, 1, 1, 256, device="cuda", dtype=k.dtype) for _ in range(2)
    ]
    host_slots = torch.arange(1, 51, dtype=torch.int64)
    host_slots[:2] = 0
    validate_slots(host_slots)
    slots = host_slots.to("cuda")
    read_indices = host_slots[host_slots > 0].to("cuda")
    consumed_k, consumed_v = [
        torch.empty(read_indices.numel(), 1, 1, 256, device="cuda", dtype=k.dtype)
        for _ in range(2)
    ]
    addresses = tuple(tensor.data_ptr() for tensor in (k, v, kc, vc, slots))

    def chain():
        run(k, v, kc, vc, slots, kernel)
        torch.index_select(kc, 0, read_indices, out=consumed_k)
        torch.index_select(vc, 0, read_indices, out=consumed_v)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        chain()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        chain()
    for factor in (0.5, -0.25):
        k.mul_(factor)
        v.mul_(factor)
        expected_k, expected_v = reference(k, v, kc, vc, host_slots)
        graph.replay()
        torch.testing.assert_close(kc, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(vc, expected_v, rtol=0, atol=0)
        torch.testing.assert_close(consumed_k, expected_k[read_indices], rtol=0, atol=0)
        torch.testing.assert_close(consumed_v, expected_v[read_indices], rtol=0, atol=0)
        assert tuple(tensor.data_ptr() for tensor in (k, v, kc, vc, slots)) == addresses


def test_two_callers_keep_separate_caches(kernel):
    generator = torch.Generator(device="cuda").manual_seed(113)
    k, v = make_sources(50, generator, strided=True)
    first_k, first_v, second_k, second_v = [
        torch.zeros(1019, 1, 1, 256, device="cuda", dtype=k.dtype) for _ in range(4)
    ]
    slots = torch.arange(1, 51, device="cuda", dtype=torch.int64)
    run(k, v, first_k, first_v, slots, kernel)
    run(k, v, second_k, second_v, slots, kernel)
    second_k.zero_()
    second_v.zero_()
    torch.testing.assert_close(first_k[slots, 0], k, rtol=0, atol=0)
    torch.testing.assert_close(first_v[slots, 0], v, rtol=0, atol=0)
