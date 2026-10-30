"""Gated residual/AdaRMS storage, rounding, streams and captured consumers."""

import pytest
import torch

from phyai_kernel import adarmsnorm
from phyai_kernel.cuda.jit.pending_gate_adarms_broadcast import (
    prepare,
    row_span_supported,
    run,
    validate,
)


def test_actual_view_stride_and_terminal_span():
    nbytes = 2 * (1 + 2 * 1025 + 1024)
    assert row_span_supported((3, 1024), (1025, 1), 4098, nbytes, 1, 1024)
    assert not row_span_supported((3, 1024), (1025, 1), 4098, nbytes - 1, 1, 1024)
    assert row_span_supported((1, 3072), (3072, 1), 4096, 6144, 0, 3072)


@pytest.mark.parametrize(
    "shape,stride,pointer,offset",
    [
        ((0, 1024), (1024, 1), 4096, 0),
        ((4097, 1024), (1024, 1), 4096, 0),
        ((3, 1023), (1024, 1), 4096, 0),
        ((3, 1024), (1025, 2), 4096, 0),
        ((3, 1024), (1023, 1), 4096, 0),
        ((3, 1024), (1024, 1), 4097, 0),
        ((3, 1024), (1024, 1), 0, 0),
        ((3, 1024), (1024, 1), 4096, -1),
    ],
)
def test_invalid_metadata(shape, stride, pointer, offset):
    assert not row_span_supported(shape, stride, pointer, 2**40, offset, 1024)


def test_cpu_tensors_rejected_without_launch():
    a, b, g, h, y = [torch.zeros(1, 1024, dtype=torch.bfloat16) for _ in range(5)]
    m = torch.zeros(1, 3072, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA BF16"):
        validate(a, b, g, m, h, y)


@pytest.mark.parametrize("epsilon", [0.0, -1e-6, float("nan"), float("inf")])
def test_invalid_epsilon_rejected_without_launch(epsilon):
    def no_launch(*args):
        pytest.fail("invalid epsilon reached the kernel")

    with pytest.raises(ValueError, match="epsilon"):
        run(None, None, None, None, None, None, epsilon, no_launch)


@pytest.fixture(scope="module")
def kernel():
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("the kernel targets SM110")
    return prepare()


def make_tensor(rows, width, generator, *, strided=False):
    if strided:
        storage = torch.empty(
            rows * (width + 1) + 1, device="cuda", dtype=torch.bfloat16
        )
        tensor = storage.as_strided((rows, width), (width + 1, 1), 1)
    else:
        tensor = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
    return tensor.normal_(std=0.2, generator=generator)


def reference(a, b, gate, modulation):
    expanded_gate = gate.repeat_interleave(a.shape[0] // gate.shape[0], dim=0)
    residual = torch.addcmul(a, b, expanded_gate)
    normalized, _ = adarmsnorm(residual, modulation, 1e-6)
    return residual, normalized


@pytest.mark.parametrize(
    "rows,gate_rows,mod_rows",
    [
        (1, 1, 1),
        (2, 1, 1),
        (50, 1, 1),
        (48, 2, 2),
        (50, 50, 50),
        (48, 1, 2),
        (48, 2, 1),
    ],
)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("output_padding", [1, 8])
def test_matches_actual_triton_chain(
    kernel, rows, gate_rows, mod_rows, strided, output_padding
):
    generator = torch.Generator(device="cuda").manual_seed(17 + rows)
    a = make_tensor(rows, 1024, generator, strided=strided)
    b = make_tensor(rows, 1024, generator, strided=strided)
    gate = make_tensor(gate_rows, 1024, generator, strided=strided)
    modulation = make_tensor(mod_rows, 3072, generator, strided=strided)
    snapshots = [tensor.clone() for tensor in (a, b, gate, modulation)]
    outputs = [
        torch.full(
            (rows * 1024 + 2 * output_padding,), 42, device="cuda", dtype=a.dtype
        )
        for _ in range(2)
    ]
    h, y = [
        storage[output_padding:-output_padding].view(rows, 1024) for storage in outputs
    ]
    expected_h, expected_y = reference(a, b, gate, modulation)
    result_h, result_y = run(a, b, gate, modulation, h, y, 1e-6, kernel)
    assert result_h is h and result_y is y
    torch.testing.assert_close(h, expected_h, rtol=0, atol=0)
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
    for tensor, snapshot in zip((a, b, gate, modulation), snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    for storage in outputs:
        sentinel = torch.full((output_padding,), 42, device="cuda", dtype=a.dtype)
        torch.testing.assert_close(storage[:output_padding], sentinel)
        torch.testing.assert_close(storage[-output_padding:], sentinel)


def test_gate_modulation_alias_and_output_rejection(kernel):
    generator = torch.Generator(device="cuda").manual_seed(113)
    a, b = [make_tensor(50, 1024, generator) for _ in range(2)]
    modulation = make_tensor(1, 3072, generator)
    gate = modulation[:, 2048:]
    h, y = torch.empty_like(a), torch.empty_like(a)
    expected_h, expected_y = reference(a, b, gate, modulation)
    run(a, b, gate, modulation, h, y, 1e-6, kernel)
    torch.testing.assert_close(h, expected_h, rtol=0, atol=0)
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
    for output_h, output_y in ((a, y), (h, h), (h, b)):
        with pytest.raises(ValueError, match="alias"):
            validate(a, b, gate, modulation, output_h, output_y)


def test_nondefault_stream_graph_replay_and_consumer(kernel):
    generator = torch.Generator(device="cuda").manual_seed(49)
    a, b = [make_tensor(50, 1024, generator) for _ in range(2)]
    modulation = make_tensor(1, 3072, generator)
    gate = modulation[:, 2048:]
    h, y = torch.empty_like(a), torch.empty_like(a)
    weight = make_tensor(1024, 64, generator)
    consumer = torch.empty(50, 64, device="cuda", dtype=a.dtype)
    addresses = tuple(
        tensor.data_ptr() for tensor in (a, b, gate, modulation, h, y, consumer)
    )

    def chain():
        run(a, b, gate, modulation, h, y, 1e-6, kernel)
        torch.mm(y, weight, out=consumer)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        chain()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        chain()
    for factor in (0.5, -0.25):
        a.mul_(factor)
        modulation.mul_(factor)
        expected_h, expected_y = reference(a, b, gate, modulation)
        expected_consumer = expected_y @ weight
        graph.replay()
        torch.testing.assert_close(h, expected_h, rtol=0, atol=0)
        torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
        torch.testing.assert_close(consumer, expected_consumer, rtol=0, atol=0)
        assert (
            tuple(
                tensor.data_ptr() for tensor in (a, b, gate, modulation, h, y, consumer)
            )
            == addresses
        )


def test_two_callers_keep_separate_workspace(kernel):
    generator = torch.Generator(device="cuda").manual_seed(200)
    a, b, gate = [make_tensor(50 if i < 2 else 1, 1024, generator) for i in range(3)]
    modulation = make_tensor(1, 3072, generator)
    first_h, first_y = torch.empty_like(a), torch.empty_like(a)
    second_h, second_y = torch.empty_like(a), torch.empty_like(a)
    run(a, b, gate, modulation, first_h, first_y, 1e-6, kernel)
    saved = first_y.clone()
    run(a, b, gate, modulation, second_h, second_y, 1e-6, kernel)
    second_y.zero_()
    torch.testing.assert_close(first_y, saved, rtol=0, atol=0)
