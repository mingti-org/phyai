"""GPU equivalence against the actual allocating FlashInfer catalog path."""

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch

from phyai.kernel.ops.rope import _flashinfer, _flashinfer_strided


@pytest.fixture(scope="module")
def implementations():
    if torch.cuda.get_device_capability() != (11, 0):
        pytest.skip("the specialized catalog row targets SM110")
    assert version("flashinfer-python") == "0.6.17"
    facts = SimpleNamespace(lookup=lambda path: False)
    return _flashinfer(facts, {}), _flashinfer_strided(facts, {})


def inputs(rows):
    generator = torch.Generator(device="cuda").manual_seed(812 + rows)
    fused = torch.randn(
        rows, 2560, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    q, k, _ = fused.split((2048, 256, 256), -1)
    angle = torch.randn(
        2048, 128, dtype=torch.float32, device="cuda", generator=generator
    )
    cache = torch.cat((angle.cos(), angle.sin()), -1).contiguous()
    positions = torch.arange(rows, dtype=torch.int32, device="cuda")
    return fused, q.view(rows, 8, 256), k.view(rows, 1, 256), cache, positions


@pytest.mark.parametrize("rows", [50, 784, 816, 880, 968])
def test_matches_current_flashinfer_path_without_changing_input(implementations, rows):
    native, specialized = implementations
    fused, q, k, cache, positions = inputs(rows)
    before = fused.clone()
    expected = native(positions, q, k, cache)
    actual = specialized(positions, q, k, cache)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert result.is_contiguous()
    torch.testing.assert_close(fused, before, rtol=0, atol=0)


def test_nondefault_stream_graph_and_consumer(implementations):
    native, specialized = implementations
    fused, q, k, cache, positions = inputs(50)
    weight = torch.randn(2048, 32, dtype=q.dtype, device=q.device)
    consumer = torch.empty(50, 32, dtype=q.dtype, device=q.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out_q, out_k = specialized(positions, q, k, cache)
            torch.mm(out_q.view(50, 2048), weight, out=consumer)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out_q, out_k = specialized(positions, q, k, cache)
            torch.mm(out_q.view(50, 2048), weight, out=consumer)
        pointers = (out_q.data_ptr(), out_k.data_ptr(), consumer.data_ptr())
        for delta in (0.125, -0.25):
            fused.add_(delta)
            positions.add_(1)
            graph.replay()
            expected_q, expected_k = native(positions, q, k, cache)
            torch.testing.assert_close(out_q, expected_q, rtol=0, atol=0)
            torch.testing.assert_close(out_k, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(consumer, expected_q.view(50, 2048) @ weight)
            assert pointers == (out_q.data_ptr(), out_k.data_ptr(), consumer.data_ptr())
    torch.cuda.current_stream().wait_stream(stream)


def test_misalignment_is_rejected_before_flashinfer(implementations):
    _, specialized = implementations
    _, _, k, cache, positions = inputs(1)
    q = torch.empty(2049, dtype=torch.bfloat16, device="cuda")[1:].view(1, 8, 256)
    assert q.is_contiguous()
    with pytest.raises(ValueError, match="aligned canonical"):
        specialized(positions, q, k, cache)
