"""Shared activation contracts, including input ownership and graph replay."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phyai.layers import SiLU


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(), (0,), (7,), (2, 3, 32)])
def test_silu_preserves_shape_dtype_and_input(device, dtype, shape):
    value = torch.randn(shape, device=device, dtype=dtype)
    if value.ndim == 3:
        value = value.transpose(0, 1)
        assert not value.is_contiguous()
    original = value.clone()
    layer = SiLU()

    actual = layer(value)

    assert actual.shape == value.shape
    assert actual.dtype == value.dtype
    assert actual.device == value.device
    torch.testing.assert_close(actual, F.silu(value), rtol=0, atol=0)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    assert not layer.state_dict()


@torch.inference_mode()
def test_silu_cuda_graph_replays_changed_input():
    layer = SiLU()
    value = torch.randn(2, 768, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(value)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(value)

    for fill in (0.5, -1.25):
        value.fill_(fill)
        graph.replay()
        torch.testing.assert_close(output, F.silu(value), rtol=0, atol=0)
