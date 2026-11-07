"""Numerics, stream ownership, and changed-input Graph replay on SM110."""

from types import SimpleNamespace

import pytest
import torch

from phyai_kernel.cuda.sm110.gemm import Gemm, configurations

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0),
    reason="requires SM110",
)


def reference(x, weight, bias):
    from flashinfer.gemm import mm_bf16

    return mm_bf16(x, weight.t(), bias=bias, out_dtype=torch.bfloat16)


def check(actual, expected):
    assert torch.isfinite(actual).all()
    relative = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative < 0.0045, relative.item()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize(
    "config", configurations(), ids=lambda row: f"{row['target']}-{row['m']}"
)
@torch.no_grad()
def test_gemm_stream_graph_and_reload(config):
    torch.manual_seed(1021)
    m, n, k = (config[name] for name in ("m", "n", "k"))
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * k**-0.5
    bias = torch.randn(n, dtype=x.dtype, device=x.device) if config["bias"] else None
    layer = SimpleNamespace(weight=weight)
    before_x, before_weight = x.clone(), weight.clone()
    kernel = Gemm(config["target"], m)
    check(kernel(layer, x, bias), reference(x, weight, bias))
    torch.testing.assert_close(x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(weight, before_weight, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            kernel(layer, x, bias)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = kernel(layer, x, bias)
        x.add_(0.125)
        output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    check(output, reference(x, weight, bias))
    # Both a new Parameter and an in-place update invalidate packed weights.
    layer.weight = weight.clone().mul_(0.75)
    check(kernel(layer, x, bias), reference(x, layer.weight, bias))
    layer.weight.mul_(0.5)
    check(kernel(layer, x, bias), reference(x, layer.weight, bias))
