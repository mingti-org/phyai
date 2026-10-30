"""Fusion arithmetic, immutable inputs, and changed-input Graph replay."""

from types import SimpleNamespace

import pytest
import torch

from phyai_kernel.cuda.sm110.fusion import DenseMLPFusion
from phyai_kernel.cuda.sm110.gemm import Gemm
from phyai_kernel.cuda.sm110.kv_scatter import scatter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0),
    reason="requires SM110",
)


class Facts:
    def __init__(self, **values):
        self.values = values

    def lookup(self, key):
        return self.values[key]


def check(actual, expected):
    assert torch.isfinite(actual).all()
    relative = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative < 0.0045, relative.item()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("rows", (1, 50, 528, 712))
@torch.no_grad()
def test_kv_scatter_strides_negative_indices_and_graph(rows):
    slots = rows + 7
    k = torch.randn(rows, 3, 256, device="cuda", dtype=torch.bfloat16)[:, :1]
    v = torch.randn_like(k).transpose(0, 1).contiguous().transpose(0, 1)
    indices = torch.arange(-rows, 0, device="cuda", dtype=torch.int64)
    kc = torch.zeros(slots, 1, 1, 256, device="cuda", dtype=k.dtype)
    vc = torch.zeros_like(kc)
    expected_k, expected_v = kc.clone(), vc.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        scatter(indices, k, v, kc, vc)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            scatter(indices, k, v, kc, vc)
        k.add_(0.25)
        v.mul_(0.5)
        kc.fill_(float("nan"))
        vc.fill_(float("nan"))
        graph.replay()
        expected_k[indices, 0] = k
        expected_v[indices, 0] = v
    stream.synchronize()
    torch.testing.assert_close(kc[indices], expected_k[indices], atol=0, rtol=0)
    torch.testing.assert_close(vc[indices], expected_v[indices], atol=0, rtol=0)


@pytest.mark.parametrize(
    "gated,rows", ((False, 512), (True, 528), (True, 560), (True, 624), (True, 712))
)
@torch.no_grad()
def test_projection_activation_direct_parent_and_graph(gated, rows):
    from flashinfer.activation import gelu_and_mul

    torch.manual_seed(1021)
    hidden, intermediate = (2048, 16384) if gated else (1152, 4304)
    width = intermediate * (2 if gated else 1)
    weight = (
        torch.randn(width, hidden, device="cuda", dtype=torch.bfloat16) * hidden**-0.5
    )
    bias = None if gated else torch.randn(width, device="cuda", dtype=weight.dtype)
    layer = SimpleNamespace(weight=weight, bias=bias, weight_post_load_hooks={})
    mlp = SimpleNamespace(gated=gated, intermediate_size=intermediate)
    if gated:
        mlp.gate_up_proj, mlp.down_proj = layer, lambda x: (x, None)
    else:
        mlp.fc1, mlp.fc2 = layer, lambda x: (x, None)
    operation = DenseMLPFusion(Facts(**{"shape.tokens": rows, "attrs.gated": gated}))
    parent = Gemm("pi05_G14" if gated else "pi05_G10", rows)
    x = torch.randn(rows, hidden, device="cuda", dtype=weight.dtype)

    def reference():
        value = parent(layer, x, bias)
        return (
            gelu_and_mul(value)
            if gated
            else torch.nn.functional.gelu(value, approximate="none")
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        operation(mlp, x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = operation(mlp, x)
        x.add_(0.0625)
        output.fill_(float("nan"))
        graph.replay()
        expected = reference()
    stream.synchronize()
    check(output, expected)
    layer.weight.mul_(0.75)
    for hook in layer.weight_post_load_hooks.values():
        hook(layer)
    graph.replay()
    check(output, reference())
    layer.weight = weight.clone().mul_(0.5)
    check(operation(mlp, x), reference())
    layer.weight.add_(0.001)
    check(operation(mlp, x), reference())
    layer.weight.data = layer.weight.data.clone()
    with pytest.raises(RuntimeError, match="Rebuild captured"):
        for hook in layer.weight_post_load_hooks.values():
            hook(layer)
