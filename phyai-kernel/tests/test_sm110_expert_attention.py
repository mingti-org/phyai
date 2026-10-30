"""Expert attention tails, page permutations and changing graph metadata."""

import pytest
import torch
import torch.nn.functional as F

from phyai_kernel.cuda.sm110.attention import ExpertAttention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0),
    reason="requires SM110",
)


@pytest.fixture(scope="module")
def kernel():
    return ExpertAttention(torch.device("cuda", torch.cuda.current_device()))


def reference(q, keys, values, pages):
    if pages.numel() == 0:
        return torch.zeros_like(q)
    k = keys[pages.long(), 0, 0].float()[None, None].expand(1, 8, -1, -1)
    v = values[pages.long(), 0, 0].float()[None, None].expand(1, 8, -1, -1)
    return (
        F.scaled_dot_product_attention(q.float().transpose(0, 1)[None], k, v)[0]
        .transpose(0, 1)
        .to(q.dtype)
    )


def check(actual, expected):
    assert torch.isfinite(actual).all()
    relative = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-12)
    assert relative < 0.0045, relative.item()
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.04)


@pytest.mark.parametrize("variant", ("c27",))
@pytest.mark.parametrize("length", (0, 1, 127, 128, 129, 578, 610, 674, 762))
@torch.no_grad()
def test_expert_attention_pages_and_tails(kernel, variant, length):
    torch.manual_seed(1021)
    # A fused QKV view has a non-contiguous token stride.
    q = torch.randn(50, 10, 256, device="cuda", dtype=torch.bfloat16)[:, :8]
    keys = torch.randn(800, 1, 1, 256, device="cuda", dtype=q.dtype)
    values = torch.randn_like(keys)
    pages = torch.randperm(800, device="cuda", dtype=torch.int32)[:length]
    indices = torch.cat((torch.zeros(3, device="cuda", dtype=torch.int32), pages))
    indptr = torch.tensor([3, length + 3], device="cuda", dtype=torch.int32)
    output = kernel(variant, q, keys, values, indptr, indices, 256**-0.5)
    check(output, reference(q, keys, values, pages))


@pytest.mark.parametrize("variant", ("c27",))
@torch.no_grad()
def test_expert_graph_changes_input_pages_and_length(kernel, variant):
    q = torch.randn(50, 8, 256, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(800, 1, 1, 256, device="cuda", dtype=q.dtype)
    values = torch.randn_like(keys)
    indices = torch.arange(800, dtype=torch.int32, device="cuda")
    indptr = torch.tensor([0, 610], dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernel(variant, q, keys, values, indptr, indices, 256**-0.5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = kernel(variant, q, keys, values, indptr, indices, 256**-0.5)
        for length in (762, 17, 0, 578):
            q.add_(0.03125)
            keys.mul_(0.99)
            values.add_(0.015625)
            indices.copy_(indices.roll(7))
            indptr.copy_(torch.tensor([0, length], device="cuda", dtype=torch.int32))
            output.fill_(float("nan"))
            graph.replay()
            check(output, reference(q, keys, values, indices[:length]))
    stream.synchronize()
