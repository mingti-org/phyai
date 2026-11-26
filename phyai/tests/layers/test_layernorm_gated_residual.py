"""Gated residual precision boundaries, affine parameters, and kernel policy."""

import pytest
import torch
import torch.nn.functional as F

from phyai.kernel.bootstrap import initialize_kernel_system
from phyai.kernel.config import KernelConfig
from phyai.layers import LayerNorm


def reference(norm, x, branch, gate):
    updated = x + branch * gate.unsqueeze(1)
    normalized = F.layer_norm(
        updated.float(),
        (x.shape[-1],),
        norm.weight.float(),
        norm.bias.float() if norm.has_bias else None,
        norm.variance_epsilon,
    ).to(x.dtype)
    return normalized, updated


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("affine,bias", [(False, False), (True, False), (True, True)])
def test_dispatch_rounding_and_strides(device, dtype, affine, bias):
    torch.manual_seed(342)
    norm = LayerNorm(
        257, elementwise_affine=affine, bias=bias, dtype=dtype, device=device
    )
    if affine:
        norm.weight.data.normal_()
        if bias:
            norm.bias.data.normal_()
    x = torch.randn(2, 14, 514, device=device, dtype=dtype)[:, ::2, ::2]
    branch = torch.randn(7, 2, 257, device=device, dtype=dtype).transpose(0, 1)
    gate = torch.randn(2, 6, 257, device=device, dtype=dtype)[:, 4]
    originals = [value.clone() for value in (x, branch, gate)]
    actual, updated = norm.forward_gated_residual(x, branch, gate)
    expected, expected_update = reference(norm, x, branch, gate)
    tolerance = {torch.float16: 0.004, torch.bfloat16: 0.032, torch.float32: 3e-6}[
        dtype
    ]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(updated, expected_update, atol=0, rtol=0)
    for value, original in zip((x, branch, gate), originals):
        torch.testing.assert_close(value, original, atol=0, rtol=0)
    assert actual.is_contiguous() and updated.is_contiguous()
    assert actual.data_ptr() != x.data_ptr() and updated.data_ptr() != x.data_ptr()
    expected_keys = (["weight"] + (["bias"] if bias else [])) if affine else []
    assert list(norm.state_dict()) == expected_keys
    backend = "torch" if device == "cpu" else "phyai_kernel"
    assert {
        handle.kernel_id for handle in norm._gated_residual_call._memo.values()
    } == {f"{backend}.layernorm_gated_residual"}


@pytest.mark.parametrize("shape", [(1, 5120), (2, 1), (1, 1)])
def test_gate_broadcast_and_cuda_graph(shape):
    norm = LayerNorm(5120, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(2, 7, 5120, device="cuda", dtype=torch.bfloat16)
    branch = torch.randn_like(x)
    gate = torch.randn(shape, device=x.device, dtype=x.dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            norm.forward_gated_residual(x, branch, gate)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, updated = norm.forward_gated_residual(x, branch, gate)
    for _ in range(2):
        x.normal_()
        branch.normal_()
        gate.normal_()
        norm.weight.data.normal_()
        norm.bias.data.normal_()
        graph.replay()
        expected, expected_update = reference(norm, x, branch, gate)
        torch.testing.assert_close(actual, expected, atol=0.032, rtol=0.032)
        torch.testing.assert_close(updated, expected_update, atol=0, rtol=0)


def test_yaml_selects_reference_without_changing_plain_norm(tmp_path):
    policy = tmp_path / "kernels.yaml"
    policy.write_text(
        "schema: phyai.kernel/v1\nprofile: static\noverrides:\n"
        "  - match: {op: layernorm_gated_residual, role: test.residual_norm}\n"
        "    use: torch.layernorm_gated_residual\n"
    )
    initialize_kernel_system(KernelConfig(config_path=str(policy)), device="cuda")
    norm = LayerNorm(
        512, dtype=torch.bfloat16, device="cuda", kernel_role="test.residual_norm"
    )
    x = torch.randn(1, 11, 512, device="cuda", dtype=torch.bfloat16)
    branch, gate = torch.randn_like(x), torch.randn_like(x[:, 0])
    plain = norm(x)
    actual = norm.forward_gated_residual(x, branch, gate)
    for value, expected in zip(actual, reference(norm, x, branch, gate)):
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
    assert {h.kernel_id for h in norm._gated_residual_call._memo.values()} == {
        "torch.layernorm_gated_residual"
    }
    torch.testing.assert_close(norm(x), plain, atol=0, rtol=0)


def test_large_width_reference_and_empty_sequence():
    for hidden, sequence in [(8193, 3), (512, 0)]:
        norm = LayerNorm(hidden, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(2, sequence, hidden, device="cuda", dtype=torch.bfloat16)
        branch, gate = (
            torch.randn_like(x),
            torch.randn(1, 1, device=x.device, dtype=x.dtype),
        )
        for actual, expected in zip(
            norm.forward_gated_residual(x, branch, gate),
            reference(norm, x, branch, gate),
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_invalid_residual_contract():
    norm = LayerNorm(16, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(2, 3, 16, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(2, 16, device=x.device, dtype=x.dtype)
    with pytest.raises(ValueError, match="matching"):
        norm.forward_gated_residual(x, x[:, :1], gate)
    with pytest.raises(ValueError, match="broadcast"):
        norm.forward_gated_residual(x, x, gate.unsqueeze(1))
    with pytest.raises(ValueError, match="share dtype"):
        norm.forward_gated_residual(x, x.float(), gate)
