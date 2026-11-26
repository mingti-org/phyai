"""Opt-in RMSNorm cast/affine fusion and kernel policy behavior."""

import pytest
import torch

from phyai.kernel.bootstrap import initialize_kernel_system
from phyai.kernel.config import KernelConfig
from phyai.layers import RMSNorm


def reference(x, weight, eps):
    value = x.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return normalized.to(x.dtype) * weight


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_dispatch(device, dtype):
    norm = RMSNorm(
        512,
        dtype=dtype,
        device=device,
        cast_before_affine=True,
        fuse_cast_before_affine=True,
    )
    x = torch.randn(2, 13, 1536, device=device, dtype=dtype)[:, :9, 512:1024]
    norm.weight.data.normal_()
    actual = norm(x, preserve_3d_layout=True)
    expected = reference(x, norm.weight, norm.variance_epsilon)
    tolerance = {torch.float16: 0.004, torch.bfloat16: 0.032, torch.float32: 2e-6}[
        dtype
    ]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    backend = "torch" if device == "cpu" else "phyai_kernel"
    assert {handle.kernel_id for handle in norm._cast_affine_call._memo.values()} == {
        f"{backend}.rmsnorm_cast_affine"
    }
    assert list(norm.state_dict()) == ["weight"]
    assert not norm._plain_call._memo


def test_yaml_and_toggle_preserve_original_path(tmp_path):
    policy = tmp_path / "kernels.yaml"
    policy.write_text(
        "schema: phyai.kernel/v1\nprofile: static\noverrides:\n"
        "  - id: reference-rounded-norm\n"
        "    match: {op: rmsnorm_cast_affine, role: test.qk_norm}\n"
        "    use: torch.rmsnorm_cast_affine\n"
    )
    initialize_kernel_system(KernelConfig(config_path=str(policy)), device="cuda")
    norm = RMSNorm(
        128,
        dtype=torch.bfloat16,
        device="cuda",
        cast_before_affine=True,
        kernel_role="test.qk_norm",
    )
    x = torch.randn(2, 7, 128, dtype=norm.weight.dtype, device=norm.weight.device)
    baseline = norm(x)
    norm.fuse_cast_before_affine = True
    fused = norm(x)
    torch.testing.assert_close(
        fused, reference(x, norm.weight, norm.variance_epsilon), atol=0, rtol=0
    )
    assert {handle.kernel_id for handle in norm._cast_affine_call._memo.values()} == {
        "torch.rmsnorm_cast_affine"
    }
    norm.fuse_cast_before_affine = False
    torch.testing.assert_close(norm(x), baseline, atol=0, rtol=0)


def test_fused_contract_and_general_rank_fallback():
    with pytest.raises(ValueError, match="requires cast_before_affine"):
        RMSNorm(128, fuse_cast_before_affine=True)
    norm = RMSNorm(
        128,
        dtype=torch.bfloat16,
        device="cuda",
        cast_before_affine=True,
        fuse_cast_before_affine=True,
    )
    x = torch.randn(2, 3, 4, 128, dtype=norm.weight.dtype, device=norm.weight.device)
    torch.testing.assert_close(
        norm(x),
        reference(x, norm.weight, norm.variance_epsilon),
        atol=0.032,
        rtol=0.032,
    )
    with pytest.raises(ValueError, match="no residual"):
        norm(x, torch.zeros_like(x))
    with pytest.raises(ValueError, match="matching activation"):
        norm(x.float())


def test_graph_replay_refreshes_strided_inputs():
    norm = RMSNorm(
        5120,
        dtype=torch.bfloat16,
        device="cuda",
        cast_before_affine=True,
        fuse_cast_before_affine=True,
    )
    storage = torch.randn(
        2, 13, 15360, dtype=norm.weight.dtype, device=norm.weight.device
    )
    x = storage[:, :7, 5120:10240]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            norm(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = norm(x)
    for _ in range(2):
        storage.normal_()
        norm.weight.data.normal_()
        graph.replay()
        torch.testing.assert_close(
            result,
            reference(x, norm.weight, norm.variance_epsilon),
            atol=0.032,
            rtol=0.032,
        )
