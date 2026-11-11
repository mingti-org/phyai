"""Selectable paired normalization/RoPE and graph input refresh."""

import pytest
import torch

from phyai.kernel.bootstrap import initialize_kernel_system
from phyai.kernel.config import KernelConfig
from phyai.layers import QKRMSNormRotary


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [512, 8448])
def test_dispatch_and_parameter_ownership(device, dtype, width):
    layer = QKRMSNormRotary(128)
    q = torch.randn(2, 7, width, device=device, dtype=dtype)
    k = torch.randn_like(q)
    qw, kw = [torch.randn(width, device=device, dtype=dtype) for _ in range(2)]
    cos, sin = torch.randn(7, 128, device=device), torch.randn(7, 128, device=device)
    result = layer(q, k, qw, kw, cos, sin, head_start=1)
    assert result[0].shape == (2, 7, width // 128 - 1, 128)
    assert layer.state_dict() == {}
    backend = "phyai_kernel" if device == "cuda" and width <= 8192 else "torch"
    assert {h.kernel_id for h in layer.call_site._memo.values()} == {
        f"{backend}.qk_rmsnorm_rope"
    }
    ref = QKRMSNormRotary(128, backend="torch")(q, k, qw, kw, cos, sin, head_start=1)
    tolerance = {torch.float16: 0.005, torch.bfloat16: 0.04, torch.float32: 3e-6}[dtype]
    for actual, expected in zip(result, ref):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_yaml_override(tmp_path):
    config = tmp_path / "kernels.yaml"
    config.write_text(
        "schema: phyai.kernel/v1\nprofile: static\noverrides:\n"
        "  - id: explicit-reference\n    match: {op: qk_rmsnorm_rope, role: test.qk}\n"
        "    use: torch.qk_rmsnorm_rope\n"
    )
    initialize_kernel_system(KernelConfig(config_path=str(config)), device="cuda")
    layer = QKRMSNormRotary(32, kernel_role="test.qk")
    q = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, device=q.device, dtype=q.dtype)
    c = torch.ones(32, device=q.device)
    layer(q, q, w, w, c, c)
    assert {h.kernel_id for h in layer.call_site._memo.values()} == {
        "torch.qk_rmsnorm_rope"
    }


def test_graph_refreshes_inputs_weights_and_phases():
    layer = QKRMSNormRotary(128)
    storage = torch.randn(2, 11, 15360, device="cuda", dtype=torch.bfloat16)
    q, k = storage[:, :7, :5120], storage[:, :7, 5120:10240]
    weights = torch.randn(2, 5120, device=q.device, dtype=q.dtype)
    phases = torch.randn(2, 7, 128, device=q.device)

    def run():
        return layer(
            q,
            k,
            weights[0],
            weights[1],
            phases[0],
            phases[1],
            head_start=20,
            local_heads=20,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for _ in range(2):
        storage.normal_()
        weights.normal_()
        phases.normal_()
        graph.replay()
        for actual, expected in zip(output, run()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_invalid_layer_contracts():
    with pytest.raises(ValueError, match="positive and even"):
        QKRMSNormRotary(31)
    with pytest.raises(ValueError, match="finite and positive"):
        QKRMSNormRotary(32, eps=-1)
    layer = QKRMSNormRotary(32)
    q = torch.randn(2, 7, 128)
    w = torch.ones(128)
    c = torch.ones(32)
    with pytest.raises(ValueError, match="head slice"):
        layer(q, q, w, w, c, c, local_heads=5)
    with pytest.raises(ValueError, match="matching projection"):
        layer(q, q, w[::2], w, c, c)
    with pytest.raises(ValueError, match="matching rank"):
        layer(q, q[:1], w, w, c, c)


@pytest.mark.parametrize("num_warps", [4, 8, 16])
def test_yaml_reduction_layout(tmp_path, num_warps):
    from phyai_kernel import qk_rmsnorm_rope

    config = tmp_path / "warp_policy.yaml"
    config.write_text(
        "schema: phyai.kernel/v1\nprofile: static\noverrides:\n"
        "  - id: reduction-layout\n    match: {op: qk_rmsnorm_rope, role: test.qk}\n"
        f"    use: phyai_kernel.qk_rmsnorm_rope\n    params: {{num_warps: {num_warps}}}\n"
    )
    initialize_kernel_system(KernelConfig(config_path=str(config)), device="cuda")
    layer = QKRMSNormRotary(128, kernel_role="test.qk")
    q = torch.randn(2, 11, 5120, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    w = torch.randn(5120, device=q.device, dtype=q.dtype)
    c = torch.randn(11, 128, device=q.device)
    result = layer(q, k, w, w, c, c)
    expected = qk_rmsnorm_rope(q, k, w, w, c, c, 1e-6, 128, num_warps=num_warps)
    for actual, target in zip(result, expected):
        torch.testing.assert_close(actual, target, atol=0, rtol=0)
    assert {
        choice.params["num_warps"] for choice in layer.call_site._memo.values()
    } == {num_warps}
