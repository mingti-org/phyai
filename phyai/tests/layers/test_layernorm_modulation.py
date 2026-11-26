"""Normalization modulation dispatch, rounding, and graph reuse contracts."""

import pytest
import torch
import torch.nn.functional as F

from phyai.kernel.bootstrap import initialize_kernel_system
from phyai.kernel.config import KernelConfig
from phyai.layers import LayerNormModulation


def reference(x, shift, scale, residual):
    updated = x if residual is None else x + residual
    normalized = F.layer_norm(updated.float(), (x.shape[-1],), eps=1e-6).to(x.dtype)
    return normalized * (1 + scale[:, None]) + shift[:, None], updated


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_residual", [False, True])
def test_layernorm_modulation_dispatch(device, dtype, with_residual):
    layer = LayerNormModulation()
    x = torch.randn(2, 7, 513, device=device, dtype=dtype)
    shift, scale = torch.randn(2, 6, 513, device=device, dtype=dtype).unbind(1)[:2]
    residual = torch.randn_like(x) if with_residual else None
    saved = x.clone()
    actual, updated = layer(x, shift, scale, residual)
    expected, expected_updated = reference(x, shift, scale, residual)

    tolerance = {torch.float16: 0.004, torch.bfloat16: 0.032, torch.float32: 2e-6}[
        dtype
    ]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(updated, expected_updated, atol=0, rtol=0)
    torch.testing.assert_close(x, saved, atol=0, rtol=0)
    assert (updated is x) == (not with_residual)
    assert not layer.state_dict()
    backend = "torch" if device == "cpu" else "phyai_kernel"
    assert {handle.kernel_id for handle in layer.call_site._memo.values()} == {
        f"{backend}.layernorm_modulate"
    }


def test_layernorm_modulation_yaml_overrides_backend_preference(tmp_path):
    policy = tmp_path / "kernels.yaml"
    policy.write_text(
        "schema: phyai.kernel/v1\n"
        "profile: static\n"
        "overrides:\n"
        "  - id: reference-norm-modulation\n"
        "    match: {op: layernorm_modulate, role: test.norm_modulation}\n"
        "    use: torch.layernorm_modulate\n"
    )
    initialize_kernel_system(KernelConfig(config_path=str(policy)), device="cuda")
    layer = LayerNormModulation(
        backend="phyai-kernel", kernel_role="test.norm_modulation"
    )
    x = torch.randn(2, 5, 128, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(1, 128, device=x.device, dtype=x.dtype)
    scale = torch.randn(2, 1, device=x.device, dtype=x.dtype)
    actual, _ = layer(x, shift, scale)
    expected, _ = reference(x, shift, scale, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert {handle.kernel_id for handle in layer.call_site._memo.values()} == {
        "torch.layernorm_modulate"
    }


def test_layernorm_modulation_graph_refreshes_inputs():
    layer = LayerNormModulation()
    x = torch.randn(2, 9, 1024, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    shift, scale = torch.randn(2, 6, 1024, device=x.device, dtype=x.dtype).unbind(1)[:2]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x, shift, scale, residual)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, updated = layer(x, shift, scale, residual)
    pointer = actual.data_ptr()
    for _ in range(2):
        x.normal_()
        residual.normal_()
        shift.normal_()
        scale.normal_()
        graph.replay()
        expected, expected_updated = reference(x, shift, scale, residual)
        torch.testing.assert_close(actual, expected, atol=0.032, rtol=0.032)
        torch.testing.assert_close(updated, expected_updated, atol=0, rtol=0)
        assert actual.data_ptr() == pointer


def test_layernorm_modulation_wide_input_uses_reference():
    layer = LayerNormModulation()
    x = torch.randn(1, 2, 8200, device="cuda", dtype=torch.bfloat16)
    condition = torch.zeros(1, 1, device=x.device, dtype=x.dtype)
    actual, updated = layer(x, condition, condition)
    expected, _ = reference(x, condition, condition, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert updated is x
    assert {handle.kernel_id for handle in layer.call_site._memo.values()} == {
        "torch.layernorm_modulate"
    }


def test_layernorm_modulation_rejects_invalid_contracts():
    for eps in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="eps"):
            LayerNormModulation(eps)
    layer = LayerNormModulation()
    x = torch.randn(2, 3, 8)
    condition = torch.randn(2, 8)
    with pytest.raises(ValueError, match="3-D"):
        layer(x[0], condition, condition)
    with pytest.raises(ValueError, match="broadcast"):
        layer(x, condition[:1, :4], condition)
    with pytest.raises(ValueError, match="dtype"):
        layer(x, condition.bfloat16(), condition)
    with pytest.raises(ValueError, match="residual"):
        layer(x, condition, condition, x[0])
