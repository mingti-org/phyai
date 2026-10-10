"""Shared modulation dispatch and parameter-free layer contracts."""

import pytest
import torch

from phyai.layers import AffineModulation


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_modulation_selects_kernel_and_preserves_rounding(device, dtype):
    layer = AffineModulation()
    x = torch.randn(2, 15, 768, device=device, dtype=dtype)
    conditions = torch.randn(2, 9 * 768, device=device, dtype=dtype)
    shift, scale = conditions.chunk(9, dim=-1)[:2]
    assert not shift.is_contiguous() and not scale.is_contiguous()

    actual = layer(x, shift, scale)

    expected = x * (1 + scale[:, None]) + shift[:, None]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected_kernel = "torch.modulate" if device == "cpu" else "phyai_kernel.modulate"
    assert {entry.kernel_id for entry in layer.call_site._memo.values()} == {
        expected_kernel
    }
    assert not layer.state_dict()


def test_modulation_torch_backend_remains_selectable_on_cuda():
    layer = AffineModulation(backend="torch")
    x = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(1, 128, device=x.device, dtype=x.dtype)
    scale = torch.randn(2, 1, device=x.device, dtype=x.dtype)
    torch.testing.assert_close(
        layer(x, shift, scale),
        x * (1 + scale[:, None]) + shift[:, None],
        atol=0,
        rtol=0,
    )
    assert {entry.kernel_id for entry in layer.call_site._memo.values()} == {
        "torch.modulate"
    }


def test_modulation_validates_backend_and_broadcast_contract():
    with pytest.raises(ValueError, match="unknown backend"):
        AffineModulation(backend="missing")
    layer = AffineModulation()
    x = torch.randn(2, 3, 8)
    condition = torch.randn(2, 8)
    with pytest.raises(ValueError, match="3-D"):
        layer(x[0], condition, condition)
    with pytest.raises(ValueError, match="broadcast"):
        layer(x, torch.randn(3, 8), condition)
    with pytest.raises(ValueError, match="dtype"):
        layer(x, condition.bfloat16(), condition)
