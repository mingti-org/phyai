import pytest
import torch

from phyai.layers.layer_norm import RMSNorm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("with_residual", [False, True])
def test_rmsnorm_rounds_before_affine(dtype, with_residual):
    torch.manual_seed(72)
    norm = RMSNorm(128, eps=1e-6, dtype=dtype, device="cuda", cast_before_affine=True)
    norm.weight.copy_(torch.randn_like(norm.weight))
    value = torch.randn(2, 11, 128, dtype=dtype, device="cuda")
    residual = torch.randn_like(value) if with_residual else None
    combined = value.float() if residual is None else value.float() + residual.float()
    expected = (
        combined.float()
        * torch.rsqrt(combined.float().square().mean(-1, keepdim=True) + 1e-6)
    ).to(dtype) * norm.weight
    actual = norm(value, residual)
    if with_residual:
        actual, residual_output = actual
        torch.testing.assert_close(residual_output, combined.to(dtype), atol=0, rtol=0)
        assert actual.data_ptr() == value.data_ptr()
        assert residual_output.data_ptr() == residual.data_ptr()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert set(norm.state_dict()) == {"weight"}
