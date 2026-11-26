"""SiTU operator parity, output reuse, and input validation."""

import pytest
import torch

from phyai_kernel import situ_and_mul


def reference(value):
    gate, hidden = value.chunk(2, dim=-1)
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2,), (7, 34), (2, 3, 16384), (0, 64)])
def test_situ_matches_unfused_operators(dtype, shape):
    torch.manual_seed(31)
    value = torch.randn(shape, dtype=dtype, device="cuda") * 12
    expected = reference(value)
    actual = situ_and_mul(value)
    tolerance = 2e-5 if dtype == torch.float32 else 0.0
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    reused = torch.empty_like(expected)
    assert situ_and_mul(value, out=reused).data_ptr() == reused.data_ptr()
    torch.testing.assert_close(reused, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_situ_extreme_values_and_strides(dtype):
    gate = torch.linspace(-120, 120, 4096, dtype=dtype, device="cuda")
    hidden = torch.linspace(-1000, 1000, 4096, dtype=dtype, device="cuda")
    packed = torch.cat((gate, hidden)).reshape(1, -1).expand(3, -1)
    value = torch.stack((packed, packed), dim=-1)[..., 0]
    assert not value.is_contiguous()
    actual = situ_and_mul(value)
    expected = reference(value)
    tolerance = 2e-5 if dtype == torch.float32 else 0.0
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_situ_validates_output_and_shape():
    value = torch.randn(2, 32, device="cuda")
    with pytest.raises(ValueError, match="even final dimension"):
        situ_and_mul(value[:, :31])
    with pytest.raises(ValueError, match="float16"):
        situ_and_mul(value.double())
    with pytest.raises(ValueError, match="output shape"):
        situ_and_mul(value, out=torch.empty_like(value))
    with pytest.raises(ValueError, match="contiguous"):
        situ_and_mul(value, out=torch.empty(16, 2, device="cuda").t())
    with pytest.raises(ValueError, match="share storage"):
        situ_and_mul(value, out=value.view(-1)[:32].view(2, 16))
    with pytest.raises(ValueError, match="CUDA"):
        situ_and_mul(value.cpu())
