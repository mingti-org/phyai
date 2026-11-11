"""Rounded RMSNorm affine fusion for strided projection layouts."""

import pytest
import torch

from phyai_kernel import rmsnorm_cast_affine, rmsnorm_hf


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden", [127, 512, 1024, 5120, 8192, 8200])
@pytest.mark.parametrize("rank", [2, 3])
def test_strided_rows_match_existing_hf_kernel(dtype, hidden, rank):
    torch.manual_seed(57)
    storage = torch.randn(4, 11, hidden * 3, dtype=dtype, device="cuda")
    x = storage[1:3, :7, hidden : 2 * hidden]
    if rank == 2:
        x = x[0]
    weight = torch.randn(hidden, dtype=dtype, device=x.device)
    original = storage.clone()
    actual = rmsnorm_cast_affine(x, weight)
    expected = rmsnorm_hf(x.contiguous(), weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(storage, original, atol=0, rtol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rounding_matches_separate_torch_operations(dtype):
    torch.manual_seed(91)
    x = torch.randn(2, 31, 5120, dtype=dtype, device="cuda") * 5
    weight = torch.randn(5120, dtype=dtype, device=x.device)
    promoted = x.float()
    normalized = promoted * torch.rsqrt(promoted.square().mean(-1, keepdim=True) + 1e-6)
    expected = normalized.to(dtype) * weight
    actual = rmsnorm_cast_affine(x, weight)
    tolerance = {torch.float16: 0.004, torch.bfloat16: 0.032, torch.float32: 2e-6}[
        dtype
    ]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    if dtype != torch.float32:
        unrounded = (normalized * weight.float()).to(dtype)
        assert not torch.equal(actual, unrounded)


def test_empty_and_broadcast_batches():
    weight = torch.randn(127, device="cuda", dtype=torch.bfloat16)
    empty = torch.empty(2, 0, 127, device="cuda", dtype=weight.dtype)
    assert rmsnorm_cast_affine(empty, weight).shape == empty.shape
    x = torch.randn(1, 3, 127, device="cuda", dtype=weight.dtype).expand(2, -1, -1)
    torch.testing.assert_close(
        rmsnorm_cast_affine(x, weight),
        rmsnorm_hf(x.contiguous(), weight),
        atol=0,
        rtol=0,
    )


def test_invalid_contracts():
    x = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device=x.device, dtype=x.dtype)
    with pytest.raises(ValueError, match="unit last stride"):
        rmsnorm_cast_affine(x[..., ::2], weight[:64])
    with pytest.raises(ValueError, match="dtype/device"):
        rmsnorm_cast_affine(x, weight.float())
    with pytest.raises(ValueError, match="unit last stride"):
        rmsnorm_cast_affine(x[None], weight)
