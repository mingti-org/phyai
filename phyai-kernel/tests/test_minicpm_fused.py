"""Numerical tests for the MiniCPM-oriented fused Triton kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import phyai_kernel


if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for phyai-kernel Triton tests", allow_module_level=True
    )


def _rmsnorm_silu_mul_reference(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = x.dtype
    normalized = x.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    normalized = weight * normalized.to(dtype)
    return (normalized * F.silu(gate.float())).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(17, 128), (2, 7, 128)])
def test_rmsnorm_silu_mul_matches_reference(dtype, shape):
    torch.manual_seed(123)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    gate = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(shape[-1], dtype=torch.float32, device="cuda")
    expected = _rmsnorm_silu_mul_reference(x, gate, weight, 1e-6)
    actual = phyai_kernel.rmsnorm_silu_mul(x, gate, weight)
    tolerance = (
        1e-5 if dtype == torch.float32 else (2e-2 if dtype == torch.bfloat16 else 2e-3)
    )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_rmsnorm_silu_mul_out_argument():
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn_like(x)
    weight = torch.randn(128, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    result = phyai_kernel.rmsnorm_silu_mul(x, gate, weight, out=out)
    assert result.data_ptr() == out.data_ptr()
    expected = _rmsnorm_silu_mul_reference(x, gate, weight, 1e-6)
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kernel_size", [1, 4, 8])
def test_causal_conv1d_silu_split_qkv_matches_reference(dtype, kernel_size):
    torch.manual_seed(456)
    batch_size, seq_len = 2, 17
    split_sizes = (96, 64, 128)
    channels = sum(split_sizes)
    storage = torch.randn(
        batch_size,
        seq_len,
        channels + 16,
        dtype=dtype,
        device="cuda",
    )
    x = storage[..., :channels]
    assert not x.is_contiguous()
    weight = torch.randn(channels, 1, kernel_size, dtype=dtype, device="cuda")

    convolved = F.conv1d(
        x.transpose(1, 2),
        weight,
        padding=kernel_size - 1,
        groups=channels,
    )[:, :, :seq_len]
    expected = F.silu(convolved).transpose(1, 2).split(split_sizes, dim=-1)
    actual = phyai_kernel.causal_conv1d_silu_split_qkv(x, weight, split_sizes)

    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    for result, reference, size in zip(actual, expected, split_sizes):
        assert result.shape == (batch_size, seq_len, size)
        assert result.is_contiguous()
        torch.testing.assert_close(
            result,
            reference,
            atol=tolerance,
            rtol=tolerance,
        )


def test_fused_kernels_reject_cpu_inputs():
    x = torch.randn(2, 128)
    with pytest.raises(RuntimeError, match="must live on CUDA"):
        phyai_kernel.rmsnorm_silu_mul(x, x, torch.ones(128))
    with pytest.raises(RuntimeError, match="must live on CUDA"):
        phyai_kernel.causal_conv1d_silu_split_qkv(
            x.view(1, 2, 128),
            torch.randn(128, 1, 4),
            (32, 32, 64),
        )
