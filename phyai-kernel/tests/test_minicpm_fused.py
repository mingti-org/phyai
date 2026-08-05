"""Numerical tests for the MiniCPM-oriented fused Triton kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import phyai_kernel


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for phyai-kernel Triton tests",
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


@CUDA_REQUIRED
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


@CUDA_REQUIRED
def test_rmsnorm_silu_mul_out_argument():
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    gate = torch.randn_like(x)
    weight = torch.randn(128, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    result = phyai_kernel.rmsnorm_silu_mul(x, gate, weight, out=out)
    assert result.data_ptr() == out.data_ptr()
    expected = _rmsnorm_silu_mul_reference(x, gate, weight, 1e-6)
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
def test_rmsnorm_silu_mul_validates_wrapper_contract():
    x = torch.randn(2, 128, device="cuda")
    gate = torch.randn_like(x)
    weight = torch.ones(128, device="cuda")

    with pytest.raises(RuntimeError, match="shapes must match"):
        phyai_kernel.rmsnorm_silu_mul(x, gate[:, :64], weight)
    with pytest.raises(RuntimeError, match="dtypes must match"):
        phyai_kernel.rmsnorm_silu_mul(x, gate.to(torch.bfloat16), weight)
    with pytest.raises(RuntimeError, match="float16, bfloat16, or float32"):
        phyai_kernel.rmsnorm_silu_mul(
            x.to(torch.float64), gate.to(torch.float64), weight
        )
    with pytest.raises(RuntimeError, match="weight must be contiguous"):
        phyai_kernel.rmsnorm_silu_mul(
            x,
            gate,
            torch.ones(256, device="cuda")[::2],
        )
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        phyai_kernel.rmsnorm_silu_mul(x, gate, weight, eps=float("nan"))
    with pytest.raises(RuntimeError, match="out must live on the same CUDA device"):
        phyai_kernel.rmsnorm_silu_mul(x, gate, weight, out=torch.empty(2, 128))


@CUDA_REQUIRED
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


@CUDA_REQUIRED
def test_causal_conv1d_silu_split_qkv_validates_wrapper_contract():
    x = torch.randn(2, 7, 12, device="cuda")
    weight = torch.randn(12, 1, 4, device="cuda")

    with pytest.raises(RuntimeError, match="float16, bfloat16, or float32"):
        phyai_kernel.causal_conv1d_silu_split_qkv(
            x.to(torch.float64),
            weight.to(torch.float64),
            (4, 4, 4),
        )
    with pytest.raises(RuntimeError, match="dtypes must match"):
        phyai_kernel.causal_conv1d_silu_split_qkv(
            x,
            weight.to(torch.bfloat16),
            (4, 4, 4),
        )
    with pytest.raises(RuntimeError, match="dimensions must be contiguous"):
        phyai_kernel.causal_conv1d_silu_split_qkv(
            x,
            torch.randn(12, 1, 8, device="cuda")[..., ::2],
            (4, 4, 4),
        )
    with pytest.raises(ValueError, match="three positive integers"):
        phyai_kernel.causal_conv1d_silu_split_qkv(x, weight, (4, True, 7))
    with pytest.raises(ValueError, match="sum to"):
        phyai_kernel.causal_conv1d_silu_split_qkv(x, weight, (4, 4, 5))


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
