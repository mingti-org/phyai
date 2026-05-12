# Reference implementations and test scaffolding adapted from SGLang
# (https://github.com/sgl-project/sglang), Copyright 2023-2024 SGLang Team,
# licensed under the Apache License, Version 2.0:
#     http://www.apache.org/licenses/LICENSE-2.0
"""Numerical-equivalence tests for the Triton RMSNorm kernels.

Validates the five public entry points against:

* ``torch.nn.RMSNorm`` (only for the standard variant — torch's RMSNorm
  matches the Llama/Qwen formulation).
* Reference Python implementations adapted from SGLang's RMSNorm
  tests and ``layernorm`` layer module (HF-cast and Gemma variants, which
  are not natively supported by ``torch.nn.RMSNorm``).

Test grid covers Qwen (head_dim 64/128, hidden 896/2048/3584/4096/8192) and
Gemma (head_dim 256, hidden 2304/3072/4608/9216) typical sizes plus a couple
of awkward shapes (non-power-of-two hidden, very large 12288/16384).
"""

from __future__ import annotations

import pytest
import torch

import phyai_kernel
import phyai_kernel.triton.rms_norm as triton_rmsnorm_mod

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for phyai-kernel Triton tests", allow_module_level=True
    )


# --------------------------------------------------------------------------- #
# Reference implementations (semantics match SGLang's test_norm.py)          #
# --------------------------------------------------------------------------- #


def _ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    orig = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf * w.float()).to(orig)


def _ref_rmsnorm_hf(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    orig = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    # HF semantics: cast back to dtype BEFORE the weight multiply.
    return xf.to(orig) * w


def _ref_fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float
):
    orig = x.dtype
    xf = x.to(torch.float32) + residual.to(torch.float32)
    new_residual = xf.to(orig)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    new_x = (xf * w.float()).to(orig)
    return new_x, new_residual


def _ref_gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    orig = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (xf * (1.0 + w.float())).to(orig)


def _ref_gemma_fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float
):
    orig = x.dtype
    summed = x + residual
    new_residual = summed
    xf = summed.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    new_x = (xf * (1.0 + w.float())).to(orig)
    return new_x, new_residual


# --------------------------------------------------------------------------- #
# Shapes                                                                      #
# --------------------------------------------------------------------------- #

# Realistic Qwen/Gemma hidden sizes plus a few awkward ones.
_HIDDEN_SIZES = [
    64,  # Qwen2 head_dim
    96,  # non-pow2
    128,  # Qwen3 head_dim, common qk-norm size
    256,  # Gemma head_dim
    896,  # Qwen2 0.5B hidden
    2048,  # Qwen2 1.5B hidden / Gemma2-2B
    2304,  # Gemma2-2B hidden alt
    3072,  # Qwen2 8B / Gemma2-9B
    3584,  # Qwen2 7B
    4096,  # Llama / Qwen2 14B
    4608,  # Gemma2-27B
    8192,  # Qwen2 32B / 72B
    9216,  # Gemma3-27B style (uncommon)
    12288,  # large
]
_BATCHES = [1, 19, 257]
_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _tols(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return (5e-5, 5e-5)
    if dtype == torch.bfloat16:
        return (2e-2, 2e-2)
    return (1e-3, 1e-3)  # fp16


def _make_inputs(
    n_rows: int, n_cols: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0xC0DE * n_rows + n_cols)
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=dtype)
    w = torch.randn(n_cols, device="cuda", dtype=dtype) * 0.5 + 1.0
    return x, w


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_rows", _BATCHES)
@pytest.mark.parametrize("n_cols", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_rmsnorm_matches_reference(n_rows: int, n_cols: int, dtype: torch.dtype):
    x, w = _make_inputs(n_rows, n_cols, dtype)
    eps = 1e-6
    expected = _ref_rmsnorm(x, w, eps)
    actual = phyai_kernel.rmsnorm(x, w, eps)
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_rows", _BATCHES)
@pytest.mark.parametrize("n_cols", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rmsnorm_matches_torch_nn(n_rows: int, n_cols: int, dtype: torch.dtype):
    """torch.nn.RMSNorm uses Llama-style (no `1+w`); compare directly."""
    x, w = _make_inputs(n_rows, n_cols, dtype)
    eps = 1e-6
    torch_norm = torch.nn.RMSNorm(n_cols, eps=eps, device="cuda", dtype=dtype)
    with torch.no_grad():
        torch_norm.weight.copy_(w)
    expected = torch_norm(x)
    actual = phyai_kernel.rmsnorm(x, w, eps)
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_rows", _BATCHES)
@pytest.mark.parametrize("n_cols", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_rmsnorm_hf_matches_reference(n_rows: int, n_cols: int, dtype: torch.dtype):
    x, w = _make_inputs(n_rows, n_cols, dtype)
    eps = 1e-6
    expected = _ref_rmsnorm_hf(x, w, eps)
    actual = phyai_kernel.rmsnorm_hf(x, w, eps)
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_rows", _BATCHES)
@pytest.mark.parametrize("n_cols", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_add_rmsnorm_matches_reference(
    n_rows: int, n_cols: int, dtype: torch.dtype
):
    x, w = _make_inputs(n_rows, n_cols, dtype)
    residual = torch.randn_like(x)
    eps = 1e-6

    exp_x, exp_residual = _ref_fused_add_rmsnorm(x.clone(), residual.clone(), w, eps)

    act_x = x.clone()
    act_residual = residual.clone()
    phyai_kernel.fused_add_rmsnorm(act_x, act_residual, w, eps)

    rtol, atol = _tols(dtype)
    torch.testing.assert_close(act_residual, exp_residual, rtol=rtol, atol=atol)
    torch.testing.assert_close(act_x, exp_x, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_rows", _BATCHES)
@pytest.mark.parametrize("n_cols", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_gemma_rmsnorm_matches_reference(n_rows: int, n_cols: int, dtype: torch.dtype):
    x, w = _make_inputs(n_rows, n_cols, dtype)
    # Gemma weight is initialized at zero (so 1+w starts at 1.0). Use a small
    # range to mimic real weights.
    w = w * 0.1 - 0.5
    eps = 1e-6
    expected = _ref_gemma_rmsnorm(x, w, eps)
    actual = phyai_kernel.gemma_rmsnorm(x, w, eps)
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("n_rows", _BATCHES)
@pytest.mark.parametrize("n_cols", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_gemma_fused_add_rmsnorm_matches_reference(
    n_rows: int, n_cols: int, dtype: torch.dtype
):
    x, w = _make_inputs(n_rows, n_cols, dtype)
    w = w * 0.1 - 0.5
    residual = torch.randn_like(x)
    eps = 1e-6

    exp_x, exp_residual = _ref_gemma_fused_add_rmsnorm(
        x.clone(), residual.clone(), w, eps
    )

    act_x = x.clone()
    act_residual = residual.clone()
    phyai_kernel.gemma_fused_add_rmsnorm(act_x, act_residual, w, eps)

    rtol, atol = _tols(dtype)
    torch.testing.assert_close(act_residual, exp_residual, rtol=rtol, atol=atol)
    torch.testing.assert_close(act_x, exp_x, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Shape and contiguity edge-cases                                             #
# --------------------------------------------------------------------------- #


def test_rmsnorm_handles_3d_input():
    """RMSNorm typically receives ``(batch, seq, hidden)``; flatten path."""
    x = torch.randn(2, 17, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, device="cuda", dtype=torch.float16)
    eps = 1e-6
    expected = _ref_rmsnorm(x, w, eps)
    actual = phyai_kernel.rmsnorm(x, w, eps)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_rmsnorm_with_explicit_out():
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    ret = phyai_kernel.rmsnorm(x, w, 1e-6, out=out)
    assert ret.data_ptr() == out.data_ptr()
    expected = _ref_rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


def test_rmsnorm_zero_rows_no_launch():
    """Empty input should produce empty output without launching the kernel."""
    x = torch.empty(0, 4096, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, device="cuda", dtype=torch.float16)
    out = phyai_kernel.rmsnorm(x, w)
    assert out.shape == (0, 4096)


def test_qwen3_qk_norm_head_dim_pattern():
    """Qwen3's q/k norm applies RMSNorm across head_dim — replicate that path."""
    # (num_tokens, num_heads, head_dim) -> view to (-1, head_dim)
    head_dim = 128
    x = torch.randn(11, 8, head_dim, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    expected = _ref_rmsnorm_hf(x, w, 1e-6)
    actual = phyai_kernel.rmsnorm_hf(x, w, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_single_block_threshold_boundary():
    """Confirm the two-pass kernel matches the single-block kernel at the boundary."""
    threshold = triton_rmsnorm_mod._SINGLE_BLOCK_MAX
    for n_cols in (threshold, threshold + 256):
        x = torch.randn(4, n_cols, device="cuda", dtype=torch.float16)
        w = torch.randn(n_cols, device="cuda", dtype=torch.float16)
        expected = _ref_rmsnorm(x, w, 1e-6)
        actual = phyai_kernel.rmsnorm(x, w, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
