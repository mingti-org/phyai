"""Numerical-equivalence tests for the Triton FP32 Router GEMM."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import phyai_kernel
import phyai_kernel.triton.router_gemm as triton_router_gemm


if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for phyai-kernel Triton tests", allow_module_level=True
    )


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # fp32_router_gemm intentionally avoids TF32.  Compare it with an IEEE
    # FP32 PyTorch reference even when the surrounding application (including
    # LingBot) selected torch's "high" matmul mode.
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        return F.linear(
            x.to(torch.float32),
            weight.to(torch.float32),
            bias,
        )
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def test_module_exposes_fp32_router_gemm() -> None:
    assert phyai_kernel.fp32_router_gemm is triton_router_gemm.fp32_router_gemm


@pytest.mark.parametrize(
    ("input_shape", "num_experts"),
    [
        ((1, 768), 32),
        ((7, 768), 32),
        ((33, 768), 32),
        ((2, 5, 768), 32),
        ((11, 127), 17),
        ((4, 1024), 65),
    ],
)
@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_fp32_router_gemm_matches_reference(
    input_shape: tuple[int, ...],
    num_experts: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    with_bias: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(0xC0DE)
    hidden_size = input_shape[-1]
    x = torch.randn(
        input_shape,
        dtype=input_dtype,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        (num_experts, hidden_size),
        dtype=weight_dtype,
        device="cuda",
        generator=generator,
    )
    bias = (
        torch.randn(
            num_experts,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        if with_bias
        else None
    )

    actual = phyai_kernel.fp32_router_gemm(x, weight, bias)
    expected = _reference(x, weight, bias)

    assert actual.shape == input_shape[:-1] + (num_experts,)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)


def test_fp32_router_gemm_out_argument() -> None:
    x = torch.randn(4, 768, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(32, 768, dtype=torch.float32, device="cuda")
    out = torch.empty(4, 32, dtype=torch.float32, device="cuda")

    returned = phyai_kernel.fp32_router_gemm(x, weight, out=out)

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        returned,
        _reference(x, weight, None),
        atol=5e-4,
        rtol=2e-4,
    )


def test_fp32_weight_values_are_not_rounded_to_bfloat16() -> None:
    x = torch.ones(1, 768, dtype=torch.bfloat16, device="cuda")
    weight = torch.full(
        (32, 768),
        1.001,
        dtype=torch.float32,
        device="cuda",
    )

    actual = phyai_kernel.fp32_router_gemm(x, weight)
    expected = _reference(x, weight, None)
    bf16_rounded = _reference(x, weight.to(torch.bfloat16), None)

    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)
    assert torch.max(torch.abs(actual - bf16_rounded)).item() > 0.5


def test_fp32_router_gemm_accepts_non_contiguous_input() -> None:
    full = torch.randn(8, 2, 768, dtype=torch.bfloat16, device="cuda")
    x = full[:, 0, :]
    weight = torch.randn(32, 768, dtype=torch.float32, device="cuda")
    assert not x.is_contiguous()

    actual = phyai_kernel.fp32_router_gemm(x, weight)
    expected = _reference(x, weight, None)

    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)


def test_fp32_router_gemm_empty_input() -> None:
    x = torch.empty(0, 768, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(32, 768, dtype=torch.float32, device="cuda")

    out = phyai_kernel.fp32_router_gemm(x, weight)

    assert out.shape == (0, 32)
    assert out.dtype == torch.float32


def test_fp32_router_gemm_rejects_non_cuda() -> None:
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    weight = torch.randn(4, 16, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must live on CUDA"):
        phyai_kernel.fp32_router_gemm(x, weight)


def test_fp32_router_gemm_rejects_shape_mismatch() -> None:
    x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 8, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="must match weight"):
        phyai_kernel.fp32_router_gemm(x, weight)


def test_fp32_router_gemm_rejects_fp16() -> None:
    x = torch.randn(2, 16, dtype=torch.float16, device="cuda")
    weight = torch.randn(4, 16, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="bfloat16 or float32"):
        phyai_kernel.fp32_router_gemm(x, weight)


def test_fp32_router_gemm_rejects_non_fp32_bias() -> None:
    x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 16, dtype=torch.float32, device="cuda")
    bias = torch.randn(4, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="bias must be float32"):
        phyai_kernel.fp32_router_gemm(x, weight, bias)
