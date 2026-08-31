"""FP32 GEMM for numerically stable token routing.

MoE routers are unusually sensitive to rounding because a small logit change
can alter the selected top-k experts.  The common fast path
``bf16 input x bf16 weight -> fp32 output`` still rounds both operands before
the multiplication.  This kernel instead:

1. loads BF16 or FP32 activations and weights;
2. converts every loaded value to FP32;
3. performs the products, reductions, and optional bias add in FP32;
4. writes FP32 router logits.

The public layout follows ``torch.nn.functional.linear``: ``x`` has shape
``(..., K)``, ``weight`` has shape ``(N, K)``, and the result has shape
``(..., N)``.  The implementation is specialized for the small ``N`` used by
routers, but it supports arbitrary positive ``K`` and ``N``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _fp32_router_gemm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    x_stride_m,
    x_stride_k,
    weight_stride_n,
    weight_stride_k,
    bias_stride,
    out_stride_m,
    out_stride_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    expert_block = tl.program_id(1)

    offsets_n = expert_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offsets_n < N
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offsets_k < K

        x = tl.load(
            x_ptr + row * x_stride_m + offsets_k * x_stride_k,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr
            + offsets_n[:, None] * weight_stride_n
            + offsets_k[None, :] * weight_stride_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Both operands are explicitly FP32 before multiplication.  Avoid
        # tl.dot here so this path cannot silently select TF32 tensor cores.
        accumulator += tl.sum(weight * x[None, :], axis=1)

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + offsets_n * bias_stride,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += bias

    tl.store(
        out_ptr + row * out_stride_m + offsets_n * out_stride_n,
        accumulator,
        mask=n_mask,
    )


def _check_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
) -> None:
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError(
            "phyai_kernel.triton.fp32_router_gemm: tensors must live on CUDA"
        )
    if x.dim() < 1:
        raise RuntimeError(
            "phyai_kernel.triton.fp32_router_gemm: input must have at least "
            "one dimension"
        )
    if weight.dim() != 2:
        raise RuntimeError(
            f"phyai_kernel.triton.fp32_router_gemm: weight must be 2D, "
            f"got {weight.dim()}D"
        )
    if x.shape[-1] != weight.shape[1]:
        raise RuntimeError(
            f"phyai_kernel.triton.fp32_router_gemm: input K={x.shape[-1]} "
            f"must match weight K={weight.shape[1]}"
        )
    if x.shape[-1] == 0 or weight.shape[0] == 0:
        raise RuntimeError(
            "phyai_kernel.triton.fp32_router_gemm: K and N must be positive"
        )
    supported_dtypes = (torch.bfloat16, torch.float32)
    if x.dtype not in supported_dtypes or weight.dtype not in supported_dtypes:
        raise RuntimeError(
            "phyai_kernel.triton.fp32_router_gemm: input and weight dtypes "
            f"must be bfloat16 or float32, got {x.dtype} and {weight.dtype}"
        )
    if x.device != weight.device:
        raise RuntimeError(
            "phyai_kernel.triton.fp32_router_gemm: input and weight must be "
            "on the same CUDA device"
        )

    output_shape = tuple(x.shape[:-1]) + (weight.shape[0],)
    if bias is not None:
        if not bias.is_cuda or bias.device != x.device:
            raise RuntimeError(
                "phyai_kernel.triton.fp32_router_gemm: bias must be on the "
                "same CUDA device as input"
            )
        if bias.dim() != 1 or bias.shape[0] != weight.shape[0]:
            raise RuntimeError(
                f"phyai_kernel.triton.fp32_router_gemm: bias shape "
                f"{tuple(bias.shape)} must be ({weight.shape[0]},)"
            )
        if bias.dtype != torch.float32:
            raise RuntimeError(
                "phyai_kernel.triton.fp32_router_gemm: bias must be float32"
            )

    if out is not None:
        if (
            tuple(out.shape) != output_shape
            or out.dtype != torch.float32
            or out.device != x.device
            or not out.is_contiguous()
        ):
            raise RuntimeError(
                "phyai_kernel.triton.fp32_router_gemm: `out` must be a "
                "contiguous float32 tensor with the expected shape and device"
            )


def fp32_router_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute router logits with FP32 operands and FP32 accumulation.

    Parameters
    ----------
    x:
        BF16 or FP32 activations with shape ``(..., K)``.
    weight:
        BF16 or FP32 router weights with shape ``(N, K)``. LingBot VLA V2's
        default BF16 deployment stores this tensor in BF16; the kernel still
        promotes every loaded value before multiplication.
    bias:
        Optional FP32 bias with shape ``(N,)``.
    out:
        Optional contiguous FP32 output buffer with shape ``(..., N)``.

    Returns
    -------
    A FP32 tensor with shape ``(..., N)``.
    """
    _check_inputs(x, weight, bias, out)

    output_shape = tuple(x.shape[:-1]) + (weight.shape[0],)
    output = (
        torch.empty(output_shape, dtype=torch.float32, device=x.device)
        if out is None
        else out
    )

    K = x.shape[-1]
    x_2d = x.reshape(-1, K)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    output_2d = output.reshape(-1, weight.shape[0])
    M = x_2d.shape[0]
    N = weight.shape[0]
    if M == 0:
        return output

    has_bias = bias is not None
    # Triton kernels cannot receive None pointers.  The constexpr removes the
    # bias load when absent, so any valid CUDA pointer is a safe placeholder.
    bias_ptr = bias if has_bias else weight
    block_n = triton.next_power_of_2(min(N, 32))
    block_k = 128
    grid = (M, triton.cdiv(N, block_n))
    _fp32_router_gemm_kernel[grid](
        x_2d,
        weight,
        bias_ptr,
        output_2d,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        bias.stride(0) if has_bias else 0,
        output_2d.stride(0),
        output_2d.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return output


__all__ = ["fp32_router_gemm"]
