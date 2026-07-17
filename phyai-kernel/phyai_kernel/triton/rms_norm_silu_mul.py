"""Triton fused RMSNorm + learned affine + SiLU gate multiplication."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


_SINGLE_BLOCK_MAX = 8192


@triton.jit
def _rmsnorm_silu_mul_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    x_row_stride,
    gate_row_stride,
    out_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(
        x_ptr + row * x_row_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normalized = x * tl.rsqrt(variance + eps)
    normalized = normalized.to(x_ptr.dtype.element_ty).to(tl.float32)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(
        gate_ptr + row * gate_row_stride + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output = normalized * weight * gate * tl.sigmoid(gate)
    tl.store(
        out_ptr + row * out_row_stride + cols,
        output.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def rmsnorm_silu_mul(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``RMSNorm(x) * SiLU(gate)`` with MiniCPM-Qwen semantics.

    RMS statistics use FP32. The normalized value is rounded to ``x.dtype``
    before multiplication by the learned weight and FP32 SiLU gate, matching
    the reference Qwen3.5 gated RMSNorm ordering.
    """

    if not x.is_cuda or not gate.is_cuda or not weight.is_cuda:
        raise RuntimeError(
            "phyai_kernel.triton.rmsnorm_silu_mul: tensors must live on CUDA"
        )
    if x.shape != gate.shape:
        raise RuntimeError(
            "phyai_kernel.triton.rmsnorm_silu_mul: x and gate shapes must match"
        )
    if x.dtype != gate.dtype:
        raise RuntimeError(
            "phyai_kernel.triton.rmsnorm_silu_mul: x and gate dtypes must match"
        )
    if not x.is_contiguous() or not gate.is_contiguous():
        raise RuntimeError(
            "phyai_kernel.triton.rmsnorm_silu_mul: x and gate must be contiguous"
        )
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise RuntimeError(
            "phyai_kernel.triton.rmsnorm_silu_mul: weight must be 1D and match "
            "the input hidden size"
        )
    n_cols = x.shape[-1]
    if n_cols > _SINGLE_BLOCK_MAX:
        raise RuntimeError(
            f"phyai_kernel.triton.rmsnorm_silu_mul: hidden size {n_cols} exceeds "
            f"the supported maximum {_SINGLE_BLOCK_MAX}"
        )

    if out is None:
        out_t = torch.empty_like(x)
    else:
        if out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
            raise RuntimeError(
                "phyai_kernel.triton.rmsnorm_silu_mul: out must match x and be contiguous"
            )
        out_t = out
    if x.numel() == 0:
        return out_t

    x_2d = x.view(-1, n_cols)
    gate_2d = gate.view(-1, n_cols)
    out_2d = out_t.view(-1, n_cols)
    block_size = triton.next_power_of_2(n_cols)
    num_warps = 4 if block_size <= 1024 else (8 if block_size <= 4096 else 16)
    _rmsnorm_silu_mul_kernel[(x_2d.shape[0],)](
        x_2d,
        gate_2d,
        weight,
        out_2d,
        x_2d.stride(0),
        gate_2d.stride(0),
        out_2d.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_t


__all__ = ["rmsnorm_silu_mul"]
