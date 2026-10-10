"""Fused affine modulation with the rounding of separate PyTorch operations."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _modulate_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    out_ptr,
    x_stride_b,
    x_stride_s,
    x_stride_d,
    shift_stride_b,
    shift_stride_d,
    scale_stride_b,
    scale_stride_d,
    ELEMENTS: tl.constexpr,
    SEQUENCE: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < ELEMENTS
    batch = offset // (SEQUENCE * DIM)
    sequence = (offset // DIM) % SEQUENCE
    channel = offset % DIM
    x_offset = batch * x_stride_b + sequence * x_stride_s + channel * x_stride_d
    shift_offset = batch * shift_stride_b + channel * shift_stride_d
    scale_offset = batch * scale_stride_b + channel * scale_stride_d
    x = tl.load(x_ptr + x_offset, valid, 0).to(tl.float32)
    shift = tl.load(shift_ptr + shift_offset, valid, 0).to(tl.float32)
    scale = tl.load(scale_ptr + scale_offset, valid, 0).to(tl.float32)

    # Preserve the rounding after each operator, including low-precision casts.
    factor = libdevice.add_rn(1.0, scale).to(x_ptr.dtype.element_ty).to(tl.float32)
    product = libdevice.mul_rn(x, factor).to(x_ptr.dtype.element_ty).to(tl.float32)
    result = libdevice.add_rn(product, shift).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + offset, result, valid)


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``x * (1 + scale[:, None, :]) + shift[:, None, :]``.

    ``x`` has shape ``(B, S, D)``. Each of ``shift`` and ``scale`` has two
    dimensions broadcastable to ``(B, D)``. Inputs may have arbitrary strides
    and must share a CPU or CUDA device and a float16, bfloat16, or float32 dtype.
    Every addition and multiplication rounds to that dtype, matching the
    unfused expression. CUDA executes one Triton launch; CPU uses PyTorch.

    The output is contiguous. An optional ``out`` must be contiguous, match
    ``x`` in shape, dtype, and device, and share no storage with any input.
    This permits reuse of an output allocation during CUDA graph capture.
    """
    if x.ndim != 3 or shift.ndim != 2 or scale.ndim != 2:
        raise ValueError("modulate requires 3-D x and 2-D shift and scale")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("modulate supports float16, bfloat16, and float32")
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("modulate requires CPU or CUDA tensors")
    tensors = (x, shift, scale)
    if any(value.dtype != x.dtype or value.device != x.device for value in tensors):
        raise ValueError("modulate inputs must have the same dtype and device")
    if any(value.layout != torch.strided for value in tensors):
        raise ValueError("modulate requires strided inputs")
    batch, sequence, dim = x.shape
    for value in (shift, scale):
        if value.shape[0] not in (1, batch) or value.shape[1] not in (1, dim):
            raise ValueError("shift and scale must be broadcastable to (B, D)")

    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    else:
        if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
            raise ValueError("out must match x in shape, dtype, and device")
        if out.layout != torch.strided or not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if any(torch._C._overlaps(out, value) for value in tensors):
            raise ValueError("out must not share storage with any input")
    if x.numel() == 0:
        return out
    if x.device.type == "cpu":
        out.copy_(x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
        return out

    block = 256
    _modulate_kernel[(triton.cdiv(x.numel(), block),)](
        x,
        shift,
        scale,
        out,
        *x.stride(),
        shift.stride(0) if shift.shape[0] != 1 else 0,
        shift.stride(1) if shift.shape[1] != 1 else 0,
        scale.stride(0) if scale.shape[0] != 1 else 0,
        scale.stride(1) if scale.shape[1] != 1 else 0,
        ELEMENTS=x.numel(),
        SEQUENCE=sequence,
        DIM=dim,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return out


__all__ = ["modulate"]
