"""Residual addition, non-affine LayerNorm, and conditional affine modulation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _layernorm_modulate_kernel(
    x_ptr,
    residual_ptr,
    shift_ptr,
    scale_ptr,
    out_ptr,
    updated_ptr,
    x_stride_b,
    x_stride_s,
    x_stride_d,
    residual_stride_b,
    residual_stride_s,
    residual_stride_d,
    shift_stride_b,
    shift_stride_d,
    scale_stride_b,
    scale_stride_d,
    EPS: tl.constexpr,
    SEQUENCE: tl.constexpr,
    DIM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // SEQUENCE
    sequence = row % SEQUENCE
    channel = tl.arange(0, BLOCK)
    valid = channel < DIM
    x_offset = batch * x_stride_b + sequence * x_stride_s + channel * x_stride_d
    x = tl.load(x_ptr + x_offset, valid, 0).to(tl.float32)
    if HAS_RESIDUAL:
        residual_offset = (
            batch * residual_stride_b
            + sequence * residual_stride_s
            + channel * residual_stride_d
        )
        residual = tl.load(residual_ptr + residual_offset, valid, 0).to(tl.float32)
        # The residual stream rounds before its mean and variance are computed.
        updated = libdevice.add_rn(x, residual).to(x_ptr.dtype.element_ty)
        tl.store(updated_ptr + row * DIM + channel, updated, valid)
        x = updated.to(tl.float32)

    mean = tl.sum(tl.where(valid, x, 0.0), axis=0) / DIM
    centered = tl.where(valid, libdevice.add_rn(x, -mean), 0.0)
    variance = tl.sum(centered * centered, axis=0) / DIM
    normalized = libdevice.mul_rn(centered, tl.rsqrt(variance + EPS))
    normalized = normalized.to(x_ptr.dtype.element_ty).to(tl.float32)

    shift = tl.load(
        shift_ptr + batch * shift_stride_b + channel * shift_stride_d, valid, 0
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + batch * scale_stride_b + channel * scale_stride_d, valid, 0
    ).to(tl.float32)
    factor = libdevice.add_rn(1.0, scale).to(x_ptr.dtype.element_ty).to(tl.float32)
    product = libdevice.mul_rn(normalized, factor)
    product = product.to(x_ptr.dtype.element_ty).to(tl.float32)
    result = libdevice.add_rn(product, shift).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + row * DIM + channel, result, valid)


def layernorm_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return modulated LayerNorm and the updated residual stream.

    ``x`` and an optional ``residual`` have shape ``(B, S, D)``. Shift and scale
    independently broadcast from two dimensions to ``(B, D)``. Inputs may be
    strided and must share a CPU or CUDA device and fp16, bf16, or fp32 dtype.
    The residual addition, normalized values, and each modulation operation
    round to the input dtype; LayerNorm statistics use fp32. Parallel reduction
    order may differ from PyTorch LayerNorm.

    The modulated output is contiguous. With a residual, the updated stream is
    also a new contiguous tensor; otherwise it is ``x`` itself. Inputs are never
    modified. CUDA widths up to 8192 use one Triton launch; larger widths and CPU
    inputs use the equivalent PyTorch operations.
    """
    if x.ndim != 3 or shift.ndim != 2 or scale.ndim != 2:
        raise ValueError("layernorm_modulate requires 3-D x and 2-D shift and scale")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("layernorm_modulate supports float16, bfloat16, and float32")
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("layernorm_modulate requires CPU or CUDA tensors")
    tensors = (x, shift, scale) + (() if residual is None else (residual,))
    if any(value.dtype != x.dtype or value.device != x.device for value in tensors):
        raise ValueError(
            "layernorm_modulate inputs must have the same dtype and device"
        )
    if any(value.layout != torch.strided for value in tensors):
        raise ValueError("layernorm_modulate requires strided inputs")
    if residual is not None and residual.shape != x.shape:
        raise ValueError("residual must have the same shape as x")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    batch, sequence, dim = x.shape
    for parameter in (shift, scale):
        if parameter.shape[0] not in (1, batch) or parameter.shape[1] not in (1, dim):
            raise ValueError("shift and scale must broadcast to (B, D)")
    if x.device.type == "cpu" or dim > 8192:
        updated = x if residual is None else (x + residual).contiguous()
        normalized = F.layer_norm(updated.float(), (dim,), None, None, eps).to(x.dtype)
        modulated = normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return modulated.contiguous(), updated

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    updated = x if residual is None else torch.empty_like(out)
    if x.numel() == 0:
        return out, updated
    residual_tensor = x if residual is None else residual
    block = triton.next_power_of_2(dim)
    _layernorm_modulate_kernel[(batch * sequence,)](
        x,
        residual_tensor,
        shift,
        scale,
        out,
        updated,
        *x.stride(),
        *residual_tensor.stride(),
        shift.stride(0) if shift.shape[0] != 1 else 0,
        shift.stride(1) if shift.shape[1] != 1 else 0,
        scale.stride(0) if scale.shape[0] != 1 else 0,
        scale.stride(1) if scale.shape[1] != 1 else 0,
        EPS=eps,
        SEQUENCE=sequence,
        DIM=dim,
        HAS_RESIDUAL=residual is not None,
        BLOCK=block,
        num_warps=4 if block <= 2048 else 8,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return out, updated


__all__ = ["layernorm_modulate"]
