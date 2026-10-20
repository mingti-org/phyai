"""Fused SiTU with the rounding of separate low-precision Torch operators."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _situ_and_mul_kernel(
    input_ptr,
    output_ptr,
    ELEMENTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < ELEMENTS
    row = offset // HIDDEN
    column = offset % HIDDEN
    gate = tl.load(input_ptr + row * (2 * HIDDEN) + column, valid, 0).to(tl.float32)
    hidden = tl.load(input_ptr + row * (2 * HIDDEN) + HIDDEN + column, valid, 0).to(
        tl.float32
    )
    scalar_type = input_ptr.dtype.element_ty

    # Every original Torch operation rounds to the activation dtype.
    gate_scaled = libdevice.mul_rn(gate, 0.25).to(scalar_type).to(tl.float32)
    gate_tanh = libdevice.tanh(gate_scaled).to(scalar_type).to(tl.float32)
    gate_bounded = libdevice.mul_rn(4.0, gate_tanh).to(scalar_type).to(tl.float32)
    gate_sigmoid = (
        libdevice.div_rn(1.0, libdevice.add_rn(1.0, libdevice.exp(-gate)))
        .to(scalar_type)
        .to(tl.float32)
    )
    gate_output = (
        libdevice.mul_rn(gate_bounded, gate_sigmoid).to(scalar_type).to(tl.float32)
    )
    hidden_scaled = libdevice.mul_rn(hidden, 1.0 / 25.0).to(scalar_type).to(tl.float32)
    hidden_tanh = libdevice.tanh(hidden_scaled).to(scalar_type).to(tl.float32)
    hidden_bounded = libdevice.mul_rn(25.0, hidden_tanh).to(scalar_type).to(tl.float32)
    result = libdevice.mul_rn(gate_output, hidden_bounded).to(scalar_type)
    tl.store(output_ptr + offset, result, valid)


def situ_and_mul(
    value: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply SiTU to concatenated ``[gate, hidden]`` along the final axis.

    Computes ``(4*tanh(gate/4)*sigmoid(gate)) * (25*tanh(hidden/25))``.
    CUDA inputs may be float16, bfloat16, or float32. The output is contiguous
    and each elementary operator rounds to the input dtype.
    """
    if value.device.type != "cuda":
        raise ValueError("situ_and_mul requires CUDA tensors")
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("situ_and_mul supports float16, bfloat16, and float32")
    if (
        value.layout != torch.strided
        or value.ndim < 1
        or value.shape[-1] <= 0
        or value.shape[-1] % 2
    ):
        raise ValueError(
            "situ_and_mul requires a strided tensor with a positive even final dimension"
        )
    hidden = value.shape[-1] // 2
    shape = (*value.shape[:-1], hidden)
    if out is None:
        out = torch.empty(shape, dtype=value.dtype, device=value.device)
    else:
        if out.shape != shape or out.dtype != value.dtype or out.device != value.device:
            raise ValueError("out must match the SiTU output shape, dtype, and device")
        if out.layout != torch.strided or not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if torch._C._overlaps(out, value):
            raise ValueError("out must not share storage with the input")
    if out.numel() == 0:
        return out
    value = value.contiguous()
    block = 256
    _situ_and_mul_kernel[(triton.cdiv(out.numel(), block),)](
        value,
        out,
        ELEMENTS=out.numel(),
        HIDDEN=hidden,
        BLOCK=block,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return out


__all__ = ["situ_and_mul"]
