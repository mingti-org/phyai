"""Fused Q/K rotation with precomputed cosine and sine tensors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.extra import libdevice


@triton.jit
def _precomputed_rope_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    cos_stride_b,
    cos_stride_s,
    cos_stride_d,
    sin_stride_b,
    sin_stride_s,
    sin_stride_d,
    SEQUENCE,
    Q_HEADS,
    K_HEADS,
    DIM: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    batch = token // SEQUENCE
    sequence = token % SEQUENCE
    pair = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    half = DIM // 2
    q_mask = pair < Q_HEADS * half
    k_pair = pair - Q_HEADS * half
    k_mask = (k_pair >= 0) & (k_pair < K_HEADS * half)
    channel = pair % half
    if INTERLEAVE:
        first = channel * 2
        second = first + 1
    else:
        first = channel
        second = channel + half

    q_base = batch * q_stride_b + sequence * q_stride_s + (pair // half) * q_stride_h
    k_base = batch * k_stride_b + sequence * k_stride_s + (k_pair // half) * k_stride_h
    q_first = tl.load(q_ptr + q_base + first * q_stride_d, mask=q_mask, other=0).to(
        tl.float32
    )
    q_second = tl.load(q_ptr + q_base + second * q_stride_d, mask=q_mask, other=0).to(
        tl.float32
    )
    k_first = tl.load(k_ptr + k_base + first * k_stride_d, mask=k_mask, other=0).to(
        tl.float32
    )
    k_second = tl.load(k_ptr + k_base + second * k_stride_d, mask=k_mask, other=0).to(
        tl.float32
    )
    x_first = tl.where(q_mask, q_first, k_first)
    x_second = tl.where(q_mask, q_second, k_second)
    valid = q_mask | k_mask
    cos_base = batch * cos_stride_b + sequence * cos_stride_s
    sin_base = batch * sin_stride_b + sequence * sin_stride_s
    cos_first = tl.load(
        cos_ptr + cos_base + first * cos_stride_d, mask=valid, other=0
    ).to(tl.float32)
    cos_second = tl.load(
        cos_ptr + cos_base + second * cos_stride_d, mask=valid, other=0
    ).to(tl.float32)
    sin_first = tl.load(
        sin_ptr + sin_base + first * sin_stride_d, mask=valid, other=0
    ).to(tl.float32)
    sin_second = tl.load(
        sin_ptr + sin_base + second * sin_stride_d, mask=valid, other=0
    ).to(tl.float32)
    # Explicit rounding also survives Inductor's reconstruction of launch options.
    out_first = libdevice.add_rn(
        libdevice.mul_rn(x_first, cos_first), libdevice.mul_rn(-x_second, sin_first)
    )
    out_second = libdevice.add_rn(
        libdevice.mul_rn(x_second, cos_second), libdevice.mul_rn(x_first, sin_second)
    )
    q_out_base = token * Q_HEADS * DIM + (pair // half) * DIM
    k_out_base = token * K_HEADS * DIM + (k_pair // half) * DIM
    tl.store(q_out_ptr + q_out_base + first, out_first, mask=q_mask)
    tl.store(q_out_ptr + q_out_base + second, out_second, mask=q_mask)
    tl.store(k_out_ptr + k_out_base + first, out_first, mask=k_mask)
    tl.store(k_out_ptr + k_out_base + second, out_second, mask=k_mask)


@triton_op("phyai_kernel::apply_rope_precomputed", mutates_args=())
def apply_rope_precomputed(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleave: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``(S,H,D)`` or ``(B,S,H,D)`` Q/K in one CUDA launch.

    Head counts may differ. Arbitrary input strides and cosine/sine broadcast
    dimensions are supported without input copies. Cosine and sine broadcast
    to ``(*q.shape[:-2], D)``; outputs are contiguous and retain each input's
    dtype. Products and sums use separate FP32 rounding, with FMA disabled.
    """
    if q.ndim not in (3, 4) or k.ndim != q.ndim:
        raise ValueError("Q/K must both have rank 3 or rank 4")
    if q.shape[:-2] != k.shape[:-2] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q/K token dimensions and head dimensions must match")
    if q.shape[-1] <= 0 or q.shape[-1] % 2:
        raise ValueError("RoPE requires a positive even head dimension")
    tensors = (q, k, cos, sin)
    if not q.is_cuda or any(value.device != q.device for value in tensors):
        raise ValueError("Q/K and cosine/sine must share a CUDA device")
    if any(
        value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        for value in tensors
    ):
        raise ValueError("RoPE supports float16, bfloat16, and float32 tensors")

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    phase_shape = (*q.shape[:-2], q.shape[-1])
    cos = torch.broadcast_to(cos, phase_shape)
    sin = torch.broadcast_to(sin, phase_shape)
    if q.ndim == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    batch, sequence, q_heads, dim = q.shape
    k_heads = k.shape[-2]
    if batch == 0 or sequence == 0 or q_heads + k_heads == 0:
        return q_out, k_out
    block = 256
    grid = (batch * sequence, triton.cdiv((q_heads + k_heads) * (dim // 2), block))
    wrap_triton(_precomputed_rope_kernel)[grid](
        q,
        k,
        cos,
        sin,
        q_out,
        k_out,
        *q.stride(),
        *k.stride(),
        *cos.stride(),
        *sin.stride(),
        sequence,
        q_heads,
        k_heads,
        dim,
        interleave,
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q_out, k_out


__all__ = ["apply_rope_precomputed"]
