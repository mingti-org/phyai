"""Paired projection-wide RMSNorm, rounded affine, and rotary embedding."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.extra.cuda import libdevice


@triton.jit
def _qk_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    qw_ptr,
    kw_ptr,
    cos_ptr,
    sin_ptr,
    qo_ptr,
    ko_ptr,
    Q_BATCH: tl.constexpr,
    Q_ROW: tl.constexpr,
    Q_COL: tl.constexpr,
    K_BATCH: tl.constexpr,
    K_ROW: tl.constexpr,
    K_COL: tl.constexpr,
    COS_BATCH: tl.constexpr,
    COS_ROW: tl.constexpr,
    COS_COL: tl.constexpr,
    SIN_BATCH: tl.constexpr,
    SIN_ROW: tl.constexpr,
    SIN_COL: tl.constexpr,
    EPS: tl.constexpr,
    SEQUENCE: tl.constexpr,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOCAL_START: tl.constexpr,
    LOCAL_WIDTH: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, branch = tl.program_id(0), tl.program_id(1)
    batch, sequence = row // SEQUENCE, row % SEQUENCE
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    input_ptr = tl.where(branch == 0, q_ptr, k_ptr)
    weight_ptr = tl.where(branch == 0, qw_ptr, kw_ptr)
    output_ptr = tl.where(branch == 0, qo_ptr, ko_ptr)
    batch_stride = tl.where(branch == 0, Q_BATCH, K_BATCH)
    row_stride = tl.where(branch == 0, Q_ROW, K_ROW)
    col_stride = tl.where(branch == 0, Q_COL, K_COL)
    x = tl.load(
        input_ptr + batch * batch_stride + sequence * row_stride + col * col_stride,
        valid,
        0,
    ).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / WIDTH
    normalized = (x * tl.rsqrt(variance + EPS)).to(output_ptr.dtype.element_ty)
    weight = tl.load(weight_ptr + col, valid, 0).to(output_ptr.dtype.element_ty)
    # Both casts are observable in low precision and precede rotary arithmetic.
    affine = (normalized * weight).to(output_ptr.dtype.element_ty).to(tl.float32)
    phase_col = col % HEAD_DIM
    if INTERLEAVE:
        partner_col = col ^ 1
        first = phase_col % 2 == 0
    else:
        first = phase_col < HEAD_DIM // 2
        partner_col = tl.where(first, col + HEAD_DIM // 2, col - HEAD_DIM // 2)
    partner = tl.gather(affine, tl.where(valid, partner_col, 0), axis=0)
    cosine = tl.load(
        cos_ptr + batch * COS_BATCH + sequence * COS_ROW + phase_col * COS_COL
    ).to(tl.float32)
    sine = tl.load(
        sin_ptr + batch * SIN_BATCH + sequence * SIN_ROW + phase_col * SIN_COL
    ).to(tl.float32)
    rotated_partner = tl.where(first, -partner, partner)
    # Match the separate FP32 products and sum of apply_rope_precomputed.
    value = libdevice.add_rn(
        libdevice.mul_rn(affine, cosine), libdevice.mul_rn(rotated_partner, sine)
    )
    local = (col >= LOCAL_START) & (col < LOCAL_START + LOCAL_WIDTH)
    tl.store(output_ptr + row * LOCAL_WIDTH + col - LOCAL_START, value, local)


@triton_op("phyai_kernel::qk_rmsnorm_rope", mutates_args=())
def qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    head_dim: int,
    interleave: bool = True,
    head_start: int = 0,
    local_heads: int | None = None,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize complete Q/K projections and rotate an optional head slice.

    Q/K have matching ``(S,D)`` or ``(B,S,D)`` shapes with arbitrary strides.
    RMS statistics cover the entire projection width, including for TP slices.
    Normalization and affine multiplication each round to the input dtype;
    RoPE then uses separate FP32 products and sums. Outputs are contiguous,
    shaped ``(*q.shape[:-1], local_heads, head_dim)``. Inputs are never modified.
    This single-pass CUDA kernel supports projection widths up to 8192.
    ``num_warps`` controls the reduction layout and accepts 4, 8, 16, or 32.
    """
    if q.ndim not in (2, 3) or q.shape != k.shape:
        raise ValueError("Q/K must have matching rank-2 or rank-3 shapes")
    width = q.shape[-1]
    if head_dim <= 0 or head_dim % 2 or width <= 0 or width % head_dim:
        raise ValueError(
            "projection width must be divisible by a positive even head_dim"
        )
    if width > 8192:
        raise ValueError("the fused kernel supports projection widths up to 8192")
    if local_heads is None:
        local_heads = width // head_dim - head_start
    if (
        head_start < 0
        or local_heads <= 0
        or (head_start + local_heads) * head_dim > width
    ):
        raise ValueError("requested head slice is outside the full projection")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if type(num_warps) is not int or num_warps not in (4, 8, 16, 32):
        raise ValueError("num_warps must be 4, 8, 16, or 32")
    tensors = (q, k, q_weight, k_weight, cos, sin)
    supported = (torch.float16, torch.bfloat16, torch.float32)
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must share a CUDA device")
    if q.dtype not in supported or k.dtype != q.dtype:
        raise ValueError("Q/K must share an fp16, bf16, or fp32 dtype")
    if cos.dtype not in supported or sin.dtype not in supported:
        raise ValueError("cosine and sine must use fp16, bf16, or fp32")
    for weight in (q_weight, k_weight):
        if (
            weight.shape != (width,)
            or not weight.is_contiguous()
            or weight.dtype != q.dtype
        ):
            raise ValueError(
                "weights must be contiguous vectors matching projection width and dtype"
            )
    cos = torch.broadcast_to(cos, (*q.shape[:-1], head_dim))
    sin = torch.broadcast_to(sin, (*q.shape[:-1], head_dim))
    output_shape = (*q.shape[:-1], local_heads, head_dim)
    q_out = torch.empty(output_shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty_like(q_out)
    if q.ndim == 2:
        q, k, cos, sin = (t.unsqueeze(0) for t in (q, k, cos, sin))
    batch, sequence, _ = q.shape
    if batch == 0 or sequence == 0:
        return q_out, k_out
    wrap_triton(_qk_rmsnorm_rope_kernel)[(batch * sequence, 2)](
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        k_out,
        *q.stride(),
        *k.stride(),
        *cos.stride(),
        *sin.stride(),
        eps,
        sequence,
        width,
        head_dim,
        head_start * head_dim,
        local_heads * head_dim,
        interleave,
        triton.next_power_of_2(width),
        num_warps=num_warps,
    )
    return q_out, k_out


__all__ = ["qk_rmsnorm_rope"]
