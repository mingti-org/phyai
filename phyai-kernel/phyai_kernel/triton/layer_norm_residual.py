"""LayerNorm fused with a separately rounded gated residual update."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _layernorm_gated_residual_kernel(
    x_ptr,
    branch_ptr,
    gate_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    updated_ptr,
    x_stride_b,
    x_stride_s,
    x_stride_d,
    branch_stride_b,
    branch_stride_s,
    branch_stride_d,
    gate_stride_b,
    gate_stride_d,
    weight_stride,
    bias_stride,
    SEQUENCE: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // SEQUENCE
    sequence = row % SEQUENCE
    channel = tl.arange(0, BLOCK)
    valid = channel < DIM
    x = tl.load(
        x_ptr + batch * x_stride_b + sequence * x_stride_s + channel * x_stride_d,
        valid,
        0,
    ).to(tl.float32)
    branch = tl.load(
        branch_ptr
        + batch * branch_stride_b
        + sequence * branch_stride_s
        + channel * branch_stride_d,
        valid,
        0,
    ).to(tl.float32)
    gate = tl.load(
        gate_ptr + batch * gate_stride_b + channel * gate_stride_d,
        valid,
        0,
    ).to(tl.float32)
    # Preserve both activation-dtype roundings before the normalization.
    gated = libdevice.mul_rn(branch, gate).to(x_ptr.dtype.element_ty).to(tl.float32)
    updated = libdevice.add_rn(x, gated).to(x_ptr.dtype.element_ty)
    tl.store(updated_ptr + row * DIM + channel, updated, valid)
    value = updated.to(tl.float32)
    mean = tl.sum(tl.where(valid, value, 0.0), 0) / DIM
    centered = tl.where(valid, libdevice.add_rn(value, -mean), 0.0)
    variance = tl.sum(centered * centered, 0) / DIM
    normalized = libdevice.mul_rn(centered, tl.rsqrt(variance + EPS))
    weight = tl.load(weight_ptr + channel * weight_stride, valid, 0).to(tl.float32)
    result = libdevice.mul_rn(normalized, weight)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel * bias_stride, valid, 0).to(tl.float32)
        result = libdevice.add_rn(result, bias)
    tl.store(out_ptr + row * DIM + channel, result.to(x_ptr.dtype.element_ty), valid)


def layernorm_gated_residual(
    x: torch.Tensor,
    branch: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(LayerNorm(updated), updated)`` for ``updated = x + branch * gate``.

    The 3-D residual tensors share shape ``(B, S, D)`` and dtype. A 2-D gate
    broadcasts to ``(B, D)``. Multiplication and addition round separately to
    the input dtype. Statistics and affine operations use fp32, followed by
    one output cast. Inputs may be strided and are never modified; both outputs
    are contiguous. CUDA widths up to 8192 use one Triton kernel. CPU tensors
    and larger widths use equivalent PyTorch operations.
    """
    floats = (torch.float16, torch.bfloat16, torch.float32)
    if x.ndim != 3 or branch.shape != x.shape or gate.ndim != 2:
        raise ValueError("expected matching 3-D x/branch and a 2-D gate")
    if x.dtype not in floats or x.device.type not in ("cpu", "cuda"):
        raise ValueError("expected float16, bfloat16, or float32 CPU/CUDA inputs")
    if any(t.dtype != x.dtype or t.device != x.device for t in (branch, gate)):
        raise ValueError("x, branch, and gate must share dtype and device")
    batch, sequence, dim = x.shape
    if dim <= 0:
        raise ValueError("hidden dimension must be positive")
    if gate.shape[0] not in (1, batch) or gate.shape[1] not in (1, dim):
        raise ValueError("gate must broadcast to (B, D)")
    parameters = (weight,) if bias is None else (weight, bias)
    if any(
        t.shape != (dim,) or t.dtype not in floats or t.device != x.device
        for t in parameters
    ):
        raise ValueError(
            "affine parameters must be floating vectors of width D on the input device"
        )
    if any(t.layout != torch.strided for t in (x, branch, gate, *parameters)):
        raise ValueError("expected strided inputs")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if x.device.type == "cpu" or dim > 8192:
        updated = (x + branch * gate.unsqueeze(1)).contiguous()
        normalized = F.layer_norm(
            updated.float(),
            (dim,),
            weight.float(),
            None if bias is None else bias.float(),
            eps,
        ).to(x.dtype)
        return normalized.contiguous(), updated
    out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    updated = torch.empty_like(out)
    if x.numel() == 0:
        return out, updated
    block = triton.next_power_of_2(dim)
    _layernorm_gated_residual_kernel[(batch * sequence,)](
        x,
        branch,
        gate,
        weight,
        weight if bias is None else bias,
        out,
        updated,
        *x.stride(),
        *branch.stride(),
        gate.stride(0) if gate.shape[0] != 1 else 0,
        gate.stride(1) if gate.shape[1] != 1 else 0,
        weight.stride(0),
        0 if bias is None else bias.stride(0),
        SEQUENCE=sequence,
        DIM=dim,
        EPS=eps,
        HAS_BIAS=bias is not None,
        BLOCK=block,
        num_warps=4 if block <= 2048 or batch > 1 else 8,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return out, updated


__all__ = ["layernorm_gated_residual"]
