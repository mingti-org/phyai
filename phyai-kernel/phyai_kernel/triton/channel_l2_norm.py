"""Channel L2 normalization with a fused, explicitly rounded affine epilogue.

The vector reduction retains PyTorch's accumulation order. One Triton launch
then clamps the denominator, divides, casts, scales, and applies the affine
parameters without materializing any intermediate activation tensors.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@torch.library.custom_op("phyai_kernel::channel_l2_vector_norm", mutates_args=())
def _channel_l2_vector_norm(
    x: torch.Tensor, dim: int, compute_dtype: torch.dtype | None
) -> torch.Tensor:
    """Keep the reference reduction order when Inductor compiles the epilogue."""
    return torch.linalg.vector_norm(
        x, ord=2, dim=dim, keepdim=True, dtype=compute_dtype
    )


@_channel_l2_vector_norm.register_fake
def _channel_l2_vector_norm_fake(
    x: torch.Tensor, dim: int, compute_dtype: torch.dtype | None
) -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] = 1
    return x.new_empty(shape, dtype=compute_dtype or x.dtype)


@triton.jit
def _channel_l2_norm_affine_kernel(
    X,
    Norm,
    Weight,
    Bias,
    Out,
    N: tl.constexpr,
    C: tl.constexpr,
    SPATIAL: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    CHANNEL_FIRST: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    if CHANNEL_FIRST:
        channel = (offset // SPATIAL) % C
        norm_offset = (offset // (C * SPATIAL)) * SPATIAL + offset % SPATIAL
    else:
        channel = offset % C
        norm_offset = offset // C

    value = tl.load(X + offset, valid, 0).to(COMPUTE_DTYPE).to(tl.float32)
    norm = tl.load(Norm + norm_offset, valid, 0).to(tl.float32)
    epsilon = tl.full((), EPS, tl.float32).to(COMPUTE_DTYPE).to(tl.float32)
    denominator = tl.maximum(norm, epsilon, propagate_nan=tl.PropagateNan.ALL)
    normalized = tl.div_rn(value, denominator).to(COMPUTE_DTYPE)
    normalized = normalized.to(Out.dtype.element_ty).to(tl.float32)
    # Each cast is an observable rounding boundary in the unfused expression.
    scaled = libdevice.mul_rn(normalized, SCALE).to(Out.dtype.element_ty).to(tl.float32)
    weight = tl.load(Weight + channel, valid, 0).to(tl.float32)
    result = libdevice.mul_rn(scaled, weight).to(Out.dtype.element_ty)
    if HAS_BIAS:
        bias = tl.load(Bias + channel, valid, 0).to(tl.float32)
        result = libdevice.add_rn(result.to(tl.float32), bias).to(Out.dtype.element_ty)
    tl.store(Out + offset, result, valid)


def channel_l2_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-12,
    *,
    channel_first: bool = True,
    compute_dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """Normalize channels and apply separately rounded scale, weight, and bias.

    CUDA inputs and affine tensors must be contiguous and have the same dtype.
    ``compute_dtype=None`` preserves the input dtype of the norm and division.
    """
    supported = (torch.bfloat16, torch.float16, torch.float32)
    if x.ndim < 2 or x.dtype not in supported:
        raise ValueError("channel_l2_norm requires at least 2-D floating inputs")
    if compute_dtype not in (None, torch.float32, x.dtype):
        raise ValueError("compute_dtype must be float32, the input dtype, or None")
    channels = x.shape[1 if channel_first else -1]
    for parameter in (weight, bias):
        if parameter is None:
            continue
        if parameter.numel() != channels or parameter.dtype != x.dtype:
            raise ValueError(
                "affine tensors must match the channel count and input dtype"
            )
        if parameter.device != x.device or not parameter.is_contiguous():
            raise ValueError(
                "affine tensors must be contiguous and share the input device"
            )
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("channel_l2_norm requires contiguous CUDA inputs")
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    dim = 1 if channel_first else -1
    if torch.compiler.is_compiling():
        norm = _channel_l2_vector_norm(x, dim, compute_dtype)
    else:
        norm = torch.linalg.vector_norm(
            x, ord=2, dim=dim, keepdim=True, dtype=compute_dtype
        )
    reduction_dtype = compute_dtype or x.dtype
    triton_dtype = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[reduction_dtype]
    _channel_l2_norm_affine_kernel[(triton.cdiv(x.numel(), 256),)](
        x,
        norm,
        weight,
        bias,
        out,
        N=x.numel(),
        C=channels,
        SPATIAL=x.numel() // (x.shape[0] * channels) if channel_first else 1,
        EPS=eps,
        SCALE=channels**0.5,
        CHANNEL_FIRST=channel_first,
        HAS_BIAS=bias is not None,
        COMPUTE_DTYPE=triton_dtype,
        BLOCK=256,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["channel_l2_norm"]
