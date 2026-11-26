"""LingBot normalization with FP32 statistics and explicit rounding boundaries."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lingbot_norm_kernel(
    X,
    Weight,
    Bias,
    Out,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    BIAS_STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    CENTER: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    remaining = row
    offset = tl.full((), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 2, -1, -1):
        offset += (remaining % SHAPE[axis]) * STRIDES[axis]
        remaining = remaining // SHAPE[axis]
    cols = tl.arange(0, BLOCK)
    mask = cols < SHAPE[-1]
    value = tl.load(X + offset + cols * STRIDES[-1], mask, 0).to(tl.float32)
    if CENTER:
        mean = tl.sum(value, 0) / SHAPE[-1]
        value = tl.where(mask, value - mean, 0.0)
    variance = tl.sum(value * value, 0) / SHAPE[-1]
    normalized = value * tl.rsqrt(variance + EPS)
    if not CENTER:
        normalized = normalized.to(Out.dtype.element_ty).to(tl.float32)
    weight = tl.load(Weight + cols * WEIGHT_STRIDE, mask, 0).to(tl.float32)
    result = normalized * weight
    if HAS_BIAS:
        bias = tl.load(Bias + cols * BIAS_STRIDE, mask, 0).to(tl.float32)
        result = result + bias
    tl.store(Out + row * SHAPE[-1] + cols, result, mask)


def _norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    *,
    center: bool,
) -> torch.Tensor:
    supported = (torch.bfloat16, torch.float16, torch.float32)
    if not x.is_cuda or x.dtype not in supported or x.ndim == 0 or x.shape[-1] == 0:
        raise ValueError(
            "LingBot norms require floating CUDA input with a positive hidden size"
        )
    for parameter in (weight, bias):
        if parameter is not None and (
            parameter.device != x.device
            or parameter.dtype not in supported
            or parameter.shape != (x.shape[-1],)
        ):
            raise ValueError(
                "affine tensors must match the input device and hidden size"
            )
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = x.numel() // x.shape[-1]
    if rows:
        block = triton.next_power_of_2(x.shape[-1])
        _lingbot_norm_kernel[(rows,)](
            x,
            weight,
            bias,
            out,
            SHAPE=tuple(x.shape),
            STRIDES=tuple(x.stride()),
            WEIGHT_STRIDE=weight.stride(0),
            BIAS_STRIDE=0 if bias is None else bias.stride(0),
            EPS=eps,
            CENTER=center,
            HAS_BIAS=bias is not None,
            BLOCK=block,
            num_warps=4 if block <= 1024 else 8,
            enable_fp_fusion=False,
        )
    return out


def lingbot_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Round normalized activations to the input dtype before applying weight."""
    return _norm(x, weight, None, eps, center=False)


def lingbot_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute centered variance and separate FP32 affine operations."""
    return _norm(x, weight, bias, eps, center=True)


@triton.jit
def _lingbot_adarmsnorm_kernel(
    X,
    Weight,
    Gamma,
    Beta,
    Out,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    GAMMA_STRIDES: tl.constexpr,
    BETA_STRIDES: tl.constexpr,
    GAMMA_BATCH: tl.constexpr,
    BETA_BATCH: tl.constexpr,
    ROWS_PER_BATCH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // ROWS_PER_BATCH
    offset = batch.to(tl.int64) * STRIDES[0]
    if len(SHAPE) == 3:
        offset += (row % ROWS_PER_BATCH) * STRIDES[1]
    cols = tl.arange(0, BLOCK)
    mask = cols < SHAPE[-1]
    value = tl.load(X + offset + cols * STRIDES[-1], mask, 0).to(tl.float32)
    variance = tl.sum(value * value, 0) / SHAPE[-1]
    normalized = value * tl.rsqrt(variance + EPS)
    weight = tl.load(Weight + cols * WEIGHT_STRIDE, mask, 0).to(tl.float32)
    weighted = weight * normalized
    gamma = tl.load(
        Gamma + (batch % GAMMA_BATCH) * GAMMA_STRIDES[0] + cols * GAMMA_STRIDES[1],
        mask,
        0,
    ).to(tl.float32)
    beta = tl.load(
        Beta + (batch % BETA_BATCH) * BETA_STRIDES[0] + cols * BETA_STRIDES[1],
        mask,
        0,
    ).to(tl.float32)
    result = (1.0 + gamma) * weighted + beta
    tl.store(Out + row * SHAPE[-1] + cols, result, mask)


def lingbot_adarmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted FP32 RMSNorm and FiLM for state or action tokens.

    ``x`` is ``(B, H)`` or ``(B, S, H)``; gamma and beta are ``(B, H)``
    or ``(1, H)``. Only the final result is rounded to the input dtype.
    """
    supported = (torch.bfloat16, torch.float16, torch.float32)
    if (
        not x.is_cuda
        or x.dtype not in supported
        or x.ndim not in (2, 3)
        or x.shape[-1] == 0
    ):
        raise ValueError("LingBot AdaRMSNorm requires 2-D or 3-D floating CUDA input")
    if weight.shape != (x.shape[-1],) or any(
        parameter.ndim != 2
        or parameter.shape[0] not in (1, x.shape[0])
        or parameter.shape[1] != x.shape[-1]
        for parameter in (gamma, beta)
    ):
        raise ValueError("weight must have shape (H,) and modulation (B, H) or (1, H)")
    if any(
        parameter.device != x.device or parameter.dtype not in supported
        for parameter in (weight, gamma, beta)
    ):
        raise ValueError("affine tensors must use floating dtypes on the input device")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = x.numel() // x.shape[-1]
    if rows:
        block = triton.next_power_of_2(x.shape[-1])
        _lingbot_adarmsnorm_kernel[(rows,)](
            x,
            weight,
            gamma,
            beta,
            out,
            SHAPE=tuple(x.shape),
            STRIDES=tuple(x.stride()),
            WEIGHT_STRIDE=weight.stride(0),
            GAMMA_STRIDES=tuple(gamma.stride()),
            BETA_STRIDES=tuple(beta.stride()),
            GAMMA_BATCH=gamma.shape[0],
            BETA_BATCH=beta.shape[0],
            ROWS_PER_BATCH=1 if x.ndim == 2 else x.shape[1],
            EPS=eps,
            BLOCK=block,
            num_warps=4 if block <= 1024 else 8,
            enable_fp_fusion=False,
        )
    return out
