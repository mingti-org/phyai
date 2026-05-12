# Portions of the kernel design and reference semantics are adapted from
# SGLang (https://github.com/sgl-project/sglang), Copyright 2023-2024 SGLang
# Team, licensed under the Apache License, Version 2.0:
#     http://www.apache.org/licenses/LICENSE-2.0
"""Triton RMSNorm kernels for the Qwen and Gemma model families.

Five public entry points:

* :func:`rmsnorm`                   — Llama/Qwen-style standard RMSNorm.
* :func:`rmsnorm_hf`                — HF semantics: cast back to ``x.dtype`` BEFORE the
  weight multiply (matches ``transformers.LlamaRMSNorm`` and Qwen3 q/k-norm).
* :func:`fused_add_rmsnorm`         — in-place ``residual += x`` then rmsnorm into ``x``.
* :func:`gemma_rmsnorm`             — Gemma/Gemma3 RMSNorm, weight is ``(1 + w)``.
* :func:`gemma_fused_add_rmsnorm`   — Gemma fused add+norm, in-place.

All kernels are single-pass: each program handles one row, the row is loaded
into registers/SRAM as a contiguous block padded to ``next_power_of_2(D)``.
This matches SGLang's ``RMSNormKernel`` shape regime; for ``D`` larger
than the on-chip block we fall back to a numerically-equivalent two-pass
kernel. Reductions and the multiply are always done in fp32 to match the
reference implementations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Largest single-block row width. 8192 fp32 lanes = 32 KiB per row, well under
# the per-SM SRAM budget on Ampere/Hopper/Blackwell. Beyond this we pivot to a
# multi-block reduction.
_SINGLE_BLOCK_MAX = 8192


# --------------------------------------------------------------------------- #
# Single-block kernels (one row per program; D fits a single tile)            #
# --------------------------------------------------------------------------- #


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    x_row_stride,
    out_row_stride,
    n_cols,
    eps,
    HF_SEMANTICS: tl.constexpr,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    x_hat = x * rstd

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if GEMMA:
        w = w + 1.0

    if HF_SEMANTICS:
        # Cast back to activation dtype before weight multiply (LlamaRMSNorm
        # in HF transformers). The narrow-dtype multiply costs a tiny bit of
        # accuracy but matches HF/Qwen exactly.
        x_hat = x_hat.to(out_ptr.dtype.element_ty)
        out = x_hat * w.to(out_ptr.dtype.element_ty)
    else:
        out = (x_hat * w).to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + row * out_row_stride + cols, out, mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    w_ptr,
    x_row_stride,
    res_row_stride,
    n_cols,
    eps,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + row * res_row_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    summed = x + r
    # Write the post-add tensor back to ``residual`` in its original dtype.
    tl.store(
        residual_ptr + row * res_row_stride + cols,
        summed.to(residual_ptr.dtype.element_ty),
        mask=mask,
    )

    var = tl.sum(summed * summed, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    x_hat = summed * rstd

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    if GEMMA:
        w = w + 1.0

    out = (x_hat * w).to(x_ptr.dtype.element_ty)
    tl.store(x_ptr + row * x_row_stride + cols, out, mask=mask)


# --------------------------------------------------------------------------- #
# Two-pass kernels for very large hidden_size (D > _SINGLE_BLOCK_MAX)         #
# --------------------------------------------------------------------------- #


@triton.jit
def _rmsnorm_fwd_kernel_large(
    x_ptr,
    w_ptr,
    out_ptr,
    x_row_stride,
    out_row_stride,
    n_cols,
    eps,
    HF_SEMANTICS: tl.constexpr,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    # Pass 1: compute sum-of-squares in fp32.
    sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        sq += x * x
    var = tl.sum(sq, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    # Pass 2: normalize and scale.
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        x_hat = x * rstd
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if GEMMA:
            w = w + 1.0
        if HF_SEMANTICS:
            x_hat = x_hat.to(out_ptr.dtype.element_ty)
            out = x_hat * w.to(out_ptr.dtype.element_ty)
        else:
            out = (x_hat * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * out_row_stride + cols, out, mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel_large(
    x_ptr,
    residual_ptr,
    w_ptr,
    x_row_stride,
    res_row_stride,
    n_cols,
    eps,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    # Pass 1: write residual = x + residual, accumulate sum-of-squares.
    sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        r = tl.load(
            residual_ptr + row * res_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        s = x + r
        tl.store(
            residual_ptr + row * res_row_stride + cols,
            s.to(residual_ptr.dtype.element_ty),
            mask=mask,
        )
        sq += s * s
    var = tl.sum(sq, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    # Pass 2: normalize, scale, store back into x.
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        s = tl.load(
            residual_ptr + row * res_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        x_hat = s * rstd
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        if GEMMA:
            w = w + 1.0
        out = (x_hat * w).to(x_ptr.dtype.element_ty)
        tl.store(x_ptr + row * x_row_stride + cols, out, mask=mask)


# --------------------------------------------------------------------------- #
# Python entry points                                                         #
# --------------------------------------------------------------------------- #


def _flatten_input(
    x: torch.Tensor, residual: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, ...]]:
    """Reshape (..., D) tensors into (N, D) and return the original shape.

    Mirrors what sglang's `_forward_impl` does at the Python layer.
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    res2d = None
    if residual is not None:
        res2d = residual.reshape(-1, orig_shape[-1])
        if not res2d.is_contiguous():
            res2d = res2d.contiguous()
    return x2d, res2d, orig_shape


def _check_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("phyai_kernel.triton.rmsnorm: tensors must live on CUDA")
    if weight.dim() != 1:
        raise RuntimeError(
            f"phyai_kernel.triton.rmsnorm: weight must be 1D, got {weight.dim()}D"
        )
    if x.shape[-1] != weight.shape[0]:
        raise RuntimeError(
            f"phyai_kernel.triton.rmsnorm: weight ({weight.shape[0]}) must match "
            f"last dim of input ({x.shape[-1]})"
        )


def _launch_rmsnorm(
    x2d: torch.Tensor,
    weight: torch.Tensor,
    out2d: torch.Tensor,
    eps: float,
    *,
    hf_semantics: bool,
    gemma: bool,
) -> None:
    n_rows, n_cols = x2d.shape
    if n_rows == 0:
        return

    if n_cols <= _SINGLE_BLOCK_MAX:
        block_size = triton.next_power_of_2(n_cols)
        # Constrain occupancy on small rows; large blocks need more warps.
        num_warps = 4 if block_size <= 1024 else (8 if block_size <= 4096 else 16)
        _rmsnorm_fwd_kernel[(n_rows,)](
            x2d,
            weight,
            out2d,
            x2d.stride(0),
            out2d.stride(0),
            n_cols,
            eps,
            HF_SEMANTICS=hf_semantics,
            GEMMA=gemma,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        block_size = 4096
        _rmsnorm_fwd_kernel_large[(n_rows,)](
            x2d,
            weight,
            out2d,
            x2d.stride(0),
            out2d.stride(0),
            n_cols,
            eps,
            HF_SEMANTICS=hf_semantics,
            GEMMA=gemma,
            BLOCK_SIZE=block_size,
            num_warps=16,
        )


def _launch_fused_add(
    x2d: torch.Tensor,
    residual2d: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    gemma: bool,
) -> None:
    n_rows, n_cols = x2d.shape
    if n_rows == 0:
        return

    if n_cols <= _SINGLE_BLOCK_MAX:
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_size <= 1024 else (8 if block_size <= 4096 else 16)
        _fused_add_rmsnorm_kernel[(n_rows,)](
            x2d,
            residual2d,
            weight,
            x2d.stride(0),
            residual2d.stride(0),
            n_cols,
            eps,
            GEMMA=gemma,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        block_size = 4096
        _fused_add_rmsnorm_kernel_large[(n_rows,)](
            x2d,
            residual2d,
            weight,
            x2d.stride(0),
            residual2d.stride(0),
            n_cols,
            eps,
            GEMMA=gemma,
            BLOCK_SIZE=block_size,
            num_warps=16,
        )


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standard Llama/Qwen RMSNorm.

    ``out = (rsqrt(mean(x^2) + eps) * x * weight).to(x.dtype)`` with the full
    reduction and weight multiply done in fp32.
    """
    _check_inputs(x, weight)
    x2d, _, orig_shape = _flatten_input(x)
    if out is None:
        out_t = torch.empty_like(x)
    else:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise RuntimeError(
                "phyai_kernel.triton.rmsnorm: `out` must match input shape and dtype"
            )
        out_t = out
    out2d = out_t.reshape(-1, orig_shape[-1])
    _launch_rmsnorm(x2d, weight, out2d, eps, hf_semantics=False, gemma=False)
    return out_t


def rmsnorm_hf(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RMSNorm with HuggingFace ``LlamaRMSNorm`` semantics.

    Differs from :func:`rmsnorm` only in that the cast back to ``x.dtype``
    happens BEFORE the weight multiply, matching HF transformers and the
    Qwen3 q/k norm path. Used by ``rmsnorm_hf`` in sglang's jit_kernel.
    """
    _check_inputs(x, weight)
    x2d, _, orig_shape = _flatten_input(x)
    if out is None:
        out_t = torch.empty_like(x)
    else:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise RuntimeError(
                "phyai_kernel.triton.rmsnorm_hf: `out` must match input shape/dtype"
            )
        out_t = out
    out2d = out_t.reshape(-1, orig_shape[-1])
    _launch_rmsnorm(x2d, weight, out2d, eps, hf_semantics=True, gemma=False)
    return out_t


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In-place ``residual += x`` then write ``rmsnorm(residual)`` into ``x``.

    Returns the same tensors that were passed in, so callers may use either
    the return value or rely on the in-place update (matches sglang's
    ``fused_add_rmsnorm`` contract).
    """
    _check_inputs(x, weight)
    if not x.is_contiguous() or not residual.is_contiguous():
        raise RuntimeError(
            "phyai_kernel.triton.fused_add_rmsnorm: x and residual must be contiguous for in-place update"
        )
    if x.shape != residual.shape:
        raise RuntimeError(
            "phyai_kernel.triton.fused_add_rmsnorm: x and residual shapes must match"
        )
    if x.dtype != residual.dtype:
        raise RuntimeError(
            "phyai_kernel.triton.fused_add_rmsnorm: x and residual dtypes must match"
        )
    x2d, res2d, _ = _flatten_input(x, residual)
    _launch_fused_add(x2d, res2d, weight, eps, gemma=False)
    return x, residual


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gemma/Gemma3 RMSNorm: scales by ``(1 + weight)`` in fp32.

    Shared between Gemma2's ``GemmaRMSNorm`` and Gemma3's ``Gemma3RMSNorm``;
    both are mathematically identical (cast back to dtype after the multiply).
    """
    _check_inputs(x, weight)
    x2d, _, orig_shape = _flatten_input(x)
    if out is None:
        out_t = torch.empty_like(x)
    else:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise RuntimeError(
                "phyai_kernel.triton.gemma_rmsnorm: `out` must match input shape/dtype"
            )
        out_t = out
    out2d = out_t.reshape(-1, orig_shape[-1])
    _launch_rmsnorm(x2d, weight, out2d, eps, hf_semantics=False, gemma=True)
    return out_t


def gemma_fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gemma fused add+RMSNorm (in-place). Scale is ``(1 + weight)``."""
    _check_inputs(x, weight)
    if x.shape != residual.shape:
        raise RuntimeError(
            "phyai_kernel.triton.gemma_fused_add_rmsnorm: shapes must match"
        )
    if x.dtype != residual.dtype:
        raise RuntimeError(
            "phyai_kernel.triton.gemma_fused_add_rmsnorm: dtypes must match"
        )
    x2d, res2d, _ = _flatten_input(x, residual)
    _launch_fused_add(x2d, res2d, weight, eps, gemma=True)
    return x, residual
