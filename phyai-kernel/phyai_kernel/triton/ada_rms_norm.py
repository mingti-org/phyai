# The AdaRMS variant (with conditional ``(scale, shift, gate)`` modulation
# replacing the standard ``(1 + w)`` Gemma RMS affine) follows the
# ``PiGemmaRMSNorm`` design from lerobot
# (https://github.com/huggingface/lerobot, ``src/lerobot/policies/pi_gemma.py``)
# and the openpi ``gemma_pytorch`` action-expert reference, both Apache-2.0.
"""Triton AdaRMSNorm kernel for the pi0.5 / pi0.6 action-expert family.

One public entry point:

* :func:`adarmsnorm` — Adaptive RMSNorm with conditional ``(scale, shift, gate)``
  modulation. Computes::

      normed = x * rsqrt(mean(x ** 2) + eps)
      scale, shift, gate = chunk(modulation, 3, dim=-1)
      out  = (normed * (1 + scale) + shift).to(x.dtype)
      gate = gate.to(x.dtype)

  This is the AdaRMS variant used by lerobot ``PiGemmaRMSNorm`` / openpi
  ``gemma_pytorch`` action expert when ``use_adarms=True``. The
  ``(1 + weight)`` term of the standard Gemma RMSNorm is *replaced* by
  ``(1 + scale)`` from the conditioning projection — there is no
  ``self.weight`` in the AdaRMS path. The caller is responsible for the
  ``modulation = dense(cond)`` projection (a regular ``nn.Linear``); this
  kernel only fuses the post-projection RMSNorm + affine + gate-cast.

Shape contract::

    x          : (..., D)                      — activations
    modulation : (..., 3 * D)                  — already-projected cond
    out        : (..., D)                      — same shape & dtype as x
    gate       : (modulation.shape[:-1], D)    — gate vector, dtype = x.dtype

When ``modulation`` has fewer leading rows than ``x`` (e.g. ``x`` is
``(B, S, D)`` and ``modulation`` is ``(B, 3*D)``), each row of
``modulation`` is broadcast across ``group_size = N_total // N_mod``
consecutive rows of ``x``. The gate buffer is written *once per modulation
row* — only the program with ``row % group_size == 0`` performs the cast
+ store, so the gate output shape mirrors the input modulation shape.

Single-pass for ``D <= _SINGLE_BLOCK_MAX``, two-pass otherwise. Reductions
and the affine run in fp32 to match the lerobot/openpi reference; output
is cast back to ``x.dtype``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Largest single-block row width. Same budget as rms_norm.py / layer_norm.py.
_SINGLE_BLOCK_MAX = 8192


# --------------------------------------------------------------------------- #
# Single-block kernel (one row per program; D fits a single tile)             #
# --------------------------------------------------------------------------- #


@triton.jit
def _adarmsnorm_fwd_kernel(
    x_ptr,
    mod_ptr,
    out_ptr,
    gate_ptr,
    x_row_stride,
    mod_row_stride,
    out_row_stride,
    gate_row_stride,
    n_cols,
    eps,
    group_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    mod_row = row // group_size
    in_group_idx = row - mod_row * group_size  # cheaper than `row % group_size`
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # 1. Load x, RMS reduction in fp32.
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)
    x_hat = x * rstd

    # 2. Load (scale, shift) from the matching modulation row in fp32.
    scale = tl.load(mod_ptr + mod_row * mod_row_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(
        mod_ptr + mod_row * mod_row_stride + n_cols + cols, mask=mask, other=0.0
    ).to(tl.float32)

    # 3. Affine transform; cast back to output dtype.
    out = x_hat * (1.0 + scale) + shift
    tl.store(
        out_ptr + row * out_row_stride + cols,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )

    # 4. Gate write — once per modulation row. Triton resolves this scalar
    # conditional at JIT time, so non-first programs skip the load+store.
    if in_group_idx == 0:
        gate = tl.load(
            mod_ptr + mod_row * mod_row_stride + 2 * n_cols + cols,
            mask=mask,
            other=0.0,
        )
        tl.store(
            gate_ptr + mod_row * gate_row_stride + cols,
            gate.to(gate_ptr.dtype.element_ty),
            mask=mask,
        )


# --------------------------------------------------------------------------- #
# Two-pass kernel for very large hidden_size (D > _SINGLE_BLOCK_MAX)          #
# --------------------------------------------------------------------------- #


@triton.jit
def _adarmsnorm_fwd_kernel_large(
    x_ptr,
    mod_ptr,
    out_ptr,
    gate_ptr,
    x_row_stride,
    mod_row_stride,
    out_row_stride,
    gate_row_stride,
    n_cols,
    eps,
    group_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    mod_row = row // group_size
    in_group_idx = row - mod_row * group_size

    # Pass 1: sum-of-squares in fp32.
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

    is_first = in_group_idx == 0

    # Pass 2: normalize, affine, store; copy gate on the first-in-group row.
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        x_hat = x * rstd
        scale = tl.load(
            mod_ptr + mod_row * mod_row_stride + cols, mask=mask, other=0.0
        ).to(tl.float32)
        shift = tl.load(
            mod_ptr + mod_row * mod_row_stride + n_cols + cols, mask=mask, other=0.0
        ).to(tl.float32)
        out = x_hat * (1.0 + scale) + shift
        tl.store(
            out_ptr + row * out_row_stride + cols,
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
        if is_first:
            gate = tl.load(
                mod_ptr + mod_row * mod_row_stride + 2 * n_cols + cols,
                mask=mask,
                other=0.0,
            )
            tl.store(
                gate_ptr + mod_row * gate_row_stride + cols,
                gate.to(gate_ptr.dtype.element_ty),
                mask=mask,
            )


# --------------------------------------------------------------------------- #
# Python entry point                                                          #
# --------------------------------------------------------------------------- #


def _check_inputs(x: torch.Tensor, modulation: torch.Tensor) -> None:
    if not x.is_cuda or not modulation.is_cuda:
        raise RuntimeError("phyai_kernel.triton.adarmsnorm: tensors must live on CUDA")
    if x.shape[-1] * 3 != modulation.shape[-1]:
        raise RuntimeError(
            f"phyai_kernel.triton.adarmsnorm: modulation last dim "
            f"({modulation.shape[-1]}) must be 3x x last dim ({x.shape[-1]})"
        )


def _flatten(x: torch.Tensor, last_dim: int) -> torch.Tensor:
    """Reshape ``(..., last_dim) -> (-1, last_dim)`` and contiguousify if needed."""
    flat = x.reshape(-1, last_dim)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    return flat


def adarmsnorm(
    x: torch.Tensor,
    modulation: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
    gate_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adaptive RMSNorm with conditional ``(scale, shift, gate)`` modulation.

    Parameters
    ----------
    x:
        Input activations of shape ``(..., D)``.
    modulation:
        Already-projected conditioning tensor, shape ``(..., 3 * D)``. The
        leading dims must be a *prefix* of ``x``'s leading dims — i.e.
        ``prod(x.shape[:-1])`` must be divisible by ``prod(modulation.shape[:-1])``.
        Each modulation row is broadcast across the matching block of ``x``
        rows (e.g. ``x=(B, S, D)`` with ``modulation=(B, 3*D)`` broadcasts
        each modulation over ``S``).
    eps:
        Numerical-stability epsilon for the variance reduction.
    out:
        Optional pre-allocated output buffer; must match ``x.shape`` and
        ``x.dtype``.
    gate_out:
        Optional pre-allocated gate buffer; must match
        ``modulation.shape[:-1] + (D,)`` and ``x.dtype``.

    Returns
    -------
    ``(out, gate)``.

    Notes
    -----
    The ``(1 + weight)`` term of standard Gemma RMSNorm is *replaced* by
    ``(1 + scale)``; there is no learned ``weight`` parameter in this
    kernel. Use :func:`gemma_rmsnorm` for the no-cond fallback.
    """
    _check_inputs(x, modulation)
    n_cols = x.shape[-1]

    x2d = _flatten(x, n_cols)
    mod2d = _flatten(modulation, 3 * n_cols)
    n_total = x2d.shape[0]
    n_mod = mod2d.shape[0]

    # Output buffers (allocated up-front so the empty-input fast path returns
    # well-shaped tensors instead of needing the caller to handle ``None``).
    if out is None:
        out_t = torch.empty_like(x)
    else:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise RuntimeError(
                "phyai_kernel.triton.adarmsnorm: `out` must match input shape and dtype"
            )
        out_t = out
    out2d = out_t.reshape(-1, n_cols)

    gate_shape = modulation.shape[:-1] + (n_cols,)
    if gate_out is None:
        gate_t = torch.empty(gate_shape, dtype=x.dtype, device=x.device)
    else:
        if tuple(gate_out.shape) != tuple(gate_shape) or gate_out.dtype != x.dtype:
            raise RuntimeError(
                f"phyai_kernel.triton.adarmsnorm: `gate_out` must have shape "
                f"{tuple(gate_shape)} and dtype {x.dtype}, got "
                f"{tuple(gate_out.shape)} / {gate_out.dtype}"
            )
        gate_t = gate_out
    gate2d = gate_t.reshape(-1, n_cols)

    if n_total == 0:
        gate_t.copy_(modulation[..., 2 * n_cols :])
        return out_t, gate_t
    if n_mod == 0 or n_total % n_mod != 0:
        raise RuntimeError(
            f"phyai_kernel.triton.adarmsnorm: x rows ({n_total}) must be a "
            f"non-zero multiple of modulation rows ({n_mod})"
        )
    group_size = n_total // n_mod

    if n_cols <= _SINGLE_BLOCK_MAX:
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_size <= 1024 else (8 if block_size <= 4096 else 16)
        _adarmsnorm_fwd_kernel[(n_total,)](
            x2d,
            mod2d,
            out2d,
            gate2d,
            x2d.stride(0),
            mod2d.stride(0),
            out2d.stride(0),
            gate2d.stride(0),
            n_cols,
            eps,
            group_size,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    else:
        block_size = 4096
        _adarmsnorm_fwd_kernel_large[(n_total,)](
            x2d,
            mod2d,
            out2d,
            gate2d,
            x2d.stride(0),
            mod2d.stride(0),
            out2d.stride(0),
            gate2d.stride(0),
            n_cols,
            eps,
            group_size,
            BLOCK_SIZE=block_size,
            num_warps=16,
        )
    return out_t, gate_t
