"""Triton fused depthwise causal Conv1d + SiLU + three-way split."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_silu_split_kernel(
    x_ptr,
    weight_ptr,
    first_ptr,
    second_ptr,
    third_ptr,
    x_batch_stride,
    x_token_stride,
    x_channel_stride,
    weight_channel_stride,
    weight_kernel_stride,
    seq_len,
    num_channels,
    FIRST_SIZE: tl.constexpr,
    SECOND_SIZE: tl.constexpr,
    THIRD_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    channel_block = tl.program_id(1)
    batch = token // seq_len
    time = token - batch * seq_len
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < num_channels

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for kernel_idx in range(KERNEL_SIZE):
        source_time = time + kernel_idx - (KERNEL_SIZE - 1)
        source_mask = channel_mask & (source_time >= 0)
        source = tl.load(
            x_ptr
            + batch * x_batch_stride
            + source_time * x_token_stride
            + channels * x_channel_stride,
            mask=source_mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr
            + channels * weight_channel_stride
            + kernel_idx * weight_kernel_stride,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        acc += source * weight

    # Match Conv1d(BF16) -> SiLU: round the convolution output before SiLU.
    conv = acc.to(x_ptr.dtype.element_ty).to(tl.float32)
    output = conv * tl.sigmoid(conv)

    first_channels = channels
    first_mask = channel_mask & (first_channels < FIRST_SIZE)
    tl.store(
        first_ptr + token * FIRST_SIZE + first_channels,
        output.to(first_ptr.dtype.element_ty),
        mask=first_mask,
    )

    second_channels = channels - FIRST_SIZE
    second_mask = (
        channel_mask & (second_channels >= 0) & (second_channels < SECOND_SIZE)
    )
    tl.store(
        second_ptr + token * SECOND_SIZE + second_channels,
        output.to(second_ptr.dtype.element_ty),
        mask=second_mask,
    )

    third_channels = channels - FIRST_SIZE - SECOND_SIZE
    third_mask = channel_mask & (third_channels >= 0) & (third_channels < THIRD_SIZE)
    tl.store(
        third_ptr + token * THIRD_SIZE + third_channels,
        output.to(third_ptr.dtype.element_ty),
        mask=third_mask,
    )


def causal_conv1d_silu_split_qkv(
    x: torch.Tensor,
    weight: torch.Tensor,
    split_sizes: tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run depthwise causal Conv1d, SiLU, and split the channel output.

    ``x`` has shape ``(B, S, C)`` and may have a padded token stride. ``weight``
    uses the PyTorch depthwise Conv1d layout ``(C, 1, K)``. The result matches
    ``F.conv1d(x.transpose(1, 2), padding=K-1, groups=C)[:, :, :S]`` followed
    by SiLU and a three-way split, but returns three contiguous ``(B, S, C_i)``
    tensors directly.
    """

    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError(
            "phyai_kernel.triton.causal_conv1d_silu_split_qkv: tensors must "
            "live on CUDA"
        )
    if x.ndim != 3:
        raise RuntimeError(
            "phyai_kernel.triton.causal_conv1d_silu_split_qkv: x must be 3D"
        )
    if weight.ndim != 3 or weight.shape[1] != 1:
        raise RuntimeError(
            "phyai_kernel.triton.causal_conv1d_silu_split_qkv: weight must "
            "have shape (C, 1, K)"
        )
    if x.stride(-1) != 1 or weight.stride(-1) != 1:
        raise RuntimeError(
            "phyai_kernel.triton.causal_conv1d_silu_split_qkv: channel and "
            "kernel dimensions must be contiguous"
        )
    if x.shape[-1] != weight.shape[0]:
        raise RuntimeError(
            "phyai_kernel.triton.causal_conv1d_silu_split_qkv: input and "
            "weight channel counts must match"
        )
    if x.dtype != weight.dtype:
        raise RuntimeError(
            "phyai_kernel.triton.causal_conv1d_silu_split_qkv: input and "
            "weight dtypes must match"
        )
    if len(split_sizes) != 3 or any(size <= 0 for size in split_sizes):
        raise ValueError(
            f"split_sizes must contain three positive values, got {split_sizes}."
        )
    if sum(split_sizes) != x.shape[-1]:
        raise ValueError(
            f"split_sizes sum to {sum(split_sizes)} but x has {x.shape[-1]} channels."
        )
    kernel_size = int(weight.shape[-1])
    if not 1 <= kernel_size <= 8:
        raise ValueError(f"kernel size must be in [1, 8], got {kernel_size}.")

    batch_size, seq_len, num_channels = x.shape
    outputs = tuple(
        torch.empty(
            batch_size,
            seq_len,
            size,
            dtype=x.dtype,
            device=x.device,
        )
        for size in split_sizes
    )
    if x.numel() == 0:
        return outputs

    block_c = 256
    grid = (batch_size * seq_len, triton.cdiv(num_channels, block_c))
    _causal_conv1d_silu_split_kernel[grid](
        x,
        weight,
        *outputs,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(2),
        seq_len,
        num_channels,
        FIRST_SIZE=split_sizes[0],
        SECOND_SIZE=split_sizes[1],
        THIRD_SIZE=split_sizes[2],
        KERNEL_SIZE=kernel_size,
        BLOCK_C=block_c,
        num_warps=4,
    )
    return outputs


__all__ = ["causal_conv1d_silu_split_qkv"]
