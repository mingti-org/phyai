"""Triton gather kernel: flatten a paged KV page-table for flashinfer.

flashinfer's paged wrappers address the KV cache through a triple
``(paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len)`` where
``paged_kv_indices`` is a *flat* array of physical slot ids and
``paged_kv_indptr`` gives each request's start/stop offset into that
flat array.

When KV slots are allocated dynamically (paging, radix-prefix sharing),
a single request's tokens live in non-contiguous physical slots, so the
flat ``paged_kv_indices`` cannot be produced by a plain ``arange``. The
mapping ``(request, logical position) -> physical slot`` is held in a
``req_to_token`` table of shape ``[max_batch, max_context_len]``; this
kernel gathers, for each request ``b``, the slice
``req_to_token[req_pool_indices[b], kv_start[b] : kv_start[b] + len[b]]``
into ``kv_indices[kv_indptr[b] : kv_indptr[b + 1]]`` in one launch with
no host synchronisation.

``kv_start_idx`` is the enabler for ``S_q != S_kv``: it selects an
arbitrary per-request KV sub-window — skip an already-computed prefix
(chunked prefill / extend), pick a sliding window, or address the
encoder segment for cross-attention.

This is the dynamic / non-contiguous counterpart to the pure-PyTorch
``build_joint_paged_kv_indices`` in the pi0.5 scheduler, which suffices
only because pi0.5's slot layout is a static contiguous slab.

Attribution
-----------
The Triton gather kernel below is adapted from SGLang's
``create_flashinfer_kv_indices_triton``:
    https://github.com/sgl-project/sglang
    (python/sglang/srt/layers/attention/utils.py)

    Copyright 2023-2024 SGLang Team
    Licensed under the Apache License, Version 2.0 (the "License").
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0

Modified by the phyai team: renamed to ``create_paged_kv_indices``,
re-typed the Python wrapper with explicit input validation, and adjusted
comments to phyai conventions. See the repository-root ``NOTICE`` file.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _create_paged_kv_indices_kernel(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr_ptr,
    kv_start_idx_ptr,
    kv_indices_ptr,
    req_to_token_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # One program per request: locate its row in req_to_token and its
    # write offset into the flat kv_indices output.
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr_ptr + pid)

    # kv_start_idx selects a per-request sub-window of the KV span. When
    # the pointer is null (kv_start == 0), the full [0, len) span is used.
    kv_start = 0
    kv_end = 0
    if kv_start_idx_ptr:
        kv_start = tl.load(kv_start_idx_ptr + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # Index into req_to_token must be int64 to avoid overflow on
        # large tables.
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr + req_pool_index * req_to_token_stride + kv_start + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


def create_paged_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_kernel_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_start_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather a flat flashinfer ``paged_kv_indices`` page-table.

    For each request ``b`` (``0 <= b < batch``), copies the physical
    slot ids ``req_to_token[req_pool_indices[b],
    kv_start[b] : kv_start[b] + page_kernel_lens[b]]`` into
    ``kv_indices[kv_indptr[b] : kv_indptr[b + 1]]``. Runs entirely on the
    device in a single launch (no host sync), so it is safe on the
    per-step planning hot path.

    Args:
        req_to_token: ``(max_batch, max_context_len)`` int32 table mapping
            ``(request slot, logical position) -> physical KV slot``.
        req_pool_indices: ``(batch,)`` int — each request's row in
            ``req_to_token``.
        page_kernel_lens: ``(batch,)`` int — per-request KV length
            (``S_kv``) to gather.
        kv_indptr: ``(batch + 1,)`` int32 exclusive-scan offsets into
            ``kv_indices`` (``kv_indptr[b]`` is request ``b``'s start).
        kv_indices: pre-allocated ``(total_kv,)`` int32 output buffer,
            ``total_kv == kv_indptr[-1]``. Written in place.
        kv_start_idx: optional ``(batch,)`` int — per-request start offset
            into the KV span (prefix skip / window / encoder segment). When
            ``None``, each request gathers from position 0. This is the
            knob that makes ``S_q != S_kv`` expressible.

    Returns:
        The same ``kv_indices`` tensor, filled in place.
    """
    if not (req_to_token.is_cuda and req_pool_indices.is_cuda and kv_indices.is_cuda):
        raise RuntimeError(
            "phyai_kernel.triton.create_paged_kv_indices: tensors must live on CUDA"
        )
    if req_to_token.dim() != 2:
        raise RuntimeError(
            f"phyai_kernel.triton.create_paged_kv_indices: req_to_token must be 2-D, "
            f"got {req_to_token.dim()}-D"
        )
    batch = req_pool_indices.shape[0]
    if page_kernel_lens.shape[0] != batch:
        raise RuntimeError(
            f"phyai_kernel.triton.create_paged_kv_indices: page_kernel_lens batch "
            f"{page_kernel_lens.shape[0]} != req_pool_indices batch {batch}"
        )
    if kv_indptr.shape[0] != batch + 1:
        raise RuntimeError(
            f"phyai_kernel.triton.create_paged_kv_indices: kv_indptr length "
            f"{kv_indptr.shape[0]} != batch + 1 ({batch + 1})"
        )
    if kv_start_idx is not None and kv_start_idx.shape[0] != batch:
        raise RuntimeError(
            f"phyai_kernel.triton.create_paged_kv_indices: kv_start_idx batch "
            f"{kv_start_idx.shape[0]} != req_pool_indices batch {batch}"
        )
    if batch == 0:
        return kv_indices

    _create_paged_kv_indices_kernel[(batch,)](
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
        req_to_token.shape[1],
    )
    return kv_indices


__all__ = ["create_paged_kv_indices"]
