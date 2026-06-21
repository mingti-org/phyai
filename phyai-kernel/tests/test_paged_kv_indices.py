"""Numerical-equivalence tests for the Triton create_paged_kv_indices kernel.

The kernel gathers, per request ``b``, the slice
``req_to_token[req_pool_indices[b], kv_start[b] : kv_start[b] + len[b]]``
into ``kv_indices[kv_indptr[b] : kv_indptr[b + 1]]``. It is a pure
indexed copy, so we expect bit-exact equality with a plain-PyTorch
reference loop.

Test grid covers:
* contiguous and shuffled (non-contiguous) physical slot layouts
* with and without a per-request kv_start_idx (prefix-skip / window)
* uneven per-request lengths and an empty request mixed in
* an empty batch edge case
"""

from __future__ import annotations

import pytest
import torch

import phyai_kernel

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for phyai-kernel Triton tests", allow_module_level=True
    )


# --------------------------------------------------------------------------- #
# Reference                                                                   #
# --------------------------------------------------------------------------- #


def _ref_paged_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    page_kernel_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_start_idx: torch.Tensor | None,
) -> torch.Tensor:
    """Plain per-request gather loop on CPU."""
    out = []
    for b in range(req_pool_indices.shape[0]):
        row = int(req_pool_indices[b])
        start = int(kv_start_idx[b]) if kv_start_idx is not None else 0
        length = int(page_kernel_lens[b])
        out.append(req_to_token[row, start : start + length])
    if not out:
        return torch.empty(0, dtype=req_to_token.dtype)
    return torch.cat(out)


def _run(req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx):
    """Drive the kernel; return (kernel_out, reference_out) on CPU."""
    batch = req_pool_indices.shape[0]
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(page_kernel_lens, dim=0)
    total = int(kv_indptr[-1])

    kv_indices = torch.empty(total, dtype=torch.int32, device="cuda")
    phyai_kernel.create_paged_kv_indices(
        req_to_token.cuda(),
        req_pool_indices.cuda(),
        page_kernel_lens.cuda(),
        kv_indptr.cuda(),
        kv_indices,
        kv_start_idx.cuda() if kv_start_idx is not None else None,
    )
    ref = _ref_paged_kv_indices(
        req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_start_idx
    )
    return kv_indices.cpu(), ref


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_contiguous_layout_no_start():
    """Contiguous slots, full span from position 0."""
    max_batch, max_ctx = 4, 32
    req_to_token = torch.arange(max_batch * max_ctx, dtype=torch.int32).reshape(
        max_batch, max_ctx
    )
    req_pool_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    page_kernel_lens = torch.tensor([5, 8, 3, 10], dtype=torch.int32)

    out, ref = _run(req_to_token, req_pool_indices, page_kernel_lens, None)
    assert torch.equal(out, ref)


def test_shuffled_noncontiguous_layout():
    """Physical slots are shuffled — the gather must follow req_to_token."""
    torch.manual_seed(0)
    max_batch, max_ctx = 6, 64
    req_to_token = (
        torch.randperm(max_batch * max_ctx, dtype=torch.int64)
        .to(torch.int32)
        .reshape(max_batch, max_ctx)
    )
    req_pool_indices = torch.tensor([5, 0, 3], dtype=torch.int32)
    page_kernel_lens = torch.tensor([12, 40, 7], dtype=torch.int32)

    out, ref = _run(req_to_token, req_pool_indices, page_kernel_lens, None)
    assert torch.equal(out, ref)


def test_with_kv_start_idx_prefix_skip():
    """kv_start_idx selects a per-request sub-window (S_q != S_kv enabler)."""
    torch.manual_seed(1)
    max_batch, max_ctx = 4, 128
    req_to_token = (
        torch.randperm(max_batch * max_ctx, dtype=torch.int64)
        .to(torch.int32)
        .reshape(max_batch, max_ctx)
    )
    req_pool_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    page_kernel_lens = torch.tensor([16, 20, 8, 30], dtype=torch.int32)
    kv_start_idx = torch.tensor([4, 0, 10, 50], dtype=torch.int32)

    out, ref = _run(req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx)
    assert torch.equal(out, ref)


def test_empty_request_in_batch():
    """A zero-length request contributes nothing and does not shift others."""
    max_batch, max_ctx = 3, 16
    req_to_token = torch.arange(max_batch * max_ctx, dtype=torch.int32).reshape(
        max_batch, max_ctx
    )
    req_pool_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    page_kernel_lens = torch.tensor([4, 0, 6], dtype=torch.int32)

    out, ref = _run(req_to_token, req_pool_indices, page_kernel_lens, None)
    assert torch.equal(out, ref)


def test_long_span_spans_multiple_blocks():
    """A span longer than the kernel's 512 BLOCK_SIZE exercises the loop."""
    max_batch, max_ctx = 2, 2048
    req_to_token = torch.arange(max_batch * max_ctx, dtype=torch.int32).reshape(
        max_batch, max_ctx
    )
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int32)
    page_kernel_lens = torch.tensor([1500, 2000], dtype=torch.int32)

    out, ref = _run(req_to_token, req_pool_indices, page_kernel_lens, None)
    assert torch.equal(out, ref)


def test_empty_batch():
    """Empty batch returns the (empty) buffer untouched."""
    req_to_token = torch.arange(16, dtype=torch.int32).reshape(1, 16).cuda()
    req_pool_indices = torch.empty(0, dtype=torch.int32, device="cuda")
    page_kernel_lens = torch.empty(0, dtype=torch.int32, device="cuda")
    kv_indptr = torch.zeros(1, dtype=torch.int32, device="cuda")
    kv_indices = torch.empty(0, dtype=torch.int32, device="cuda")

    out = phyai_kernel.create_paged_kv_indices(
        req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_indices, None
    )
    assert out.shape[0] == 0
