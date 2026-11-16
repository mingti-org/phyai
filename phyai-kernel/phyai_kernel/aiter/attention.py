"""aiter paged-KV attention (CK batch prefill) over a FlashInfer-style page table."""

from __future__ import annotations

import torch


def batch_prefill_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Attend ragged ``q (tokens, H, D)`` to the paged KV cache.

    ``max_seqlen_q`` / ``max_seqlen_kv`` are host integers that a captured
    CUDA graph freezes into the launch, so callers that replay with longer
    sequences pass static upper bounds; real lengths must stay within them.
    With page size 1, the number of KV page indices bounds the KV tokens.
    """

    from aiter import mha_batch_prefill_func

    return mha_batch_prefill_func(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        kv_indptr,
        kv_indices,
        max_seqlen_q,
        max_seqlen_kv,
        softmax_scale=softmax_scale,
        causal=causal,
        kv_last_page_lens=kv_last_page_lens,
    )


__all__ = ["batch_prefill_paged"]
