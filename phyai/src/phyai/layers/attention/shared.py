"""Fact derivation shared by the four attention stacks.

The stacks differ in their tensor contracts — the cacheless one is the only one
supporting a rectangular ``S_q != S_kv``, the paged pair scatter K/V
themselves, and GDN takes eight tensors — but they derive their *selection
facts* the same way. Six call sites used to re-implement this, two of them with
a different expression for the 4-D token count.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


def token_count(tensor: torch.Tensor) -> int:
    """Tokens in a 3-D ragged or 4-D padded attention tensor."""

    if tensor.ndim == 4:
        return int(tensor.shape[0] * tensor.shape[1])
    return int(tensor.shape[0])


def attention_shape(
    layer: object, q: torch.Tensor, k: torch.Tensor | None = None, **extra: int
) -> dict[str, Any]:
    """Shape facts for an attention call, read off the layer and the tensors."""

    tokens = token_count(q)
    facts: dict[str, Any] = {"tokens": tokens}
    if k is not None:
        facts["kv_tokens"] = token_count(k)
    for source, target in (
        ("num_heads", "heads"),
        ("num_query_heads", "heads"),
        ("num_kv_heads", "kv_heads"),
        ("num_key_heads", "key_heads"),
        ("num_value_heads", "value_heads"),
        ("num_state_heads", "state_heads"),
        ("head_dim", "head_dim"),
    ):
        value = getattr(layer, source, None)
        if value is not None and target not in facts:
            facts[target] = int(value)
    facts.update(extra)
    return facts


def attention_dtypes(
    q: torch.Tensor,
    k: torch.Tensor | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Dtype roles for an attention call.

    ``key`` and ``value`` default to ``q``'s dtype so a caller that only has
    one tensor to hand still produces a complete set — the FlashInfer contracts
    constrain all three and would otherwise report them as "not provided".
    """

    dtypes: dict[str, object] = {"input": q.dtype, "output": q.dtype}
    if k is not None:
        dtypes["key"] = k.dtype
        dtypes["value"] = k.dtype
    if extra:
        dtypes.update(extra)
    dtypes.setdefault("key", q.dtype)
    dtypes.setdefault("value", q.dtype)
    return dtypes


def select_paged_backend(
    op: str,
    layer: object,
    runner: object,
    *,
    device: torch.device,
    params_dtype: torch.dtype,
    num_tokens: int,
    capture: bool,
    runner_tag: str,
):
    """Select and construct a paged attention backend for a runner.

    Paged backends own runner-scoped buffers — a FlashInfer wrapper, a
    workspace, static indptr arrays — so they are constructed *with* the runner
    at the runner's own lifecycle boundary, not at selection time. That is
    declared on the operation as ``returns_instance(constructed_with=("runner",))``
    rather than being a fourth ad-hoc calling convention.

    ``capture`` picks the mode fact, so a runner that will capture a CUDA graph
    only ever sees capture-safe implementations.
    """

    from phyai.kernel.call import select

    selection = select(
        op,
        role=getattr(layer, "kernel_role", op.removeprefix("attention_")),
        device=device,
        dtype={"input": params_dtype, "output": params_dtype},
        shape=_paged_shape(layer, num_tokens),
        attrs={
            "layout": "paged",
            "causal": bool(getattr(layer, "causal", True)),
            "layer_id": getattr(layer, "layer_id", None),
            "runner": runner_tag,
            "batch_size": getattr(runner, "batch_size", None),
        },
        mode="capture" if capture else "eager",
        prefer=getattr(layer, "prefer", ()),
    )
    return selection.implementation(runner)


def _paged_shape(layer: object, num_tokens: int) -> dict[str, Any]:
    facts: dict[str, Any] = {"tokens": num_tokens}
    for source, target in (
        ("num_heads", "heads"),
        ("num_kv_heads", "kv_heads"),
        ("head_dim", "head_dim"),
    ):
        value = getattr(layer, source, None)
        if value is not None:
            facts[target] = int(value)
    return facts


__all__ = [
    "attention_dtypes",
    "attention_shape",
    "select_paged_backend",
    "token_count",
]
