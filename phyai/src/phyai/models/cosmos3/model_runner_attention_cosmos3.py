"""Reusable attention metadata for Cosmos3's dense Generator."""

from __future__ import annotations

import torch
from dataclasses import replace

from phyai.kernel.types import KernelMode
from phyai.kernel.bootstrap import get_kernel_selector
from phyai.layers.attention.enums import AttnLayout
from phyai.layers.attention.nocache.base import AttnCtx
from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3GenDecoderLayer


class Cosmos3AttentionContexts:
    """Keep metadata for one Generator shape, independent of token values."""

    def __init__(self) -> None:
        self.contexts: tuple[AttnCtx, ...] = ()
        self.signature: tuple | None = None
        self.revision = 0

    def reset(self) -> None:
        seen = set()
        for context in self.contexts:
            backend = context.backend
            if id(backend) not in seen and getattr(
                backend, "share_across_layers", False
            ):
                backend.release()
                seen.add(id(backend))
        self.contexts = ()
        self.signature = None
        self.revision += 1

    def prepare(
        self,
        layers: torch.nn.ModuleList,
        *,
        batch_size: int,
        query_tokens: int,
        text_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        use_cuda_graph: bool = False,
    ) -> tuple[AttnCtx, ...]:
        if batch_size != 1:
            raise ValueError("Cached Cosmos3 attention metadata requires batch_size=1")
        mode = KernelMode.CAPTURE if use_cuda_graph else KernelMode.EAGER
        signature = (
            tuple(layers),
            get_kernel_selector(),
            batch_size,
            query_tokens,
            text_tokens,
            dtype,
            device,
            mode,
        )
        if self.signature == signature:
            return self.contexts
        self.reset()
        contexts = []
        shared_backends = {}
        for layer in layers:
            if not isinstance(layer, Cosmos3GenDecoderLayer):
                raise TypeError("Cosmos3 attention metadata requires Generator layers")
            attention = layer.cross_attention
            query = torch.empty(
                batch_size,
                query_tokens,
                attention.num_local_heads,
                attention.head_dim,
                dtype=dtype,
                device=device,
            )
            key = torch.empty(
                batch_size,
                query_tokens + text_tokens,
                attention.num_local_kv_heads,
                attention.head_dim,
                dtype=dtype,
                device=device,
            )
            context = attention.attn._build_default_ctx(
                query, key, key, AttnLayout.PADDED_4D, None, None, mode=mode
            )
            if getattr(context.backend, "share_across_layers", False):
                key = (type(context.backend), context.backend.sharing_key)
                backend = shared_backends.setdefault(key, context.backend)
                context = replace(context, backend=backend)
            contexts.append(context)
        self.contexts = tuple(contexts)
        self.signature = signature
        self.revision += 1
        return self.contexts


__all__ = ["Cosmos3AttentionContexts"]
