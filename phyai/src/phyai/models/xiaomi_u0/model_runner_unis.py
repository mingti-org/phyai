"""Single-sequence UNIS runner with externally planned paged attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.cache.kv_cache_pool import KVCachePool
from phyai.layers.attention.enums import AttnLayout, AttnMode
from phyai.layers.attention.paged.base import (
    PagedAttentionBackend,
    PagedAttnCtx,
    PagedAttnMetadata,
    PagedAttnPlanHandle,
)
from phyai.layers.attention.shared import select_paged_backend
from phyai.models.xiaomi_u0.modeling_unis_ar import UNISForCausalLM
from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class UNISForwardBatch:
    """One unpadded sequence with explicit absolute positions."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor


class UNISModelRunner(ModelRunner):
    """Own the KV pool and backend; consume scheduler-supplied metadata."""

    def __init__(
        self,
        model: UNISForCausalLM,
        *,
        capacity: int,
        paged_backend: str = "flashinfer",
    ) -> None:
        if capacity <= 0 or capacity > model.model.max_position_embeddings:
            raise ValueError("capacity must be within the configured position range.")
        if paged_backend not in {"flashinfer", "torch_reference"}:
            raise ValueError("paged_backend must be flashinfer or torch_reference.")
        self.paged_backend = paged_backend
        self.model = model
        self.capacity = capacity
        weight = model.model.embed_tokens.weight
        self.device = weight.device
        self.params_dtype = weight.dtype
        self.kv_pool: KVCachePool | None = None
        self.attn_backend: PagedAttentionBackend | None = None
        self._meta: PagedAttnMetadata | None = None
        self._plan: PagedAttnPlanHandle | None = None

    def setup(self) -> None:
        if self.kv_pool is not None:
            raise RuntimeError("Runner is already set up.")
        proto = self.model.model.layers[0].self_attn.paged_attention
        pool = KVCachePool(
            num_layers=len(self.model.model.layers),
            num_slots=self.capacity,
            num_kv_heads=proto.num_kv_heads,
            head_dim=proto.head_dim,
            dtype=self.params_dtype,
            device=self.device,
            page_size=1,
        )
        if self.paged_backend == "torch_reference":
            from phyai.models.xiaomi_u0.attention_torch_reference import (
                UNISTorchReferenceBackend,
            )

            self.kv_pool = pool
            self.attn_backend = UNISTorchReferenceBackend()
            return
        backend = select_paged_backend(
            "attention_paged",
            proto,
            self,
            device=self.device,
            params_dtype=self.params_dtype,
            num_tokens=self.capacity,
            capture=False,
            runner_tag="xiaomi_u0",
        )
        # This backend requires static buffer setup even without graph capture.
        backend.init_cuda_graph_state(
            max_batch_size=1,
            max_num_tokens=self.capacity,
            max_paged_kv_indices=self.capacity,
            device=self.device,
            params_dtype=self.params_dtype,
            layer_proto=proto,
        )
        # Seed the wrapper at capacity before planning shorter requests.
        # Planning allocates execution state; it does not execute the model
        # or write any K/V tokens into the pool.
        capacity = self.capacity
        cu = torch.tensor([0, capacity], dtype=torch.int32, device=self.device)
        seed = PagedAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=1,
            num_query_tokens=capacity,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu.clone(),
            seq_lens_kv=torch.tensor([capacity], dtype=torch.int32, device=self.device),
            paged_kv_indptr=cu.clone(),
            paged_kv_indices=torch.arange(
                capacity, dtype=torch.int32, device=self.device
            ),
            paged_kv_last_page_len=torch.ones(1, dtype=torch.int32, device=self.device),
            write_indices=torch.arange(capacity, dtype=torch.int64, device=self.device),
            position_ids=torch.arange(capacity, dtype=torch.int64, device=self.device),
        )
        backend.init_forward_metadata(seed)
        self.kv_pool = pool
        self.attn_backend = backend

    def plan_inference(self, meta: PagedAttnMetadata) -> None:
        if self.attn_backend is None:
            raise RuntimeError("Call setup before planning.")
        if meta.batch_size != 1 or meta.layout != AttnLayout.RAGGED_3D:
            raise ValueError("Initial UNIS runner requires one ragged sequence.")
        if not 0 < meta.num_query_tokens <= self.capacity:
            raise ValueError("Invalid query token count.")
        if meta.write_indices is None or meta.position_ids is None:
            raise ValueError("Explicit cache slots and positions are required.")
        if meta.write_indices.shape != (meta.num_query_tokens,):
            raise ValueError("Cache write slots must match query token count.")
        if meta.write_indices.dtype != torch.int64:
            raise ValueError("Cache write slots must be int64.")
        if meta.write_indices.device != self.device:
            raise ValueError("Cache write slots must be on the runner device.")
        if meta.paged_kv_indices is None:
            raise ValueError("Visible cache slots are required.")
        if meta.paged_kv_indices.numel() > self.capacity:
            raise ValueError("Visible cache span exceeds capacity.")
        self._plan = self.attn_backend.init_forward_metadata(meta)
        self._meta = meta

    def forward(self, batch: UNISForwardBatch) -> torch.Tensor:
        if self._meta is None or self._plan is None:
            raise RuntimeError("Plan this step before forward.")
        if self.kv_pool is None or self.attn_backend is None:
            raise RuntimeError("Runner is not set up.")
        meta = self._meta
        if batch.input_ids.shape != (1, meta.num_query_tokens):
            raise ValueError("Input shape does not match the planned step.")
        if batch.input_ids.device != self.device:
            raise ValueError("Input IDs must be on the runner device.")
        if batch.position_ids.shape != batch.input_ids.shape:
            raise ValueError("Positions must match the input shape.")
        if not torch.equal(batch.position_ids.reshape(-1), meta.position_ids):
            raise ValueError("Input positions differ from the planned positions.")
        ctx = PagedAttnCtx(
            backend=self.attn_backend,
            plan=self._plan,
            mode=meta.mode,
            layout=meta.layout,
            kv_pool=self.kv_pool,
            write_indices=meta.write_indices,
            cu_seqlens_q=meta.cu_seqlens_q,
            cu_seqlens_kv=meta.cu_seqlens_kv,
        )
        # Consume the plan so a second forward cannot silently reuse stale metadata.
        self._meta = None
        self._plan = None
        return self.model(batch.input_ids, batch.position_ids, ctx)

    def reset(self) -> None:
        """Clear request state while retaining pool and backend allocations."""
        if self.kv_pool is None:
            raise RuntimeError("Runner is not set up.")
        self.kv_pool.zero_()
        self._meta = None
        self._plan = None

    def close(self) -> None:
        self._meta = None
        self._plan = None
        self.attn_backend = None
        self.kv_pool = None
