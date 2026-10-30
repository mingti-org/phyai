"""Opt-in SM110 Expert backend with explicit graph variants."""

import torch

from phyai.kernel.execution import current_execution

from phyai.layers.attention.paged.backends.flashinfer import FlashInferPagedBackend
from phyai.layers.attention.paged.base import PagedAttnMetadata


class SM110ExpertBackend(FlashInferPagedBackend):
    name = "phyai_kernel"

    def __init__(self, runner=None, *, prefill_backend=None):
        super().__init__(runner, prefill_backend=prefill_backend)
        self.variant = "flashinfer"
        self.kernel = None

    def graph_capture_keys(self, *, rollout=False):
        if rollout:
            return ("flashinfer",)
        return ("c27", "flashinfer")

    def set_graph_capture_key(self, key):
        if key not in self.graph_capture_keys():
            raise ValueError(f"Unknown SM110 Expert graph variant {key!r}")
        self.variant = key

    def graph_replay_key(self, meta: PagedAttnMetadata, *, rollout=False):
        if rollout:
            return "flashinfer"
        if meta.batch_size != 1 or meta.num_query_tokens != 50:
            return "flashinfer"
        # Capacity 610 uses FlashInfer to preserve model-level numerical parity.
        if meta.max_kv_tokens in {578, 674, 762}:
            return "c27"
        return "flashinfer"

    def init_cuda_graph_state(self, **kwargs):
        from phyai_kernel.cuda.sm110.attention import ExpertAttention

        super().init_cuda_graph_state(**kwargs)
        self.kernel = ExpertAttention(torch.device(kwargs["device"]))

    def init_forward_metadata(self, meta):
        self.variant = self.graph_replay_key(meta)
        return super().init_forward_metadata(meta)

    def forward(self, layer, q, k, v, ctx):
        if ctx.mode.is_idle():
            return q.new_zeros(q.shape)
        if self.variant == "flashinfer" or current_execution() == "rollout":
            return super().forward(layer, q, k, v, ctx)
        ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)
        keys, values = ctx.kv_pool.kv_buffer(layer.layer_id)
        return self.kernel(
            self.variant,
            q,
            keys,
            values,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            layer.scale,
        )
