"""Request-local condition and prefix KV ownership for Qwen-Image 2.1."""

from __future__ import annotations

import torch

from phyai.kernel.call import select
from phyai.layers.attention import AttnCtx, AttnLayout, AttnMetadata, AttnMode
from phyai.parallel.state import graph_capture
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner
from phyai.utils import get_logger

from phyai.models.qwen_image_21.modeling_qwen_image_21 import (
    PrefixKV,
    QwenImage21PreparedCondition,
    QwenImage21Transformer,
)


logger = get_logger(__name__)


class QwenImage21Runner(ModelRunner):
    def __init__(
        self,
        transformer: QwenImage21Transformer,
        *,
        use_kv_cache: bool = True,
        torch_compile: bool = False,
        use_cuda_graph: bool = False,
    ) -> None:
        self.transformer = transformer
        self.use_kv_cache = use_kv_cache and transformer.config.causal_condition
        self.torch_compile = torch_compile
        self.use_cuda_graph = use_cuda_graph and self.use_kv_cache
        self.conditions: dict[str, QwenImage21PreparedCondition] = {}
        self.prefixes: dict[str, tuple[PrefixKV, ...]] = {}
        self.graphs: dict[str, CudaGraph] = {}
        self.graph_contexts: dict[str, AttnCtx] = {}

    def setup(self) -> None:
        if self.torch_compile:
            self.transformer.enable_compile()

    def reset(self) -> None:
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
        self.graph_contexts.clear()
        self.conditions.clear()
        self.prefixes.clear()

    def capture_attention_context(
        self, hidden_states: torch.Tensor, condition: QwenImage21PreparedCondition
    ) -> AttnCtx:
        """Plan the shared cached-attention geometry outside the graph."""
        layer = self.transformer.transformer_blocks[0].attn.full_attention
        batch, tokens, _ = hidden_states.shape
        kv_tokens = condition.prefix_length + tokens
        selection = select(
            "attention",
            role=layer.kernel_role,
            device=hidden_states.device,
            dtype={
                "input": hidden_states.dtype,
                "key": hidden_states.dtype,
                "value": hidden_states.dtype,
                "output": hidden_states.dtype,
            },
            shape={
                "tokens": batch * tokens,
                "kv_tokens": batch * kv_tokens,
                "heads": layer.num_heads,
                "kv_heads": layer.num_kv_heads,
                "head_dim": layer.head_dim,
            },
            attrs={
                "layout": "padded",
                "mask_kind": None,
                "causal": False,
                "sliding_window": None,
                "logits_soft_cap": None,
            },
            mode="capture",
            prefer=layer.prefer,
        )
        backend = selection.implementation(self)
        cu_q = torch.arange(
            0,
            (batch + 1) * tokens,
            tokens,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        cu_kv = torch.arange(
            0,
            (batch + 1) * kv_tokens,
            kv_tokens,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        meta = AttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.PADDED_4D,
            batch_size=batch,
            num_query_tokens=batch * tokens,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            layer_proto=layer,
            q_dtype=hidden_states.dtype,
            kv_dtype=hidden_states.dtype,
        )
        return AttnCtx(
            backend=backend,
            plan=backend.init_capture_metadata(meta),
            mode=meta.mode,
            layout=meta.layout,
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
        )

    def captured_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        condition: QwenImage21PreparedCondition,
        prefix: tuple[PrefixKV, ...],
        branch: str,
    ) -> torch.Tensor:
        graph = self.graphs.get(branch)
        if graph is None:
            attn_ctx = self.capture_attention_context(hidden_states, condition)

            def forward(hidden_states: torch.Tensor, timestep: torch.Tensor):
                return self.transformer(
                    hidden_states,
                    timestep,
                    condition,
                    prefix_kv=prefix,
                    attn_ctx=attn_ctx,
                ).sample

            inputs = {"hidden_states": hidden_states, "timestep": timestep}
            # Resolve capture-mode kernels and communicators before stream capture.
            with graph_capture():
                forward(**inputs)
            graph = CudaGraph()
            graph.capture(forward, inputs)
            # Plans own metadata allocated before capture; replay keeps its addresses.
            self.graph_contexts[branch] = attn_ctx
            self.graphs[branch] = graph
        output = graph.replay({"hidden_states": hidden_states, "timestep": timestep})
        # Preserve the runner's independent-output contract across graph replays.
        return output.clone()

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        branch: str = "cond",
    ) -> torch.Tensor:
        condition = self.conditions.get(branch)
        if condition is None:
            condition = self.transformer.prepare_condition(
                encoder_hidden_states,
                img_shapes,
                img_mask,
                encoder_hidden_states_mask,
            )
            self.conditions[branch] = condition
        prefix = self.prefixes.get(branch)
        if prefix is not None:
            hidden_states = hidden_states[:, -condition.target_tokens :]
            if self.use_cuda_graph and hidden_states.is_cuda:
                if condition.key_valid is None:
                    return self.captured_forward(
                        hidden_states, timestep, condition, prefix, branch
                    )
                logger.warning_once(
                    "Qwen-Image 2.1 uses uncaptured execution for padded prompts; "
                    "their variable-length attention packing is not graph safe."
                )
        output = self.transformer(
            hidden_states,
            timestep,
            condition,
            prefix_kv=prefix,
            return_prefix_kv=self.use_kv_cache and prefix is None,
        )
        if output.prefix_kv is not None:
            self.prefixes[branch] = output.prefix_kv
        return output.sample[:, -condition.target_tokens :]

    def close(self) -> None:
        self.reset()
