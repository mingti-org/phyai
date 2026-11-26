"""VPP2 condition caches and reusable CUDA graphs for action rollouts."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import torch

from phyai.layers.attention import AttnCtx, AttnLayout
from phyai.models.vpp2.modeling_vpp2 import (
    VPP2Condition,
    VPP2CrossKV,
    VPP2ExpertCondition,
    VPP2Model,
    VPP2VideoCache,
)
from phyai.parallel.state import graph_capture
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner
from phyai.utils import get_logger


logger = get_logger(__name__)
ActionForward = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ActionRollout = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, ActionForward], torch.Tensor
]


@dataclass
class VPP2ForwardBatch:
    sample: torch.Tensor
    timestep: torch.Tensor
    mode: Literal["action", "video"] = "action"
    branch: str = "cond"
    condition_latents: torch.Tensor | None = None


@dataclass(frozen=True)
class VPP2PrefillBinding:
    graph: CudaGraph
    condition: VPP2ExpertCondition
    video_cache: VPP2VideoCache


class VPP2Runner(ModelRunner):
    def __init__(
        self,
        model: VPP2Model,
        *,
        torch_compile: bool = False,
        use_cuda_graph: bool = False,
        use_prefill_cuda_graph: bool = True,
        bind_prefill_outputs: bool = True,
        use_action_kv_buffer: bool = True,
        optimize_pointwise: bool = True,
        optimize_qk_norm_layout: bool = True,
        optimize_qk_norm_fusion: bool = True,
        optimize_qk_norm_rope_fusion: bool = True,
        optimize_layernorm_modulation: bool = True,
        optimize_gated_layernorm: bool = True,
        fuse_prepare_modulation: bool = True,
        max_cached_graphs: int = 1,
    ) -> None:
        if max_cached_graphs < 1:
            raise ValueError("max_cached_graphs must be positive.")
        self.model = model
        self.config = model.config
        self.torch_compile = torch_compile
        self.use_cuda_graph = use_cuda_graph
        self.use_prefill_cuda_graph = use_prefill_cuda_graph
        self.bind_prefill_outputs = bind_prefill_outputs
        self.use_action_kv_buffer = use_action_kv_buffer
        self.optimize_pointwise = optimize_pointwise
        self.optimize_qk_norm_layout = optimize_qk_norm_layout
        self.optimize_qk_norm_fusion = optimize_qk_norm_fusion
        self.optimize_qk_norm_rope_fusion = optimize_qk_norm_rope_fusion
        self.optimize_layernorm_modulation = optimize_layernorm_modulation
        self.optimize_gated_layernorm = optimize_gated_layernorm
        self.fuse_prepare_modulation = fuse_prepare_modulation
        self.max_cached_graphs = max_cached_graphs
        self.conditions: dict[str, VPP2Condition] = {}
        self.video_caches: dict[str, VPP2VideoCache] = {}
        self.graphs: dict[tuple[object, ...], CudaGraph] = {}
        self.graph_contexts: dict[
            tuple[object, ...], tuple[tuple[AttnCtx, AttnCtx], ...]
        ] = {}
        self.action_graph_bindings: dict[tuple[object, ...], VPP2PrefillBinding] = {}
        self.graph_capture_count = 0
        self.graph_replay_count = 0
        self.prefill_graphs: dict[tuple[object, ...], CudaGraph] = {}
        self.prefill_sources: dict[str, CudaGraph] = {}
        self.prefill_graph_capture_count = 0
        self.prefill_graph_replay_count = 0
        self.action_forward = model.forward_action
        self.video_forward = model.forward_video
        self.ready = False

    @property
    def device(self) -> torch.device:
        return self.model.video_expert.patch_embedding.weight.device

    @property
    def params_dtype(self) -> torch.dtype:
        return self.model.video_expert.patch_embedding.weight.dtype

    @property
    def supports_action(self) -> bool:
        return self.model.action_expert is not None

    def setup(self) -> None:
        if self.ready:
            return
        if self.optimize_pointwise:
            self.model.enable_pointwise_fusion()
        self.model.enable_qk_norm_layout(self.optimize_qk_norm_layout)
        self.model.enable_qk_norm_fusion(self.optimize_qk_norm_fusion)
        self.model.enable_qk_norm_rope_fusion(self.optimize_qk_norm_rope_fusion)
        self.model.enable_gated_layernorm(self.optimize_gated_layernorm)
        self.model.enable_layernorm_modulation(
            self.optimize_layernorm_modulation,
            include_prepare=self.fuse_prepare_modulation,
        )
        if self.torch_compile:
            options = {"emulate_precision_casts": True, "triton.cudagraphs": False}
            self.action_forward = torch.compile(
                self.model.forward_action, dynamic=False, options=options
            )
            self.video_forward = torch.compile(
                self.model.forward_video, dynamic=False, options=options
            )
        self.ready = True

    def reset(self) -> None:
        self.release_video_cache()
        self.conditions.clear()

    def release_video_cache(self) -> None:
        self.video_caches.clear()
        self.prefill_sources.clear()

    def release_action_graph(self, key: tuple[object, ...]) -> None:
        self.graphs.pop(key).reset()
        self.graph_contexts.pop(key, None)
        self.action_graph_bindings.pop(key, None)

    def release_prefill_dependents(self, source: CudaGraph) -> None:
        for key, binding in tuple(self.action_graph_bindings.items()):
            if binding.graph is source:
                self.release_action_graph(key)
        for branch, graph in tuple(self.prefill_sources.items()):
            if graph is source:
                self.prefill_sources.pop(branch)

    def clear_cuda_graphs(self) -> None:
        for key in tuple(self.graphs):
            self.release_action_graph(key)
        for graph in self.prefill_graphs.values():
            graph.reset()
        self.prefill_graphs.clear()
        self.prefill_sources.clear()

    @torch.no_grad()
    def prepare_condition(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        clip_fea: torch.Tensor,
        *,
        proprio: torch.Tensor | None = None,
        branch: str = "cond",
    ) -> None:
        inputs = self.condition_inputs(context, context_mask, clip_fea, proprio)
        self.conditions[branch] = self.model.encode_condition(**inputs)
        self.video_caches.pop(branch, None)
        self.prefill_sources.pop(branch, None)

    def condition_inputs(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        clip_fea: torch.Tensor,
        proprio: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if not self.ready:
            raise RuntimeError("Call VPP2Runner.setup() before inference.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask is not None and context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError("context must have shape [1,L,D].")
        if context_mask is not None and context_mask.shape != context.shape[:2]:
            raise ValueError("context_mask must match the [1,L] context dimensions.")
        if context_mask is not None:
            if not bool(context_mask.bool().all()):
                # Text positions are already encoded; remove masked K/V rows once.
                valid = context_mask[0].to(device=context.device, dtype=torch.bool)
                context = context[:, valid]
            context_mask = None
        if context.shape[1] == 0 and proprio is None:
            raise ValueError(
                "At least one text token or proprioception token is required."
            )
        if clip_fea.ndim != 3 or clip_fea.shape[0] != 1:
            raise ValueError("clip_fea must have shape [1,L,D].")
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError("proprio must have shape [D] or [1,D].")
            proprio = proprio.to(device=self.device, dtype=self.params_dtype)
        inputs = {
            "context": context.to(device=self.device, dtype=self.params_dtype),
            "clip_fea": clip_fea.to(device=self.device, dtype=self.params_dtype),
        }
        if proprio is not None:
            inputs["proprio"] = proprio
        return inputs

    @staticmethod
    def tensor_metadata(inputs: dict[str, torch.Tensor]) -> tuple[object, ...]:
        return tuple(
            (name, tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
            for name, value in inputs.items()
        )

    @torch.no_grad()
    def prepare_policy(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        clip_fea: torch.Tensor,
        video: torch.Tensor,
        timestep: torch.Tensor,
        condition_latents: torch.Tensor,
        *,
        proprio: torch.Tensor | None = None,
        action_seq_len: int,
        return_prediction: bool = False,
        branch: str = "cond",
    ) -> torch.Tensor | None:
        use_graph = (
            self.use_cuda_graph
            and self.use_prefill_cuda_graph
            and video.is_cuda
            and self.config.video.video_attention_mask_mode
            in ("bidirectional", "three_frame")
        )
        if not use_graph:
            self.prepare_condition(
                context, context_mask, clip_fea, proprio=proprio, branch=branch
            )
            return self.prefill_video(
                video,
                timestep,
                condition_latents,
                action_seq_len=action_seq_len,
                return_prediction=return_prediction,
                branch=branch,
            )
        inputs = self.condition_inputs(context, context_mask, clip_fea, proprio)
        inputs.update(
            video=video, timestep=timestep, condition_latents=condition_latents
        )
        graph_key = (action_seq_len, return_prediction, self.tensor_metadata(inputs))
        graph = self.prefill_graphs.get(graph_key)
        if graph is None:
            if len(self.prefill_graphs) >= self.max_cached_graphs:
                oldest = next(iter(self.prefill_graphs))
                evicted = self.prefill_graphs.pop(oldest)
                self.release_prefill_dependents(evicted)
                evicted.reset()

            def graph_forward(**buffers: torch.Tensor):
                condition = self.model.encode_condition(
                    buffers["context"],
                    clip_fea=buffers["clip_fea"],
                    proprio=buffers.get("proprio"),
                )
                cache = self.model.prefill_video(
                    buffers["video"],
                    buffers["timestep"],
                    condition,
                    condition_latents=buffers["condition_latents"],
                    action_seq_len=action_seq_len,
                    return_prediction=return_prediction,
                )
                return condition, cache

            with graph_capture():
                graph_forward(**inputs)
            graph = CudaGraph(num_warmup_iters=1)
            graph.capture(graph_forward, inputs)
            self.prefill_graphs[graph_key] = graph
            self.prefill_graph_capture_count += 1
        condition, cache = graph.replay(inputs)
        self.conditions[branch] = condition
        self.video_caches[branch] = cache
        self.prefill_sources[branch] = graph
        self.prefill_graph_replay_count += 1
        return cache.prediction

    def capture_action_contexts(
        self, sample: torch.Tensor, branch: str
    ) -> tuple[tuple[AttnCtx, AttnCtx], ...]:
        expert = self.model.action_expert
        if expert is None:
            raise ValueError("Action inference requires an action checkpoint.")
        block = expert.blocks[0]
        cache = self.video_caches[branch]
        condition = self.conditions[branch].action
        if condition is None:
            raise ValueError("Action conditions are not prepared.")
        query = sample.new_empty(
            (
                *sample.shape[:2],
                block.self_attn.num_local_heads,
                block.self_attn.head_dim,
            )
        )
        video_key = cache.layers[0][0]
        key = query.new_empty(
            (query.shape[0], video_key.shape[1] + query.shape[1], *query.shape[2:])
        )
        self_context = block.self_attn.attn._build_default_ctx(
            query, key, key, AttnLayout.PADDED_4D, None, None, mode="capture"
        )
        text_kv = condition.cross_kv[0]
        cross_context = block.cross_attn.attn._build_default_ctx(
            query,
            text_kv.key,
            text_kv.value,
            AttnLayout.PADDED_4D,
            None,
            None,
            mode="capture",
        )
        return ((self_context, cross_context),) * len(expert.blocks)

    @staticmethod
    def action_kv_buffers(
        cache: VPP2VideoCache, action_tokens: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        layers = []
        for key, value in cache.layers:
            shape = (key.shape[0], key.shape[1] + action_tokens, *key.shape[2:])
            full_key = key.new_empty(shape)
            full_value = value.new_empty(shape)
            full_key[:, : cache.video_seq_len].copy_(key)
            full_value[:, : cache.video_seq_len].copy_(value)
            layers.append((full_key, full_value))
        return tuple(layers)

    @torch.no_grad()
    def execute_action_rollout(
        self,
        rollout: ActionRollout,
        initial: torch.Tensor,
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        *,
        branch: str = "cond",
    ) -> torch.Tensor:
        condition = self.conditions[branch].action
        if condition is None or self.model.action_expert is None:
            raise ValueError("Action inference requires an action checkpoint.")
        use_graph = self.use_cuda_graph and initial.is_cuda
        if use_graph and condition.context_mask is not None:
            logger.warning_once(
                "VPP2 uses eager execution for masked text conditions; "
                "the attention packing path does not support CUDA graph capture."
            )
            use_graph = False
        if not use_graph:
            if self.use_action_kv_buffer:
                cache = self.video_caches[branch]
                self.video_caches[branch] = replace(
                    cache,
                    action_kv_buffers=self.action_kv_buffers(cache, initial.shape[1]),
                )

            def forward(sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                return self.forward(VPP2ForwardBatch(sample, timestep, branch=branch))

            return rollout(initial, timesteps, deltas, forward)
        cache = self.video_caches[branch]
        source = self.prefill_sources.get(branch) if self.bind_prefill_outputs else None
        inputs = {
            "initial": initial,
            "timesteps": timesteps,
            "deltas": deltas,
        }
        if source is None:
            inputs["context"] = condition.context
            for index, (key, value) in enumerate(cache.layers):
                inputs[f"video_key_{index}"] = key
                inputs[f"video_value_{index}"] = value
            for index, kv in enumerate(condition.cross_kv):
                inputs[f"text_key_{index}"] = kv.key
                inputs[f"text_value_{index}"] = kv.value
        graph_key = (
            getattr(rollout, "__func__", rollout),
            source,
            self.use_action_kv_buffer,
            cache.tokens_per_frame,
            cache.video_seq_len,
            tuple(
                (
                    name,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.dtype,
                    value.device,
                )
                for name, value in inputs.items()
            ),
        )
        graph = self.graphs.get(graph_key)
        if graph is None:
            if len(self.graphs) >= self.max_cached_graphs:
                oldest = next(iter(self.graphs))
                self.release_action_graph(oldest)
            contexts = self.capture_action_contexts(initial, branch)
            expert = self.model.action_expert
            layer_count = len(cache.layers)
            tokens_per_frame = cache.tokens_per_frame
            video_seq_len = cache.video_seq_len

            def graph_forward(**buffers: torch.Tensor) -> torch.Tensor:
                if source is None:
                    local_condition = VPP2ExpertCondition(
                        context=buffers["context"],
                        context_mask=None,
                        cross_kv=tuple(
                            VPP2CrossKV(
                                buffers[f"text_key_{i}"], buffers[f"text_value_{i}"]
                            )
                            for i in range(layer_count)
                        ),
                    )
                    local_cache = VPP2VideoCache(
                        layers=tuple(
                            (buffers[f"video_key_{i}"], buffers[f"video_value_{i}"])
                            for i in range(layer_count)
                        ),
                        tokens_per_frame=tokens_per_frame,
                        video_seq_len=video_seq_len,
                    )
                else:
                    local_condition = condition
                    local_cache = cache
                if self.use_action_kv_buffer:
                    local_cache = replace(
                        local_cache,
                        action_kv_buffers=self.action_kv_buffers(
                            local_cache, buffers["initial"].shape[1]
                        ),
                    )

                def forward(
                    sample: torch.Tensor, timestep: torch.Tensor
                ) -> torch.Tensor:
                    return expert(
                        sample,
                        timestep,
                        condition=local_condition,
                        video_cache=local_cache,
                        attention_contexts=contexts,
                    )

                return rollout(
                    buffers["initial"], buffers["timesteps"], buffers["deltas"], forward
                )

            with graph_capture():
                graph_forward(**inputs)
            graph = CudaGraph(num_warmup_iters=1)
            graph.capture(graph_forward, inputs)
            self.graph_contexts[graph_key] = contexts
            self.graphs[graph_key] = graph
            if source is not None:
                # The captured executable does not retain its Python closure.
                # Own every borrowed tensor until its dependent graph is destroyed.
                self.action_graph_bindings[graph_key] = VPP2PrefillBinding(
                    source, condition, cache
                )
            self.graph_capture_count += 1
        self.graph_replay_count += 1
        # Preparation graphs refresh borrowed tensors at stable addresses. Other
        # callers refresh condition copies along with the noise and schedule.
        return graph.replay(inputs).clone()

    @torch.no_grad()
    def prefill_video(
        self,
        video: torch.Tensor,
        timestep: torch.Tensor,
        condition_latents: torch.Tensor,
        *,
        action_seq_len: int,
        return_prediction: bool = False,
        branch: str = "cond",
    ) -> torch.Tensor | None:
        cache = self.model.prefill_video(
            video,
            timestep,
            self.conditions[branch],
            condition_latents=condition_latents,
            action_seq_len=action_seq_len,
            return_prediction=return_prediction,
        )
        self.video_caches[branch] = cache
        self.prefill_sources.pop(branch, None)
        return cache.prediction

    @torch.no_grad()
    def forward(self, batch: VPP2ForwardBatch) -> torch.Tensor:
        if not self.ready:
            raise RuntimeError("Call VPP2Runner.setup() before inference.")
        if batch.branch not in self.conditions:
            raise ValueError(f"Condition branch {batch.branch!r} is not prepared.")
        condition = self.conditions[batch.branch]
        if batch.mode == "video":
            return self.video_forward(
                batch.sample,
                batch.timestep,
                condition,
                condition_latents=batch.condition_latents,
            )
        if batch.mode != "action":
            raise ValueError(f"Unknown VPP2 forward mode: {batch.mode!r}.")
        if batch.branch not in self.video_caches:
            raise ValueError("Action inference requires a video prefill first.")
        return self.action_forward(
            batch.sample,
            batch.timestep,
            condition,
            self.video_caches[batch.branch],
        )

    def close(self) -> None:
        self.reset()
        self.clear_cuda_graphs()
        self.ready = False
