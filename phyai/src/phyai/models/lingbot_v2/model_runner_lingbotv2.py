"""LingBot-VLA 2.0 runners: vision, cached Qwen prefix, and action expert."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from phyai.cache import KVCachePool, StaticCache
from phyai.utils import get_logger
from phyai.layers.attention import (
    AttnMode,
    AttnCtx,
    AttnMetadata,
    AttnLayout,
    PagedAttnCtx,
    PagedAttention,
    PagedAttnMetadata,
    PagedAttnPlanHandle,
)
from phyai.runtime.model_runner import ModelRunner
from phyai.layers.attention.shared import select_paged_backend
from phyai.layers.linear import LinearBase, resolve_linear_kernel
from phyai.layers.rotary_embedding import InterleavedMRotaryEmbedding
from phyai.parallel.state import graph_capture, resolve_mesh
from phyai.runtime.cuda_graph_manager import CudaGraph, CudaGraphError
from phyai.models.lingbot_v2.modeling_lingbotv2 import (
    LingBotV2Model,
    Qwen3VLTextModel,
    LingBotV2ExpertStack,
    LingBotV2TokenMoE,
    LingBotV2ActionTimeHeads,
    build_lingbot_v2_mrope_position_ids,
)

from phyai.models.lingbot_v2.vision import Qwen3VLVisionModel, VisionPositionInputs

logger = get_logger(__name__)


@dataclass
class LingBotV2VisionForwardBatch:
    """Packed patch vectors and one CPU (t,h,w) row per image."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


@dataclass
class LingBotV2PrefixForwardBatch:
    """Fixed-stride prefix rows plus DeepStack data for real image positions."""

    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    write_indices: torch.Tensor
    visual_token_indices: torch.Tensor | None = None
    deepstack_visual_embeds: list[torch.Tensor] | None = None


@dataclass(frozen=True)
class LingBotV2PrefixEmbeddings:
    """Request-dependent text plus cached learned prefix embeddings."""

    language: torch.Tensor
    current_query: torch.Tensor
    future_query: torch.Tensor
    vision_start: torch.Tensor | None
    vision_end: torch.Tensor | None


@dataclass(frozen=True)
class LingBotV2CacheAllocation:
    """KV slot indices reserved for one scheduler request."""

    prefix_slots: torch.Tensor
    suffix_slots: torch.Tensor


@dataclass
class LingBotV2ExpertForwardBatch:
    """One flow-matching velocity evaluation."""

    state: torch.Tensor
    x_t: torch.Tensor
    time: torch.Tensor


def _ar_attn_proto(text_model: Qwen3VLTextModel) -> PagedAttention:
    if not text_model.layers:
        raise ValueError("text model has no layers.")
    attention = text_model.layers[0].attn
    if not isinstance(attention, PagedAttention):
        raise TypeError("LingBot prefix runner requires PagedAttention.")
    return attention


def _diffusion_attn_proto(
    expert_stack: LingBotV2ExpertStack,
) -> PagedAttention:
    if not expert_stack.layers:
        raise ValueError("expert stack has no layers.")
    attention = expert_stack.layers[0].attn
    if not isinstance(attention, PagedAttention):
        raise TypeError("LingBot expert runner requires PagedAttention.")
    return attention


def _close_backend(*backends: object) -> None:
    first_error = None
    for backend in backends:
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                else:
                    first_error.add_note(
                        f"Additional backend cleanup failure: {error!r}"
                    )
    if first_error is not None:
        raise first_error


@dataclass
class _LingBotV2EulerGraphState:
    """A captured Euler loop and the backend plans it owns."""

    graph: CudaGraph
    state_plan: PagedAttnPlanHandle
    action_plan: PagedAttnPlanHandle
    wavefront: _LingBotV2WavefrontState | None = None


@dataclass(frozen=True)
class _LingBotV2WavefrontState:
    """Keep the state stream and per-step layer events alive with their graph."""

    stream: torch.cuda.Stream
    events: tuple[tuple[torch.cuda.Event, ...], ...]


class LingBotV2VisionRunner(ModelRunner):
    """Cache vision positions by grid and plan attention once per forward."""

    def __init__(
        self,
        vision_model: Qwen3VLVisionModel,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.vision_model = vision_model
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        self._positions: dict[tuple[tuple[int, ...], ...], VisionPositionInputs] = {}

    def setup(self) -> None:
        """Invalidate cached positions after weights have been loaded."""
        self._positions.clear()

    def reset(self) -> None:
        """Keep reusable geometry across requests."""

    def close(self) -> None:
        """Release cached geometry and positional embeddings."""
        self._positions.clear()

    def forward(
        self, batch: LingBotV2VisionForwardBatch
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode packed image patches with one shared attention context."""
        grid = batch.image_grid_thw
        if grid.device.type != "cpu":
            raise ValueError("LingBot vision planning requires CPU image_grid_thw.")
        key = tuple(tuple(row) for row in grid.tolist())
        positions = self._positions.get(key)
        if positions is None:
            positions = self.vision_model.prepare_positions(grid)
            self._positions[key] = positions

        attention = self.vision_model.blocks[0].attn.attn
        query_shape = positions.pos_embeds[
            :, : attention.num_heads * attention.head_dim
        ].view(positions.pos_embeds.shape[0], attention.num_heads, attention.head_dim)
        backend = attention._ensure_backend(q=query_shape, layout=AttnLayout.RAGGED_3D)
        metadata = AttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=positions.cu_seqlens.numel() - 1,
            num_query_tokens=positions.pos_embeds.shape[0],
            cu_seqlens_q=positions.cu_seqlens,
            cu_seqlens_kv=positions.cu_seqlens,
            layer_proto=attention,
            q_dtype=self.params_dtype,
            kv_dtype=self.params_dtype,
        )
        context = AttnCtx(
            backend=backend,
            plan=backend.init_forward_metadata(metadata),
            mode=metadata.mode,
            layout=metadata.layout,
            cu_seqlens_q=positions.cu_seqlens,
            cu_seqlens_kv=positions.cu_seqlens,
        )
        return self.vision_model(
            batch.pixel_values.to(device=self.device, dtype=self.params_dtype),
            positions,
            attn_ctx=context,
        )


class LingBotV2PrefixRunner(ModelRunner):
    """Qwen3-VL prefix runtime with shared KV storage."""

    def __init__(
        self,
        model: LingBotV2Model,
        *,
        max_batch_size: int,
        prefix_capacity: int,
        suffix_capacity: int,
        params_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if prefix_capacity <= 0 or suffix_capacity <= 0:
            raise ValueError("prefix_capacity and suffix_capacity must be positive.")

        self.model = model
        self.config = model.config
        self.text_model = model.text
        self.mrope = model.mrope

        self.max_batch_size = int(max_batch_size)
        self.max_num_tokens = int(prefix_capacity)
        self.max_paged_kv_indices = int(prefix_capacity)

        self.params_dtype = params_dtype
        self.device = torch.device(device)

        self.sentinel_slot = 0
        self.prefix_base = 1
        self.suffix_base = self.prefix_base + int(prefix_capacity)

        self.attn_proto = _ar_attn_proto(self.text_model)
        self.kv_pool = KVCachePool(
            num_layers=self.config.num_layers,
            num_slots=1 + int(prefix_capacity) + int(suffix_capacity),
            num_kv_heads=self.attn_proto.num_kv_heads,
            head_dim=self.config.text.head_dim,
            # Joint attention retains FP32 K/V after MRoPE.
            dtype=torch.float32,
            device=self.device,
        )

        self.prefix_cache = StaticCache(
            self.kv_pool,
            base_offset=self.prefix_base,
            capacity=int(prefix_capacity),
        )

        self.suffix_cache = StaticCache(
            self.kv_pool,
            base_offset=self.suffix_base,
            capacity=int(suffix_capacity),
        )

        self.attn_backend = select_paged_backend(
            "attention_paged",
            self.attn_proto,
            self,
            device=self.device,
            params_dtype=torch.float32,
            num_tokens=self.max_num_tokens,
            capture=False,
            runner_tag="lingbot_v2.prefix",
        )

        self._plan: PagedAttnPlanHandle | None = None
        self._current_query: torch.Tensor | None = None
        self._future_query: torch.Tensor | None = None
        self._vision_start: torch.Tensor | None = None
        self._vision_end: torch.Tensor | None = None

    def setup(self) -> None:
        """Allocate prefix attention metadata and caches."""

        self.attn_backend.init_cuda_graph_state(
            max_batch_size=self.max_batch_size,
            max_num_tokens=self.max_num_tokens,
            max_paged_kv_indices=self.max_paged_kv_indices,
            device=self.device,
            params_dtype=torch.float32,
            layer_proto=self.attn_proto,
        )

        with torch.no_grad():
            current_query, future_query = self.model.dual_query(1)
            self._current_query = current_query[0].to(self.params_dtype)
            self._future_query = future_query[0].to(self.params_dtype)

            if self.config.use_vision_boundaries:
                self._vision_start = self.model.embed_special(
                    self.config.vision_start_token_id
                ).to(self.params_dtype)

                self._vision_end = self.model.embed_special(
                    self.config.vision_end_token_id
                ).to(self.params_dtype)

        if (
            self._future_query is None
            or self._future_query.shape[0]
            != self.config.dual_query.future_query_token_count
        ):
            raise RuntimeError("unexpected LingBot V2 future-query layout.")

    def reset(self) -> None:
        """Clear request-local plans and rewind KV allocators."""

        self._plan = None
        self.prefix_cache.reset()
        self.suffix_cache.reset()

    def allocate_request_cache(
        self,
        *,
        prefix_tokens: int,
        suffix_tokens: int,
    ) -> LingBotV2CacheAllocation:
        """Reserve compact prefix/suffix KV slots."""

        return LingBotV2CacheAllocation(
            prefix_slots=self.prefix_cache.allocate(prefix_tokens),
            suffix_slots=self.suffix_cache.allocate(suffix_tokens),
        )

    def prepare_prefix_embeddings(
        self,
        input_ids: torch.Tensor,
        *,
        batch_size: int,
    ) -> LingBotV2PrefixEmbeddings:
        """Run text embedding and reuse learned conditions."""

        if self._current_query is None or self._future_query is None:
            raise RuntimeError("prefix runner setup must run before embedding inputs.")

        language = self.model.embed_language(
            input_ids.to(device=self.device, dtype=torch.int64)
        ).to(self.params_dtype)

        return LingBotV2PrefixEmbeddings(
            language=language,
            current_query=self._current_query.unsqueeze(0).expand(batch_size, -1, -1),
            future_query=self._future_query.unsqueeze(0).expand(batch_size, -1, -1),
            vision_start=self._vision_start,
            vision_end=self._vision_end,
        )

    def build_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Build LingBot 3-D MRoPE coordinates."""

        return build_lingbot_v2_mrope_position_ids(
            input_ids,
            attention_mask,
            image_grid_thw,
            image_token_id=self.config.image_token_id,
            spatial_merge_size=self.config.vision.spatial_merge_size,
        )

    def plan_inference(
        self,
        metadata: PagedAttnMetadata,
    ) -> None:
        """Build the attention plan for the next prefix request."""

        self._plan = self.attn_backend.init_forward_metadata(metadata)

    def forward(
        self,
        batch: LingBotV2PrefixForwardBatch,
    ) -> None:
        """Run prefix transformer and write K/V cache."""

        if self._plan is None:
            raise RuntimeError("plan_inference must be called before prefix forward.")

        cos, sin = self.mrope.get_cos_sin(batch.position_ids)

        context = PagedAttnCtx(
            backend=self.attn_backend,
            plan=self._plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=batch.write_indices,
        )

        k, v = self.text_model.prefill(
            batch.hidden_states,
            cos=cos,
            sin=sin,
            attn_ctx=context,
            visual_token_indices=batch.visual_token_indices,
            deepstack_visual_embeds=batch.deepstack_visual_embeds,
        )
        self.kv_pool.write_kv(
            self.text_model.layers[-1].layer_idx, batch.write_indices, k, v
        )

    def close(self) -> None:
        """Release prefix attention resources."""

        self.reset()
        _close_backend(self.attn_backend)
        self._current_query = None
        self._future_query = None
        self._vision_start = None
        self._vision_end = None


@dataclass(frozen=True)
class _LingBotV2StepCondition:
    """Time features and affine parameters shared by state and action tokens."""

    cond: torch.Tensor
    input_modulations: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    post_modulations: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class LingBotV2ExpertRunner(ModelRunner):
    """Run the action expert with cached conditions and optional Euler graphs."""

    def __init__(
        self,
        expert_stack: LingBotV2ExpertStack,
        heads: LingBotV2ActionTimeHeads,
        mrope: InterleavedMRotaryEmbedding,
        kv_pool: KVCachePool,
        *,
        max_batch_size: int,
        chunk_size: int,
        max_action_dim: int,
        num_inference_steps: int,
        max_num_tokens: int,
        max_paged_kv_indices: int,
        params_dtype: torch.dtype,
        device: torch.device | str,
        use_cuda_graph: bool = True,
        use_wavefront_pipeline: bool = False,
    ) -> None:
        """Select attention backends and check Wavefront execution requirements."""
        self.expert_stack = expert_stack
        self.heads = heads
        self.mrope = mrope
        self.kv_pool = kv_pool
        self.max_batch_size = int(max_batch_size)
        self.chunk_size = int(chunk_size)
        self.max_action_dim = int(max_action_dim)
        self.num_inference_steps = int(num_inference_steps)
        self.max_num_tokens = int(max_num_tokens)
        self.max_paged_kv_indices = int(max_paged_kv_indices)
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        self.attn_proto = _diffusion_attn_proto(expert_stack)
        capture = bool(use_cuda_graph) and self.device.type == "cuda"
        self.state_backend = select_paged_backend(
            "attention_paged",
            self.attn_proto,
            self,
            device=self.device,
            params_dtype=torch.float32,
            num_tokens=self.max_batch_size,
            capture=capture,
            runner_tag="lingbot_v2.state",
        )
        self.action_backend = select_paged_backend(
            "attention_paged",
            self.attn_proto,
            self,
            device=self.device,
            params_dtype=torch.float32,
            num_tokens=self.max_num_tokens,
            capture=capture,
            runner_tag="lingbot_v2.action",
        )
        self.use_cuda_graph = (
            capture
            and self.state_backend.supports_capture()
            and self.action_backend.supports_capture()
        )
        self.use_wavefront_pipeline = bool(use_wavefront_pipeline)
        if self.use_wavefront_pipeline and not self.use_cuda_graph:
            raise ValueError("LingBot wavefront pipeline requires Euler CUDA Graph.")
        if self.use_wavefront_pipeline and any(
            backend.name != "sdpa"
            for backend in (self.state_backend, self.action_backend)
        ):
            raise ValueError(
                "LingBot wavefront pipeline requires SDPA paged attention; "
                f"got state={self.state_backend.name}, action={self.action_backend.name}."
            )
        self._time_schedule: torch.Tensor | None = None
        self._dt: torch.Tensor | None = None
        self._step_conditions: tuple[_LingBotV2StepCondition, ...] = ()
        self._euler_graphs: dict[tuple[object, ...], _LingBotV2EulerGraphState] = {}
        self._failed_graph_keys: set[tuple[object, ...]] = set()
        self.reset()

    def setup(self) -> None:
        """Validate Wavefront kernels and prepare the fixed Euler conditions."""
        self._euler_graphs.clear()
        self._failed_graph_keys.clear()
        self._step_conditions = ()
        self.reset()
        if self.use_wavefront_pipeline:
            with torch.no_grad():
                state = torch.zeros(
                    1,
                    self.heads.flow_config.max_state_dim,
                    dtype=self.params_dtype,
                    device=self.device,
                )
                self._validate_wavefront_kernels(
                    self.heads.embed_state(state).reshape(1, -1)
                )
        common = {
            "max_batch_size": self.max_batch_size,
            "max_paged_kv_indices": self.max_paged_kv_indices,
            "device": self.device,
            "params_dtype": torch.float32,
            "layer_proto": self.attn_proto,
        }
        self.state_backend.init_cuda_graph_state(
            max_num_tokens=self.max_batch_size, **common
        )
        self.action_backend.init_cuda_graph_state(
            max_num_tokens=self.max_num_tokens, **common
        )
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        self._dt = torch.tensor(
            -1.0 / self.num_inference_steps,
            dtype=self.params_dtype,
            device=self.device,
        )
        time = torch.tensor(1.0, dtype=self.params_dtype, device=self.device)
        schedule = []
        for _ in range(self.num_inference_steps):
            schedule.append(time.clone())
            time += self._dt
        self._time_schedule = torch.stack(schedule)
        with torch.no_grad():
            self._step_conditions = tuple(
                self._prepare_condition(time.expand(self.max_batch_size))
                for time in self._time_schedule
            )

    def _prepare_condition(self, time: torch.Tensor) -> _LingBotV2StepCondition:
        cond = self.heads.time_condition(time)
        return _LingBotV2StepCondition(
            cond=cond,
            input_modulations=tuple(
                layer.input_layernorm.compute_modulation(cond)
                for layer in self.expert_stack.layers
            ),
            post_modulations=tuple(
                layer.post_attention_layernorm.compute_modulation(cond)
                for layer in self.expert_stack.layers
            ),
        )

    def reset(self) -> None:
        """Discard request-specific attention plans and rotary positions."""
        self._state_metadata: PagedAttnMetadata | None = None
        self._action_metadata: PagedAttnMetadata | None = None
        self._state_plan: PagedAttnPlanHandle | None = None
        self._action_plan: PagedAttnPlanHandle | None = None
        self._position_inputs: dict[str, torch.Tensor] = {}
        self._layout_key: tuple[object, ...] = ()

    def plan_inference(
        self,
        state_metadata: PagedAttnMetadata,
        action_metadata: PagedAttnMetadata,
    ) -> None:
        """Plan from CPU offsets and compute rotary embeddings from GPU positions."""
        layout = []
        for name, metadata in (("state", state_metadata), ("action", action_metadata)):
            if metadata.position_ids is None or metadata.write_indices is None:
                raise ValueError(
                    "LingBot expert metadata requires positions and KV write indices."
                )
            if metadata.cu_seqlens_q is None or metadata.paged_kv_indptr is None:
                raise ValueError(
                    "LingBot expert metadata requires query and KV offsets."
                )
            if (
                metadata.cu_seqlens_q.device.type != "cpu"
                or metadata.paged_kv_indptr.device.type != "cpu"
            ):
                raise ValueError("LingBot query and KV offsets must remain on CPU.")
            cos, sin = self.mrope.get_cos_sin(metadata.position_ids)
            if name == "state":
                cos, sin = (
                    cos.reshape(-1, cos.shape[-1]),
                    sin.reshape(-1, sin.shape[-1]),
                )
            self._position_inputs[f"{name}_cos"] = cos
            self._position_inputs[f"{name}_sin"] = sin
            self._position_inputs[f"{name}_write_indices"] = metadata.write_indices
            layout.append(
                (
                    tuple(metadata.cu_seqlens_q.tolist()),
                    tuple(metadata.paged_kv_indptr.tolist()),
                )
            )
        self._layout_key = tuple(layout)
        self._state_metadata = state_metadata
        self._action_metadata = action_metadata
        self._state_plan = self.state_backend.init_forward_metadata(state_metadata)
        self._action_plan = self.action_backend.init_forward_metadata(action_metadata)

    def _contexts(
        self,
        state_plan: PagedAttnPlanHandle,
        action_plan: PagedAttnPlanHandle,
        state_write_indices: torch.Tensor,
        action_write_indices: torch.Tensor,
    ) -> tuple[PagedAttnCtx, PagedAttnCtx]:
        return (
            PagedAttnCtx(
                backend=self.state_backend,
                plan=state_plan,
                mode=AttnMode.PREFILL,
                layout=AttnLayout.RAGGED_3D,
                kv_pool=self.kv_pool,
                write_indices=state_write_indices,
            ),
            PagedAttnCtx(
                backend=self.action_backend,
                plan=action_plan,
                mode=AttnMode.PREFILL,
                layout=AttnLayout.RAGGED_3D,
                kv_pool=self.kv_pool,
                write_indices=action_write_indices,
            ),
        )

    def _velocity(
        self,
        state_hidden: torch.Tensor,
        x_t: torch.Tensor,
        condition: _LingBotV2StepCondition,
        *,
        state_context: PagedAttnCtx,
        action_context: PagedAttnCtx,
        state_cos: torch.Tensor,
        state_sin: torch.Tensor,
        action_cos: torch.Tensor,
        action_sin: torch.Tensor,
        wavefront: _LingBotV2WavefrontState | None = None,
        step: int = 0,
    ) -> torch.Tensor:
        """Run State/Action layers with ordered KV reads and a final stream join."""
        batch_size = state_hidden.shape[0]
        cond = condition.cond[:batch_size]
        action_hidden = self.heads.embed_action(x_t, cond)
        main_stream = None
        if wavefront is not None:
            main_stream = torch.cuda.current_stream(self.device)
            wavefront.stream.wait_stream(main_stream)
        for index, layer in enumerate(self.expert_stack.layers):
            input_modulation = tuple(
                t[:batch_size] for t in condition.input_modulations[index]
            )
            post_modulation = tuple(
                t[:batch_size] for t in condition.post_modulations[index]
            )
            with (
                torch.cuda.stream(wavefront.stream)
                if wavefront is not None
                else nullcontext()
            ):
                state_hidden = layer(
                    state_hidden,
                    cond=cond,
                    input_modulation=input_modulation,
                    post_modulation=post_modulation,
                    rope=self.mrope,
                    cos=state_cos,
                    sin=state_sin,
                    attn_ctx=state_context,
                )
                if wavefront is not None:
                    wavefront.events[step][index].record(wavefront.stream)
            if wavefront is not None:
                # Action reads this layer's state KV; state never reads action KV.
                main_stream.wait_event(wavefront.events[step][index])
            action_hidden = layer(
                action_hidden,
                cond=cond,
                input_modulation=input_modulation,
                post_modulation=post_modulation,
                rope=self.mrope,
                cos=action_cos,
                sin=action_sin,
                attn_ctx=action_context,
            )
        if wavefront is not None:
            # Finish the fork before the next step can overwrite the shared KV.
            main_stream.wait_stream(wavefront.stream)
        return self.heads.project_action(self.expert_stack.norm(action_hidden))

    def forward(self, batch: LingBotV2ExpertForwardBatch) -> torch.Tensor:
        """Evaluate one supplied diffusion time, outside the fixed Euler schedule."""
        if self._state_plan is None or self._action_plan is None:
            raise RuntimeError("plan_inference must run before expert forward.")
        inputs = self._position_inputs
        state_context, action_context = self._contexts(
            self._state_plan,
            self._action_plan,
            inputs["state_write_indices"],
            inputs["action_write_indices"],
        )
        return self._velocity(
            self.heads.embed_state(batch.state).reshape(batch.state.shape[0], -1),
            batch.x_t,
            self._prepare_condition(batch.time),
            state_context=state_context,
            action_context=action_context,
            state_cos=inputs["state_cos"],
            state_sin=inputs["state_sin"],
            action_cos=inputs["action_cos"],
            action_sin=inputs["action_sin"],
        )

    def _euler_loop(
        self,
        *,
        state_hidden: torch.Tensor,
        noise: torch.Tensor,
        state_plan: PagedAttnPlanHandle,
        action_plan: PagedAttnPlanHandle,
        state_cos: torch.Tensor,
        state_sin: torch.Tensor,
        action_cos: torch.Tensor,
        action_sin: torch.Tensor,
        state_write_indices: torch.Tensor,
        action_write_indices: torch.Tensor,
        wavefront: _LingBotV2WavefrontState | None = None,
    ) -> torch.Tensor:
        """Advance action noise through the cached Euler conditions."""
        if not self._step_conditions or self._dt is None:
            raise RuntimeError("setup must run before the LingBot V2 Euler loop.")
        state_context, action_context = self._contexts(
            state_plan, action_plan, state_write_indices, action_write_indices
        )
        x_t = noise.clone()
        for step, condition in enumerate(self._step_conditions):
            velocity = self._velocity(
                state_hidden,
                x_t,
                condition,
                state_context=state_context,
                action_context=action_context,
                state_cos=state_cos,
                state_sin=state_sin,
                action_cos=action_cos,
                action_sin=action_sin,
                wavefront=wavefront,
                step=step,
            )
            x_t += self._dt * velocity
        return x_t

    def _validate_wavefront_kernels(self, state_hidden: torch.Tensor) -> None:
        """Reject collectives and unsafe GEMMs for every supported batch size."""
        for layer in self.expert_stack.layers:
            if (
                isinstance(layer.mlp, LingBotV2TokenMoE)
                and layer.mlp.mesh.group_size("moe_tp") != 1
            ):
                raise ValueError("LingBot wavefront pipeline requires moe_tp=1.")
            for component in (layer.qkv_proj, layer.o_proj, layer.mlp):
                for linear in component.modules():
                    if not isinstance(linear, LinearBase):
                        continue
                    group = getattr(linear, "group", None)
                    if group is not None:
                        mesh = resolve_mesh(linear.mesh_name)
                        if mesh.group_size(group) != 1:
                            raise ValueError(
                                f"LingBot wavefront pipeline requires {group}=1."
                            )
                    for batch_size in range(1, self.max_batch_size + 1):
                        for capture in (False, True):
                            with graph_capture() if capture else nullcontext():
                                for tokens in (
                                    batch_size,
                                    batch_size * self.chunk_size,
                                ):
                                    selection = resolve_linear_kernel(
                                        linear,
                                        state_hidden,
                                        M=tokens,
                                        N=linear.output_size_per_partition,
                                        K=linear.input_size_per_partition,
                                    )
                                    if selection.kernel_id != "torch.gemm.bf16":
                                        raise ValueError(
                                            "LingBot wavefront pipeline requires "
                                            "torch.gemm.bf16 for concurrent linear layers; "
                                            f"{linear.prefix} selects {selection.kernel_id} "
                                            f"with M={tokens}, capture={capture}."
                                        )

    def _capture_euler_graph(
        self, inputs: dict[str, torch.Tensor]
    ) -> _LingBotV2EulerGraphState:
        """Capture Euler steps and retain their plans, stream, and events."""
        assert self._state_metadata is not None and self._action_metadata is not None
        state_plan = self.state_backend.init_capture_metadata(self._state_metadata)
        action_plan = self.action_backend.init_capture_metadata(self._action_metadata)
        wavefront = (
            _LingBotV2WavefrontState(
                stream=torch.cuda.Stream(device=self.device),
                events=tuple(
                    tuple(torch.cuda.Event() for _ in self.expert_stack.layers)
                    for _ in self._step_conditions
                ),
            )
            if self.use_wavefront_pipeline
            else None
        )

        def captured_loop(**inputs: torch.Tensor) -> torch.Tensor:
            return self._euler_loop(
                state_plan=state_plan,
                action_plan=action_plan,
                wavefront=wavefront,
                **inputs,
            )

        graph = CudaGraph()
        try:
            graph.capture(captured_loop, inputs)
        except BaseException:
            graph.reset()
            raise
        return _LingBotV2EulerGraphState(graph, state_plan, action_plan, wavefront)

    def forward_euler(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Embed the state once, then run or replay the full Euler loop."""
        if self._state_plan is None or self._action_plan is None:
            raise RuntimeError("plan_inference must run before the Euler loop.")
        inputs = {
            "state_hidden": self.heads.embed_state(state).reshape(state.shape[0], -1),
            "noise": noise,
            **self._position_inputs,
        }
        key = (
            self.use_wavefront_pipeline,
            self._layout_key,
            tuple(
                (name, tuple(t.shape), t.dtype, t.device) for name, t in inputs.items()
            ),
        )
        if self.use_cuda_graph and key not in self._failed_graph_keys:
            graph_state = self._euler_graphs.get(key)
            if graph_state is None:
                try:
                    graph_state = self._capture_euler_graph(inputs)
                except (CudaGraphError, RuntimeError) as error:
                    self._failed_graph_keys.add(key)
                    logger.warning_once(
                        "LingBot V2 Euler graph capture failed; using uncaptured execution: %s",
                        error,
                    )
                else:
                    self._euler_graphs[key] = graph_state
            if graph_state is not None:
                assert (
                    self._state_metadata is not None
                    and self._action_metadata is not None
                )
                self.state_backend.replay_metadata(
                    graph_state.state_plan, self._state_metadata
                )
                self.action_backend.replay_metadata(
                    graph_state.action_plan, self._action_metadata
                )
                return graph_state.graph.replay(inputs)
        return self._euler_loop(
            state_plan=self._state_plan, action_plan=self._action_plan, **inputs
        )

    @property
    def cuda_graph_active(self) -> bool:
        return self.use_cuda_graph and bool(self._euler_graphs)

    @property
    def cuda_graph_count(self) -> int:
        return len(self._euler_graphs)

    @property
    def cuda_graph_fallback_count(self) -> int:
        return len(self._failed_graph_keys)

    def close(self) -> None:
        """Release condition tables, graph state, and attention backends."""
        self.reset()
        self._euler_graphs.clear()
        self._failed_graph_keys.clear()
        self._step_conditions = ()
        self._time_schedule = None
        self._dt = None
        _close_backend(self.state_backend, self.action_backend)


__all__ = [
    "LingBotV2CacheAllocation",
    "LingBotV2ExpertForwardBatch",
    "LingBotV2ExpertRunner",
    "LingBotV2PrefixEmbeddings",
    "LingBotV2PrefixForwardBatch",
    "LingBotV2PrefixRunner",
    "LingBotV2VisionForwardBatch",
    "LingBotV2VisionRunner",
]
