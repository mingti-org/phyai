"""LingBot-VLA 2.0 runners: vision, cached Qwen prefix, and action expert."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from phyai.cache import KVCachePool, StaticCache
from phyai.layers.attention import (
    ARAttention,
    ARAttentionBackend,
    ARAttnCtx,
    ARAttnMetadata,
    ARAttnPlanHandle,
    AttnLayout,
    AttnMode,
    DiffusionAttention,
    DiffusionAttentionBackend,
    DiffusionAttnCtx,
    DiffusionAttnMetadata,
    DiffusionAttnPlanHandle,
)
from phyai.layers.rotary_embedding import InterleavedMRotaryEmbedding
from phyai.models.lingbot_v2.modeling_lingbotv2 import (
    LingBotV2ActionTimeHeads,
    LingBotV2ExpertStack,
    LingBotV2Model,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
    build_lingbot_v2_mrope_position_ids,
)
from phyai.runtime.cuda_graph_manager import CudaGraph, CudaGraphError
from phyai.runtime.model_runner import ModelRunner
from phyai.utils import all_ranks_log

logger = logging.getLogger(__name__)


@dataclass
class LingBotV2VisionForwardBatch:
    """Packed patch vectors and one (t,h,w) row per image."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


@dataclass
class LingBotV2PrefixForwardBatch:
    """Fixed-stride prefix rows plus DeepStack data for real image positions."""

    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    write_indices: torch.Tensor
    visual_pos_masks: torch.Tensor | None = None
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


def _ar_attn_proto(text_model: Qwen3VLTextModel) -> ARAttention:
    if not text_model.layers:
        raise ValueError("text model has no layers.")
    attention = text_model.layers[0].attn
    if not isinstance(attention, ARAttention):
        raise TypeError("LingBot prefix runner requires text attn_kind='ar'.")
    return attention


def _diffusion_attn_proto(expert_stack: LingBotV2ExpertStack) -> DiffusionAttention:
    if not expert_stack.layers:
        raise ValueError("expert stack has no layers.")
    return expert_stack.layers[0].attn


def _close_backend(backend: object) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


@dataclass(frozen=True)
class _LingBotV2EagerPlan(ARAttnPlanHandle, DiffusionAttnPlanHandle):
    """Host-planned ragged slices for official FP32 eager attention."""

    q_slices: tuple[tuple[int, int], ...]
    kv_slices: tuple[tuple[int, int], ...]
    paged_kv_indices: torch.Tensor


@dataclass
class _LingBotV2EulerGraphState:
    """One fixed-layout full-Euler CUDA graph and its bound metadata."""

    graph: CudaGraph
    state_plan: _LingBotV2EagerPlan
    action_plan: _LingBotV2EagerPlan
    state_position_ids: torch.Tensor
    action_position_ids: torch.Tensor
    state_write_indices: torch.Tensor
    action_write_indices: torch.Tensor


def _offset_slices(offsets: torch.Tensor, name: str) -> tuple[tuple[int, int], ...]:
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError(f"{name} must be a 1-D offset tensor with at least 2 entries.")
    values = tuple(int(value) for value in offsets.detach().cpu().tolist())
    return tuple(zip(values[:-1], values[1:]))


def _build_eager_plan(
    meta: ARAttnMetadata | DiffusionAttnMetadata,
) -> _LingBotV2EagerPlan:
    if (
        meta.cu_seqlens_q is None
        or meta.paged_kv_indptr is None
        or meta.paged_kv_indices is None
    ):
        raise ValueError(
            "LingBot official eager attention requires cu_seqlens_q, "
            "paged_kv_indptr, and paged_kv_indices."
        )
    q_slices = _offset_slices(meta.cu_seqlens_q, "cu_seqlens_q")
    kv_slices = _offset_slices(meta.paged_kv_indptr, "paged_kv_indptr")
    if len(q_slices) != len(kv_slices):
        raise ValueError(
            f"query batch size {len(q_slices)} does not match KV batch "
            f"size {len(kv_slices)}."
        )
    return _LingBotV2EagerPlan(
        q_slices=q_slices,
        kv_slices=kv_slices,
        paged_kv_indices=meta.paged_kv_indices.to(torch.int64),
    )


def _official_eager_paged_attention(
    layer: ARAttention | DiffusionAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx: ARAttnCtx | DiffusionAttnCtx,
) -> torch.Tensor:
    """Match LingBot V2's FP32 eager joint attention over paged K/V."""

    if ctx.mode.is_idle():
        return q.new_zeros(q.shape)
    if not isinstance(ctx.plan, _LingBotV2EagerPlan):
        raise TypeError(
            "LingBot official eager attention expected _LingBotV2EagerPlan, "
            f"got {type(ctx.plan).__name__}."
        )

    q = q.float()
    k = k.float()
    v = v.float()
    ctx.kv_pool.write_kv(layer.layer_id, ctx.write_indices, k, v)

    outputs: list[torch.Tensor] = []
    for (q_start, q_end), (kv_start, kv_end) in zip(
        ctx.plan.q_slices,
        ctx.plan.kv_slices,
    ):
        query = q[q_start:q_end]
        slots = ctx.plan.paged_kv_indices[kv_start:kv_end]
        key, value = ctx.kv_pool.gather_kv(layer.layer_id, slots)
        if layer.num_heads != layer.num_kv_heads:
            groups = layer.num_heads // layer.num_kv_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)

        query_heads = query.transpose(0, 1)
        key_heads = key.transpose(0, 1)
        value_heads = value.transpose(0, 1)
        attention_scores = torch.matmul(
            query_heads,
            key_heads.transpose(-2, -1),
        )
        attention_scores = attention_scores * layer.scale
        if layer.causal:
            query_length = query.shape[0]
            key_length = key.shape[0]
            query_positions = (
                torch.arange(query_length, device=q.device) + key_length - query_length
            )
            key_positions = torch.arange(key_length, device=q.device)
            causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(
                ~causal_mask.unsqueeze(0),
                -2.3819763e38,
            )
        probabilities = torch.softmax(
            attention_scores,
            dim=-1,
            dtype=torch.float32,
        )
        outputs.append(torch.matmul(probabilities, value_heads).transpose(0, 1))
    return torch.cat(outputs, dim=0)


class _LingBotV2OfficialARBackend(ARAttentionBackend):
    """Model-local AR backend matching the official eager FP32 path."""

    name = "lingbot-v2-official-eager"

    def init_forward_metadata(self, meta: ARAttnMetadata) -> ARAttnPlanHandle:
        """Create an eager plan for one autoregressive attention layout."""

        return _build_eager_plan(meta)

    def forward(
        self,
        layer: ARAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ARAttnCtx,
    ) -> torch.Tensor:
        """Execute the official eager autoregressive attention path."""

        return _official_eager_paged_attention(layer, q, k, v, ctx)


class _LingBotV2OfficialDiffusionBackend(DiffusionAttentionBackend):
    """Model-local action backend matching the official eager FP32 path."""

    name = "lingbot-v2-official-eager"

    def supports_capture(self) -> bool:
        """The fixed-layout Torch operator sequence is CUDA-graph safe."""

        return True

    def init_forward_metadata(
        self,
        meta: DiffusionAttnMetadata,
    ) -> DiffusionAttnPlanHandle:
        """Create an eager plan for one diffusion attention layout."""

        return _build_eager_plan(meta)

    def forward(
        self,
        layer: DiffusionAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: DiffusionAttnCtx,
    ) -> torch.Tensor:
        """Execute the official eager diffusion attention path."""

        return _official_eager_paged_attention(layer, q, k, v, ctx)


class LingBotV2VisionRunner(ModelRunner):
    """Variable-grid Qwen3-VL vision runner.

    The vision helpers inspect grid_thw on the host, so this path remains
    eager until PHYAI has a graph-safe, bucketed Qwen3-VL vision planner.
    """

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

    def setup(self) -> None:
        """Initialize the eager attention backend and vision model state."""

        all_ranks_log(
            logger,
            logging.INFO,
            "LingBotV2VisionRunner uses eager variable-grid execution.",
        )

    def reset(self) -> None:
        """Vision execution has no request-local runtime state."""

    def close(self) -> None:
        """Vision eager execution owns no external runtime resources."""

    def forward(
        self, batch: LingBotV2VisionForwardBatch
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode packed image patches and return features with rotary inputs."""

        return self.vision_model(
            batch.pixel_values.to(device=self.device, dtype=self.params_dtype),
            batch.image_grid_thw.to(device=self.device, dtype=torch.int64),
        )


class LingBotV2PrefixRunner(ModelRunner):
    """Qwen3-VL prefix runtime, including shared KV storage and condition cache."""

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
        self.kv_pool = KVCachePool(
            num_layers=self.config.num_layers,
            num_slots=1 + int(prefix_capacity) + int(suffix_capacity),
            num_kv_heads=self.config.text.num_key_value_heads,
            head_dim=self.config.text.head_dim,
            # Official V2 promotes Q/K/V before MRoPE and stores that FP32 K/V
            # dictionary for reuse during every Euler step.
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
        self.attn_proto = _ar_attn_proto(self.text_model)
        self.attn_backend: ARAttentionBackend = _LingBotV2OfficialARBackend()
        self._plan: ARAttnPlanHandle | None = None
        self._current_query: torch.Tensor | None = None
        self._future_query: torch.Tensor | None = None
        self._vision_start: torch.Tensor | None = None
        self._vision_end: torch.Tensor | None = None

    def setup(self) -> None:
        """Allocate prefix attention metadata and model caches."""

        all_ranks_log(logger, logging.INFO, "Entering LingBotV2PrefixRunner.setup")
        self.attn_backend.init_cuda_graph_state(
            max_batch_size=self.max_batch_size,
            max_num_tokens=self.max_num_tokens,
            max_paged_kv_indices=self.max_paged_kv_indices,
            device=self.device,
            params_dtype=self.params_dtype,
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
        """Clear request-local plans and rewind the shared KV allocators."""

        self._plan = None
        self.prefix_cache.reset()
        self.suffix_cache.reset()

    def allocate_request_cache(
        self,
        *,
        prefix_tokens: int,
        suffix_tokens: int,
    ) -> LingBotV2CacheAllocation:
        """Reserve compact prefix/suffix KV slots for the current request."""

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
        """Run request text embedding and reuse setup-time learned conditions."""

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
        """Build the model-specific 3-D MRoPE coordinates through the runner."""

        return build_lingbot_v2_mrope_position_ids(
            input_ids,
            attention_mask,
            image_grid_thw,
            image_token_id=self.config.image_token_id,
            spatial_merge_size=self.config.vision.spatial_merge_size,
        )

    def plan_inference(self, metadata: ARAttnMetadata) -> None:
        """Build the attention plan for the next prefix request."""

        self._plan = self.attn_backend.init_forward_metadata(metadata)

    def forward(self, batch: LingBotV2PrefixForwardBatch) -> None:
        """Run the prefix transformer and write its key/value cache."""

        if self._plan is None:
            raise RuntimeError("plan_inference must be called before prefix forward.")
        cos, sin = self.mrope.get_cos_sin(batch.position_ids)
        context = ARAttnCtx(
            backend=self.attn_backend,
            plan=self._plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=batch.write_indices,
        )
        self.text_model(
            batch.hidden_states,
            cos=cos,
            sin=sin,
            attn_ctx=context,
            visual_pos_masks=batch.visual_pos_masks,
            deepstack_visual_embeds=batch.deepstack_visual_embeds,
        )
        return None

    def close(self) -> None:
        """Release prefix attention resources and clear cached tensors."""

        self.reset()
        _close_backend(self.attn_backend)
        self._current_query = None
        self._future_query = None
        self._vision_start = None
        self._vision_end = None


class LingBotV2ExpertRunner(ModelRunner):
    """State/action expert runtime with an optional full-Euler CUDA graph."""

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
    ) -> None:
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
        self.state_backend: DiffusionAttentionBackend = (
            _LingBotV2OfficialDiffusionBackend()
        )
        self.action_backend: DiffusionAttentionBackend = (
            _LingBotV2OfficialDiffusionBackend()
        )
        self.cuda_graph_requested = bool(use_cuda_graph)
        self.use_cuda_graph = (
            self.cuda_graph_requested
            and self.device.type == "cuda"
            and self.state_backend.supports_capture()
            and self.action_backend.supports_capture()
        )
        self._state_plan: DiffusionAttnPlanHandle | None = None
        self._action_plan: DiffusionAttnPlanHandle | None = None
        self._state_position_ids: torch.Tensor | None = None
        self._action_position_ids: torch.Tensor | None = None
        self._state_write_indices: torch.Tensor | None = None
        self._action_write_indices: torch.Tensor | None = None
        self._time_schedule: torch.Tensor | None = None
        self._dt: torch.Tensor | None = None
        self._euler_graphs: dict[tuple[object, ...], _LingBotV2EulerGraphState] = {}
        self._failed_graph_keys: set[tuple[object, ...]] = set()

    def setup(self) -> None:
        """Initialize expert attention backends and lazy graph bookkeeping."""

        all_ranks_log(logger, logging.INFO, "Entering LingBotV2ExpertRunner.setup")
        common = dict(
            max_batch_size=self.max_batch_size,
            max_paged_kv_indices=self.max_paged_kv_indices,
            device=self.device,
            params_dtype=self.params_dtype,
            layer_proto=self.attn_proto,
        )
        self.state_backend.init_cuda_graph_state(
            max_num_tokens=self.max_batch_size,
            **common,
        )
        self.action_backend.init_cuda_graph_state(
            max_num_tokens=self.max_num_tokens,
            **common,
        )
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        # Match official FlowMatchingV2.sample_actions: dt and time live in the
        # model dtype, and time advances through repeated ``time += dt`` rather
        # than a Python-float closed form. Recording the resulting table keeps
        # that exact low-precision sequence inside the unrolled graph.
        self._dt = torch.tensor(
            -1.0 / self.num_inference_steps,
            dtype=self.params_dtype,
            device=self.device,
        )
        time = torch.tensor(1.0, dtype=self.params_dtype, device=self.device)
        schedule: list[torch.Tensor] = []
        for _ in range(self.num_inference_steps):
            schedule.append(time.clone())
            time += self._dt
        self._time_schedule = torch.stack(schedule)
        all_ranks_log(
            logger,
            logging.INFO,
            "LingBotV2ExpertRunner CUDA Graph is %s; full Euler graphs are "
            "captured lazily per fixed attention layout.",
            "enabled" if self.use_cuda_graph else "disabled",
        )

    def reset(self) -> None:
        """Clear all request-specific diffusion plans and position buffers."""

        self._state_plan = None
        self._action_plan = None
        self._state_position_ids = None
        self._action_position_ids = None
        self._state_write_indices = None
        self._action_write_indices = None

    def plan_inference(
        self,
        state_metadata: DiffusionAttnMetadata,
        action_metadata: DiffusionAttnMetadata,
    ) -> None:
        """Plan state and action attention for one diffusion inference."""

        if state_metadata.position_ids is None or action_metadata.position_ids is None:
            raise ValueError("LingBot expert metadata requires 3-D MRoPE position_ids.")
        if (
            state_metadata.write_indices is None
            or action_metadata.write_indices is None
        ):
            raise ValueError("LingBot expert metadata requires KV write_indices.")
        self._state_position_ids = state_metadata.position_ids
        self._action_position_ids = action_metadata.position_ids
        self._state_write_indices = state_metadata.write_indices
        self._action_write_indices = action_metadata.write_indices
        self._state_plan = self.state_backend.init_forward_metadata(state_metadata)
        self._action_plan = self.action_backend.init_forward_metadata(action_metadata)

    def forward(self, batch: LingBotV2ExpertForwardBatch) -> torch.Tensor:
        """Run the expert stack and return the predicted action chunk."""

        required = (
            self._state_plan,
            self._action_plan,
            self._state_position_ids,
            self._action_position_ids,
            self._state_write_indices,
            self._action_write_indices,
        )
        if any(value is None for value in required):
            raise RuntimeError("plan_inference must be called before expert forward.")

        return self._forward_with_runtime(
            state=batch.state,
            x_t=batch.x_t,
            time=batch.time,
            state_plan=self._require_eager_plan(self._state_plan, "state"),
            action_plan=self._require_eager_plan(self._action_plan, "action"),
            state_position_ids=self._state_position_ids,
            action_position_ids=self._action_position_ids,
            state_write_indices=self._state_write_indices,
            action_write_indices=self._action_write_indices,
        )

    @staticmethod
    def _require_eager_plan(
        plan: DiffusionAttnPlanHandle | None,
        name: str,
    ) -> _LingBotV2EagerPlan:
        if not isinstance(plan, _LingBotV2EagerPlan):
            raise TypeError(
                f"LingBot V2 {name} attention expected an eager plan, "
                f"got {type(plan).__name__}."
            )
        return plan

    def _forward_with_runtime(
        self,
        *,
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
        state_plan: _LingBotV2EagerPlan,
        action_plan: _LingBotV2EagerPlan,
        state_position_ids: torch.Tensor,
        action_position_ids: torch.Tensor,
        state_write_indices: torch.Tensor,
        action_write_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate one velocity using explicit graph-stable runtime tensors."""

        batch_size = int(state.shape[0])
        state_hidden = self.heads.embed_state(state).reshape(batch_size, -1)
        action_hidden, cond = self.heads.embed_action_time(x_t, time)

        state_cos, state_sin = self.mrope.get_cos_sin(state_position_ids)
        action_cos, action_sin = self.mrope.get_cos_sin(action_position_ids)
        state_cos = state_cos.reshape(-1, state_cos.shape[-1])
        state_sin = state_sin.reshape(-1, state_sin.shape[-1])

        state_context = DiffusionAttnCtx(
            backend=self.state_backend,
            plan=state_plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=state_write_indices,
        )
        action_context = DiffusionAttnCtx(
            backend=self.action_backend,
            plan=action_plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=action_write_indices,
        )

        for layer in self.expert_stack.layers:
            state_hidden = layer(
                state_hidden,
                cond=cond,
                rope=self.mrope,
                cos=state_cos,
                sin=state_sin,
                attn_ctx=state_context,
            )
            action_hidden = layer(
                action_hidden,
                cond=cond,
                rope=self.mrope,
                cos=action_cos,
                sin=action_sin,
                attn_ctx=action_context,
            )
        action_hidden = self.expert_stack.norm(action_hidden)
        return self.heads.project_action(action_hidden)

    def _euler_loop(
        self,
        *,
        state: torch.Tensor,
        noise: torch.Tensor,
        state_plan: _LingBotV2EagerPlan,
        action_plan: _LingBotV2EagerPlan,
        state_position_ids: torch.Tensor,
        action_position_ids: torch.Tensor,
        state_write_indices: torch.Tensor,
        action_write_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run the exact fixed BF16 time schedule and Euler update sequence."""

        if self._time_schedule is None or self._dt is None:
            raise RuntimeError("setup must run before the LingBot V2 Euler loop.")
        batch_size = int(state.shape[0])
        x_t = noise.clone()
        for step in range(self.num_inference_steps):
            velocity = self._forward_with_runtime(
                state=state,
                x_t=x_t,
                time=self._time_schedule[step].expand(batch_size),
                state_plan=state_plan,
                action_plan=action_plan,
                state_position_ids=state_position_ids,
                action_position_ids=action_position_ids,
                state_write_indices=state_write_indices,
                action_write_indices=action_write_indices,
            )
            x_t += self._dt * velocity
        return x_t

    def _current_graph_key(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[object, ...]:
        state_plan = self._require_eager_plan(self._state_plan, "state")
        action_plan = self._require_eager_plan(self._action_plan, "action")
        return (
            tuple(state.shape),
            state.dtype,
            state.device,
            tuple(noise.shape),
            noise.dtype,
            noise.device,
            state_plan.q_slices,
            state_plan.kv_slices,
            tuple(state_plan.paged_kv_indices.shape),
            action_plan.q_slices,
            action_plan.kv_slices,
            tuple(action_plan.paged_kv_indices.shape),
        )

    def _capture_euler_graph(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> _LingBotV2EulerGraphState:
        state_plan = self._require_eager_plan(self._state_plan, "state")
        action_plan = self._require_eager_plan(self._action_plan, "action")
        assert self._state_position_ids is not None
        assert self._action_position_ids is not None
        assert self._state_write_indices is not None
        assert self._action_write_indices is not None

        captured_state_plan = _LingBotV2EagerPlan(
            q_slices=state_plan.q_slices,
            kv_slices=state_plan.kv_slices,
            paged_kv_indices=state_plan.paged_kv_indices.clone(),
        )
        captured_action_plan = _LingBotV2EagerPlan(
            q_slices=action_plan.q_slices,
            kv_slices=action_plan.kv_slices,
            paged_kv_indices=action_plan.paged_kv_indices.clone(),
        )
        state_position_ids = self._state_position_ids.clone()
        action_position_ids = self._action_position_ids.clone()
        state_write_indices = self._state_write_indices.clone()
        action_write_indices = self._action_write_indices.clone()

        def _captured_loop(*, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
            return self._euler_loop(
                state=state,
                noise=noise,
                state_plan=captured_state_plan,
                action_plan=captured_action_plan,
                state_position_ids=state_position_ids,
                action_position_ids=action_position_ids,
                state_write_indices=state_write_indices,
                action_write_indices=action_write_indices,
            )

        graph = CudaGraph()
        graph.capture(_captured_loop, {"state": state, "noise": noise})
        return _LingBotV2EulerGraphState(
            graph=graph,
            state_plan=captured_state_plan,
            action_plan=captured_action_plan,
            state_position_ids=state_position_ids,
            action_position_ids=action_position_ids,
            state_write_indices=state_write_indices,
            action_write_indices=action_write_indices,
        )

    def _refresh_graph_metadata(self, graph_state: _LingBotV2EulerGraphState) -> None:
        state_plan = self._require_eager_plan(self._state_plan, "state")
        action_plan = self._require_eager_plan(self._action_plan, "action")
        assert self._state_position_ids is not None
        assert self._action_position_ids is not None
        assert self._state_write_indices is not None
        assert self._action_write_indices is not None
        graph_state.state_plan.paged_kv_indices.copy_(state_plan.paged_kv_indices)
        graph_state.action_plan.paged_kv_indices.copy_(action_plan.paged_kv_indices)
        graph_state.state_position_ids.copy_(self._state_position_ids)
        graph_state.action_position_ids.copy_(self._action_position_ids)
        graph_state.state_write_indices.copy_(self._state_write_indices)
        graph_state.action_write_indices.copy_(self._action_write_indices)

    def forward_euler(self, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Run all Euler steps eagerly or replay a fixed-layout CUDA graph."""

        required = (
            self._state_plan,
            self._action_plan,
            self._state_position_ids,
            self._action_position_ids,
            self._state_write_indices,
            self._action_write_indices,
        )
        if any(value is None for value in required):
            raise RuntimeError("plan_inference must be called before the Euler loop.")
        batch_size = int(state.shape[0])
        state_plan = self._require_eager_plan(self._state_plan, "state")
        action_plan = self._require_eager_plan(self._action_plan, "action")
        if not self.use_cuda_graph:
            return self._euler_loop(
                state=state,
                noise=noise,
                state_plan=state_plan,
                action_plan=action_plan,
                state_position_ids=self._state_position_ids,
                action_position_ids=self._action_position_ids,
                state_write_indices=self._state_write_indices,
                action_write_indices=self._action_write_indices,
            )

        key = self._current_graph_key(state, noise)
        if key in self._failed_graph_keys:
            return self._euler_loop(
                state=state,
                noise=noise,
                state_plan=state_plan,
                action_plan=action_plan,
                state_position_ids=self._state_position_ids,
                action_position_ids=self._action_position_ids,
                state_write_indices=self._state_write_indices,
                action_write_indices=self._action_write_indices,
            )
        graph_state = self._euler_graphs.get(key)
        if graph_state is None:
            all_ranks_log(
                logger,
                logging.INFO,
                "Capturing LingBot V2 full %d-step Euler CUDA graph for "
                "batch_size=%d.",
                self.num_inference_steps,
                batch_size,
            )
            try:
                graph_state = self._capture_euler_graph(state, noise)
            except (CudaGraphError, RuntimeError) as error:
                self._failed_graph_keys.add(key)
                all_ranks_log(
                    logger,
                    logging.WARNING,
                    "LingBot V2 CUDA Graph capture failed for batch_size=%d; "
                    "falling back to eager Euler execution: %s",
                    batch_size,
                    error,
                )
                return self._euler_loop(
                    state=state,
                    noise=noise,
                    state_plan=state_plan,
                    action_plan=action_plan,
                    state_position_ids=self._state_position_ids,
                    action_position_ids=self._action_position_ids,
                    state_write_indices=self._state_write_indices,
                    action_write_indices=self._action_write_indices,
                )
            self._euler_graphs[key] = graph_state
        self._refresh_graph_metadata(graph_state)
        return graph_state.graph.replay({"state": state, "noise": noise})

    @property
    def cuda_graph_active(self) -> bool:
        """Whether at least one full-Euler graph was captured successfully."""

        return self.use_cuda_graph and bool(self._euler_graphs)

    @property
    def cuda_graph_count(self) -> int:
        """Number of fixed-layout Euler graphs currently cached."""

        return len(self._euler_graphs)

    @property
    def cuda_graph_fallback_count(self) -> int:
        """Number of layouts that failed capture and now execute eagerly."""

        return len(self._failed_graph_keys)

    def close(self) -> None:
        """Release expert backends, graph captures, and request metadata."""

        self.reset()
        self._euler_graphs.clear()
        self._failed_graph_keys.clear()
        _close_backend(self.state_backend)
        _close_backend(self.action_backend)


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
