"""pi0 model runners: vision, LLM backbone, action expert.

The runner split mirrors pi0.5, but the expert runner is pi0-specific:
it embeds a numeric state token, fuses scalar timestep into action
tokens, runs the expert over ``[state, action_0, ...]``, and projects
only the action-token outputs back to action velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch

from phyai.cache import KVCachePool
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
    get_ar_backend_factory,
    get_diffusion_backend_factory,
)
from phyai.layers.rotary_embedding import RotaryEmbedding
from phyai.models.pi0.modeling_pi0 import (
    ActionTimeHeads,
    PaliGemmaLanguageModel,
    PI0ExpertStack,
    PI0VisionTower,
)
from phyai.payload import LLMForwardBatch, VisionForwardBatch
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner
from phyai.utils import all_ranks_log

logger = logging.getLogger(__name__)


@dataclass
class PI0ExpertForwardBatch:
    """Per-step pi0 expert inputs."""

    state: torch.Tensor
    x_t: torch.Tensor
    time: torch.Tensor


def _ar_attn_proto(stack_layers) -> ARAttention:
    if len(stack_layers) == 0:
        raise ValueError("stack has no layers; cannot read attention metadata.")
    return stack_layers[0].attn


def _diffusion_attn_proto(stack_layers) -> DiffusionAttention:
    if len(stack_layers) == 0:
        raise ValueError("stack has no layers; cannot read attention metadata.")
    return stack_layers[0].attn


class PI0VisionRunner(ModelRunner):
    """SigLIP vision-tower runner with optional CUDA-graph capture."""

    def __init__(
        self,
        vision_tower: PI0VisionTower,
        *,
        num_images: int = 3,
        params_dtype: torch.dtype,
        device: torch.device | str,
        use_cuda_graph: bool = True,
    ) -> None:
        self.vision_tower = vision_tower
        self.num_images = int(num_images)
        if self.num_images <= 0:
            raise ValueError(f"num_images must be positive, got {num_images}.")
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.image_size = int(vision_tower.config.image_size)
        self.num_channels = int(vision_tower.config.num_channels)
        self.graph: CudaGraph | None = None

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI0VisionRunner.setup")
        if not self.use_cuda_graph or self.device.type != "cuda":
            return
        all_ranks_log(
            logger,
            logging.INFO,
            "Entering PI0VisionRunner.setup: capturing vision-tower CUDA graph "
            "at fixed shape (%d, %d, %d, %d).",
            self.num_images,
            self.num_channels,
            self.image_size,
            self.image_size,
        )
        example = {
            "pixel_values": torch.zeros(
                self.num_images,
                self.num_channels,
                self.image_size,
                self.image_size,
                dtype=self.params_dtype,
                device=self.device,
            ),
        }
        self.graph = CudaGraph()
        self.graph.capture(self._fwd, example)

    def _fwd(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_tower(pixel_values)

    def forward(self, batch: VisionForwardBatch) -> torch.Tensor:
        if self.graph is not None:
            return self.graph.replay({"pixel_values": batch.pixel_values})
        return self.vision_tower(batch.pixel_values)


class PI0LLMRunner(ModelRunner):
    """PaliGemma prefix runner."""

    def __init__(
        self,
        paligemma_lm: PaliGemmaLanguageModel,
        rope: RotaryEmbedding,
        kv_pool: KVCachePool,
        *,
        batch_size: int,
        n_per_sample: int,
        params_dtype: torch.dtype,
        device: torch.device | str,
        use_cuda_graph: bool = True,
        max_paged_kv_indices: int | None = None,
    ) -> None:
        self.paligemma_lm = paligemma_lm
        self.rope = rope
        self.kv_pool = kv_pool
        self.batch_size = int(batch_size)
        self.n_per_sample = int(n_per_sample)
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        self.attn_proto: ARAttention = _ar_attn_proto(paligemma_lm.layers)
        self.num_heads = self.attn_proto.num_heads
        self.num_kv_heads = self.attn_proto.num_kv_heads
        self.head_dim = self.attn_proto.head_dim
        self.hidden_size = int(paligemma_lm.config.hidden_size)
        self.max_paged_kv_indices = int(
            max_paged_kv_indices
            if max_paged_kv_indices is not None
            else self.batch_size * self.n_per_sample
        )

        factory = get_ar_backend_factory(self.attn_proto.backend)
        self.attn_backend: ARAttentionBackend = factory(self)
        self.use_cuda_graph = (
            bool(use_cuda_graph)
            and self.attn_backend.supports_capture()
            and self.device.type == "cuda"
        )
        self._capture_plan: ARAttnPlanHandle | None = None
        self.graph: CudaGraph | None = None

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI0LLMRunner.setup")
        self.attn_backend.init_cuda_graph_state(
            max_batch_size=self.batch_size,
            max_num_tokens=self.batch_size * self.n_per_sample,
            max_paged_kv_indices=self.max_paged_kv_indices,
            device=self.device,
            params_dtype=self.params_dtype,
            layer_proto=self.attn_proto,
        )
        if self.use_cuda_graph:
            self._capture_plan = self.attn_backend.init_capture_metadata(
                self._capture_seed_metadata()
            )
            self._capture_graph()

    def _capture_seed_metadata(self) -> ARAttnMetadata:
        cu_q = torch.arange(
            0,
            (self.batch_size + 1) * self.n_per_sample,
            self.n_per_sample,
            dtype=torch.int32,
            device=self.device,
        )
        per_sample_real = min(
            self.n_per_sample, self.max_paged_kv_indices // self.batch_size
        )
        kv_indptr = torch.arange(
            0,
            (self.batch_size + 1) * per_sample_real,
            per_sample_real,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indices = torch.arange(
            self.batch_size * per_sample_real,
            dtype=torch.int32,
            device=self.device,
        )
        last_page = torch.ones(self.batch_size, dtype=torch.int32, device=self.device)
        write_indices = torch.zeros(
            self.batch_size * self.n_per_sample,
            dtype=torch.int64,
            device=self.device,
        )
        return ARAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=self.batch_size,
            num_query_tokens=self.batch_size * self.n_per_sample,
            cu_seqlens_q=cu_q,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page,
            write_indices=write_indices,
        )

    def _capture_graph(self) -> None:
        n = self.batch_size * self.n_per_sample
        example = {
            "hidden_states": torch.zeros(
                n, self.hidden_size, dtype=self.params_dtype, device=self.device
            ),
            "position_ids": torch.zeros(n, dtype=torch.int32, device=self.device),
            "write_indices": torch.zeros(n, dtype=torch.int64, device=self.device),
        }
        self.graph = CudaGraph()
        self.graph.capture(self._fwd, example)

    def _fwd(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        write_indices: torch.Tensor,
    ) -> torch.Tensor:
        ctx = ARAttnCtx(
            backend=self.attn_backend,
            plan=self._capture_plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=write_indices,
        )
        return self.paligemma_lm(hidden_states, position_ids, self.rope, ctx)

    def plan_inference(self, meta: ARAttnMetadata) -> None:
        if self.use_cuda_graph:
            self.attn_backend.replay_metadata(self._capture_plan, meta)
        else:
            self._capture_plan = self.attn_backend.init_forward_metadata(meta)

    def forward(self, batch: LLMForwardBatch) -> None:
        if self.graph is not None:
            self.graph.replay(
                {
                    "hidden_states": batch.hidden_states,
                    "position_ids": batch.position_ids,
                    "write_indices": batch.write_indices,
                }
            )
            return None
        self._fwd(
            hidden_states=batch.hidden_states,
            position_ids=batch.position_ids,
            write_indices=batch.write_indices,
        )
        return None


class PI0ExpertRunner(ModelRunner):
    """One pi0 Euler denoise step with pi0's state/action block mask.

    Each expert layer is run in two calls:

    1. state token attends to cached prefix + fresh state;
    2. action tokens attend to cached prefix + fresh state + fresh actions.

    This matches pi0's prefix/state/action block mask without requiring a
    custom per-query attention mask in the paged backend.
    """

    def __init__(
        self,
        expert_stack: PI0ExpertStack,
        heads: ActionTimeHeads,
        rope: RotaryEmbedding,
        kv_pool: KVCachePool,
        *,
        batch_size: int,
        suffix_len: int,
        chunk_size: int,
        max_state_dim: int,
        max_action_dim: int,
        params_dtype: torch.dtype,
        device: torch.device | str,
        use_cuda_graph: bool = True,
        max_paged_kv_indices: int | None = None,
    ) -> None:
        self.expert_stack = expert_stack
        self.heads = heads
        self.rope = rope
        self.kv_pool = kv_pool
        self.batch_size = int(batch_size)
        self.suffix_len = int(suffix_len)
        self.chunk_size = int(chunk_size)
        self.max_state_dim = int(max_state_dim)
        self.max_action_dim = int(max_action_dim)
        self.expert_hidden = int(heads.expert_hidden)
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        self.attn_proto: DiffusionAttention = _diffusion_attn_proto(expert_stack.layers)
        self.num_heads = self.attn_proto.num_heads
        self.num_kv_heads = self.attn_proto.num_kv_heads
        self.head_dim = self.attn_proto.head_dim
        max_indices = (
            int(max_paged_kv_indices)
            if max_paged_kv_indices is not None
            else self.batch_size * self.suffix_len * 32
        )
        self.max_paged_kv_indices_state = max_indices
        self.max_paged_kv_indices_action = max_indices

        self.pos_ids_state_buf = torch.zeros(
            self.batch_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.pos_ids_action_buf = torch.zeros(
            self.batch_size * self.chunk_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.write_indices_state_buf = torch.zeros(
            self.batch_size,
            dtype=torch.int64,
            device=self.device,
        )
        self.write_indices_action_buf = torch.zeros(
            self.batch_size * self.chunk_size,
            dtype=torch.int64,
            device=self.device,
        )

        factory = get_diffusion_backend_factory(self.attn_proto.backend)
        self.state_attn_backend: DiffusionAttentionBackend = factory(self)
        self.action_attn_backend: DiffusionAttentionBackend = factory(self)
        self.use_cuda_graph = (
            bool(use_cuda_graph)
            and self.state_attn_backend.supports_capture()
            and self.action_attn_backend.supports_capture()
            and self.device.type == "cuda"
        )
        self._state_capture_plan: DiffusionAttnPlanHandle | None = None
        self._action_capture_plan: DiffusionAttnPlanHandle | None = None
        self.graph: CudaGraph | None = None

    def set_write_indices(
        self,
        write_indices_state: torch.Tensor,
        write_indices_action: torch.Tensor,
    ) -> None:
        if write_indices_state.shape != self.write_indices_state_buf.shape:
            raise ValueError(
                f"write_indices_state shape {tuple(write_indices_state.shape)} "
                f"!= {tuple(self.write_indices_state_buf.shape)}."
            )
        if write_indices_action.shape != self.write_indices_action_buf.shape:
            raise ValueError(
                f"write_indices_action shape {tuple(write_indices_action.shape)} "
                f"!= {tuple(self.write_indices_action_buf.shape)}."
            )
        self.write_indices_state_buf.copy_(write_indices_state.to(torch.int64))
        self.write_indices_action_buf.copy_(write_indices_action.to(torch.int64))

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI0ExpertRunner.setup")
        self.state_attn_backend.init_cuda_graph_state(
            max_batch_size=self.batch_size,
            max_num_tokens=self.batch_size,
            max_paged_kv_indices=self.max_paged_kv_indices_state,
            device=self.device,
            params_dtype=self.params_dtype,
            layer_proto=self.attn_proto,
        )
        self.action_attn_backend.init_cuda_graph_state(
            max_batch_size=self.batch_size,
            max_num_tokens=self.batch_size * self.chunk_size,
            max_paged_kv_indices=self.max_paged_kv_indices_action,
            device=self.device,
            params_dtype=self.params_dtype,
            layer_proto=self.attn_proto,
        )
        if self.use_cuda_graph:
            self._state_capture_plan = self.state_attn_backend.init_capture_metadata(
                self._capture_seed_metadata_state()
            )
            self._action_capture_plan = self.action_attn_backend.init_capture_metadata(
                self._capture_seed_metadata_action()
            )
            self._capture_graph()

    def _capture_seed_metadata_state(self) -> DiffusionAttnMetadata:
        cu_q = torch.arange(
            0,
            self.batch_size + 1,
            1,
            dtype=torch.int32,
            device=self.device,
        )
        per_sample_kv = min(
            self.suffix_len * 4,
            self.max_paged_kv_indices_state // self.batch_size,
        )
        kv_indptr = torch.arange(
            0,
            (self.batch_size + 1) * per_sample_kv,
            per_sample_kv,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indices = torch.arange(
            self.batch_size * per_sample_kv,
            dtype=torch.int32,
            device=self.device,
        )
        last_page = torch.ones(self.batch_size, dtype=torch.int32, device=self.device)
        return DiffusionAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=self.batch_size,
            num_query_tokens=self.batch_size,
            cu_seqlens_q=cu_q,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page,
            write_indices=self.write_indices_state_buf,
        )

    def _capture_seed_metadata_action(self) -> DiffusionAttnMetadata:
        cu_q = torch.arange(
            0,
            (self.batch_size + 1) * self.chunk_size,
            self.chunk_size,
            dtype=torch.int32,
            device=self.device,
        )
        per_sample_kv = min(
            self.suffix_len * 4,
            self.max_paged_kv_indices_action // self.batch_size,
        )
        kv_indptr = torch.arange(
            0,
            (self.batch_size + 1) * per_sample_kv,
            per_sample_kv,
            dtype=torch.int32,
            device=self.device,
        )
        kv_indices = torch.arange(
            self.batch_size * per_sample_kv,
            dtype=torch.int32,
            device=self.device,
        )
        last_page = torch.ones(self.batch_size, dtype=torch.int32, device=self.device)
        return DiffusionAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=self.batch_size,
            num_query_tokens=self.batch_size * self.chunk_size,
            cu_seqlens_q=cu_q,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page,
            write_indices=self.write_indices_action_buf,
        )

    def _capture_graph(self) -> None:
        example = {
            "state": torch.zeros(
                self.batch_size,
                self.max_state_dim,
                dtype=self.params_dtype,
                device=self.device,
            ),
            "x_t": torch.zeros(
                self.batch_size,
                self.chunk_size,
                self.max_action_dim,
                dtype=self.params_dtype,
                device=self.device,
            ),
            "time": torch.zeros(
                self.batch_size, dtype=torch.float32, device=self.device
            ),
        }
        self.graph = CudaGraph()
        self.graph.capture(self._fwd, example)

    def _fwd(
        self,
        *,
        state: torch.Tensor,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        state_emb = self.heads.embed_state(state).unsqueeze(1)
        action_emb = self.heads.embed_action_time(x_t, time)
        suffix_h = torch.cat([state_emb, action_emb], dim=1)
        state_h = suffix_h[:, :1, :].reshape(self.batch_size, -1)
        action_h = suffix_h[:, 1:, :].reshape(self.batch_size * self.chunk_size, -1)

        state_ctx = DiffusionAttnCtx(
            backend=self.state_attn_backend,
            plan=self._state_capture_plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=self.write_indices_state_buf,
        )
        action_ctx = DiffusionAttnCtx(
            backend=self.action_attn_backend,
            plan=self._action_capture_plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=self.write_indices_action_buf,
        )
        for layer in self.expert_stack.layers:
            state_h = layer(
                state_h,
                self.pos_ids_state_buf,
                self.rope,
                state_ctx,
            )
            action_h = layer(
                action_h,
                self.pos_ids_action_buf,
                self.rope,
                action_ctx,
            )
        action_h = self.expert_stack.norm(action_h)
        action_out = action_h.view(self.batch_size, self.chunk_size, -1)
        return self.heads.project_action(action_out)

    def plan_inference(
        self,
        state_meta: DiffusionAttnMetadata,
        action_meta: DiffusionAttnMetadata,
    ) -> None:
        if state_meta.position_ids is None:
            raise ValueError(
                "PI0ExpertRunner.plan_inference requires state position_ids."
            )
        if action_meta.position_ids is None:
            raise ValueError("PI0ExpertRunner.plan_inference requires position_ids.")
        self.pos_ids_state_buf.copy_(state_meta.position_ids.to(torch.int32))
        self.pos_ids_action_buf.copy_(action_meta.position_ids.to(torch.int32))
        if self.use_cuda_graph:
            self.state_attn_backend.replay_metadata(
                self._state_capture_plan,
                state_meta,
            )
            self.action_attn_backend.replay_metadata(
                self._action_capture_plan,
                action_meta,
            )
        else:
            self._state_capture_plan = self.state_attn_backend.init_forward_metadata(
                state_meta
            )
            self._action_capture_plan = self.action_attn_backend.init_forward_metadata(
                action_meta
            )

    def forward(self, batch: PI0ExpertForwardBatch) -> torch.Tensor:
        if self.graph is not None:
            return self.graph.replay(
                {
                    "state": batch.state,
                    "x_t": batch.x_t,
                    "time": batch.time,
                }
            )
        return self._fwd(state=batch.state, x_t=batch.x_t, time=batch.time)


__all__ = [
    "PI0ExpertForwardBatch",
    "PI0ExpertRunner",
    "PI0LLMRunner",
    "PI0VisionRunner",
]
