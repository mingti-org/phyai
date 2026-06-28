"""pi0 single-card (world_size=1) end-to-end inference scheduler.

The scheduler mirrors the pi0.5 scheduler for the vision and language
prefix, then uses pi0's three-block suffix layout:

* prefix: image + language, cached once;
* state: one numeric robot-state token, attends to prefix + state;
* action: full action chunk, attends to prefix + state + action.

This is the ``ws1`` variant: single-card / non-distributed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.cache import KVCachePool, StaticCache
from phyai.layers.attention import (
    ARAttnMetadata,
    AttnLayout,
    AttnMode,
    DiffusionAttnMetadata,
)
from phyai.models.pi0.configuration_pi0 import PI0Config
from phyai.models.pi0.model_runner_pi0 import (
    PI0ExpertForwardBatch,
    PI0ExpertRunner,
    PI0LLMRunner,
    PI0VisionRunner,
)
from phyai.models.pi0.modeling_pi0 import PI0Model
from phyai.payload import LLMForwardBatch, VisionForwardBatch
from phyai.runtime.schedule import Scheduler
from phyai.utils.profile import event_scope


def pack_prefix_per_sample_padded(
    image_embs: torch.Tensor,
    lang_embs: torch.Tensor,
    lang_lens: torch.Tensor,
    *,
    n_per_sample: int,
) -> torch.Tensor:
    """Assemble per-sample-padded ``(B * n_per_sample, D)`` prefix buffer."""

    B = image_embs.shape[0]
    n_img = image_embs.shape[1]
    D = image_embs.shape[-1]
    device = image_embs.device
    dtype = image_embs.dtype

    packed = torch.zeros(B * n_per_sample, D, dtype=dtype, device=device)
    lang_lens_list = lang_lens.tolist()
    for b in range(B):
        L_lang = int(lang_lens_list[b])
        base = b * n_per_sample
        packed[base : base + n_img] = image_embs[b]
        if L_lang > 0:
            packed[base + n_img : base + n_img + L_lang] = lang_embs[b, :L_lang]
    return packed


def build_prefix_padded_write_indices(
    real_lens: torch.Tensor,
    *,
    n_per_sample: int,
    prefix_slot_base: int,
    sentinel_slot: int = 0,
) -> torch.Tensor:
    """KV-pool slot index per padded prefix token."""

    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    cu_real = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_real[1:] = torch.cumsum(real64, 0)

    j = torch.arange(n_per_sample, dtype=torch.int64, device=device).unsqueeze(0)
    real_at_b = real64.unsqueeze(1)
    cu_at_b = cu_real[:-1].unsqueeze(1)
    is_real = j < real_at_b
    real_slot = prefix_slot_base + cu_at_b + j
    write = torch.where(
        is_real, real_slot, torch.full_like(real_slot, int(sentinel_slot))
    )
    return write.flatten().to(torch.int64)


def build_state_pos_ids(real_lens: torch.Tensor) -> torch.Tensor:
    """RoPE positions for pi0 state tokens, ``(B,)`` int32."""

    return real_lens.to(torch.int32)


def build_action_pos_ids(real_lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """RoPE positions for pi0 action tokens, ``(B * chunk_size,)`` int32."""

    device = real_lens.device
    base = real_lens.to(torch.int64).unsqueeze(1) + 1
    j = torch.arange(chunk_size, dtype=torch.int64, device=device).unsqueeze(0)
    return (base + j).flatten().to(torch.int32)


def build_pi0_state_paged_kv_indices(
    real_lens: torch.Tensor,
    suffix_len: int,
    *,
    prefix_slot_base: int,
    suffix_slot_base: int,
) -> torch.Tensor:
    """Per-sample KV slots visible to the state query: prefix + state."""

    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    cu_p = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_p[1:] = torch.cumsum(real64, 0)

    total_lens = real64 + 1
    cu_full = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_full[1:] = torch.cumsum(total_lens, 0)
    n_full = int(cu_full[-1])

    arange_full = torch.arange(n_full, dtype=torch.int64, device=device)
    seg_id = torch.searchsorted(cu_full[1:], arange_full, right=True)
    pos_within = arange_full - cu_full[seg_id]
    real_at_seg = real64[seg_id]
    is_prefix = pos_within < real_at_seg
    prefix_slot = prefix_slot_base + cu_p[seg_id] + pos_within
    state_slot = suffix_slot_base + seg_id * suffix_len
    return torch.where(is_prefix, prefix_slot, state_slot).to(torch.int32)


def build_pi0_action_paged_kv_indices(
    real_lens: torch.Tensor,
    suffix_len: int,
    *,
    prefix_slot_base: int,
    suffix_slot_base: int,
) -> torch.Tensor:
    """Per-sample KV slots visible to action queries: prefix + state + action."""

    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    cu_p = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_p[1:] = torch.cumsum(real64, 0)

    total_lens = real64 + suffix_len
    cu_full = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_full[1:] = torch.cumsum(total_lens, 0)
    n_full = int(cu_full[-1])

    arange_full = torch.arange(n_full, dtype=torch.int64, device=device)
    seg_id = torch.searchsorted(cu_full[1:], arange_full, right=True)
    pos_within = arange_full - cu_full[seg_id]
    real_at_seg = real64[seg_id]
    is_prefix = pos_within < real_at_seg
    prefix_slot = prefix_slot_base + cu_p[seg_id] + pos_within
    suffix_slot = suffix_slot_base + seg_id * suffix_len + (pos_within - real_at_seg)
    return torch.where(is_prefix, prefix_slot, suffix_slot).to(torch.int32)


@dataclass
class PI0Request:
    """One pi0 inference request."""

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    lang_lens: torch.Tensor
    state: torch.Tensor
    noise: torch.Tensor | None = None


class PI0WS1Scheduler(Scheduler):
    """Single-card pi0 inference orchestrator with fixed max batch size."""

    def __init__(
        self,
        model: PI0Model,
        *,
        max_batch_size: int = 1,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        cfg: PI0Config = model.config
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.params_dtype = model.params_dtype
        self.cfg = cfg
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")
        self.num_images = cfg.num_images
        self.model = model

        self.image_token_count = cfg.vision.num_patches * self.num_images
        self.n_per_sample = self.image_token_count + cfg.tokenizer_max_length
        self.suffix_len = cfg.suffix_len

        prefix_capacity = self.max_batch_size * self.n_per_sample
        suffix_capacity = self.max_batch_size * self.suffix_len
        self.sentinel_slot = 0
        self.prefix_base = 1
        self.suffix_base = self.prefix_base + prefix_capacity
        total_slots = 1 + prefix_capacity + suffix_capacity

        self.kv_pool = KVCachePool(
            num_layers=cfg.num_layers,
            num_slots=total_slots,
            num_kv_heads=cfg.text.num_key_value_heads,
            head_dim=cfg.text.head_dim,
            dtype=self.params_dtype,
            device=self.device,
        )
        self.prefix_static = StaticCache(
            self.kv_pool, base_offset=self.prefix_base, capacity=prefix_capacity
        )
        self.suffix_static = StaticCache(
            self.kv_pool, base_offset=self.suffix_base, capacity=suffix_capacity
        )

        self.vision_runner = PI0VisionRunner(
            model.vision,
            num_images=self.num_images,
            params_dtype=self.params_dtype,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
        )
        self.llm_runner = PI0LLMRunner(
            model.paligemma_lm,
            model.rope,
            self.kv_pool,
            batch_size=self.max_batch_size,
            n_per_sample=self.n_per_sample,
            params_dtype=self.params_dtype,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
            max_paged_kv_indices=self.max_batch_size * self.n_per_sample,
        )
        self.expert_runner = PI0ExpertRunner(
            model.expert_stack,
            model.heads,
            model.rope,
            self.kv_pool,
            batch_size=self.max_batch_size,
            suffix_len=self.suffix_len,
            chunk_size=cfg.chunk_size,
            max_state_dim=cfg.max_state_dim,
            max_action_dim=cfg.max_action_dim,
            params_dtype=self.params_dtype,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
            max_paged_kv_indices=self.max_batch_size
            * (self.n_per_sample + self.suffix_len),
        )

    def setup(self) -> None:
        self.vision_runner.setup()
        self.llm_runner.setup()
        self.expert_runner.setup()

        write_indices_state = (
            self.suffix_base
            + torch.arange(
                self.max_batch_size,
                dtype=torch.int64,
                device=self.device,
            )
            * self.suffix_len
        )
        action_offsets = torch.arange(
            1,
            self.suffix_len,
            dtype=torch.int64,
            device=self.device,
        )
        write_indices_action = (
            write_indices_state.unsqueeze(1) + action_offsets.unsqueeze(0)
        ).flatten()
        self.expert_runner.set_write_indices(
            write_indices_state,
            write_indices_action,
        )

    @torch.no_grad()
    def step(self, request: PI0Request) -> torch.Tensor:
        """Run one inference; return action chunk ``(B, chunk, action_dim)``."""

        cfg = self.cfg
        device = self.device
        dtype = self.params_dtype
        self._validate(request)

        actual_B = int(request.pixel_values.shape[0])
        max_B = self.max_batch_size

        lang_lens_actual = request.lang_lens.to(device=device, dtype=torch.int32)
        real_lens = torch.zeros(max_B, dtype=torch.int32, device=device)
        real_lens[:actual_B] = lang_lens_actual + self.image_token_count
        n_real_total = int(real_lens.sum())

        with event_scope("pi0.vision_loop"):
            pixel_values = request.pixel_values.to(device=device, dtype=dtype)
            image_embs: torch.Tensor | None = None
            for b in range(actual_B):
                vision_out = self.vision_runner.forward(
                    VisionForwardBatch(pixel_values=pixel_values[b])
                )
                flat = vision_out.flatten(0, 1)
                if image_embs is None:
                    image_embs = torch.zeros(
                        max_B, *flat.shape, dtype=flat.dtype, device=flat.device
                    )
                image_embs[b] = flat
            assert image_embs is not None

        with event_scope("pi0.lang_pack"):
            if actual_B < max_B:
                input_ids_padded = torch.zeros(
                    max_B,
                    cfg.tokenizer_max_length,
                    dtype=torch.int64,
                    device=device,
                )
                input_ids_padded[:actual_B] = request.input_ids.to(
                    device=device, dtype=torch.int64
                )
                lang_lens_padded = torch.zeros(max_B, dtype=torch.int64, device=device)
                lang_lens_padded[:actual_B] = lang_lens_actual.to(torch.int64)
            else:
                input_ids_padded = request.input_ids.to(
                    device=device, dtype=torch.int64
                )
                lang_lens_padded = lang_lens_actual.to(torch.int64)

            lang_embs = self.model.paligemma_lm.embed_lang(input_ids_padded)
            if lang_embs.dtype != dtype:
                lang_embs = lang_embs.to(dtype)
            packed = pack_prefix_per_sample_padded(
                image_embs,
                lang_embs,
                lang_lens_padded,
                n_per_sample=self.n_per_sample,
            )

        with event_scope("pi0.llm_prefix_plan"):
            self.prefix_static.reset()
            self.suffix_static.reset()
            _ = self.prefix_static.allocate(n_real_total)
            _ = self.suffix_static.allocate(max_B * self.suffix_len)

            cu_q_prefix = torch.arange(
                0,
                (max_B + 1) * self.n_per_sample,
                self.n_per_sample,
                dtype=torch.int32,
                device=device,
            )
            paged_kv_indptr_prefix = torch.zeros(
                max_B + 1, dtype=torch.int32, device=device
            )
            paged_kv_indptr_prefix[1:] = torch.cumsum(real_lens, 0)
            paged_kv_indices_prefix = torch.arange(
                self.prefix_base,
                self.prefix_base + n_real_total,
                dtype=torch.int32,
                device=device,
            )
            paged_kv_last_prefix = (real_lens > 0).to(torch.int32)
            write_indices = build_prefix_padded_write_indices(
                real_lens,
                n_per_sample=self.n_per_sample,
                prefix_slot_base=self.prefix_base,
                sentinel_slot=self.sentinel_slot,
            )
            position_ids = torch.arange(
                self.n_per_sample, dtype=torch.int32, device=device
            ).repeat(max_B)
            prefix_meta = ARAttnMetadata(
                mode=AttnMode.PREFILL,
                layout=AttnLayout.RAGGED_3D,
                batch_size=max_B,
                num_query_tokens=max_B * self.n_per_sample,
                cu_seqlens_q=cu_q_prefix,
                paged_kv_indptr=paged_kv_indptr_prefix,
                paged_kv_indices=paged_kv_indices_prefix,
                paged_kv_last_page_len=paged_kv_last_prefix,
                write_indices=write_indices,
                position_ids=position_ids,
            )
            self.llm_runner.plan_inference(prefix_meta)

        with event_scope("pi0.llm_prefix_fwd"):
            self.llm_runner.forward(
                LLMForwardBatch(
                    hidden_states=packed,
                    position_ids=position_ids,
                    write_indices=write_indices,
                )
            )

        with event_scope("pi0.expert_plan"):
            cu_q_state = torch.arange(
                0,
                max_B + 1,
                1,
                dtype=torch.int32,
                device=device,
            )
            state_indptr = torch.zeros(max_B + 1, dtype=torch.int32, device=device)
            state_indptr[1:] = torch.cumsum(real_lens + 1, 0)
            state_indices = build_pi0_state_paged_kv_indices(
                real_lens,
                self.suffix_len,
                prefix_slot_base=self.prefix_base,
                suffix_slot_base=self.suffix_base,
            )
            state_meta = DiffusionAttnMetadata(
                mode=AttnMode.PREFILL,
                layout=AttnLayout.RAGGED_3D,
                batch_size=max_B,
                num_query_tokens=max_B,
                cu_seqlens_q=cu_q_state,
                paged_kv_indptr=state_indptr,
                paged_kv_indices=state_indices,
                paged_kv_last_page_len=torch.ones(
                    max_B, dtype=torch.int32, device=device
                ),
                position_ids=build_state_pos_ids(real_lens),
            )

            cu_q_action = torch.arange(
                0,
                (max_B + 1) * cfg.chunk_size,
                cfg.chunk_size,
                dtype=torch.int32,
                device=device,
            )
            action_indptr = torch.zeros(max_B + 1, dtype=torch.int32, device=device)
            action_indptr[1:] = torch.cumsum(real_lens + self.suffix_len, 0)
            action_indices = build_pi0_action_paged_kv_indices(
                real_lens,
                self.suffix_len,
                prefix_slot_base=self.prefix_base,
                suffix_slot_base=self.suffix_base,
            )
            action_meta = DiffusionAttnMetadata(
                mode=AttnMode.PREFILL,
                layout=AttnLayout.RAGGED_3D,
                batch_size=max_B,
                num_query_tokens=max_B * cfg.chunk_size,
                cu_seqlens_q=cu_q_action,
                paged_kv_indptr=action_indptr,
                paged_kv_indices=action_indices,
                paged_kv_last_page_len=torch.ones(
                    max_B, dtype=torch.int32, device=device
                ),
                position_ids=build_action_pos_ids(real_lens, cfg.chunk_size),
            )
            self.expert_runner.plan_inference(state_meta, action_meta)

        with event_scope("pi0.expert_loop"):
            state = torch.zeros(
                max_B,
                cfg.max_state_dim,
                dtype=dtype,
                device=device,
            )
            state[:actual_B] = request.state.to(device=device, dtype=dtype)
            if request.noise is None:
                x_t = torch.randn(
                    max_B,
                    cfg.chunk_size,
                    cfg.max_action_dim,
                    dtype=dtype,
                    device=device,
                )
            else:
                x_t = torch.zeros(
                    max_B,
                    cfg.chunk_size,
                    cfg.max_action_dim,
                    dtype=dtype,
                    device=device,
                )
                x_t[:actual_B] = request.noise.to(device=device, dtype=dtype)

            dt = -1.0 / cfg.num_inference_steps
            for step in range(cfg.num_inference_steps):
                with event_scope("pi0.expert_step"):
                    t = 1.0 + step * dt
                    time = torch.full(
                        (max_B,),
                        t,
                        dtype=torch.float32,
                        device=device,
                    )
                    v_t = self.expert_runner.forward(
                        PI0ExpertForwardBatch(state=state, x_t=x_t, time=time)
                    )
                    x_t = x_t + dt * v_t.to(x_t.dtype)

        return x_t[:actual_B]

    def _validate(self, req: PI0Request) -> None:
        cfg = self.cfg
        if req.pixel_values.dim() != 5:
            raise ValueError(
                f"pixel_values must be 5-D (B, num_images, C, image_size, "
                f"image_size); got shape {tuple(req.pixel_values.shape)}."
            )
        actual_B = int(req.pixel_values.shape[0])
        if not 1 <= actual_B <= self.max_batch_size:
            raise ValueError(
                f"pixel_values batch dim {actual_B} not in "
                f"[1, max_batch_size={self.max_batch_size}]."
            )
        if req.pixel_values.shape[1] != self.num_images:
            raise ValueError(
                f"pixel_values has {req.pixel_values.shape[1]} cameras, "
                f"expected {self.num_images}."
            )
        if req.pixel_values.shape[2] != cfg.vision.num_channels:
            raise ValueError(
                f"pixel_values channel dim {req.pixel_values.shape[2]} != "
                f"num_channels {cfg.vision.num_channels}."
            )
        if req.pixel_values.shape[-2:] != (
            cfg.vision.image_size,
            cfg.vision.image_size,
        ):
            raise ValueError(
                f"pixel_values H/W {tuple(req.pixel_values.shape[-2:])} != "
                f"image_size {cfg.vision.image_size}."
            )
        if req.input_ids.shape != (actual_B, cfg.tokenizer_max_length):
            raise ValueError(
                f"input_ids shape {tuple(req.input_ids.shape)} != "
                f"(B, tokenizer_max_length)=({actual_B}, "
                f"{cfg.tokenizer_max_length})."
            )
        if req.lang_lens.shape != (actual_B,):
            raise ValueError(
                f"lang_lens shape {tuple(req.lang_lens.shape)} != (B,)=({actual_B},)."
            )
        if req.state.shape != (actual_B, cfg.max_state_dim):
            raise ValueError(
                f"state shape {tuple(req.state.shape)} != "
                f"(B, max_state_dim)=({actual_B}, {cfg.max_state_dim})."
            )
        if req.noise is not None and req.noise.shape != (
            actual_B,
            cfg.chunk_size,
            cfg.max_action_dim,
        ):
            raise ValueError(
                f"noise shape {tuple(req.noise.shape)} != "
                f"(B, chunk_size, max_action_dim)="
                f"({actual_B}, {cfg.chunk_size}, {cfg.max_action_dim})."
            )


__all__ = [
    "PI0Request",
    "PI0WS1Scheduler",
    "build_action_pos_ids",
    "build_pi0_action_paged_kv_indices",
    "build_pi0_state_paged_kv_indices",
    "build_prefix_padded_write_indices",
    "build_state_pos_ids",
    "pack_prefix_per_sample_padded",
]
