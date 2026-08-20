"""pi0.5 single-card (world_size=1) end-to-end inference scheduler.

The scheduler is the user-facing entry point for pi0.5 inference on a
single card. It owns:

* The :class:`KVCachePool` and the prefix / suffix
  :class:`StaticCache` slabs over it.
* Three runners — :class:`PI05VisionRunner`,
  :class:`PI05LLMRunner`, :class:`PI05ExpertRunner` — each with its
  own captured CUDA graph (when the engine backend supports paged
  attention).

A single ``step()`` call runs deterministic inference. ``rollout_step()``
reuses the same prefix path and runs an RLinf-compatible ODE/SDE trajectory,
returning action chains and policy log probabilities for RL training.

Multi-batch support — fixed ``max_batch_size`` at construction; each
``step()`` accepts any ``B ∈ [1, max_batch_size]``. Smaller batches pad
up to ``max_batch_size`` along the sample axis with sentinel-routed
unused rows so the captured graphs always run at constant shape. The
padding mechanism reuses the per-token sentinel slot already in place
for intra-sample lang padding: padded samples claim zero real prefix
tokens, so all of their q-rows route to slot 0 (sentinel) and never
contribute to (or read from) any real sample's attention.

This is the ``ws1`` variant — single-card / non-distributed. Continuous
batching, preemption, and tensor parallel are out of scope here.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import torch

from phyai.cache import KVCachePool, StaticCache
from phyai.layers.attention import (
    ARAttnMetadata,
    AttnLayout,
    AttnMode,
    DiffusionAttnMetadata,
)
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.model_runner_pi05 import (
    PI05ExpertRunner,
    PI05LLMRunner,
    PI05VisionRunner,
)
from phyai.models.pi05.modeling_pi05 import PI05Model, value_from_prefix
from phyai.payload import (
    LLMForwardBatch,
    VisionForwardBatch,
)
from phyai.runtime.schedule import Scheduler
from phyai.utils.profile import event_scope

# ============================================================================ #
# Batch-layout helpers — pi0.5 specific, only consumed by step() below.        #
# ============================================================================ #


def build_prefix_padded_write_indices(
    real_lens: torch.Tensor,
    *,
    n_per_sample: int,
    prefix_slot_base: int,
    sentinel_slot: int = 0,
) -> torch.Tensor:
    """KV-pool slot index per padded prefix token, ``(B * n_per_sample,)`` int64.

    Real token ``b * n_per_sample + j`` (with ``j < real_lens[b]``) writes
    to ``prefix_slot_base + cu_real[b] + j``. Padding rows write to
    ``sentinel_slot`` (typically 0). Directly consumable by
    :meth:`KVCachePool.write_kv`.

    The same ``j < real_lens[b]`` mask handles inter-sample padding too:
    setting ``real_lens[b] = 0`` for unused samples routes every one of
    their ``n_per_sample`` rows to the sentinel slot.
    """
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


def build_suffix_pos_ids(real_lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """RoPE positions for suffix tokens, ``(B * chunk_size,)`` int32.

    Sample ``b``'s chunk sits at positions
    ``[real_len_b, real_len_b + 1, ..., real_len_b + chunk_size - 1]``
    so the joint attention K layout (cached prefix + fresh suffix) sees
    one coherent ``[0..real_len_b + chunk_size - 1]`` per sample.
    Padded samples (``real_len_b == 0``) get positions ``[0, chunk_size)``
    — fine because they self-attend only over their own chunk.
    """
    device = real_lens.device
    base = real_lens.to(torch.int64).unsqueeze(1)
    j = torch.arange(chunk_size, dtype=torch.int64, device=device).unsqueeze(0)
    return (base + j).flatten().to(torch.int32)


def build_joint_paged_kv_indices(
    real_lens: torch.Tensor,
    chunk_size: int,
    *,
    prefix_slot_base: int,
    suffix_slot_base: int,
    n_full: int | None = None,
) -> torch.Tensor:
    """Per-sample interleaved slot list for the joint-attention wrapper.

    Output layout per sample:
    ``[prefix_b0_slots (real_len_0), suffix_b0_slots (chunk_size),
       prefix_b1_slots (real_len_1), suffix_b1_slots (chunk_size), ...]``
    concatenated end-to-end, returned as ``(N_full,)`` int32 where
    ``N_full = sum(real_lens) + B * chunk_size``.

    Padded samples (``real_len_b == 0``) contribute only their suffix
    slots — the expert step for those rows self-attends within its own
    chunk, with no real-prefix participation.

    ``n_full`` may be supplied (``sum(real_lens) + B * chunk_size``) when
    the caller already knows it on the host, to skip the blocking
    ``int(cu_full[-1])`` device→host read.
    """
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)

    cu_p = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_p[1:] = torch.cumsum(real64, 0)
    full_lens = real64 + chunk_size
    cu_full = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_full[1:] = torch.cumsum(full_lens, 0)
    if n_full is None:
        n_full = int(cu_full[-1])

    arange_full = torch.arange(n_full, dtype=torch.int64, device=device)
    seg_id = torch.searchsorted(cu_full[1:], arange_full, right=True)
    pos_within = arange_full - cu_full[seg_id]
    real_at_seg = real64[seg_id]
    is_prefix = pos_within < real_at_seg

    prefix_slot = prefix_slot_base + cu_p[seg_id] + pos_within
    suffix_slot = suffix_slot_base + seg_id * chunk_size + (pos_within - real_at_seg)
    return torch.where(is_prefix, prefix_slot, suffix_slot).to(torch.int32)


@dataclass
class PI05Request:
    """One pi0.5 inference request — canonical, already-preprocessed tensors.

    ``phyai`` is strict about inputs: image resize/normalize, tokenization, and
    state discretization happen in the caller's processor
    (``phyai_utils_tools.models.pi05.PI05Processor``), **not** here. The
    scheduler accepts only the canonical tensors that processor produces:

    ``pixel_values`` is the stacked camera tensor
    ``(B, num_images, C, image_size, image_size)`` — already resized to the
    SigLIP grid, on the model device/dtype. ``num_images`` is fixed at scheduler
    construction; ``B`` may be any value in ``[1, max_batch_size]`` (the
    scheduler pads up internally). H/W must equal ``image_size`` (validated; no
    resize is performed).

    ``input_ids`` is the padded language-token tensor (``(B,
    tokenizer_max_length)`` int64); ``lang_lens`` carries the
    un-padded real lengths.

    ``noise`` is optional; if ``None`` the scheduler samples a fresh
    Gaussian (over the *real* batch only — the padded tail uses zeros
    that get discarded with the rest of the padded output).
    """

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    lang_lens: torch.Tensor
    noise: torch.Tensor | None = None


@dataclass
class PI05RolloutRequest(PI05Request):
    """Pi0.5 trajectory-sampling request for an RL rollout backend.

    The canonical model inputs are inherited from :class:`PI05Request`.
    ``noise`` is the initial Gaussian action state. ``step_noise`` supplies
    one standard-normal sample per denoise transition and makes a rollout
    exactly reproducible when provided.

    RLinf's default pi0.5 policy uses ``flow_sde`` at one randomly selected
    denoise step and ordinary flow ODE updates elsewhere. The selected step
    is shared by the request batch, matching RLinf's rollout implementation.
    Supplying ``denoise_inds`` accepts either that scalar step or RLinf's
    expanded ``(B, num_steps)`` representation.

    ``action_chunk`` and ``action_dim`` only crop ``prev_logprobs``; the
    returned model action and chains always retain the full model-space
    ``(chunk_size, max_action_dim)`` geometry needed by actor recomputation.
    """

    step_noise: torch.Tensor | None = None
    denoise_inds: int | torch.Tensor | None = None
    noise_method: Literal["flow_ode", "flow_sde"] = "flow_sde"
    noise_level: float = 0.5
    action_chunk: int | None = None
    action_dim: int | None = None
    joint_logprob: bool = False
    ignore_last: bool = False
    safe_get_logprob: bool = False
    compute_values: bool = False


@dataclass
class PI05RolloutOutput:
    """Model-space trajectory data produced by :meth:`rollout_step`.

    Field names, shapes, and reductions match RLinf's OpenPI rollout output:
    ``actions`` is ``(B, chunk, action_dim)``, ``chains`` is
    ``(B, num_steps + 1, chunk, action_dim)``, and ``denoise_inds`` is
    ``(B, num_steps)``. ``prev_logprobs`` uses the caller-requested action
    crop. ``prev_values`` is ``(B, 1)``: either one prefix-pooled value or the
    mean suffix value over denoise steps, according to ``value_after_vlm``;
    it is zero when values are not requested. Environment-specific action
    unnormalization intentionally remains in the caller's processor.
    """

    actions: torch.Tensor
    chains: torch.Tensor
    denoise_inds: torch.Tensor
    prev_logprobs: torch.Tensor
    prev_values: torch.Tensor


@dataclass
class _PI05Layout:
    """Per-``(actual_B, lang_lens)`` attention layout, cached across steps.

    Everything here depends only on the batch shape and the per-sample
    real lengths — not on the image/text *values* — so for a fixed
    request shape it is identical on every ``step()`` and across the 10
    Euler steps. Building it once and reusing it eliminates the
    per-step metadata rebuild and (more importantly) the host↔device
    syncs that used to serialize the pipeline (``int(real_lens.sum())``,
    ``int(cu_full[-1])``, the ``lang_lens.tolist()`` pack loop).

    ``lang_mask`` is the per-bucket bool mask used by the vectorized prefix
    pack. ``prefix_mask`` extends it with valid image tokens and is the exact
    mask used by RLinf-compatible prefix value pooling. ``n_real_total`` is
    the host-side count of real prefix tokens.
    """

    n_real_total: int
    n_per_sample: int
    position_ids: torch.Tensor
    write_indices: torch.Tensor
    lang_mask: torch.Tensor
    prefix_mask: torch.Tensor
    prefix_meta: ARAttnMetadata
    joint_meta: DiffusionAttnMetadata


@dataclass
class _PI05PreparedPrefix:
    """Request-local result of prefix preparation."""

    actual_batch_size: int
    prefix_values: torch.Tensor | None = None


class PI05WS1Scheduler(Scheduler):
    """Single-card (world_size=1) pi0.5 inference orchestrator with multi-batch support."""

    def __init__(
        self,
        model: PI05Model,
        *,
        max_batch_size: int = 1,
        num_images: int = 3,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
        capture_rollout: bool = False,
    ) -> None:
        cfg: PI05Config = model.config
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.params_dtype = model.params_dtype
        self.cfg = cfg
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")
        self.num_images = int(num_images)
        if self.num_images <= 0:
            raise ValueError(f"num_images must be positive, got {num_images}.")
        self.model = model

        # Layout knobs.
        self.image_token_count = cfg.vision.num_patches * self.num_images
        self.n_per_sample = self.image_token_count + cfg.tokenizer_max_length

        # Prefix-length buckets. Each LLM prefix runs at a fixed graph
        # shape, so instead of always padding the lang budget to
        # ``tokenizer_max_length`` (wasting LLM GEMM work on padding rows),
        # we capture one graph per bucket and pad up to the smallest bucket
        # >= the real lang length. ``_lang_buckets`` are lang-token counts
        # (capped at and always including ``tokenizer_max_length`` as the
        # fallback); ``_n_per_sample_buckets`` are the matching prefix
        # lengths the LLM runner captures graphs for.
        tok_max = cfg.tokenizer_max_length
        self._lang_buckets = sorted(
            {b for b in (16, 48, 112) if 0 < b < tok_max} | {tok_max}
        )
        self._n_per_sample_buckets = [
            self.image_token_count + b for b in self._lang_buckets
        ]

        # KVCachePool: sentinel slot 0 + prefix slab + suffix slab.
        # Sized for ``max_batch_size``; smaller actual batches reuse a
        # subset and route their padded tail to the sentinel slot.
        prefix_capacity = self.max_batch_size * self.n_per_sample
        suffix_capacity = self.max_batch_size * cfg.chunk_size
        # +1 slot for the sentinel so padding rows (both intra-sample
        # lang padding and inter-sample batch padding) have somewhere
        # harmless to land.
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

        # Runners (constructed but not yet warmed up; setup() does it).
        # Each runner takes the individual sub-modules it needs — there is
        # no PI05Model dependency at the runner layer. The LLM and expert
        # runners are sized at ``max_batch_size`` because their captured
        # graphs run at constant shape.
        self.vision_runner = PI05VisionRunner(
            model.vision,
            num_images=self.num_images,
            params_dtype=self.params_dtype,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
        )
        # max paged-kv-indices for prefix self-attn: max_B * n_per_sample
        # (worst case all real tokens). For joint attn during the expert
        # phase: max_B * (n_per_sample + chunk_size).
        self.llm_runner = PI05LLMRunner(
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
        self.expert_runner = PI05ExpertRunner(
            model.expert_stack,
            model.heads,
            model.rope,
            self.kv_pool,
            batch_size=self.max_batch_size,
            chunk_size=cfg.chunk_size,
            max_action_dim=cfg.max_action_dim,
            params_dtype=self.params_dtype,
            device=self.device,
            suffix_value_head=(model.value_head if not cfg.value_after_vlm else None),
            critic_action_chunk=cfg.critic_action_chunk,
            chunk_critic_input=cfg.chunk_critic_input,
            detach_critic_input=cfg.detach_critic_input,
            use_cuda_graph=use_cuda_graph,
            capture_rollout=capture_rollout,
            max_paged_kv_indices=self.max_batch_size
            * (self.n_per_sample + cfg.chunk_size),
        )

        # ``(num_inference_steps, expert_hidden)`` lookup table —
        # populated by :meth:`setup` after ``load_pretrained``. Each
        # row is the full time-MLP output for one Euler step's ``t``;
        # broadcast to ``(max_B, expert_hidden)`` via ``expand`` and fed
        # straight to the expert runner's captured input buffer.
        self.time_emb_table: torch.Tensor | None = None

        # Attention-layout cache, keyed by (actual_B, tuple(lang_lens)).
        # A fixed request shape hits the cache on every step, so the
        # metadata is built (and the flashinfer wrappers re-planned) only
        # when the shape actually changes. ``_last_layout_key`` gates the
        # per-step ``plan_inference`` calls: when the layout is unchanged
        # the wrappers' static buffers already hold the right values, so
        # re-planning (and its host sync) is skipped entirely.
        self._layout_cache: dict[tuple, _PI05Layout] = {}
        self._last_layout_key: tuple | None = None

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def setup(self) -> None:
        """Warm up runners and capture every CUDA graph.

        Must be called **after** ``load_pretrained`` — :attr:`time_emb_table`
        is computed here from the (now-real) ``time_mlp_in/out`` weights,
        so calling this on freshly-allocated weights would bake garbage
        values into the table. The table (and the Euler schedule) is bound
        to the expert runner **before** it captures, because the expert
        graph now unrolls the whole denoise loop and reads the table
        in-graph.
        """
        self.vision_runner.setup()
        self.llm_runner.setup(self._n_per_sample_buckets)

        # Precompute the time-embedding table for the linear flow-matching
        # schedule ``t = 1.0 + step * (-1/N)``. ``embed_time`` is the full
        # MLP (sinusoidal -> Linear -> SiLU -> Linear -> SiLU); its output
        # depends only on the time scalar and the time-MLP weights, both
        # fixed across inferences, so a one-time precompute keeps those
        # matmuls out of the captured Euler loop.
        N = self.cfg.num_inference_steps
        ts = torch.linspace(1.0, 1.0 / N, N, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self.time_emb_table = self.model.heads.embed_time(ts).contiguous()
        # Bind the schedule before the expert runner captures: its graph
        # unrolls all N steps and reads ``time_emb_table[step]`` in-graph.
        self.expert_runner.bind_euler_schedule(
            self.time_emb_table, dt=-1.0 / N, num_steps=N
        )

        # Suffix write indices are constant — bind them once, sized for
        # max_batch_size. Sample ``b``'s chunk_size tokens write to
        # ``[suffix_slot_base + b*chunk_size, suffix_slot_base + (b+1)*chunk_size)``.
        # Padded samples (b >= actual_B) write garbage K/V to their slot
        # range, but only their own expert step ever reads back from
        # those slots (and we discard the padded rows of the output).
        write_indices_suffix = torch.arange(
            self.suffix_base,
            self.suffix_base + self.max_batch_size * self.cfg.chunk_size,
            dtype=torch.int64,
            device=self.device,
        )
        self.expert_runner.set_write_indices(write_indices_suffix)

        # Capture the expert graph last — the schedule + write indices it
        # reads in-graph must already be bound above.
        self.expert_runner.setup()

    # ------------------------------------------------------------------ #
    # Step (one inference)                                               #
    # ------------------------------------------------------------------ #

    def _prepare_prefix(self, request: PI05Request) -> _PI05PreparedPrefix:
        """Validate inputs, populate prefix K/V, and plan expert attention."""
        cfg = self.cfg
        device = self.device
        dtype = self.params_dtype
        self._validate(request)

        actual_B = int(request.pixel_values.shape[0])
        if isinstance(request, PI05RolloutRequest):
            self._validate_rollout(request, actual_B)
        max_B = self.max_batch_size

        # Resolve the attention layout for this request shape. It depends
        # only on (actual_B, per-sample real lengths) — not on the image /
        # text values — so a fixed request shape hits the cache on every
        # step and across all 10 Euler steps. Reading ``lang_lens`` to the
        # host (one tiny sync for the key) replaces the several large syncs
        # the per-step rebuild used to incur; a cache hit rebuilds nothing.
        lang_lens_cpu = tuple(int(x) for x in request.lang_lens.tolist())
        n_per_sample = self._bucket_n_per_sample(max(lang_lens_cpu, default=0))
        key = (actual_B, lang_lens_cpu)
        layout = self._layout_cache.get(key)
        if layout is None:
            layout = self._build_layout(actual_B, lang_lens_cpu, n_per_sample)
            self._layout_cache[key] = layout
        plan_changed = key != self._last_layout_key

        # 1. Vision pass — replay the captured (num_images, C, H, W) graph
        # once per real robot. The runner's output aliases its static buffer,
        # so we eagerly copy each replay into a pre-allocated
        # ``(max_B, image_token_count, D)`` slot before the next replay
        # overwrites it. Padded rows stay zero (their image embeddings
        # are written into the packed prefix at sentinel-routed rows,
        # so the values don't propagate anywhere). ``pixel_values`` is the
        # caller's canonical, already-resized tensor; we only move it to the
        # model device/dtype here (no resize — phyai is strict).
        with event_scope("pi05.vision_loop"):
            pixel_values = request.pixel_values.to(device=device, dtype=dtype)
            image_embs: torch.Tensor | None = None
            for b in range(actual_B):
                vision_out = self.vision_runner.forward(
                    VisionForwardBatch(pixel_values=pixel_values[b])
                )
                flat = vision_out.flatten(0, 1)  # (image_token_count, D)
                if image_embs is None:
                    image_embs = torch.zeros(
                        max_B, *flat.shape, dtype=flat.dtype, device=flat.device
                    )
                image_embs[b] = flat
            assert image_embs is not None  # actual_B >= 1 enforced by _validate

        # 2. Lang embed (eager) + vectorized per-sample padded packing.
        # The pack is a fixed-stride scatter (image block then a
        # mask-zeroed lang block) driven by the cached ``lang_mask`` — no
        # host sync, no Python per-sample loop, bit-identical to the old
        # per-sample gather.
        with event_scope("pi05.lang_pack"):
            if actual_B < max_B:
                input_ids_padded = torch.zeros(
                    max_B, cfg.tokenizer_max_length, dtype=torch.int64, device=device
                )
                input_ids_padded[:actual_B] = request.input_ids.to(
                    device=device, dtype=torch.int64
                )
            else:
                input_ids_padded = request.input_ids.to(
                    device=device, dtype=torch.int64
                )

            lang_embs = self.model.paligemma_lm.embed_lang(input_ids_padded)
            if lang_embs.dtype != dtype:
                lang_embs = lang_embs.to(dtype)
            packed = self._pack_prefix(image_embs, lang_embs, layout)

        # 3. Plan the prefix layout — only when the layout changed. The
        # flashinfer wrapper keeps its planned buffers across steps
        # (use_cuda_graph=True), so an unchanged layout needs no re-plan
        # and incurs no host sync.
        with event_scope("pi05.llm_prefix_plan"):
            if plan_changed:
                self.prefix_static.reset()
                self.suffix_static.reset()
                _ = self.prefix_static.allocate(layout.n_real_total)
                _ = self.suffix_static.allocate(max_B * cfg.chunk_size)
                self.llm_runner.plan_inference(layout.prefix_meta)

        # 4. Prefix forward — populates the cache pool. ``packed`` is a
        # fresh contiguous tensor; the runner's ``replay`` will copy it
        # into the captured input buffer, so no scheduler-side staging.
        # Attention metadata was already staged on the runner via
        # ``plan_inference`` above; the batch carries only the per-call
        # variable inputs.
        with event_scope("pi05.llm_prefix_fwd"):
            llm_batch = LLMForwardBatch(
                hidden_states=packed,
                position_ids=layout.position_ids,
                write_indices=layout.write_indices,
            )
            prefix_out = self.llm_runner.forward(
                llm_batch, n_per_sample=layout.n_per_sample
            )

        with event_scope("pi05.prefix_value"):
            prefix_values = None
            if (
                isinstance(request, PI05RolloutRequest)
                and request.compute_values
                and cfg.value_after_vlm
            ):
                assert self.model.value_head is not None  # checked by validation
                prefix_out_3d = prefix_out.view(max_B, layout.n_per_sample, -1)
                prefix_values = value_from_prefix(
                    self.model.value_head,
                    prefix_out_3d[:actual_B],
                    layout.prefix_mask[:actual_B],
                    mode=cfg.value_vlm_mode,
                )

        # 5. Plan the expert joint-attention metadata — same gating as the
        # prefix: only re-plan when the layout changed.
        with event_scope("pi05.expert_plan"):
            if plan_changed:
                self.expert_runner.plan_inference(layout.joint_meta)
        self._last_layout_key = key

        return _PI05PreparedPrefix(actual_B, prefix_values)

    @torch.no_grad()
    def step(self, request: PI05Request) -> torch.Tensor:
        """Run deterministic inference and return the full model action chunk."""
        cfg = self.cfg
        device = self.device
        dtype = self.params_dtype
        prepared = self._prepare_prefix(request)
        actual_B = prepared.actual_batch_size
        max_B = self.max_batch_size

        with event_scope("pi05.expert_loop"):
            if self.time_emb_table is None:
                raise RuntimeError(
                    "PI05WS1Scheduler.step() called before setup(); "
                    "the time-embedding lookup table has not been built."
                )
            if request.noise is None:
                noise = torch.randn(
                    max_B,
                    cfg.chunk_size,
                    cfg.max_action_dim,
                    dtype=dtype,
                    device=device,
                )
            else:
                noise = torch.zeros(
                    max_B,
                    cfg.chunk_size,
                    cfg.max_action_dim,
                    dtype=dtype,
                    device=device,
                )
                noise[:actual_B] = request.noise.to(device=device, dtype=dtype)
            x_t = self.expert_runner.forward(noise)
        # ``x_t`` aliases the captured graph's static output buffer — clone
        # it (and drop the padded tail) so the result survives the next step.
        return x_t[:actual_B].clone()

    @torch.no_grad()
    def rollout_step(self, request: PI05RolloutRequest) -> PI05RolloutOutput:
        """Sample an RL trajectory with RLinf-compatible flow semantics.

        Prefix preparation and attention planning are identical to ordinary
        inference. The expert phase keeps fp32 integration state, injects all
        random inputs explicitly, and returns the trajectory data required to
        recompute policy log probabilities during actor training.
        """
        prepared = self._prepare_prefix(request)
        actual_B = prepared.actual_batch_size

        noise = self._prepare_rollout_noise(request, actual_B)
        step_noise = self._prepare_step_noise(request, actual_B)
        denoise_inds = self._resolve_denoise_inds(request, actual_B)
        sigmas = self._build_rollout_sigmas(request, denoise_inds, actual_B)
        safe_logprob = torch.tensor(
            request.safe_get_logprob, dtype=torch.bool, device=self.device
        )

        with event_scope("pi05.expert_rollout_loop"):
            actions, chains, transition_logprobs, step_values = (
                self.expert_runner.forward_rollout(
                    noise, step_noise, sigmas, safe_logprob
                )
            )

        action_chunk = (
            request.action_chunk
            if request.action_chunk is not None
            else self.cfg.critic_action_chunk
        )
        action_dim = (
            request.action_dim
            if request.action_dim is not None
            else self.cfg.max_action_dim
        )
        cropped = transition_logprobs[:actual_B, :, :action_chunk, :action_dim]
        if request.joint_logprob:
            initial_noise = noise[:actual_B, :action_chunk, :action_dim]
            initial = self._initial_logprob(
                initial_noise, safe=request.safe_get_logprob
            )
            prev_logprobs = torch.cat([initial[:, None], cropped], dim=1).mean(dim=1)
        else:
            batch_indices = torch.arange(actual_B, device=self.device)
            prev_logprobs = cropped[batch_indices, denoise_inds[:actual_B, 0]]
        if request.compute_values:
            if self.cfg.value_after_vlm:
                assert prepared.prefix_values is not None
                prev_values = prepared.prefix_values[:, None]
            else:
                prev_values = step_values[:actual_B].mean(dim=-1, keepdim=True)
        else:
            prev_values = torch.zeros(
                actual_B, 1, dtype=torch.float32, device=self.device
            )

        # Clone every graph-backed result before a subsequent replay can
        # overwrite its static output buffers.
        return PI05RolloutOutput(
            actions=actions[:actual_B].clone(),
            chains=chains[:actual_B].clone(),
            denoise_inds=denoise_inds[:actual_B].clone(),
            prev_logprobs=prev_logprobs.clone(),
            prev_values=prev_values.clone(),
        )

    def _prepare_rollout_noise(
        self, request: PI05RolloutRequest, actual_B: int
    ) -> torch.Tensor:
        """Return fp32 initial noise padded to the captured batch size."""
        shape = (actual_B, self.cfg.chunk_size, self.cfg.max_action_dim)
        if request.noise is None:
            real_noise = torch.randn(shape, dtype=torch.float32, device=self.device)
        else:
            real_noise = request.noise.to(device=self.device, dtype=torch.float32)
        noise = torch.zeros(
            self.max_batch_size,
            self.cfg.chunk_size,
            self.cfg.max_action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        noise[:actual_B] = real_noise
        return noise

    def _prepare_step_noise(
        self, request: PI05RolloutRequest, actual_B: int
    ) -> torch.Tensor:
        """Return one fp32 standard-normal tensor per denoise transition."""
        cfg = self.cfg
        if request.step_noise is None:
            # Separate draws mirror RLinf's one ``sample_noise`` call per step.
            real_noise = torch.stack(
                [
                    torch.randn(
                        actual_B,
                        cfg.chunk_size,
                        cfg.max_action_dim,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    for _ in range(cfg.num_inference_steps)
                ],
                dim=1,
            )
        else:
            real_noise = request.step_noise.to(device=self.device, dtype=torch.float32)
        step_noise = torch.zeros(
            self.max_batch_size,
            cfg.num_inference_steps,
            cfg.chunk_size,
            cfg.max_action_dim,
            dtype=torch.float32,
            device=self.device,
        )
        step_noise[:actual_B] = real_noise
        return step_noise

    def _resolve_denoise_inds(
        self, request: PI05RolloutRequest, actual_B: int
    ) -> torch.Tensor:
        """Return RLinf's expanded ``(max_B, num_steps)`` index tensor."""
        num_steps = self.cfg.num_inference_steps
        provided = request.denoise_inds
        if provided is None:
            if request.joint_logprob:
                real = torch.arange(num_steps, dtype=torch.int64)[None].expand(
                    actual_B, -1
                )
            else:
                last = num_steps - 2 if request.ignore_last else num_steps - 1
                selected = random.randint(0, last)
                real = torch.full((actual_B, num_steps), selected, dtype=torch.int64)
        elif isinstance(provided, int):
            real = torch.full((actual_B, num_steps), provided, dtype=torch.int64)
        else:
            real = provided.to(dtype=torch.int64, device="cpu")
            if real.ndim == 0:
                real = real.expand(actual_B, num_steps)
            elif real.shape == (actual_B,):
                real = real[:, None].expand(-1, num_steps)
            elif real.shape != (actual_B, num_steps):
                raise ValueError(
                    "denoise_inds shape must be scalar, (B,), or "
                    f"(B, num_steps)=({actual_B}, {num_steps}); got "
                    f"{tuple(real.shape)}."
                )

        if bool(((real < 0) | (real >= num_steps)).any()):
            raise ValueError(
                f"denoise_inds must be in [0, {num_steps}), got {real.tolist()}."
            )
        if request.joint_logprob:
            expected = torch.arange(num_steps, dtype=torch.int64)[None].expand(
                actual_B, -1
            )
            if not torch.equal(real, expected):
                raise ValueError(
                    "joint_logprob=True requires denoise_inds to be "
                    "arange(num_steps) for every sample."
                )
        elif not bool((real == real[0, 0]).all()):
            raise ValueError(
                "non-joint RLinf rollout requires one denoise index shared "
                "across the entire request batch."
            )
        padded = torch.zeros(
            self.max_batch_size,
            num_steps,
            dtype=torch.int64,
            device=self.device,
        )
        padded[:actual_B] = real.to(self.device)
        return padded

    def _build_rollout_sigmas(
        self,
        request: PI05RolloutRequest,
        denoise_inds: torch.Tensor,
        actual_B: int,
    ) -> torch.Tensor:
        """Build per-sample, per-step SDE sigmas using RLinf's schedule."""
        num_steps = self.cfg.num_inference_steps
        sigmas = torch.zeros(
            self.max_batch_size,
            num_steps,
            dtype=torch.float32,
            device=self.device,
        )
        if request.noise_method == "flow_ode" or request.noise_level == 0:
            return sigmas

        timesteps = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=self.device)
        timesteps = torch.cat([timesteps, torch.zeros(1, device=self.device)])
        denominator = torch.where(timesteps == 1, timesteps[1], timesteps)
        schedule = (
            request.noise_level * torch.sqrt(timesteps / (1.0 - denominator))[:-1]
        )
        if request.joint_logprob:
            sigmas[:actual_B] = schedule[None]
        else:
            step_ids = torch.arange(num_steps, device=self.device)[None]
            selected = denoise_inds[:actual_B, :1]
            sigmas[:actual_B] = torch.where(step_ids == selected, schedule[None], 0.0)
        return sigmas

    @staticmethod
    def _standard_normal_logprob(sample: torch.Tensor) -> torch.Tensor:
        """Elementwise log probability under a standard normal."""
        return (
            -0.5 * torch.log(torch.full_like(sample, 2.0 * torch.pi))
            - 0.5 * sample.square()
        )

    @classmethod
    def _initial_logprob(cls, sample: torch.Tensor, *, safe: bool) -> torch.Tensor:
        """Match RLinf's initial joint-policy log-probability convention."""
        if safe:
            return -sample.square()
        return cls._standard_normal_logprob(sample)

    # ------------------------------------------------------------------ #
    # Layout build + pack (cached, sync-free)                            #
    # ------------------------------------------------------------------ #

    def _bucket_n_per_sample(self, max_lang: int) -> int:
        """Per-sample prefix length for the smallest bucket covering ``max_lang``.

        ``max_lang`` is the largest real lang length in the request. Picks
        the smallest ``_lang_buckets`` entry ``>= max_lang`` (falling back
        to ``tokenizer_max_length``) and adds the image-token count.
        """
        lang_bucket = next(
            (b for b in self._lang_buckets if b >= max_lang),
            self.cfg.tokenizer_max_length,
        )
        return self.image_token_count + lang_bucket

    def _build_layout(
        self, actual_B: int, lang_lens_cpu: tuple[int, ...], n_per_sample: int
    ) -> _PI05Layout:
        """Build the attention layout for ``(actual_B, lang_lens_cpu)``.

        ``n_per_sample`` is the per-sample prefix length for the chosen
        length bucket (``image_token_count + lang_bucket``); the lang
        budget ``lang_bucket = n_per_sample - image_token_count`` must be
        ``>= max(lang_lens_cpu)`` so no real lang token is dropped.

        Every per-sample scalar comes from ``lang_lens_cpu`` (host ints),
        so this is free of device→host syncs even on a cache miss. Padded
        samples (``b >= actual_B``) get ``real_len = 0`` — their q-rows
        route to the sentinel slot and contribute no real-prefix K/V, so
        no real token ever attends to them.
        """
        cfg = self.cfg
        device = self.device
        max_B = self.max_batch_size
        n_img = self.image_token_count
        n_ps = n_per_sample
        chunk = cfg.chunk_size
        lang_bucket = n_ps - n_img

        real_list = [0] * max_B
        lang_list = [0] * max_B
        for b in range(actual_B):
            lang_list[b] = lang_lens_cpu[b]
            real_list[b] = lang_lens_cpu[b] + n_img
        n_real_total = sum(real_list)
        real_lens = torch.tensor(real_list, dtype=torch.int32, device=device)

        # Vectorized-pack mask over this bucket's lang budget: real lang
        # positions True, padding False.
        lang_lens_t = torch.tensor(lang_list, dtype=torch.int64, device=device)
        lang_mask = (
            torch.arange(lang_bucket, device=device)[None, :] < lang_lens_t[:, None]
        )
        prefix_mask = torch.zeros(max_B, n_ps, dtype=torch.bool, device=device)
        prefix_mask[:actual_B, :n_img] = True
        prefix_mask[:, n_img:] = lang_mask

        # --- Prefix (LLM self-attention) layout ---
        cu_q_prefix = torch.arange(
            0, (max_B + 1) * n_ps, n_ps, dtype=torch.int32, device=device
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
            n_per_sample=n_ps,
            prefix_slot_base=self.prefix_base,
            sentinel_slot=self.sentinel_slot,
        )
        position_ids = torch.arange(n_ps, dtype=torch.int32, device=device).repeat(
            max_B
        )
        prefix_meta = ARAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=max_B,
            num_query_tokens=max_B * n_ps,
            cu_seqlens_q=cu_q_prefix,
            paged_kv_indptr=paged_kv_indptr_prefix,
            paged_kv_indices=paged_kv_indices_prefix,
            paged_kv_last_page_len=paged_kv_last_prefix,
            write_indices=write_indices,
            position_ids=position_ids,
        )

        # --- Joint (expert) attention layout ---
        pos_ids_suffix = build_suffix_pos_ids(real_lens, chunk)
        cu_q_suffix = torch.arange(
            0, (max_B + 1) * chunk, chunk, dtype=torch.int32, device=device
        )
        paged_kv_indptr_full = torch.zeros(max_B + 1, dtype=torch.int32, device=device)
        paged_kv_indptr_full[1:] = torch.cumsum(real_lens + chunk, 0)
        # Host-side total joint length — passing it skips the blocking
        # ``int(cu_full[-1])`` read inside build_joint_paged_kv_indices.
        n_full = n_real_total + max_B * chunk
        paged_kv_indices_full = build_joint_paged_kv_indices(
            real_lens,
            chunk,
            prefix_slot_base=self.prefix_base,
            suffix_slot_base=self.suffix_base,
            n_full=n_full,
        )
        paged_kv_last_full = torch.ones(max_B, dtype=torch.int32, device=device)
        joint_meta = DiffusionAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=max_B,
            num_query_tokens=max_B * chunk,
            cu_seqlens_q=cu_q_suffix,
            paged_kv_indptr=paged_kv_indptr_full,
            paged_kv_indices=paged_kv_indices_full,
            paged_kv_last_page_len=paged_kv_last_full,
            position_ids=pos_ids_suffix,
        )

        return _PI05Layout(
            n_real_total=n_real_total,
            n_per_sample=n_ps,
            position_ids=position_ids,
            write_indices=write_indices,
            lang_mask=lang_mask,
            prefix_mask=prefix_mask,
            prefix_meta=prefix_meta,
            joint_meta=joint_meta,
        )

    def _pack_prefix(
        self,
        image_embs: torch.Tensor,
        lang_embs: torch.Tensor,
        layout: _PI05Layout,
    ) -> torch.Tensor:
        """Vectorized per-sample-padded prefix pack at the layout's bucket.

        Per sample the layout is ``[image (n_img), lang[:L_lang], pad]``
        over ``layout.n_per_sample`` rows. The image block is a
        fixed-stride copy; the lang block is ``lang_embs`` truncated to the
        bucket's lang budget and zeroed past each sample's real length via
        ``layout.lang_mask`` (multiply by a {0,1} mask — exact in IEEE, so
        real-token rows are bit-identical to the old per-sample gather).
        """
        max_B = self.max_batch_size
        n_ps = layout.n_per_sample
        n_img = self.image_token_count
        lang_bucket = n_ps - n_img
        D = image_embs.shape[-1]
        packed = torch.zeros(
            max_B * n_ps, D, dtype=image_embs.dtype, device=image_embs.device
        )
        pv = packed.view(max_B, n_ps, D)
        pv[:, :n_img] = image_embs
        mask = layout.lang_mask.to(lang_embs.dtype)[
            ..., None
        ]  # (max_B, lang_bucket, 1)
        pv[:, n_img:] = lang_embs[:, :lang_bucket] * mask
        return packed

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #

    def _validate_rollout(self, req: PI05RolloutRequest, actual_B: int) -> None:
        """Validate rollout-only sampling controls and tensor shapes."""
        cfg = self.cfg
        if req.noise_method not in ("flow_ode", "flow_sde"):
            raise ValueError(
                "noise_method must be 'flow_ode' or 'flow_sde', got "
                f"{req.noise_method!r}."
            )
        if not math.isfinite(req.noise_level) or req.noise_level < 0:
            raise ValueError(
                f"noise_level must be finite and non-negative, got {req.noise_level}."
            )
        if req.ignore_last and cfg.num_inference_steps < 2:
            raise ValueError("ignore_last=True requires at least two inference steps.")
        if req.compute_values and self.model.value_head is None:
            raise ValueError(
                "compute_values=True requires PI05Config(add_value_head=True)."
            )
        action_chunk = (
            req.action_chunk
            if req.action_chunk is not None
            else cfg.critic_action_chunk
        )
        action_dim = (
            req.action_dim if req.action_dim is not None else cfg.max_action_dim
        )
        if not 1 <= action_chunk <= cfg.chunk_size:
            raise ValueError(
                f"action_chunk must be in [1, {cfg.chunk_size}], got {action_chunk}."
            )
        if not 1 <= action_dim <= cfg.max_action_dim:
            raise ValueError(
                f"action_dim must be in [1, {cfg.max_action_dim}], got {action_dim}."
            )
        expected_step_noise = (
            actual_B,
            cfg.num_inference_steps,
            cfg.chunk_size,
            cfg.max_action_dim,
        )
        if req.step_noise is not None and req.step_noise.shape != expected_step_noise:
            raise ValueError(
                f"step_noise shape {tuple(req.step_noise.shape)} != "
                f"(B, num_steps, chunk_size, max_action_dim)="
                f"{expected_step_noise}."
            )

    def _validate(self, req: PI05Request) -> None:
        """Strictly validate the canonical request tensors.

        ``phyai`` is strict: it does not resize or reshape. ``pixel_values``
        must already be the stacked canonical tensor
        ``(B, num_images, C, image_size, image_size)``; this checks the camera
        count, H/W, batch-dim bound, and that the text / noise tensors agree
        with the batch size. Resizing non-square inputs is the caller's
        processor's job (``phyai_utils_tools.models.pi05.PI05Processor``).
        """
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
                f"expected num_images={self.num_images}."
            )
        if req.pixel_values.shape[-2:] != (
            cfg.vision.image_size,
            cfg.vision.image_size,
        ):
            raise ValueError(
                f"pixel_values H/W {tuple(req.pixel_values.shape[-2:])} != "
                f"(image_size, image_size)=({cfg.vision.image_size}, "
                f"{cfg.vision.image_size}). Resize in the caller's processor."
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
    "PI05Request",
    "PI05RolloutOutput",
    "PI05RolloutRequest",
    "PI05WS1Scheduler",
]
