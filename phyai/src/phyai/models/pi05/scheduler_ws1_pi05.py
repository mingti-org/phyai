"""pi0.5 single-card (world_size=1) end-to-end inference scheduler.

The scheduler is the user-facing entry point for pi0.5 inference on a
single card. It owns:

* The :class:`KVCachePool` and the prefix / suffix
  :class:`StaticCache` slabs over it.
* Three runners — :class:`PI05VisionRunner`,
  :class:`PI05LLMRunner`, :class:`PI05ExpertRunner` — each with its
  own captured CUDA graph (when the engine backend supports paged
  attention).

A single ``step()`` call runs the full inference: vision tower per
camera-stack (replayed once per real robot), prefix forward writing K/V
into the pool, then a 10-step Euler loop reading those K/V plus the
freshly-computed suffix K/V.

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

from dataclasses import dataclass

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
from phyai.models.pi05.modeling_pi05 import PI05Model
from phyai.payload import (
    ExpertForwardBatch,
    LLMForwardBatch,
    VisionForwardBatch,
)
from phyai.runtime.schedule import Scheduler
from phyai.utils.profile import event_scope


# ============================================================================ #
# Batch-layout helpers — pi0.5 specific, only consumed by step() below.        #
# ============================================================================ #


def pack_prefix_per_sample_padded(
    image_embs: torch.Tensor,
    lang_embs: torch.Tensor,
    lang_lens: torch.Tensor,
    *,
    n_per_sample: int,
) -> torch.Tensor:
    """Assemble per-sample-padded ``(B * n_per_sample, D)`` prefix buffer.

    Per sample ``b`` the layout is
    ``[image_b (N_img), lang_b[:lang_lens[b]] (real_lang_b), padding (rest)]``
    where ``N_img + tokenizer_max_length == n_per_sample`` and the
    padding region holds zeros. The LLM runner is captured at fixed
    shape, so padding rows are required even though they will write K/V
    to the sentinel slot.

    Padded *samples* (``b >= actual_B``) are produced by the caller
    setting ``lang_lens[b] = 0`` and zero-filling ``image_embs[b]`` —
    pack still walks them, but only writes the (zero) image block; their
    rows are routed to sentinel by ``build_prefix_padded_write_indices``.
    """
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
    """
    device = real_lens.device
    B = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)

    cu_p = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_p[1:] = torch.cumsum(real64, 0)
    full_lens = real64 + chunk_size
    cu_full = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_full[1:] = torch.cumsum(full_lens, 0)
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
    """One pi0.5 inference request.

    ``pixel_values`` carries every robot's three cameras stacked along
    the camera axis: shape ``(B, 3, 3, H, W)`` — ``B`` robots x 3 cameras
    x 3 channels x ``image_size``x``image_size``. ``B`` may be any value
    in ``[1, max_batch_size]`` (the scheduler pads up internally).

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


class PI05WS1Scheduler(Scheduler):
    """Single-card (world_size=1) pi0.5 inference orchestrator with multi-batch support."""

    def __init__(
        self,
        model: PI05Model,
        *,
        max_batch_size: int = 1,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
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
        self.model = model

        # Layout knobs.
        self.image_token_count = cfg.vision.num_patches * 3  # 3 cameras per robot
        self.n_per_sample = self.image_token_count + cfg.tokenizer_max_length

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
            use_cuda_graph=use_cuda_graph,
            max_paged_kv_indices=self.max_batch_size
            * (self.n_per_sample + cfg.chunk_size),
        )

        # ``(num_inference_steps, expert_hidden)`` lookup table —
        # populated by :meth:`setup` after ``load_pretrained``. Each
        # row is the full time-MLP output for one Euler step's ``t``;
        # broadcast to ``(max_B, expert_hidden)`` via ``expand`` and fed
        # straight to the expert runner's captured input buffer.
        self.time_emb_table: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Warm up runners and capture every CUDA graph.

        Must be called **after** ``load_pretrained`` — :attr:`time_emb_table`
        is computed here from the (now-real) ``time_mlp_in/out`` weights,
        so calling this on freshly-allocated weights would bake garbage
        values into the table.
        """
        self.vision_runner.setup()
        self.llm_runner.setup()
        self.expert_runner.setup()

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

        # Precompute the time-embedding table for the linear flow-matching
        # schedule ``t = 1.0 + step * (-1/N)``. ``embed_time`` is the full
        # MLP (sinusoidal -> Linear -> SiLU -> Linear -> SiLU); its output
        # depends only on the time scalar and the time-MLP weights, both
        # fixed across inferences, so a one-time precompute eliminates
        # the matmuls from every Euler step's captured graph.
        N = self.cfg.num_inference_steps
        ts = 1.0 + torch.arange(N, dtype=torch.float32, device=self.device) * (-1.0 / N)
        with torch.no_grad():
            self.time_emb_table = self.model.heads.embed_time(ts).contiguous()

    # ------------------------------------------------------------------ #
    # Step (one inference)                                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, request: PI05Request) -> torch.Tensor:
        """Run one inference; return the action chunk ``(actual_B, chunk, action_dim)``.

        ``actual_B`` is ``request.pixel_values.shape[0]`` and may be any
        value in ``[1, max_batch_size]``. The internal forward passes
        run at the full ``max_batch_size`` shape (constant captured
        graphs); the padded tail is sliced off before returning.
        """
        cfg = self.cfg
        device = self.device
        dtype = self.params_dtype
        self._validate(request)

        actual_B = int(request.pixel_values.shape[0])
        max_B = self.max_batch_size

        # Build per-sample real_lens with explicit zeros for the padded
        # tail. Zero entries make ``build_prefix_padded_write_indices``
        # route every padded q-row to the sentinel slot and make
        # ``build_joint_paged_kv_indices`` emit zero real-prefix slots
        # for those samples. flashinfer's wrapper interprets the
        # zero-length per-sample K segment as an empty attend set —
        # padded q-rows then produce zeros (or NaNs that we discard).
        # No K/V written to slot 0 is ever read by attention.
        lang_lens_actual = request.lang_lens.to(device=device, dtype=torch.int32)
        real_lens = torch.zeros(max_B, dtype=torch.int32, device=device)
        real_lens[:actual_B] = lang_lens_actual + self.image_token_count
        n_real_total = int(real_lens.sum())

        # 1. Vision pass — replay the captured (3, 3, H, W) graph once
        # per real robot. The runner's output aliases its static buffer,
        # so we eagerly copy each replay into a pre-allocated
        # ``(max_B, image_token_count, D)`` slot before the next replay
        # overwrites it. Padded rows stay zero (their image embeddings
        # are written into the packed prefix at sentinel-routed rows,
        # so the values don't propagate anywhere).
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

        # 2. Lang embed (eager) + per-sample padded packing. input_ids
        # and lang_lens are zero-padded along dim 0 for the unused
        # samples; lang_lens=0 for padded means pack writes nothing
        # from those rows beyond their (zero) image block.
        with event_scope("pi05.lang_pack"):
            if actual_B < max_B:
                input_ids_padded = torch.zeros(
                    max_B, cfg.tokenizer_max_length, dtype=torch.int64, device=device
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

        # 3. Reset caches and plan the prefix layout.
        with event_scope("pi05.llm_prefix_plan"):
            self.prefix_static.reset()
            self.suffix_static.reset()
            # The static cache's role is bookkeeping; the actual write goes
            # to slots inferred from real_lens / prefix_slot_base. We
            # explicitly bump the cursor to ``n_real_total`` to record the
            # allocation. Suffix is sized at ``max_B * chunk_size`` because
            # every sample (real or padded) gets its own suffix slot range.
            _ = self.prefix_static.allocate(n_real_total)
            _ = self.suffix_static.allocate(max_B * cfg.chunk_size)

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
            # Real prefix tokens are written contiguously starting at
            # ``prefix_base`` (see ``build_prefix_padded_write_indices``), so the
            # paged-kv-indices for the prefix self-attention wrapper is just
            # the contiguous range. Padded samples contribute 0 entries.
            paged_kv_indices_prefix = torch.arange(
                self.prefix_base,
                self.prefix_base + n_real_total,
                dtype=torch.int32,
                device=device,
            )
            # ``page_size=1``: empty samples have last-page-len 0, non-empty 1.
            paged_kv_last_prefix = (real_lens > 0).to(torch.int32)
            write_indices = build_prefix_padded_write_indices(
                real_lens,
                n_per_sample=self.n_per_sample,
                prefix_slot_base=self.prefix_base,
                sentinel_slot=self.sentinel_slot,
            )
            # Per-sample local positions: every padded row gets its in-sample
            # index ``j``; padding rows produce K rotations that go to the
            # sentinel slot and are never read by attention.
            position_ids = torch.arange(
                self.n_per_sample, dtype=torch.int32, device=device
            ).repeat(max_B)

            # Plan the LLM runner's wrapper / sdpa metadata buffers.
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

        # 4. Prefix forward — populates the cache pool. ``packed`` is a
        # fresh contiguous tensor; the runner's ``replay`` will copy it
        # into the captured input buffer, so no scheduler-side staging.
        # Attention metadata was already staged on the runner via
        # ``plan_inference`` above; the batch carries only the per-call
        # variable inputs.
        with event_scope("pi05.llm_prefix_fwd"):
            llm_batch = LLMForwardBatch(
                hidden_states=packed,
                position_ids=position_ids,
                write_indices=write_indices,
            )
            self.llm_runner.forward(llm_batch)

        # 5. Plan the expert runner's joint-attention metadata.
        with event_scope("pi05.expert_plan"):
            pos_ids_suffix = build_suffix_pos_ids(real_lens, cfg.chunk_size)
            cu_q_suffix = torch.arange(
                0,
                (max_B + 1) * cfg.chunk_size,
                cfg.chunk_size,
                dtype=torch.int32,
                device=device,
            )
            paged_kv_indptr_full = torch.zeros(
                max_B + 1, dtype=torch.int32, device=device
            )
            paged_kv_indptr_full[1:] = torch.cumsum(real_lens + cfg.chunk_size, 0)
            paged_kv_indices_full = build_joint_paged_kv_indices(
                real_lens,
                cfg.chunk_size,
                prefix_slot_base=self.prefix_base,
                suffix_slot_base=self.suffix_base,
            )
            # Joint last-page-len: every sample has chunk_size > 0 suffix
            # tokens under page_size=1, so the last page always holds
            # exactly one token — including padded samples (their suffix is
            # garbage but lives in real slots and self-attends correctly).
            paged_kv_last_full = torch.ones(max_B, dtype=torch.int32, device=device)
            joint_meta = DiffusionAttnMetadata(
                mode=AttnMode.PREFILL,
                layout=AttnLayout.RAGGED_3D,
                batch_size=max_B,
                num_query_tokens=max_B * cfg.chunk_size,
                cu_seqlens_q=cu_q_suffix,
                paged_kv_indptr=paged_kv_indptr_full,
                paged_kv_indices=paged_kv_indices_full,
                paged_kv_last_page_len=paged_kv_last_full,
                position_ids=pos_ids_suffix,
            )
            self.expert_runner.plan_inference(joint_meta)

        # 6. Sample noise (or use the user's, padded) and run 10-step Euler.
        with event_scope("pi05.expert_loop"):
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

            if self.time_emb_table is None:
                raise RuntimeError(
                    "PI05WS1Scheduler.step() called before setup(); "
                    "the time-embedding lookup table has not been built."
                )

            dt = -1.0 / cfg.num_inference_steps
            for step in range(cfg.num_inference_steps):
                with event_scope("pi05.expert_step"):
                    # Look up the precomputed AdaRMS condition for this step's t
                    # and broadcast across all max_B samples (all batch elements
                    # share the same t in this scheduler). The expand view is fed
                    # straight to the expert runner — its ``replay`` materializes
                    # the broadcast into the captured input buffer.
                    time_emb = self.time_emb_table[step : step + 1].expand(max_B, -1)
                    expert_batch = ExpertForwardBatch(x_t=x_t, time_emb=time_emb)
                    v_t = self.expert_runner.forward(expert_batch)
                    # Capture-time output buffer is reused; clone is implicit
                    # via the addition (allocates a fresh tensor each step,
                    # which we then bind to ``x_t`` for the next iteration).
                    x_t = x_t + dt * v_t.to(x_t.dtype)
        # Drop the padded tail before returning.
        return x_t[:actual_B]

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #

    def _validate(self, req: PI05Request) -> None:
        cfg = self.cfg
        actual_B = int(req.pixel_values.shape[0])
        if not 1 <= actual_B <= self.max_batch_size:
            raise ValueError(
                f"pixel_values batch dim {actual_B} not in "
                f"[1, max_batch_size={self.max_batch_size}]."
            )
        if req.pixel_values.shape[1] != 3:
            raise ValueError(
                f"pixel_values must have 3 cameras, got {req.pixel_values.shape[1]}."
            )
        if req.pixel_values.shape[-1] != cfg.vision.image_size:
            raise ValueError(
                f"pixel_values H/W {req.pixel_values.shape[-2:]} != "
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


__all__ = ["PI05Request", "PI05WS1Scheduler"]
