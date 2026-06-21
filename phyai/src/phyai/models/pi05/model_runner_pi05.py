"""pi0.5 model runners: vision, LLM backbone, action expert."""

from __future__ import annotations

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
from phyai.models.pi05.modeling_pi05 import (
    ActionTimeHeads,
    ExpertModulationTables,
    PaliGemmaLanguageModel,
    PI05ExpertStack,
    PI05VisionTower,
)
from phyai.payload import (
    LLMForwardBatch,
    VisionForwardBatch,
)
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner
from phyai.utils import all_ranks_log


logger = logging.getLogger(__name__)


def _ar_attn_proto(stack_layers) -> ARAttention:
    """Return the first layer's :class:`ARAttention` instance.

    Used by the LLM runner to read ``num_heads`` / ``num_kv_heads`` /
    ``head_dim`` / ``backend`` for the backend factory and capture-shape
    seed. Every layer in a pi0.5 paligemma stack has the same attention
    config; only ``layer_id`` differs.
    """
    if len(stack_layers) == 0:
        raise ValueError("stack has no layers; cannot read attention metadata.")
    return stack_layers[0].attn


def _diffusion_attn_proto(stack_layers) -> DiffusionAttention:
    """Return the first layer's :class:`DiffusionAttention` instance.

    Used by the expert runner; same role as :func:`_ar_attn_proto` but
    typed for the diffusion / action-expert stack.
    """
    if len(stack_layers) == 0:
        raise ValueError("stack has no layers; cannot read attention metadata.")
    return stack_layers[0].attn


def _modulation_tables_match(
    a: ExpertModulationTables, b: ExpertModulationTables
) -> bool:
    """True if two modulation table sets share layout, dtype, and device.

    Used to decide whether a re-bound schedule can be copied into the
    existing tensors (preserving captured-graph storage) or needs a fresh
    assignment.
    """
    if len(a.layers) != len(b.layers):
        return False

    def _same(x: torch.Tensor, y: torch.Tensor) -> bool:
        return x.shape == y.shape and x.dtype == y.dtype and x.device == y.device

    if not _same(a.final, b.final):
        return False
    return all(
        _same(ai, bi) and _same(ap, bp)
        for (ai, ap), (bi, bp) in zip(a.layers, b.layers)
    )


def _copy_modulation_tables_(
    dst: ExpertModulationTables, src: ExpertModulationTables
) -> None:
    """Copy ``src`` into ``dst``'s tensors in place (shapes must match).

    Keeps ``dst``'s storage stable so an already-captured graph that reads the
    tables keeps seeing the same addresses after a same-shape schedule re-bind.
    """
    dst.final.copy_(src.final)
    for (dst_in, dst_post), (src_in, src_post) in zip(dst.layers, src.layers):
        dst_in.copy_(src_in)
        dst_post.copy_(src_post)


# ============================================================================ #
# Vision runner                                                                #
# ============================================================================ #


class PI05VisionRunner(ModelRunner):
    """SigLIP vision-tower runner with optional CUDA-graph capture.

    pi0.5 runs all of a robot's cameras in one tower call; the runner is
    captured at fixed shape ``(num_images, C, image_size, image_size)`` and
    replayed once per robot in the scheduler's batch (``B`` times when
    ``B > 1``). ``num_images`` is fixed at construction (3 for pi05_base).
    """

    def __init__(
        self,
        vision_tower: PI05VisionTower,
        *,
        num_images: int = 3,
        params_dtype: torch.dtype,
        device: torch.device | str,
        use_cuda_graph: bool = True,
    ) -> None:
        self.vision_tower = vision_tower
        self.num_images = int(num_images)
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.image_size = int(vision_tower.config.image_size)
        self.num_channels = int(vision_tower.config.num_channels)
        self.graph: CudaGraph | None = None

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI05VisionRunner.setup")
        if not self.use_cuda_graph or self.device.type != "cuda":
            return
        all_ranks_log(
            logger,
            logging.INFO,
            "Entering PI05VisionRunner.setup: capturing vision-tower CUDA graph "
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


# ============================================================================ #
# LLM backbone runner (prefix phase)                                           #
# ============================================================================ #


class PI05LLMRunner(ModelRunner):
    """PaliGemma backbone runner — runs paligemma's 18 layers over the
    per-sample-padded prefix and writes K/V to ``kv_pool``.

    Captured at fixed shape ``(B * n_per_sample, hidden_size)``. Owns
    a single :class:`ARAttentionBackend` instance built via
    :func:`get_ar_backend_factory`; the backend allocates its own static
    buffers in :meth:`init_cuda_graph_state` and
    re-plans them in place per inference via
    :meth:`replay_metadata` (graph) or
    :meth:`init_forward_metadata` (non-cuda-graph path).

    Returns ``None`` from :meth:`forward` — the cache pool side-effect
    is the only output the scheduler consumes.
    """

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
        # Read attention metadata from the first layer; every layer's
        # config is identical, only layer_id differs.
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

        # Build the runner-scoped backend instance. Backend reads runner
        # state for buffer sizing in init_cuda_graph_state.
        factory = get_ar_backend_factory(self.attn_proto.backend)
        self.attn_backend: ARAttentionBackend = factory(self)
        self.use_cuda_graph = (
            bool(use_cuda_graph)
            and self.attn_backend.supports_capture()
            and self.device.type == "cuda"
        )

        self._capture_plan: ARAttnPlanHandle | None = None
        # One captured graph per prefix-length bucket (keyed by
        # ``n_per_sample``). Shorter buckets pad the lang budget to fewer
        # tokens, so a short prompt skips the dense GEMM work on padding
        # rows. All buckets share the single attention wrapper (re-planned
        # per inference); ``n_per_sample`` here is the *largest* bucket and
        # sizes the wrapper's static buffers.
        self.graphs: dict[int, CudaGraph] = {}

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    def setup(self, n_per_sample_buckets: list[int] | None = None) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI05LLMRunner.setup")
        # Always allocate static buffers + build wrapper — graph mode
        # needs them, and the non-cuda-graph path benefits from stable
        # addresses. They are sized for the *largest* bucket
        # (``self.n_per_sample``); shorter buckets under-fill them.
        self.attn_backend.init_cuda_graph_state(
            max_batch_size=self.batch_size,
            max_num_tokens=self.batch_size * self.n_per_sample,
            max_paged_kv_indices=self.max_paged_kv_indices,
            device=self.device,
            params_dtype=self.params_dtype,
            layer_proto=self.attn_proto,
        )
        if not self.use_cuda_graph:
            return

        # Capture one graph per prefix-length bucket. The buckets must not
        # exceed the max ``n_per_sample`` the buffers were sized for.
        buckets = sorted(set(n_per_sample_buckets or [self.n_per_sample]))
        if buckets[-1] > self.n_per_sample or buckets[0] <= 0:
            raise ValueError(
                f"n_per_sample buckets {buckets} must be in (0, {self.n_per_sample}]."
            )
        all_ranks_log(
            logger,
            logging.INFO,
            "Entering PI05LLMRunner.setup: capturing %d prefix-forward CUDA "
            "graph(s) at B*n_per_sample in %s (hidden_size=%d).",
            len(buckets),
            [self.batch_size * n for n in buckets],
            self.hidden_size,
        )
        # Seed the (shared) wrapper plan handle once, then re-plan it per
        # bucket right before capturing that bucket's graph.
        self._capture_plan = self.attn_backend.init_capture_metadata(
            self._capture_seed_metadata(self.n_per_sample)
        )
        for n_ps in buckets:
            self.attn_backend.replay_metadata(
                self._capture_plan, self._capture_seed_metadata(n_ps)
            )
            self.graphs[n_ps] = self._capture_graph(n_ps)

    def _capture_seed_metadata(self, n_per_sample: int) -> ARAttnMetadata:
        # Per-sample padded q layout for this bucket — fixed across all
        # inferences that route to it.
        cu_q = torch.arange(
            0,
            (self.batch_size + 1) * n_per_sample,
            n_per_sample,
            dtype=torch.int32,
            device=self.device,
        )
        # A plausible "all real" KV layout for the warmup plan: each
        # sample contributes ``min(n_per_sample, max_indices/B)`` real
        # tokens. Real values arrive at the first ``plan_inference`` call.
        per_sample_real = min(
            n_per_sample, self.max_paged_kv_indices // self.batch_size
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
            self.batch_size * n_per_sample,
            dtype=torch.int64,
            device=self.device,
        )
        return ARAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=self.batch_size,
            num_query_tokens=self.batch_size * n_per_sample,
            cu_seqlens_q=cu_q,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=last_page,
            write_indices=write_indices,
        )

    def _capture_graph(self, n_per_sample: int) -> CudaGraph:
        n = self.batch_size * n_per_sample
        example = {
            "hidden_states": torch.zeros(
                n, self.hidden_size, dtype=self.params_dtype, device=self.device
            ),
            "position_ids": torch.zeros(n, dtype=torch.int32, device=self.device),
            "write_indices": torch.zeros(n, dtype=torch.int64, device=self.device),
        }
        graph = CudaGraph()
        graph.capture(self._fwd, example)
        return graph

    # ------------------------------------------------------------------ #
    # Forward path                                                       #
    # ------------------------------------------------------------------ #

    def _fwd(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        write_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run paligemma's 18 layers, writing K/V into ``self.kv_pool``."""
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
        """Stage attention metadata for the next ``forward`` call.

        Graph mode: re-plan the captured backend buffers in place via
        :meth:`ARAttentionBackend.replay_metadata`. Non-graph mode: build
        a fresh plan via :meth:`ARAttentionBackend.init_forward_metadata`.
        """
        if self.use_cuda_graph:
            self.attn_backend.replay_metadata(self._capture_plan, meta)
        else:
            self._capture_plan = self.attn_backend.init_forward_metadata(meta)

    def forward(
        self, batch: LLMForwardBatch, *, n_per_sample: int | None = None
    ) -> None:
        """Run the prefix forward for the given prefix-length bucket.

        ``n_per_sample`` selects which captured graph to replay (the
        scheduler builds ``batch`` at that bucket's length). ``None`` uses
        the largest bucket. Ignored in the non-cuda-graph path.
        """
        if self.use_cuda_graph and self.graphs:
            n_ps = n_per_sample if n_per_sample is not None else self.n_per_sample
            graph = self.graphs.get(n_ps)
            if graph is None:
                raise ValueError(
                    f"no captured LLM graph for n_per_sample={n_ps}; "
                    f"captured buckets: {sorted(self.graphs)}."
                )
            graph.replay(
                {
                    "hidden_states": batch.hidden_states,
                    "position_ids": batch.position_ids,
                    "write_indices": batch.write_indices,
                }
            )
            return None
        # Non-cuda-graph path.
        self._fwd(
            hidden_states=batch.hidden_states,
            position_ids=batch.position_ids,
            write_indices=batch.write_indices,
        )
        return None


# ============================================================================ #
# Action expert runner (one Euler step)                                        #
# ============================================================================ #


class PI05ExpertRunner(ModelRunner):
    """The full flow-matching Euler loop captured as one CUDA graph.

    :meth:`forward` takes the initial ``noise`` ``(B, chunk_size,
    max_action_dim)`` and runs all ``num_steps`` denoise steps
    (``embed_action -> 18 expert layers -> project_action`` then
    ``x_t <- x_t + dt * v_t``) internally, returning the final ``x_t``.
    The per-step conditioning is read in-graph from the constant
    ``time_emb_table`` bound via :meth:`bind_euler_schedule` (the
    scheduler precomputes it from the time MLP). Unrolling the loop into
    a single graph removes the N-1 extra graph launches and the eager
    between-step update; within one inference all steps share the same
    cache layout, so :meth:`plan_inference` refreshes the wrapper buffers
    once before the (single) replay.

    Two runner-owned static tensors on top of the backend's own static
    state:

    * ``pos_ids_suffix_buf`` — refreshed per inference via
      :meth:`plan_inference`.
    * ``write_indices_suffix_buf`` — constant across inferences;
      bound once at scheduler setup via :meth:`set_write_indices`.

    Both are baked into the captured graph by Python identity (read
    off ``self`` inside :meth:`_one_step`), so they stay outside the
    backend's :meth:`replay_metadata` contract.
    """

    def __init__(
        self,
        expert_stack: PI05ExpertStack,
        heads: ActionTimeHeads,
        rope: RotaryEmbedding,
        kv_pool: KVCachePool,
        *,
        batch_size: int,
        chunk_size: int,
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
        self.chunk_size = int(chunk_size)
        self.max_action_dim = int(max_action_dim)
        self.expert_hidden = int(heads.expert_hidden)
        self.params_dtype = params_dtype
        self.device = torch.device(device)
        self.attn_proto: DiffusionAttention = _diffusion_attn_proto(expert_stack.layers)
        self.num_heads = self.attn_proto.num_heads
        self.num_kv_heads = self.attn_proto.num_kv_heads
        self.head_dim = self.attn_proto.head_dim
        self.max_paged_kv_indices = int(
            max_paged_kv_indices
            if max_paged_kv_indices is not None
            else self.batch_size * self.chunk_size * 32
        )

        # pos_ids_suffix is per-inference (depends on real_lens) but
        # not per-Euler-step. Static buffer; runner refreshes once per
        # inference.
        self.pos_ids_suffix_buf = torch.zeros(
            self.batch_size * self.chunk_size,
            dtype=torch.int32,
            device=self.device,
        )
        # write_indices_suffix is constant across inferences (the suffix
        # slab base never moves once the scheduler is set up). The
        # scheduler hands it to the runner via :meth:`set_write_indices`
        # at startup.
        self.write_indices_suffix_buf = torch.zeros(
            self.batch_size * self.chunk_size,
            dtype=torch.int64,
            device=self.device,
        )

        factory = get_diffusion_backend_factory(self.attn_proto.backend)
        self.attn_backend: DiffusionAttentionBackend = factory(self)
        self.use_cuda_graph = (
            bool(use_cuda_graph)
            and self.attn_backend.supports_capture()
            and self.device.type == "cuda"
        )

        self._capture_plan: DiffusionAttnPlanHandle | None = None
        self.graph: CudaGraph | None = None

        # Euler schedule, bound by the scheduler via ``bind_euler_schedule``
        # *before* ``setup()``. The captured graph unrolls all
        # ``num_steps`` denoise steps internally and reads
        # ``time_emb_table[step]`` in-graph (a constant lookup), so the
        # whole flow-matching loop is one replay instead of N.
        self._time_emb_table: torch.Tensor | None = None
        self._dt: float = 0.0
        self._num_steps: int = 0

        # Per-norm AdaRMS modulation tables for the whole expert stack across
        # all Euler steps, built from ``_time_emb_table`` in
        # ``bind_euler_schedule``. ``_one_step`` slices step ``i`` out and
        # hands it to the (stateless) norms, so the ``dense`` projections stay
        # out of the captured graph. ``None`` until the schedule is bound.
        self._mod_tables: ExpertModulationTables | None = None

    def set_write_indices(self, write_indices_suffix: torch.Tensor) -> None:
        """Bind the suffix-slab slot indices once at scheduler setup."""
        if write_indices_suffix.shape != self.write_indices_suffix_buf.shape:
            raise ValueError(
                f"write_indices_suffix shape {tuple(write_indices_suffix.shape)} "
                f"!= {tuple(self.write_indices_suffix_buf.shape)}."
            )
        self.write_indices_suffix_buf.copy_(write_indices_suffix.to(torch.int64))

    def bind_euler_schedule(
        self, time_emb_table: torch.Tensor, *, dt: float, num_steps: int
    ) -> None:
        """Bind the flow-matching Euler schedule (call before :meth:`setup`).

        ``time_emb_table`` is the ``(num_steps, expert_hidden)`` constant
        lookup the scheduler precomputes from the time MLP; row ``step``
        is the AdaRMS conditioning for that Euler step. ``dt`` is the
        (constant) step size. The captured graph reads these in-graph, so
        they must be bound before the graph is captured.

        The schedule is input-independent, so every AdaRMS modulation is a
        constant of ``step``. We project them all once here (see
        :meth:`_build_modulation_tables`); the captured loop then selects each
        step's modulation by index instead of re-running the projection,
        keeping those GEMMs out of the graph.
        """
        if time_emb_table.shape[0] != num_steps:
            raise ValueError(
                f"time_emb_table has {time_emb_table.shape[0]} rows but "
                f"num_steps={num_steps}."
            )
        self._time_emb_table = time_emb_table
        self._dt = float(dt)
        self._num_steps = int(num_steps)
        self._build_modulation_tables()

    def _build_modulation_tables(self) -> None:
        """Build (or refresh) the runner-held AdaRMS modulation tables.

        Each step's conditioning is ``time_emb_table[step]`` (the per-token
        ``cond`` is just that row broadcast over the action tokens), and the
        schedule is input-independent, so one projection of the whole table
        per norm yields a ``(num_steps, 3*D)`` table that ``_one_step`` slices
        by index. The (stateless) norms read these via ``forward(modulation=...)``,
        so the ``dense`` GEMMs stay out of the captured graph.

        On a same-shape re-bind the new tables are copied *in place* into the
        existing tensors so an already-captured graph keeps reading the same
        storage (matches the CUDA-graph stability the rest of this runner
        relies on); otherwise they are assigned fresh.
        """
        assert self._time_emb_table is not None
        new_tables = self.expert_stack.build_modulation_tables(self._time_emb_table)
        if self._mod_tables is not None and _modulation_tables_match(
            self._mod_tables, new_tables
        ):
            _copy_modulation_tables_(self._mod_tables, new_tables)
        else:
            self._mod_tables = new_tables

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        all_ranks_log(logger, logging.INFO, "Entering PI05ExpertRunner.setup")
        if self._time_emb_table is None or self._num_steps <= 0:
            raise RuntimeError(
                "PI05ExpertRunner.setup() called before bind_euler_schedule(); "
                "the captured graph unrolls the Euler loop and needs the "
                "time-embedding table and step count."
            )
        self.attn_backend.init_cuda_graph_state(
            max_batch_size=self.batch_size,
            max_num_tokens=self.batch_size * self.chunk_size,
            max_paged_kv_indices=self.max_paged_kv_indices,
            device=self.device,
            params_dtype=self.params_dtype,
            layer_proto=self.attn_proto,
        )
        if self.use_cuda_graph:
            all_ranks_log(
                logger,
                logging.INFO,
                "Entering PI05ExpertRunner.setup: capturing the full %d-step "
                "Euler loop as one CUDA graph at fixed shape "
                "(B=%d, chunk_size=%d, max_action_dim=%d).",
                self._num_steps,
                self.batch_size,
                self.chunk_size,
                self.max_action_dim,
            )
            self._capture_plan = self.attn_backend.init_capture_metadata(
                self._capture_seed_metadata()
            )
            self._capture_graph()

    def _capture_seed_metadata(self) -> DiffusionAttnMetadata:
        # cu_q is fixed [0, chunk, 2*chunk, ...] across all inferences.
        cu_q = torch.arange(
            0,
            (self.batch_size + 1) * self.chunk_size,
            self.chunk_size,
            dtype=torch.int32,
            device=self.device,
        )
        # Seed kv layout with a "small" plausible joint length so plan() succeeds.
        per_sample_kv = min(
            self.chunk_size * 4,
            self.max_paged_kv_indices // self.batch_size,
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
            write_indices=self.write_indices_suffix_buf,
        )

    def _capture_graph(self) -> None:
        example = {
            "noise": torch.zeros(
                self.batch_size,
                self.chunk_size,
                self.max_action_dim,
                dtype=self.params_dtype,
                device=self.device,
            ),
        }
        self.graph = CudaGraph()
        self.graph.capture(self._fwd_loop, example)

    # ------------------------------------------------------------------ #
    # Forward path                                                       #
    # ------------------------------------------------------------------ #

    def _one_step(
        self,
        x_t: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """One Euler denoise step: ``embed_action -> 18 layers -> project``.

        ``step`` selects this step's AdaRMS modulation from the runner-held
        tables (built in :meth:`bind_euler_schedule`) and hands it to the
        stateless norms, so the per-token ``cond`` projection never runs
        in-graph.
        """
        assert self._mod_tables is not None  # built in bind_euler_schedule()
        action_emb = self.heads.embed_action(x_t)
        suffix_h = action_emb.reshape(self.batch_size * self.chunk_size, -1)
        ctx = DiffusionAttnCtx(
            backend=self.attn_backend,
            plan=self._capture_plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            kv_pool=self.kv_pool,
            write_indices=self.write_indices_suffix_buf,
        )
        suffix_out = self.expert_stack(
            suffix_h,
            self.pos_ids_suffix_buf,
            None,
            self.rope,
            ctx,
            modulation=self._mod_tables.step(step),
        )
        suffix_out_3d = suffix_out.view(self.batch_size, self.chunk_size, -1)
        return self.heads.project_action(suffix_out_3d)

    def _fwd_loop(self, *, noise: torch.Tensor) -> torch.Tensor:
        """Run the full ``num_steps``-step Euler loop, returning final ``x_t``.

        Unrolled at a constant ``num_steps`` so it captures into one CUDA
        graph. Each step selects its AdaRMS modulation by index from the
        runner-held tables (a static in-graph lookup) and applies the
        flow-matching update ``x_t <- x_t + dt * v_t`` — equivalent to the
        old scheduler-driven loop, just without the per-step graph launch, the
        eager between-step update, and the in-graph modulation projections.
        """
        assert self._time_emb_table is not None  # bound in setup()
        x_t = noise
        for step in range(self._num_steps):
            v_t = self._one_step(x_t, step)
            x_t = x_t + self._dt * v_t.to(x_t.dtype)
        return x_t

    def plan_inference(self, meta: DiffusionAttnMetadata) -> None:
        """Refresh metadata for one inference (all Euler steps share it).

        ``pos_ids_suffix`` is updated from ``meta.position_ids``;
        ``cu_q`` and ``write_indices_suffix`` are constants and stay on
        the runner.
        """
        if meta.position_ids is None:
            raise ValueError(
                "PI05ExpertRunner.plan_inference requires meta.position_ids."
            )
        self.pos_ids_suffix_buf.copy_(meta.position_ids.to(torch.int32))
        if self.use_cuda_graph:
            self.attn_backend.replay_metadata(self._capture_plan, meta)
        else:
            self._capture_plan = self.attn_backend.init_forward_metadata(meta)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Run the whole Euler loop from ``noise``; return final ``x_t``.

        The returned tensor aliases the captured graph's static output
        buffer (refilled every replay) — clone it if it must outlive the
        next call.
        """
        if self.graph is not None:
            return self.graph.replay({"noise": noise})
        return self._fwd_loop(noise=noise)


__all__ = [
    "PI05ExpertRunner",
    "PI05LLMRunner",
    "PI05VisionRunner",
]
