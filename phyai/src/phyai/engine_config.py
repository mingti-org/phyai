"""Process-level engine config: layered, validated defaults consulted by every model constructor.

:class:`EngineConfig` is the root config that bundles four frozen
sub-configs by concern:

* :class:`BackendConfig` — kernel/backend selection
  (``attn`` / ``norm`` / ``linear`` / ``vgpu``), all resolved through
  the corresponding registry's canonical name. ``attn`` names the
  attention backend; the same name is used across the three attention
  stacks (``attention`` / ``ar`` / ``diffusion``) where it is
  registered, with each stack's per-construction lookup picking the
  right concrete class.
* :class:`DeviceConfig` — data layout (``target`` device, ``params_dtype``).
* :class:`ParallelConfig` — parallelism topology (``world_size`` plus
  ``dp_size`` / ``ep_size`` / ``sp_size`` / ``cp_size`` / ``tp_size``);
  ``world_size`` is an explicit user input — not a product of the per-
  axis sizes — because real parallelism strategies overlap (EP / CP / SP
  often carve into TP rather than multiplying with it).
* :class:`RuntimeConfig` — runtime mode switches and tunables
  (``use_cuda_graph``, ``flashinfer_workspace_bytes``,
  ``force_linear_kernel``).

Construction
------------
Three entry points, in order of typical use:

* :meth:`EngineConfig.auto` — derive sensible defaults from the host
  (CUDA + flashinfer + bf16 if CUDA is up; CPU + sdpa + fp32 otherwise).
* :meth:`EngineConfig.from_env` — start from ``auto()`` and overlay
  ``PHYAI_*`` env-var overrides registered in :mod:`phyai.env`.
* Direct ``EngineConfig(...)`` — explicit kwargs; every sub-config
  validates itself in :meth:`__post_init__` so bad values fail at
  construction, not later in some deep tensor allocation.

Singleton
---------
:func:`get_engine_config` returns the process singleton (lazily
populated via :meth:`EngineConfig.from_env` on first read);
:func:`set_engine_config` / :func:`init_engine_config` install a fresh
config — typically once at program startup or in test setup. Mutation
after model construction does *not* retroactively fix already-allocated
tensors; later sub-modules will see the new values, so the safest
pattern is to install the config first and then build models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from threading import Lock

import torch

from phyai.env import envs


def _canonical_backend_name(name: str) -> str:
    return name.lower().replace("_", "-")


# flashinfer paged-prefill kernels (BatchPrefillWithPagedKVCacheWrapper).
# Per flashinfer's own ctor docstring the accepted names are
# auto / fa2 / fa3 / cudnn / trtllm-gen ("cute-dsl" is explicitly rejected
# for paged KV). "auto" lets flashinfer pick. We validate against this set
# so a typo fails at config construction rather than deep in wrapper init;
# whether a given kernel actually supports a model's head_dim / dtype is
# flashinfer's call at plan/run time.
_VALID_FLASHINFER_PREFILL_BACKENDS: frozenset[str] = frozenset(
    {"auto", "fa2", "fa3", "cudnn", "trtllm-gen"}
)


# ---------------------------------------------------------------------- #
# Sub-configs                                                            #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackendConfig:
    """Kernel / backend selection.

    Each name is resolved through the corresponding module's registry
    in :meth:`__post_init__`; an unknown name fails fast with the
    list of available names.

    Fields
    ------
    attn:
        Attention backend name. The same canonical name is registered
        across the three attention stacks
        (:class:`~phyai.layers.attention.Attention`,
        :class:`~phyai.layers.attention.ARAttention`,
        :class:`~phyai.layers.attention.DiffusionAttention`); each
        stack's per-construction lookup resolves it against its own
        registry. Production names: ``"flashinfer"`` (default) /
        ``"sdpa"`` / ``"eager"``. ``"sdpa"`` only registers in the
        no-cache stack — code that picks paged backends for AR or
        diffusion derives ``"eager"`` as the fallback when ``attn`` is
        ``"sdpa"`` (see ``_engine_to_paged_backend`` in pi05's
        ``modeling_pi05``).
    norm:
        :class:`~phyai.layers.layer_norm.RMSNorm` / ``LayerNorm``
        backend (``"flashinfer"`` / ``"phyai-kernel"``).
    linear:
        Optional :class:`~phyai.layers.linear.LinearKernel` name to
        prefer above all others; ``None`` defers to the registry's
        ``prefer_for`` ordering (FlashInfer-then-Torch).
    vgpu:
        Optional :mod:`phyai.vgpu` backend name (``"flashinfer"`` /
        ``"torch"``); ``None`` defers to vgpu auto-resolve.
    """

    attn: str = "flashinfer"
    norm: str = "flashinfer"
    linear: str | None = None
    vgpu: str | None = None

    def __post_init__(self) -> None:
        self._validate_attn()
        self._validate_norm()
        self._validate_linear()
        self._validate_vgpu()

    def _validate_attn(self) -> None:
        from phyai.layers.attention.ar.registry import (
            list_backends as list_ar_backends,
        )
        from phyai.layers.attention.attention.registry import (
            list_backends as list_attention_backends,
        )
        from phyai.layers.attention.diffusion.registry import (
            list_backends as list_diffusion_backends,
        )

        canonical_attn = _canonical_backend_name(self.attn)
        # The same name may register in any of the three subpackages.
        # Accept if it appears in at least one — pi05 (and other models)
        # then look the name up against the right subpackage's registry
        # at layer construction time.
        all_names = (
            set(list_attention_backends())
            | set(list_ar_backends())
            | set(list_diffusion_backends())
        )
        if canonical_attn not in all_names:
            raise ValueError(
                f"BackendConfig.attn={self.attn!r} is not registered in any "
                f"of the three attention stacks. Available: "
                f"attention={list_attention_backends()}, "
                f"ar={list_ar_backends()}, "
                f"diffusion={list_diffusion_backends()}"
            )
        object.__setattr__(self, "attn", canonical_attn)

    def _validate_norm(self) -> None:
        from phyai.layers.layer_norm import list_norm_backends

        canonical = _canonical_backend_name(self.norm)
        available = list_norm_backends()
        if canonical not in available:
            raise ValueError(
                f"BackendConfig.norm={self.norm!r} is not a registered "
                f"norm backend. Available: {available}"
            )
        object.__setattr__(self, "norm", canonical)

    def _validate_linear(self) -> None:
        if self.linear is None:
            return
        from phyai.layers.linear.registry import list_registered_linear_kernels

        names = [cls.name for cls, _ in list_registered_linear_kernels()]
        if self.linear not in names:
            raise ValueError(
                f"BackendConfig.linear={self.linear!r} is not a registered "
                f"LinearKernel. Available: {names}"
            )

    def _validate_vgpu(self) -> None:
        if self.vgpu is None:
            return
        from phyai.vgpu.backend import known_backends

        names = known_backends()
        if self.vgpu not in names:
            raise ValueError(
                f"BackendConfig.vgpu={self.vgpu!r} is not a registered "
                f"vgpu backend. Available: {names}"
            )


@dataclass(frozen=True)
class DeviceConfig:
    """Data layout — device + default parameter dtype.

    Fields
    ------
    target:
        ``"cuda"`` / ``"cuda:N"`` / ``"cpu"``; whatever
        :func:`torch.device` accepts.
    params_dtype:
        Default ``torch.dtype`` for newly-allocated parameters. Set
        process-wide by :func:`phyai.utils.cuda.init_cuda` so layer
        constructors that omit ``dtype=`` land in the same precision
        as the loaded weights.
    """

    target: str = "cuda"
    params_dtype: torch.dtype = field(default=torch.bfloat16)

    def __post_init__(self) -> None:
        self._validate_target()
        self._validate_dtype()

    def _validate_target(self) -> None:
        try:
            torch.device(self.target)
        except (RuntimeError, TypeError) as e:
            raise ValueError(
                f"DeviceConfig.target={self.target!r} is not a valid torch.device."
            ) from e

    def _validate_dtype(self) -> None:
        if not isinstance(self.params_dtype, torch.dtype):
            raise ValueError(
                f"DeviceConfig.params_dtype must be a torch.dtype, got "
                f"{type(self.params_dtype).__name__}."
            )


@dataclass(frozen=True)
class ParallelConfig:
    """Parallelism topology — per-axis user inputs + global world size.

    ``world_size`` is the size of the global process group and is an
    *explicit* user input, not derived from the per-axis sizes. Real
    parallelism strategies frequently overlap (e.g. EP / CP / SP carve
    *into* a TP group rather than multiplying with it; common
    constraints look like ``tp_size % attn_cp_size == 0`` and
    ``ep_size * moe_dp_size <= tp_size``), so the product of
    ``dp_size x ep_size x sp_size x cp_size x tp_size`` is *not* a
    reliable substitute for the actual world size. Pass it in
    directly, or let
    :meth:`EngineConfig.from_env` pick it up from ``PHYAI_WORLD_SIZE``
    (which under ``torchrun`` you'll typically set to ``$WORLD_SIZE``).

    Each field defaults to ``1`` so a single-rank run keeps
    ``ParallelConfig()`` valid and degenerate. Cross-axis consistency
    (e.g. mesh layout product matches the live ``dist.get_world_size()``)
    is checked downstream by :func:`phyai.parallel.init`, where the
    process group is actually up and the deployment-specific overlap
    rules apply.

    Axes (outer -> inner in the mesh built by :class:`~phyai.engine.Engine`):

    * ``dp`` — data parallel: full model replica per group, different
      micro-batches.
    * ``ep`` — expert parallel: distinct MoE experts per rank.
    * ``sp`` — sequence parallel: shards activations along the sequence
      axis (Megatron-style sub-mode of TP).
    * ``cp`` — context parallel: shards long contexts across ranks for
      attention. Triggers the MagiAttention path when set (>1).
    * ``tp`` — tensor parallel: shards Linear / attention weights across
      ranks. Innermost so the heaviest collectives stay on the fastest
      interconnect.

    Single-axis runs leave the unused sizes at 1; the engine still
    builds a 5-axis mesh so model code addressing ``axis="tp"`` keeps
    working unchanged.
    """

    world_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    sp_size: int = 1
    cp_size: int = 1
    tp_size: int = 1

    def __post_init__(self) -> None:
        for name in (
            "world_size",
            "dp_size",
            "ep_size",
            "sp_size",
            "cp_size",
            "tp_size",
        ):
            v = getattr(self, name)
            if not isinstance(v, int) or v < 1:
                raise ValueError(
                    f"ParallelConfig.{name} must be a positive int, got {v!r}."
                )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime mode switches and tunables.

    Fields
    ------
    use_cuda_graph:
        Capture-and-replay path on CUDA. Off on CPU.
    flashinfer_workspace_bytes:
        Size of the process-global flashinfer split-k scratch. One
        buffer per device, allocated lazily on first use. Default is
        128 MiB (1x flashinfer's recommendation); bump it for larger
        head counts or long-context prefill that pushes split-k off
        the fast path. Consumed by
        :func:`phyai.layers.attention.utils.resolve_workspace_bytes`.
    flashinfer_prefill_backend:
        Which flashinfer prefill kernel the paged-KV attention wrappers
        request. ``None`` (the default) defers to flashinfer's ``"auto"``
        heuristic; otherwise one of the names
        ``BatchPrefillWithPagedKVCacheWrapper`` accepts —
        ``"fa2"`` / ``"fa3"`` / ``"cudnn"`` / ``"trtllm-gen"``. The right
        choice is *shape-dependent* (e.g. the FA2 kernel beats the
        auto-selected FA3 by ~2.5x on the pi0.5 action-expert's tiny-query
        joint attention at head_dim 256), so this is a tunable rather than
        a baked-in default — models that know their shape ship a
        recommendation (see ``PI05RecommendedEngineConfig``) instead of forcing
        it here. Consumed by
        :func:`phyai.layers.attention.utils.resolve_prefill_backend`.
    force_linear_kernel:
        Hard override for :class:`~phyai.layers.linear.KernelDispatcher`
        — when set, every :meth:`select` returns the kernel registered
        under that name regardless of (spec, regime). Useful for A/B
        comparisons; ``None`` lets the registry's ``prefer_for``
        ordering decide.
    debug_tensor_dump_dir:
        Base directory for forward-hook activation dumps. ``None`` (the
        default) disables dumping. When set, the engine records every
        selected leaf operator's output to
        ``<dir>/rank{R}_pid{P}/pass{N}.pt`` — one file per
        :meth:`~phyai.engine.Engine.step`. Because a captured CUDA graph
        replays without re-entering Python (so forward hooks never fire),
        the engine **forces ``use_cuda_graph`` off** whenever this is set;
        expect eager-mode speed while dumping. See
        :mod:`phyai.runtime.tensor_dump`.
    debug_tensor_dump_filter:
        Optional tuple of regex strings selecting which operators to dump,
        matched against each operator's full dotted name
        (``model.expert_stack.layers.0.o_proj``). A leaf is recorded if
        **any** pattern ``re.search``-matches (a union). ``None`` records
        every operator. Mutually exclusive with
        ``debug_tensor_dump_filter_fn``. No effect unless
        ``debug_tensor_dump_dir`` is also set.
    debug_tensor_dump_filter_fn:
        Optional ``"pkg.module:func"`` / ``"/path/to/file.py:func"`` path
        to a ``(name, module) -> bool`` predicate, for selection logic a
        regex can't express. Mutually exclusive with
        ``debug_tensor_dump_filter``. No effect unless
        ``debug_tensor_dump_dir`` is also set.
    """

    use_cuda_graph: bool = True
    flashinfer_workspace_bytes: int = 128 * 1024 * 1024
    flashinfer_prefill_backend: str | None = None
    force_linear_kernel: str | None = None
    debug_tensor_dump_dir: str | None = None
    debug_tensor_dump_filter: tuple[str, ...] | None = None
    debug_tensor_dump_filter_fn: str | None = None

    def __post_init__(self) -> None:
        v = self.flashinfer_workspace_bytes
        if not isinstance(v, int) or v <= 0:
            raise ValueError(
                f"RuntimeConfig.flashinfer_workspace_bytes must be a "
                f"positive int (bytes), got {v!r}."
            )
        be = self.flashinfer_prefill_backend
        if be is not None and be not in _VALID_FLASHINFER_PREFILL_BACKENDS:
            raise ValueError(
                f"RuntimeConfig.flashinfer_prefill_backend={be!r} must be "
                f"None or one of {sorted(_VALID_FLASHINFER_PREFILL_BACKENDS)} "
                f"(the names BatchPrefillWithPagedKVCacheWrapper accepts; "
                f"'cute-dsl' is paged-incompatible)."
            )
        flt = self.debug_tensor_dump_filter
        if flt is not None:
            if not isinstance(flt, tuple) or not all(isinstance(x, str) for x in flt):
                raise ValueError(
                    f"RuntimeConfig.debug_tensor_dump_filter must be None or a "
                    f"tuple of regex strings, got {flt!r}."
                )
            for pat in flt:
                try:
                    re.compile(pat)
                except re.error as e:
                    raise ValueError(
                        f"RuntimeConfig.debug_tensor_dump_filter has an invalid "
                        f"regex {pat!r}: {e}"
                    ) from e
        if (
            self.debug_tensor_dump_filter is not None
            and self.debug_tensor_dump_filter_fn is not None
        ):
            raise ValueError(
                "RuntimeConfig.debug_tensor_dump_filter and "
                "debug_tensor_dump_filter_fn are mutually exclusive; set at "
                "most one."
            )
        # ``force_linear_kernel`` is *not* validated against the global
        # linear-kernel registry: a kernel registered into a
        # dispatcher-local registry (tests, custom builds) may not be
        # present globally. The :class:`~phyai.layers.linear.registry.ForcedPolicy`
        # already raises clearly at ``select()`` time when the name
        # doesn't resolve, so late failure is enough.


# ---------------------------------------------------------------------- #
# Root EngineConfig                                                      #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngineConfig:
    """Process-level engine config — composes four frozen sub-configs.

    Read sub-configs through ``cfg.backends`` / ``cfg.device`` /
    ``cfg.parallel`` / ``cfg.runtime``. Replace either via
    :meth:`replace` (sub-config-level kwargs) or by constructing fresh
    sub-configs and passing them to :class:`EngineConfig` directly.
    """

    backends: BackendConfig = field(default_factory=BackendConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def auto(cls) -> "EngineConfig":
        """Pick sensible defaults from the host hardware.

        On a CUDA host: ``flashinfer`` everywhere, bf16 params, cuda
        graph on. Without CUDA: ``sdpa`` attention,
        ``phyai-kernel`` norm (Triton; falls back to torch eager
        elsewhere), fp32 params, cuda graph off.
        """
        cuda = torch.cuda.is_available()
        return cls(
            backends=BackendConfig(
                attn="flashinfer" if cuda else "sdpa",
                norm="flashinfer" if cuda else "phyai-kernel",
                linear=None,
                vgpu=None,
            ),
            device=DeviceConfig(
                target="cuda" if cuda else "cpu",
                params_dtype=torch.bfloat16 if cuda else torch.float32,
            ),
            parallel=ParallelConfig(),
            runtime=RuntimeConfig(use_cuda_graph=cuda),
        )

    @classmethod
    def from_env(cls, base: "EngineConfig | None" = None) -> "EngineConfig":
        """Start from ``base`` (default ``auto()``) and overlay env-var overrides.

        Reads :class:`phyai.env.envs` and applies any set ``PHYAI_*``
        var on top of the base config.
        """
        if base is None:
            base = cls.auto()

        backends_kw: dict[str, object] = {}
        if (v := envs.PHYAI_ATTN_BACKEND.get()) is not None:
            backends_kw["attn"] = v
        if (v := envs.PHYAI_NORM_BACKEND.get()) is not None:
            backends_kw["norm"] = v
        if (v := envs.PHYAI_LINEAR_BACKEND.get()) is not None:
            backends_kw["linear"] = v
        if (v := envs.PHYAI_VGPU_BACKEND.get()) is not None:
            backends_kw["vgpu"] = v

        device_kw: dict[str, object] = {}
        if (v := envs.PHYAI_DEVICE.get()) is not None:
            device_kw["target"] = v
        if (v := envs.PHYAI_PARAMS_DTYPE.get()) is not None:
            device_kw["params_dtype"] = v

        parallel_kw: dict[str, object] = {}
        if (v := envs.PHYAI_WORLD_SIZE.get()) is not None:
            parallel_kw["world_size"] = v
        if (v := envs.PHYAI_DP_SIZE.get()) is not None:
            parallel_kw["dp_size"] = v
        if (v := envs.PHYAI_EP_SIZE.get()) is not None:
            parallel_kw["ep_size"] = v
        if (v := envs.PHYAI_SP_SIZE.get()) is not None:
            parallel_kw["sp_size"] = v
        if (v := envs.PHYAI_CP_SIZE.get()) is not None:
            parallel_kw["cp_size"] = v
        if (v := envs.PHYAI_TP_SIZE.get()) is not None:
            parallel_kw["tp_size"] = v

        runtime_kw: dict[str, object] = {}
        if (v := envs.PHYAI_USE_CUDA_GRAPH.get()) is not None:
            runtime_kw["use_cuda_graph"] = v
        if (v := envs.PHYAI_FLASHINFER_WORKSPACE_BYTES.get()) is not None:
            runtime_kw["flashinfer_workspace_bytes"] = v
        if (v := envs.PHYAI_FLASHINFER_PREFILL_BACKEND.get()) is not None:
            runtime_kw["flashinfer_prefill_backend"] = v
        if (v := envs.PHYAI_FORCE_LINEAR_KERNEL.get()) is not None:
            runtime_kw["force_linear_kernel"] = v
        if (v := envs.PHYAI_DEBUG_TENSOR_DUMP_DIR.get()) is not None:
            runtime_kw["debug_tensor_dump_dir"] = v
        if (v := envs.PHYAI_DEBUG_TENSOR_DUMP_FILTER.get()) is not None:
            runtime_kw["debug_tensor_dump_filter"] = v
        if (v := envs.PHYAI_DEBUG_TENSOR_DUMP_FILTER_FN.get()) is not None:
            runtime_kw["debug_tensor_dump_filter_fn"] = v

        return cls(
            backends=replace(base.backends, **backends_kw)
            if backends_kw
            else base.backends,
            device=replace(base.device, **device_kw) if device_kw else base.device,
            parallel=replace(base.parallel, **parallel_kw)
            if parallel_kw
            else base.parallel,
            runtime=replace(base.runtime, **runtime_kw) if runtime_kw else base.runtime,
        )

    def replace(
        self,
        *,
        backends: BackendConfig | None = None,
        device: DeviceConfig | None = None,
        parallel: ParallelConfig | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> "EngineConfig":
        """Return a new :class:`EngineConfig` with the given sub-configs swapped in.

        Sub-config-level granularity: pass a freshly-constructed
        :class:`BackendConfig` / :class:`DeviceConfig` / etc. for any
        group you want to override; omitted groups carry over verbatim.
        """
        return replace(
            self,
            backends=backends if backends is not None else self.backends,
            device=device if device is not None else self.device,
            parallel=parallel if parallel is not None else self.parallel,
            runtime=runtime if runtime is not None else self.runtime,
        )


# ---------------------------------------------------------------------- #
# Process-level singleton                                                #
# ---------------------------------------------------------------------- #


_config: EngineConfig | None = None
_lock = Lock()


def get_engine_config() -> EngineConfig:
    """Return the process-level :class:`EngineConfig`.

    Lazily initialises with :meth:`EngineConfig.from_env` on first
    read so importing ``phyai.models`` without an explicit init Just
    Works on whatever host you happen to be on (CUDA box -> flashinfer
    + bf16; CPU dev box -> sdpa + fp32; ``PHYAI_*`` env overrides
    layered on top).
    """
    global _config
    if _config is None:
        with _lock:
            if _config is None:
                _config = EngineConfig.from_env()
    return _config


def set_engine_config(cfg: EngineConfig) -> None:
    """Replace the process-level :class:`EngineConfig`.

    Pass a freshly-constructed instance (or
    ``get_engine_config().replace(...)``). Every subsequent model
    constructor that consults the singleton picks up the new values;
    already-allocated tensors keep their original device/dtype.
    """
    global _config
    with _lock:
        _config = cfg


def init_engine_config(cfg: EngineConfig) -> EngineConfig:
    """Install ``cfg`` as the process singleton and return it.

    Thin wrapper over :func:`set_engine_config` that returns the
    installed value so callers can chain — :class:`Engine` calls this
    as the first step of its init sequence:

        cfg = init_engine_config(EngineConfig.from_env())
        init_cuda(cfg.device.target, cfg.device.params_dtype)
        ...
    """
    set_engine_config(cfg)
    return cfg


def resolve_engine_defaults(
    params_dtype: torch.dtype | None,
    attn_backend: str | None,
    norm_backend: str | None,
) -> tuple[torch.dtype, str, str]:
    """Fill in ``None`` overrides from the process :class:`EngineConfig`.

    Short-circuits when every argument is already a concrete override, so a
    parent that has already resolved defaults can pass them through to a
    child constructor without a second singleton read. Callers that
    don't need ``attn_backend`` (norm-only sub-modules) just discard the
    returned value.
    """
    if (
        params_dtype is not None
        and attn_backend is not None
        and norm_backend is not None
    ):
        return params_dtype, attn_backend, norm_backend
    ec = get_engine_config()
    return (
        ec.device.params_dtype if params_dtype is None else params_dtype,
        ec.backends.attn if attn_backend is None else attn_backend,
        ec.backends.norm if norm_backend is None else norm_backend,
    )


__all__ = [
    "BackendConfig",
    "DeviceConfig",
    "EngineConfig",
    "ParallelConfig",
    "RuntimeConfig",
    "get_engine_config",
    "init_engine_config",
    "resolve_engine_defaults",
    "set_engine_config",
]
