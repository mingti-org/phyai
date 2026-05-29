"""Profile utilities for phyai benchmarks and runtime tracing.

Two collection backends are wired in here behind one ABC, exclusive
of each other — pick one per run:

* :class:`TorchProfiler` — wraps :class:`torch.profiler.profile` and
  exports a ``.trace.json.gz`` chrome trace. The chrome-trace JSON is
  the same format Perfetto's web UI (``ui.perfetto.dev``) consumes
  directly, so the same file feeds both Chrome ``about:tracing`` and
  Perfetto without conversion.
* :class:`NsysProfiler` — toggles ``cudaProfilerStart/Stop`` so an
  enclosing ``nsys profile --capture-range=cudaProfilerApi`` recording
  picks up exactly the window we care about, and emits NVTX ranges so
  every :func:`event_scope` block becomes a labelled lane in
  Nsight Systems. Note: ``nsys`` only signals the *external* nsys
  recorder; phyai never spawns nsys itself, so a run with
  ``backend="nsys"`` and no enclosing ``nsys profile`` collects
  nothing.

Both backends accept the same :func:`event_scope` calls, so business
code never branches on backend. The default :class:`NoOpProfiler` makes
``with event_scope(...)`` cost a dict lookup plus a ``yield``, so the
hooks can stay in hot paths without conditional guards.

Usage
-----
::

    from phyai.utils.profile import (
        ProfilerConfig, install_profiler, event_scope,
    )

    install_profiler(ProfilerConfig(backend="torch",
                                    output_dir=Path("./prof")))

    profiler = get_profiler()
    profiler.start()
    for step in range(n):
        with event_scope("step"):
            with event_scope("phase_a"): run_a()
            with event_scope("phase_b"): run_b()
    profiler.stop()

CLI integration
---------------
Benchmark scripts wire the same flags via
:func:`add_profile_cli_args` + :func:`profile_config_from_args` so the
``--profile-*`` namespace stays consistent across every bench script.
"""

from __future__ import annotations

import abc
import argparse
import contextlib
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, ContextManager, Iterator, Literal, Sequence

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


ProfilerBackendName = Literal["none", "torch", "nsys"]
_VALID_BACKENDS: tuple[ProfilerBackendName, ...] = ("none", "torch", "nsys")
_VALID_ACTIVITIES: tuple[str, ...] = ("CPU", "GPU")


# ============================================================================ #
# Config                                                                       #
# ============================================================================ #


@dataclass(frozen=True)
class ProfilerConfig:
    """Process-level profile knobs.

    ``backend`` picks the underlying collector(s). Other fields are
    forwarded to whichever backend reads them — :class:`TorchProfiler`
    consumes ``activities`` / ``with_stack`` / ``record_shapes`` /
    ``profile_memory``; :class:`NsysProfiler` consumes
    ``emit_module_nvtx``.

    ``only_rank`` is a guard for distributed runs: when set,
    :func:`make_profiler` substitutes :class:`NoOpProfiler` on every
    rank that isn't this one, so traces don't pile up N copies of the
    same workload. ``None`` records on every rank.
    """

    backend: ProfilerBackendName = "none"
    activities: tuple[str, ...] = ("CPU", "GPU")
    with_stack: bool = True
    record_shapes: bool = False
    profile_memory: bool = False
    output_dir: Path = field(default_factory=lambda: Path("./phyai_profile"))
    file_prefix: str = "profile"
    run_name: str = "default"
    emit_module_nvtx: bool = False
    only_rank: int | None = 0

    def __post_init__(self) -> None:
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(f"backend={self.backend!r} not in {_VALID_BACKENDS!r}")
        for a in self.activities:
            if a not in _VALID_ACTIVITIES:
                raise ValueError(f"activity {a!r} not in {_VALID_ACTIVITIES!r}")


# ============================================================================ #
# Backend ABC + No-op                                                          #
# ============================================================================ #


class Profiler(abc.ABC):
    """Profile collector ABC.

    Lifecycle: one :meth:`start` → many :meth:`event_scope` /
    :meth:`mark_instant` calls → one :meth:`stop`. :meth:`stop` is
    responsible for flushing / exporting whatever the backend collected.
    """

    name: ClassVar[str] = "abstract"

    def __init__(self) -> None:
        self._active: bool = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        """Begin collecting. Idempotent: a second start while active is a no-op."""
        if self._active:
            return
        self._do_start()
        self._active = True

    def stop(self) -> None:
        """Finish collecting and flush output. Idempotent on second call."""
        if not self._active:
            return
        self._do_stop()
        self._active = False

    def _do_start(self) -> None:  # pragma: no cover - default no-op
        return None

    def _do_stop(self) -> None:  # pragma: no cover - default no-op
        return None

    @abc.abstractmethod
    def event_scope(self, name: str, **args: Any) -> ContextManager[None]:
        """Open a named scoped event. ``args`` are best-effort attached as metadata."""

    @abc.abstractmethod
    def mark_instant(self, name: str, **args: Any) -> None:
        """Emit a zero-duration marker at the current time."""

    def attach_module_hooks(self, module: torch.nn.Module, prefix: str = "") -> None:
        """Auto-instrument every submodule's forward with an event_scope.

        Default: no-op. NVTX-capable backends override.
        """
        return None


class NoOpProfiler(Profiler):
    """Zero-cost default. ``event_scope`` is ``nullcontext``."""

    name: ClassVar[str] = "none"

    def event_scope(self, name: str, **args: Any) -> ContextManager[None]:
        return contextlib.nullcontext()

    def mark_instant(self, name: str, **args: Any) -> None:
        return None


# ============================================================================ #
# Torch profiler backend                                                       #
# ============================================================================ #


def _activities_to_torch(activities: Sequence[str]) -> list[Any]:
    mapping = {
        "CPU": torch.profiler.ProfilerActivity.CPU,
        "GPU": torch.profiler.ProfilerActivity.CUDA,
    }
    return [mapping[a] for a in activities if a in mapping]


class TorchProfiler(Profiler):
    """Wraps :class:`torch.profiler.profile` and exports a chrome trace.

    The exported ``.trace.json.gz`` is loadable both by Chrome
    ``about:tracing`` and by Perfetto's web UI without further
    conversion. ``event_scope`` uses
    :func:`torch.autograd.profiler.record_function` so the scope shows
    up in the trace as a named span on the CPU lane.
    """

    name: ClassVar[str] = "torch"

    def __init__(self, cfg: ProfilerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._profiler: torch.profiler.profile | None = None
        self._seq: int = 0

    def _do_start(self) -> None:
        self._profiler = torch.profiler.profile(
            activities=_activities_to_torch(self.cfg.activities),
            with_stack=self.cfg.with_stack,
            record_shapes=self.cfg.record_shapes,
            profile_memory=self.cfg.profile_memory,
        )
        self._profiler.start()

    def _do_stop(self) -> None:
        assert self._profiler is not None
        try:
            self._profiler.stop()
            output_path = _build_output_path(
                self.cfg, suffix=".trace.json.gz", seq=self._seq
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._profiler.export_chrome_trace(str(output_path))
            logger.info(
                "TorchProfiler: chrome trace written to %s (Perfetto-loadable)",
                output_path,
            )
            self._seq += 1
        finally:
            self._profiler = None

    def event_scope(self, name: str, **args: Any) -> ContextManager[None]:
        # record_function carries no kwargs in pre-2.4 builds; we ignore
        # ``args`` rather than crashing on older torch versions.
        return torch.autograd.profiler.record_function(name)

    def mark_instant(self, name: str, **args: Any) -> None:
        # Best-effort: a 1-step record_function is small enough to read
        # as an "instant" in the trace viewer.
        with torch.autograd.profiler.record_function(name):
            pass


# ============================================================================ #
# Nsight Systems backend (cudaProfilerStart/Stop + NVTX)                       #
# ============================================================================ #


def _have_cuda() -> bool:
    return torch.cuda.is_available()


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    """NVTX push/pop with a graceful fallback when CUDA is not present."""
    if not _have_cuda():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class NsysProfiler(Profiler):
    """``cudaProfilerStart/Stop`` + NVTX scopes for an enclosing nsys run.

    Pair with ``nsys profile --capture-range=cudaProfilerApi
    --capture-range-end=stop`` so the recording covers exactly the
    window between :meth:`start` and :meth:`stop`. NVTX ranges land on
    the timeline as labelled spans under the "NVTX" lane.
    """

    name: ClassVar[str] = "nsys"

    def __init__(self, cfg: ProfilerConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def _do_start(self) -> None:
        if not _have_cuda():
            logger.warning(
                "NsysProfiler.start: CUDA unavailable; "
                "cudaProfilerStart will be skipped (NVTX scopes still run)."
            )
            return
        torch.cuda.cudart().cudaProfilerStart()
        logger.info(
            "NsysProfiler: cudaProfilerStart called "
            "(nsys begins recording if attached)."
        )

    def _do_stop(self) -> None:
        if not _have_cuda():
            return
        torch.cuda.cudart().cudaProfilerStop()
        logger.info("NsysProfiler: cudaProfilerStop called.")

    def event_scope(self, name: str, **args: Any) -> ContextManager[None]:
        return _nvtx_range(name)

    def mark_instant(self, name: str, **args: Any) -> None:
        if not _have_cuda():
            return
        # No torch wrapper for nvtxMark; push+pop emits a near-zero span
        # which Nsight renders as a marker.
        torch.cuda.nvtx.range_push(name)
        torch.cuda.nvtx.range_pop()

    def attach_module_hooks(self, module: torch.nn.Module, prefix: str = "") -> None:
        """Register per-submodule NVTX ranges via forward pre/post hooks.

        Slim variant of sglang's :class:`PytHooks`: only emits the module
        path (no tensor-shape introspection). Best paired with
        ``emit_module_nvtx=True`` in :class:`ProfilerConfig`.
        """
        for name, sub in module.named_modules(prefix=prefix):
            if not name:
                continue

            def _pre(mod: torch.nn.Module, _inp: Any, _name: str = name) -> None:
                torch.cuda.nvtx.range_push(_name)

            def _post(mod: torch.nn.Module, _inp: Any, _out: Any) -> None:
                torch.cuda.nvtx.range_pop()

            sub.register_forward_pre_hook(_pre)
            sub.register_forward_hook(_post)


# ============================================================================ #
# Factory + singleton                                                          #
# ============================================================================ #


def make_profiler(cfg: ProfilerConfig) -> Profiler:
    """Construct the right :class:`Profiler` from ``cfg``.

    Substitutes :class:`NoOpProfiler` when ``cfg.only_rank`` excludes
    the current distributed rank, so callers can install the same
    config on every rank and trust the gating to apply.
    """
    if not _rank_allowed(cfg.only_rank):
        return NoOpProfiler()
    if cfg.backend == "none":
        return NoOpProfiler()
    if cfg.backend == "torch":
        return TorchProfiler(cfg)
    if cfg.backend == "nsys":
        return NsysProfiler(cfg)
    raise ValueError(f"unknown backend: {cfg.backend!r}")


_PROFILER: Profiler = NoOpProfiler()


def get_profiler() -> Profiler:
    """Return the process-level profiler singleton (NoOp by default)."""
    return _PROFILER


def set_profiler(p: Profiler) -> None:
    """Install ``p`` as the process-level profiler singleton."""
    global _PROFILER
    _PROFILER = p


def install_profiler(cfg: ProfilerConfig) -> Profiler:
    """Construct from ``cfg`` and install it as the singleton; return the new profiler."""
    p = make_profiler(cfg)
    set_profiler(p)
    return p


def event_scope(name: str, **args: Any) -> ContextManager[None]:
    """Wrap a block with the active profiler's scoped event.

    Cheap when no profiler is installed — defaults to ``nullcontext``.
    Always safe to drop in hot code.
    """
    return _PROFILER.event_scope(name, **args)


def mark_instant(name: str, **args: Any) -> None:
    """Emit a zero-duration marker through the active profiler."""
    _PROFILER.mark_instant(name, **args)


# ============================================================================ #
# Helpers                                                                      #
# ============================================================================ #


def _current_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _rank_allowed(only_rank: int | None) -> bool:
    if only_rank is None:
        return True
    return _current_rank() == only_rank


def _build_output_path(cfg: ProfilerConfig, *, suffix: str, seq: int = 0) -> Path:
    rank = _current_rank()
    ts = time.strftime("%Y%m%d-%H%M%S")
    parts = [cfg.file_prefix, cfg.run_name, ts, f"rank{rank}", f"seq{seq:03d}"]
    name = "_".join(parts) + suffix
    return Path(cfg.output_dir) / name


# ============================================================================ #
# CLI helpers                                                                  #
# ============================================================================ #


def add_profile_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add the common ``--profile-*`` flag group to ``parser``.

    Pair with :func:`profile_config_from_args` to build the
    :class:`ProfilerConfig` from the parsed namespace.
    """
    group = parser.add_argument_group("profile")
    group.add_argument(
        "--profile-backend",
        choices=list(_VALID_BACKENDS),
        default="none",
        help="Which profiler backend to activate.",
    )
    group.add_argument(
        "--profile-activities",
        nargs="+",
        default=["CPU", "GPU"],
        choices=list(_VALID_ACTIVITIES),
        help="torch.profiler activities (CPU / GPU).",
    )
    group.add_argument(
        "--profile-with-stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record python / cpp stacks in torch profiler output.",
    )
    group.add_argument(
        "--profile-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record tensor shapes in torch profiler output (slower).",
    )
    group.add_argument(
        "--profile-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record allocator events in torch profiler output.",
    )
    group.add_argument(
        "--profile-output-dir",
        type=Path,
        default=Path("./phyai_profile"),
        help="Directory for profile trace files.",
    )
    group.add_argument(
        "--profile-file-prefix",
        type=str,
        default="profile",
        help="Filename prefix for profile traces.",
    )
    group.add_argument(
        "--profile-emit-module-nvtx",
        action="store_true",
        help="Auto-attach per-submodule NVTX hooks at install time "
        "(the bench script is responsible for calling "
        "profiler.attach_module_hooks(model) after model construction).",
    )
    group.add_argument(
        "--profile-only-rank",
        type=int,
        default=0,
        help="Distributed rank to record on. Pass -1 to record on every rank.",
    )


def profile_config_from_args(args: argparse.Namespace) -> ProfilerConfig:
    """Lift the ``--profile-*`` namespace into a :class:`ProfilerConfig`.

    ``--profile-only-rank=-1`` is mapped back to ``only_rank=None``
    (record everywhere) since argparse can't express ``None`` directly.
    """
    only_rank = args.profile_only_rank
    if only_rank is not None and only_rank < 0:
        only_rank = None
    return ProfilerConfig(
        backend=args.profile_backend,
        activities=tuple(args.profile_activities),
        with_stack=args.profile_with_stack,
        record_shapes=args.profile_record_shapes,
        profile_memory=args.profile_memory,
        output_dir=Path(args.profile_output_dir),
        file_prefix=args.profile_file_prefix,
        run_name=getattr(args, "run_name", "default"),
        emit_module_nvtx=args.profile_emit_module_nvtx,
        only_rank=only_rank,
    )


__all__ = [
    "NoOpProfiler",
    "NsysProfiler",
    "Profiler",
    "ProfilerBackendName",
    "ProfilerConfig",
    "TorchProfiler",
    "add_profile_cli_args",
    "event_scope",
    "get_profiler",
    "install_profiler",
    "make_profiler",
    "mark_instant",
    "profile_config_from_args",
    "set_profiler",
]
