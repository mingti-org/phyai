"""Process-level engine for registered model plugins."""

from __future__ import annotations

import abc
import os
import time
import threading
from concurrent.futures import Future
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Generator
from datetime import timedelta
from contextlib import contextmanager
from dataclasses import replace, dataclass
from enum import Enum

import torch
import torch.distributed as dist
from torch import nn

import phyai.parallel as P
from phyai.utils import get_logger
from phyai.utils.cuda import init_cuda, format_gib, init_cublas, available_memory_bytes
from phyai.kernel.call import freeze_kernel_choices
from phyai.layers.attention.utils import release_global_fi_workspaces
from phyai.kernel.types import ModelContext
from phyai.engine_config import EngineConfig, ParallelConfig, init_engine_config
from phyai.parallel.config import DOMAINS, MODEL_DOMAINS
from phyai.parallel.layout import build_rank_layout
from phyai.server.deployment import DeploymentConfig, build_dispatcher
from phyai.server.lifecycle import EngineUnavailableError
from phyai.parallel.dist import init_dist
from phyai.utils.logging import configure_logging
from phyai.utils.env_setup import init_env, set_ulimit, init_process_debug
from phyai.kernel.bootstrap import reset_kernel_selector, initialize_kernel_system
from phyai.utils.torch_setup import init_seed, disable_grad, init_threads
from phyai.runtime.tensor_dump import (
    TensorDumper,
    load_filter_fn,
    register_tensor_dumper,
)

logger = get_logger(__name__)


def _runtime_replica_world_size(parallel: ParallelConfig) -> int:
    """Return the physical rank pool for this process or launcher."""
    launcher_world = os.environ.get("WORLD_SIZE")
    if dist.is_initialized():
        actual = dist.get_world_size()
        if launcher_world is not None and int(launcher_world) != actual:
            raise ValueError("WORLD_SIZE does not match the initialized process group.")
        return actual
    if launcher_world is None:
        return parallel.infer_replica_world_size()
    try:
        world = int(launcher_world)
    except ValueError as error:
        raise ValueError("WORLD_SIZE must be an integer launcher value.") from error
    if world < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world}.")
    return world


def _force_eager_for_dump(cfg: EngineConfig) -> EngineConfig:
    """Disable CUDA graphs in ``cfg`` when tensor dumping is enabled."""
    if not cfg.runtime.use_cuda_graph:
        return cfg
    return cfg.replace(runtime=replace(cfg.runtime, use_cuda_graph=False))


def _force_eager_without_cuda(cfg: EngineConfig) -> EngineConfig:
    """Disable CUDA graphs in ``cfg`` when the target is not CUDA."""
    if not cfg.runtime.use_cuda_graph:
        return cfg
    if torch.device(cfg.device.target).type == "cuda":
        return cfg
    return cfg.replace(runtime=replace(cfg.runtime, use_cuda_graph=False))


def _resolve_engine_config(args: "EngineArgs") -> EngineConfig:
    """Resolve config without performing CUDA or distributed initialization."""
    resolved = EngineConfig.from_env(base=args.config)
    if resolved.runtime.debug_tensor_dump_dir is not None:
        forced = _force_eager_for_dump(resolved)
        if forced is not resolved:
            logger.warning(
                "Tensor dump enabled (debug_tensor_dump_dir=%s): forcing "
                "use_cuda_graph=False.",
                resolved.runtime.debug_tensor_dump_dir,
            )
        resolved = forced
    return _force_eager_without_cuda(resolved)


@dataclass
class EntryArgs:
    """Base class for plugin argument dataclasses."""


class Entry(abc.ABC):
    """Interface for a model plugin's setup, inference, and cleanup."""

    name: ClassVar[str]
    args_cls: ClassVar[type[EntryArgs]]
    # Logical parallel domains this plugin actually uses. Empty means
    # single-rank only, so unsupported topology cannot silently duplicate a
    # complete model on every rank.
    parallel_domains: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def validate_parallel(
        cls, parallel: "ParallelConfig", replica_world_size: int | None = None
    ) -> None:
        """Reject parallel domains this plugin does not implement.

        Plugins declare the domains they implement through
        :attr:`parallel_domains` and may override this method to add tighter
        constraints (calling ``super`` first). Resolution is against the
        launcher's physical rank pool.

        A multi-rank ``pipeline`` or ``cfg`` dimension must be declared. A
        model domain the plugin does not declare may only follow a declared
        one: it stays TP-only and its TP groups have the same members as a
        declared domain's TP groups, so the plugin's layers see one layout
        without implementing anything for that domain.
        """
        world = (
            parallel.infer_replica_world_size()
            if replica_world_size is None
            else replica_world_size
        )
        resolved = parallel.resolve(world)
        unsupported: dict[str, str] = {}
        for name, size in (
            ("pipeline", resolved.outer.pipeline_size),
            ("cfg", resolved.outer.cfg_size),
        ):
            if size > 1 and name not in cls.parallel_domains:
                unsupported[name] = f"size {size}"
        if resolved.scope_size > 1:
            declared = [name for name in MODEL_DOMAINS if name in cls.parallel_domains]
            if not declared:
                unsupported["model_scope"] = f"size {resolved.scope_size}"
            else:
                layout = build_rank_layout(resolved)
                declared_tp = {layout.groups_for(f"{name}_tp") for name in declared}
                for name in MODEL_DOMAINS:
                    if name in cls.parallel_domains:
                        continue
                    if not resolved.domain(name).is_tp_only:
                        unsupported[name] = "uses more than tensor parallelism"
                    elif layout.groups_for(f"{name}_tp") not in declared_tp:
                        unsupported[name] = (
                            f"tp groups differ from {', '.join(declared)}"
                        )
        if unsupported:
            raise ValueError(
                f"plugin {cls.name!r} supports parallel domains "
                f"{sorted(cls.parallel_domains)!r}; unsupported: {unsupported!r}."
            )

    @abc.abstractmethod
    def setup(self, args: EntryArgs) -> None:
        """Build the model, load weights, prepare runners / scheduler."""

    @abc.abstractmethod
    def step(self, request: Any) -> Any:
        """Run one inference round. Request / response shape is plugin-defined."""

    def rollout_step(self, request: Any, **kwargs: Any) -> Any:
        """Run an opt-in training rollout round with plugin-specific state."""
        del request, kwargs
        raise NotImplementedError(
            f"Plugin {self.name!r} does not support training rollout output."
        )

    def close(self) -> None:
        """Release pinned GPU resources. Default: no-op."""
        return None

    def begin_weight_update(self) -> None:
        """Begin one streamed in-memory weight update."""
        raise NotImplementedError(
            f"Plugin {self.name!r} does not support hot weight updates."
        )

    def update_weights(
        self,
        weights: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Apply one named-tensor batch to the active update session."""
        del weights
        raise NotImplementedError(
            f"Plugin {self.name!r} does not support hot weight updates."
        )

    def finish_weight_update(self) -> Any:
        """Finalize the active update and refresh derived runtime state."""
        raise NotImplementedError(
            f"Plugin {self.name!r} does not support hot weight updates."
        )

    def abort_weight_update(self) -> None:
        """Discard bookkeeping for an interrupted update session."""

    def dump_targets(self) -> dict[str, nn.Module]:
        """Return the root modules that should be included in tensor dumps."""
        return {}


@dataclass
class EngineArgs:
    """Select a plugin and provide its arguments and optional config."""

    plugin: str
    plugin_args: EntryArgs
    config: EngineConfig | None = None


class _WeightUpdateState(Enum):
    """Lifecycle state for streamed in-memory weight updates."""

    IDLE = "idle"
    UPDATING_CLEAN = "updating_clean"
    UPDATING_DIRTY = "updating_dirty"
    POISONED = "poisoned"


class EngineCore:
    """In-process model runtime used by every GPU worker."""

    _plugins: ClassVar[dict[str, type[Entry]]] = {}

    @classmethod
    def register(cls, entry_cls: type[Entry]) -> type[Entry]:
        """Register a plugin entry class and return it unchanged."""
        if not isinstance(entry_cls, type) or not issubclass(entry_cls, Entry):
            raise TypeError(
                f"Engine.register expected an Entry subclass, got {entry_cls!r}."
            )
        name = getattr(entry_cls, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(f"{entry_cls.__name__}.name must be a non-empty string.")
        args_cls = getattr(entry_cls, "args_cls", None)
        if not isinstance(args_cls, type) or not issubclass(args_cls, EntryArgs):
            raise TypeError(
                f"{entry_cls.__name__}.args_cls must be an EntryArgs subclass."
            )
        existing = cls._plugins.get(name)
        if existing is not None and existing is not entry_cls:
            raise ValueError(
                f"plugin name {name!r} is already registered to {existing.__name__}."
            )
        valid_domains = set(DOMAINS)
        unknown_domains = set(entry_cls.parallel_domains) - valid_domains
        if unknown_domains:
            raise ValueError(
                f"{entry_cls.__name__}.parallel_domains contains unknown domains "
                f"{sorted(unknown_domains)!r}; valid domains: {sorted(valid_domains)!r}."
            )
        cls._plugins[name] = entry_cls
        return entry_cls

    @classmethod
    def registered(cls) -> tuple[str, ...]:
        """Return all registered plugin names in registration order."""
        return tuple(cls._plugins.keys())

    @classmethod
    def plugin_class(cls, name: str) -> type[Entry]:
        """Resolve one preinstalled plugin without importing wire-provided code."""
        try:
            return cls._plugins[name]
        except KeyError as error:
            raise ValueError(
                f"unknown plugin {name!r}; registered: {list(cls._plugins)!r}."
            ) from error

    def __init__(self, args: EngineArgs) -> None:
        # Resolve the plugin contract before touching CUDA or distributed state.
        # Invalid public arguments should fail without allocating a process group.
        entry_cls = self._plugins.get(args.plugin)
        if entry_cls is None:
            raise ValueError(
                f"unknown plugin {args.plugin!r}; registered: {list(self._plugins)!r}."
            )
        if not isinstance(args.plugin_args, entry_cls.args_cls):
            raise TypeError(
                f"plugin {entry_cls.name!r} expects "
                f"{entry_cls.args_cls.__name__}; got "
                f"{type(args.plugin_args).__name__}."
            )
        self.args = args
        self.entry: Entry | None = None
        self._owns_pg = False
        self._closed = False

        # 1. Resolve the effective engine configuration.
        resolved = _resolve_engine_config(args)
        world_size = _runtime_replica_world_size(resolved.parallel)
        entry_cls.validate_parallel(resolved.parallel, world_size)
        self._dump_enabled = resolved.runtime.debug_tensor_dump_dir is not None
        if self._dump_enabled:
            forced = _force_eager_for_dump(resolved)
            if forced is not resolved:
                logger.warning_rank0(
                    "Forward hooks cannot fire during CUDA-graph replay; "
                    "activation capture runs eager-only."
                )
            resolved = forced
        resolved = _force_eager_without_cuda(resolved)

        device_type = torch.device(resolved.device.target).type

        # 2. Initialize logging and the process environment.
        configure_logging()
        init_env(world_size=world_size, device_type=device_type)
        set_ulimit()
        init_process_debug()

        self.config: EngineConfig = init_engine_config(resolved)
        self._replica_world_size = world_size
        self._dumper: TensorDumper | None = None
        self._t_start = time.perf_counter()

        # 3. Initialize PyTorch process state.
        with self._stage("torch_setup"):
            init_threads(
                device_type=device_type, num_threads=self.config.runtime.num_threads
            )
            init_seed(self.config.runtime.seed)
            disable_grad()

        # 4. Initialize CUDA and cuBLAS.
        with self._stage("cuda"):
            init_cuda(self.config.device.target, self.config.device.params_dtype)
            init_cublas()

        # 5. Initialize the distributed process group.
        with self._stage("dist"):
            self._owns_pg: bool = init_dist(
                world_size=world_size,
                device_type=device_type,
                timeout=timedelta(seconds=self.config.runtime.dist_timeout_s),
                require_launcher=world_size > 1,
                device=self.config.device.target,
            )

        # 6. Initialize the parallel groups and warm their communicators.
        with self._stage("mesh"):
            mesh = P.init(
                self.config.parallel,
                replica_world_size=world_size,
                device=device_type,
            )

        process_title = f"phyai::{args.plugin}"
        for group in mesh.distinct_groups():
            process_title += f"_{group.upper()}{mesh.group_rank(group)}"
        init_process_debug(title=process_title)

        if world_size > 1:
            # Create communicators before any graph capture.
            with self._stage("collectives_warmup"):
                warmed = P.warmup_collectives()
            if warmed:
                logger.info_rank0("warmed collectives on groups %s", warmed)

        # 7. Initialize kernel selection.
        with self._stage("kernel"):
            self.kernel_selector = initialize_kernel_system(
                self.config.kernel,
                device=self.config.device.target,
                model=ModelContext(family=args.plugin),
            )

        # 8. Initialize the selected model plugin.
        self.entry = entry_cls()
        with self._stage("plugin_setup"):
            self.entry.setup(args.plugin_args)
        self._model_lock = threading.Lock()
        self._weight_update_lock = threading.Lock()
        self._weight_update_state = _WeightUpdateState.IDLE
        self._version = 0

        # 9. Finalize kernel choices and attach debugging hooks.
        if self.config.runtime.freeze_kernel_choices:
            freeze_kernel_choices()

        if self._dump_enabled:
            self._dumper = self._build_dumper()

        free_now = available_memory_bytes(self.config.device.target)
        logger.info_rank0(
            "Engine ready (plugin=%s). total=%.2fs free=%s GiB",
            args.plugin,
            time.perf_counter() - self._t_start,
            format_gib(free_now),
        )

    @contextmanager
    def _stage(self, name: str) -> Generator[None]:
        """Time a bootstrap stage and log its device-memory change."""
        free_before = available_memory_bytes(self.config.device.target)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            free_after = available_memory_bytes(self.config.device.target)
            logger.info_rank0(
                "init stage=%s elapsed=%.2fs used=%s GiB free=%s GiB",
                name,
                elapsed,
                format_gib(max(0, free_before - free_after)),
                format_gib(free_after),
            )

    def _build_dumper(self) -> TensorDumper | None:
        """Build a tensor dumper from the plugin's dump targets."""
        if self.entry is None:
            raise RuntimeError("cannot build a tensor dumper before plugin setup.")
        runtime = self.config.runtime
        targets = self.entry.dump_targets()
        if not targets:
            logger.warning_rank0(
                "Tensor dump is enabled but plugin %r exposes no dump_targets(); "
                "nothing will be recorded. Override Entry.dump_targets() to return "
                "the model module(s) to capture.",
                self.args.plugin,
            )
            return None
        filter_spec = self._resolve_dump_filter()
        return register_tensor_dumper(
            targets,
            dump_dir=runtime.debug_tensor_dump_dir,
            filter=filter_spec,
        )

    def _resolve_dump_filter(self):
        """Resolve the configured tensor-dump filter."""
        runtime = self.config.runtime
        if runtime.debug_tensor_dump_filter_fn is not None:
            return load_filter_fn(runtime.debug_tensor_dump_filter_fn)
        return runtime.debug_tensor_dump_filter

    def step(self, request: Any) -> Any:
        """Run one inference round and flush tensor dumps when enabled."""
        with self._model_lock:
            self._require_usable()
            if self._closed:
                raise EngineUnavailableError("cannot execute on a closed EngineCore.")
            assert self.entry is not None
            result = self.entry.step(request)
            if self._dumper is not None:
                self._dumper.flush_pass()
            return result

    def rollout_step(self, request: Any, **kwargs: Any) -> Any:
        """Return plugin-specific training state without changing ``step``."""
        with self._model_lock:
            self._require_usable()
            if self._closed:
                raise EngineUnavailableError("cannot execute on a closed EngineCore.")
            assert self.entry is not None
            result = self.entry.rollout_step(request, **kwargs)
            if self._dumper is not None:
                self._dumper.flush_pass()
            return result

    @property
    def version(self) -> int:
        """Last fully committed in-memory weight version."""
        return self._version

    def _require_usable(self) -> None:
        if self._weight_update_state is _WeightUpdateState.POISONED:
            raise RuntimeError(
                "Engine is unavailable after a partially applied weight update; "
                "recreate it before further inference."
            )

    def begin_weight_update(self) -> None:
        """Begin one streamed hot update on the active plugin."""
        with self._weight_update_lock:
            if self._closed:
                raise EngineUnavailableError(
                    "cannot update weights on a closed EngineCore."
                )
            if self._weight_update_state is not _WeightUpdateState.IDLE:
                self._require_usable()
                raise RuntimeError("A weight update is already active.")
            self._model_lock.acquire()
            try:
                self._require_usable()
                assert self.entry is not None
                self.entry.begin_weight_update()
                self._weight_update_state = _WeightUpdateState.UPDATING_CLEAN
            except Exception:
                self._model_lock.release()
                raise

    def update_weights(
        self,
        weights: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Forward one named-tensor batch to the active plugin."""
        with self._weight_update_lock:
            if self._weight_update_state not in (
                _WeightUpdateState.UPDATING_CLEAN,
                _WeightUpdateState.UPDATING_DIRTY,
            ):
                raise RuntimeError("begin_weight_update() must be called first.")
            assert self.entry is not None
            # An entry may mutate model state before reporting an error, so mark
            # the session dirty before dispatching the batch.
            self._weight_update_state = _WeightUpdateState.UPDATING_DIRTY
            self.entry.update_weights(weights)

    def finish_weight_update(self, version: int | None = None) -> Any:
        """Finalize an update and commit its version only after validation."""
        with self._weight_update_lock:
            if self._weight_update_state not in (
                _WeightUpdateState.UPDATING_CLEAN,
                _WeightUpdateState.UPDATING_DIRTY,
            ):
                raise RuntimeError("No weight update is active.")
            try:
                assert self.entry is not None
                report = self.entry.finish_weight_update()
                self._version = self._version + 1 if version is None else int(version)
                self._weight_update_state = _WeightUpdateState.IDLE
                return report
            except Exception:
                self._weight_update_state = (
                    _WeightUpdateState.POISONED
                    if self._weight_update_state is _WeightUpdateState.UPDATING_DIRTY
                    else _WeightUpdateState.IDLE
                )
                raise
            finally:
                self._model_lock.release()

    def _abort_weight_update_locked(self) -> None:
        """Abort an active update while ``_weight_update_lock`` is held."""
        state = self._weight_update_state
        if state not in (
            _WeightUpdateState.UPDATING_CLEAN,
            _WeightUpdateState.UPDATING_DIRTY,
        ):
            return
        try:
            assert self.entry is not None
            self.entry.abort_weight_update()
        finally:
            self._weight_update_state = (
                _WeightUpdateState.POISONED
                if state is _WeightUpdateState.UPDATING_DIRTY
                else _WeightUpdateState.IDLE
            )
            self._model_lock.release()

    def abort_weight_update(self) -> None:
        """Abort bookkeeping for the active plugin update."""
        with self._weight_update_lock:
            self._abort_weight_update_locked()

    def close(self) -> None:
        """Release plugin resources and process-level runtime services."""
        close_error: BaseException | None = None
        with self._weight_update_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._abort_weight_update_locked()
            except BaseException as error:  # noqa: BLE001
                close_error = error
            self._model_lock.acquire()
        try:
            if self._dumper is not None:
                try:
                    self._dumper.detach()
                except BaseException as error:
                    if close_error is None:
                        close_error = error
                self._dumper = None
            if self.entry is not None:
                try:
                    self.entry.close()
                except BaseException as error:
                    if close_error is None:
                        close_error = error
                self.entry = None
        finally:
            self._model_lock.release()
        try:
            # Release the collective backends P.init built (direct NCCL
            # communicators, bootstrap groups, the mesh/dispatcher
            # singletons) before the torch process group goes away.
            # Single-rank engines register nothing that needs releasing and
            # may share a process with sibling engines, so they leave the
            # process-level registry alone.
            if self._replica_world_size > 1:
                P.shutdown()
        except BaseException as error:
            if close_error is None:
                close_error = error
        try:
            if self._owns_pg and dist.is_initialized():
                dist.destroy_process_group()
        except BaseException as error:
            if close_error is None:
                close_error = error
        finally:
            self._owns_pg = False
            try:
                reset_kernel_selector()
            except BaseException as error:
                if close_error is None:
                    close_error = error
            try:
                release_global_fi_workspaces()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        if close_error is not None:
            raise close_error


class Engine:
    """Public facade over inline, managed, distributed, and external execution."""

    _plugins: ClassVar[dict[str, type[Entry]]] = EngineCore._plugins

    @classmethod
    def register(cls, entry_cls: type[Entry]) -> type[Entry]:
        """Register a model plugin for both the facade and EngineCore."""
        return EngineCore.register(entry_cls)

    @classmethod
    def registered(cls) -> tuple[str, ...]:
        """Return registered model plugin names in registration order."""
        return EngineCore.registered()

    def __init__(
        self,
        args: EngineArgs,
        *,
        deployment: DeploymentConfig | None = None,
    ) -> None:
        if not isinstance(args, EngineArgs):
            raise TypeError(f"Engine expects EngineArgs, got {type(args).__name__}.")
        if deployment is not None and not isinstance(deployment, DeploymentConfig):
            raise TypeError(
                "deployment must be a DeploymentConfig, got "
                f"{type(deployment).__name__}."
            )
        deployment_config = deployment or DeploymentConfig()
        resolved = _resolve_engine_config(args)
        resolved_args = replace(args, config=resolved)
        entry_cls = self._plugins.get(args.plugin)
        if entry_cls is None:
            raise ValueError(
                f"unknown plugin {args.plugin!r}; registered: {list(self._plugins)!r}."
            )
        if not isinstance(args.plugin_args, entry_cls.args_cls):
            raise TypeError(
                f"plugin {entry_cls.name!r} expects "
                f"{entry_cls.args_cls.__name__}; got "
                f"{type(args.plugin_args).__name__}."
            )
        world_size = _runtime_replica_world_size(resolved.parallel)
        entry_cls.validate_parallel(resolved.parallel, world_size)
        if not 0 <= deployment_config.output_rank < world_size:
            raise ValueError(
                f"output_rank={deployment_config.output_rank} is outside the model "
                f"replica world_size={world_size}."
            )
        self.args = resolved_args
        self.config = resolved
        self.deployment = deployment_config
        self._closed = False
        self._mode, self._dispatcher = build_dispatcher(
            EngineCore,
            resolved_args,
            world_size,
            deployment_config,
        )
        self._core = self._dispatcher.core
        if deployment_config.auto_start:
            try:
                self._dispatcher.setup()
            except BaseException:
                self._dispatcher.close()
                raise

    @property
    def entry(self) -> Entry | None:
        """Return the local plugin entry, or ``None`` for managed parents."""
        return None if self._core is None else self._core.entry

    @property
    def mode(self) -> str:
        """Return the resolved execution mode: inline, local, or external."""
        return self._mode

    def step(self, request: Any) -> Any:
        """Run one request synchronously through the selected backend."""
        if self._closed:
            raise EngineUnavailableError("cannot execute on a closed Engine.")
        return self._dispatcher.step(request)

    def rollout_step(self, request: Any, **kwargs: Any) -> Any:
        """Run a training rollout on the local core."""
        if self._closed:
            raise EngineUnavailableError("cannot execute on a closed Engine.")
        if self._core is None:
            raise EngineUnavailableError(
                "rollout_step requires an inline or single external core."
            )
        return self._core.rollout_step(request, **kwargs)

    @property
    def version(self) -> int:
        """Return the last committed local weight version."""
        if self._core is None:
            raise EngineUnavailableError(
                "weight versions require an inline or single external core."
            )
        return self._core.version

    def begin_weight_update(self) -> None:
        """Begin a streamed hot update on the local core."""
        if self._closed:
            raise EngineUnavailableError("cannot execute on a closed Engine.")
        if self._core is None:
            raise EngineUnavailableError(
                "weight updates require an inline or single external core."
            )
        self._core.begin_weight_update()

    def update_weights(
        self,
        weights: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Apply one named-tensor batch to the local core."""
        if self._closed:
            raise EngineUnavailableError("cannot execute on a closed Engine.")
        if self._core is None:
            raise EngineUnavailableError(
                "weight updates require an inline or single external core."
            )
        self._core.update_weights(weights)

    def finish_weight_update(self, version: int | None = None) -> Any:
        """Finalize a streamed update on the local core."""
        if self._closed:
            raise EngineUnavailableError("cannot execute on a closed Engine.")
        if self._core is None:
            raise EngineUnavailableError(
                "weight updates require an inline or single external core."
            )
        return self._core.finish_weight_update(version=version)

    def abort_weight_update(self) -> None:
        """Abort a streamed update on the local core."""
        if self._closed:
            raise EngineUnavailableError("cannot execute on a closed Engine.")
        if self._core is None:
            raise EngineUnavailableError(
                "weight updates require an inline or single external core."
            )
        self._core.abort_weight_update()

    def setup(self) -> None:
        """Ensure a managed worker group is started; inline engines are ready."""
        if self._closed:
            raise EngineUnavailableError("cannot set up a closed Engine.")
        self._dispatcher.setup()

    def submit(self, request: Any) -> Future[Any]:
        """Submit a request and return a backend-independent Future.

        On the inline backend a queued (not yet started) request can still be
        cancelled through ``Future.cancel()``; managed backends dispatch to
        worker pipes immediately, so their futures are never cancellable.
        """
        if self._closed:
            future: Future[Any] = Future()
            future.set_exception(
                EngineUnavailableError("cannot execute on a closed Engine.")
            )
            return future
        return self._dispatcher.submit(request)

    def close(self) -> None:
        """Close the selected backend exactly once."""
        if self._closed:
            return
        self._closed = True
        self._dispatcher.close()

    def __enter__(self) -> "Engine":
        self.setup()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


__all__ = [
    "EngineArgs",
    "Engine",
    "DeploymentConfig",
    "EngineUnavailableError",
    "Entry",
    "EntryArgs",
]


# Import plugin modules after defining Engine so their decorators register
# each Entry subclass. Add new plugin imports here.

from phyai.models.pi0 import main_pi0 as _main_pi0  # noqa: E402, F401
from phyai.models.pi05 import main_pi05 as _main_pi05  # noqa: E402, F401
from phyai.models.cosmos3 import main_cosmos3 as _main_cosmos3  # noqa: E402, F401
from phyai.models.cosmos3 import (  # noqa: E402, F401
    main_cosmos3_policy as _main_cosmos3_policy,
)
from phyai.models.gr00t_n17 import main_gr00t_n17 as _main_gr00t_n17  # noqa: E402, F401
from phyai.models.minicpm_gr00t import (  # noqa: E402, F401
    main_minicpm_gr00t as _main_minicpm_gr00t,
)
