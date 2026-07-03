"""Process-level engine that drives a single registered model plugin.

The :class:`Engine` is a thin programmatic dispatcher; it does *not*
parse argv or read environment variables. A future server (HTTP /
gRPC / RPC / ...) is expected to wrap this engine, translate external
requests into a model-specific request object, and call
:meth:`Engine.step`. Until that server lands, the intended caller is
in-process Python (tests, scripts, notebooks).

Three concepts:

* :class:`EntryArgs` — empty dataclass base. Each plugin declares a
  subclass with its own fields (e.g. ``PI05Args(checkpoint_dir=...)``).
  EntryArgs are programmatic: typed, defaulted, IDE-friendly. No
  argparse, no string parsing.
* :class:`Entry` — ABC every plugin inherits. Defines :meth:`setup`
  (build model + scheduler from args), :meth:`step` (one inference
  round; request / response shapes are plugin-defined), and
  :meth:`close` (release GPU resources).
* :class:`Engine` — owns the plugin registry, applies
  :class:`~phyai.engine_config.EngineConfig` so every downstream model
  constructor picks up the same defaults, and orchestrates a fixed
  sequence of discrete ``init_*`` functions
  (:func:`~phyai.engine_config.init_engine_config`,
  :func:`~phyai.utils.cuda.init_cuda`,
  :func:`~phyai.utils.cuda.init_cublas`,
  :func:`~phyai.parallel.dist.init_dist`,
  :func:`phyai.parallel.init`,
  :func:`phyai.layers.linear.init`) in :meth:`__init__`. Each ``init_*``
  is independently callable so tests / advanced users can opt into
  pieces without committing to the full sequence.

Plugin discovery is explicit: this module imports each plugin's
``main_*`` module at the bottom of the file, which in turn invokes
``@Engine.register`` at class-definition time. To add a new model,
add one import line at the bottom of this file.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch
import torch.distributed as dist
from torch import nn

import phyai.layers.linear as L
import phyai.parallel as P
from phyai.engine_config import EngineConfig, init_engine_config
from phyai.parallel.dist import init_dist
from phyai.runtime.tensor_dump import (
    TensorDumper,
    load_filter_fn,
    register_tensor_dumper,
)
from phyai.utils import this_rank_log
from phyai.utils.cuda import init_cublas, init_cuda


logger = logging.getLogger(__name__)


def _force_eager_for_dump(cfg: EngineConfig) -> EngineConfig:
    """Return ``cfg`` with ``use_cuda_graph`` forced off for tensor dumping.

    The forward-hook tensor dumper records module outputs from Python
    callbacks, which never run during a captured CUDA-graph replay. So
    activation capture is only possible in eager mode — when a dump
    directory is configured the engine routes the config through here
    before building any runner. Returns ``cfg`` unchanged when it is
    already eager (so the caller can detect whether a flip happened by
    identity).
    """
    if not cfg.runtime.use_cuda_graph:
        return cfg
    return cfg.replace(runtime=replace(cfg.runtime, use_cuda_graph=False))


@dataclass
class EntryArgs:
    """Base for every plugin's args dataclass.

    Empty on purpose — every field is plugin-specific, declared by the
    concrete subclass (``PI05Args``, future ``GR00TArgs``, ...). The
    base exists so :class:`EngineArgs.plugin_args` has a meaningful
    type annotation and ``isinstance`` checks compose cleanly.
    """


class Entry(abc.ABC):
    """Per-model plugin: build, run, release.

    A subclass must declare two class-level attributes:

    * ``name`` — short ASCII id; used for :attr:`EngineArgs.plugin`
      lookups and diagnostic strings.
    * ``args_cls`` — the concrete :class:`EntryArgs` subclass this
      entry expects in :meth:`setup`. The engine validates the runtime
      type before dispatching.

    Lifecycle: one :meth:`setup` per engine construction, then
    arbitrarily many :meth:`step` calls, then one :meth:`close`. The
    engine holds the entry instance for as long as it lives.
    """

    name: ClassVar[str]
    args_cls: ClassVar[type[EntryArgs]]

    @abc.abstractmethod
    def setup(self, args: EntryArgs) -> None:
        """Build the model, load weights, prepare runners / scheduler."""

    @abc.abstractmethod
    def step(self, request: Any) -> Any:
        """Run one inference round. Request / response shape is plugin-defined."""

    def close(self) -> None:
        """Release pinned GPU resources. Default: no-op."""
        return None

    def dump_targets(self) -> dict[str, nn.Module]:
        """Modules to attach the debug tensor dumper to.

        Returns a mapping of *root name* to ``nn.Module``; the engine
        registers forward hooks on every leaf submodule of each, so each
        :meth:`step` records that module's activations to disk (see
        :class:`~phyai.runtime.tensor_dump.TensorDumper`). The root name
        prefixes every recorded operator key
        (``{"model": m}`` -> ``model.<...>.o_proj``).

        Default: ``{}`` — the plugin opts out of tensor dumping and the
        engine simply never builds a dumper. Plugins that want it override
        this to expose their top-level model module(s). Only consulted
        when :attr:`RuntimeConfig.debug_tensor_dump_dir` is set; returning
        an empty mapping while a dump dir is configured is reported once
        as a warning so a mis-wired plugin doesn't fail silently.
        """
        return {}


@dataclass
class EngineArgs:
    """Plugin selection + optional engine config override.

    Mandatory fields: which plugin to run and the typed arg bundle the
    plugin's :meth:`Entry.setup` consumes. Engine-wide defaults
    (device, dtype, backends, parallelism, runtime knobs) live on
    :class:`~phyai.engine_config.EngineConfig`.

    ``config`` is used as the *base* the ``PHYAI_*`` env vars overlay on
    top of: the engine resolves the effective config via
    :meth:`EngineConfig.from_env(base=config) <EngineConfig.from_env>`,
    so any set env var overrides the matching field while every unset
    field carries over from ``config`` verbatim. Leave ``config`` as
    ``None`` to start from :meth:`EngineConfig.auto` (host-appropriate
    defaults) before the same env overlay. This means an env var is
    always honoured — even when an explicit ``config`` is supplied —
    which is what makes ``PHYAI_*`` usable as a run-time toggle (e.g.
    flipping on tensor dump for one run without editing the caller).
    """

    plugin: str
    plugin_args: EntryArgs
    config: EngineConfig | None = None


class Engine:
    """Thin in-process dispatcher around one registered :class:`Entry`."""

    _plugins: ClassVar[dict[str, type[Entry]]] = {}

    @classmethod
    def register(cls, entry_cls: type[Entry]) -> type[Entry]:
        """Register a plugin entry class. Use as a decorator at class definition.

        ::

            @Engine.register
            class PI05Entry(Entry):
                name = "pi05"
                args_cls = PI05Args
                ...
        """
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
        cls._plugins[name] = entry_cls
        return entry_cls

    @classmethod
    def registered(cls) -> tuple[str, ...]:
        """Return all registered plugin names in registration order."""
        return tuple(cls._plugins.keys())

    def __init__(self, args: EngineArgs) -> None:
        # 1. Resolve EngineConfig and seed the process singleton so every
        #    model constructor downstream picks up the requested
        #    device / dtype / backends without explicit plumbing.
        #    ``from_env(base=args.config)`` makes the explicit config the
        #    base and overlays any set ``PHYAI_*`` env var on top, so a
        #    var like PHYAI_DEBUG_TENSOR_DUMP_DIR is honoured even when the
        #    caller passed a config (env = run-time toggle). ``args.config``
        #    is None -> the base falls back to ``auto()`` inside from_env.
        #    When a tensor-dump directory ends up set, force eager *before*
        #    installing the singleton: the dumper's forward hooks can't
        #    fire inside a captured CUDA-graph replay, and the pi05
        #    scheduler reads ``use_cuda_graph`` off this singleton at
        #    setup time, so the flip has to land before any runner builds.
        resolved = EngineConfig.from_env(base=args.config)
        self._dump_enabled = resolved.runtime.debug_tensor_dump_dir is not None
        if self._dump_enabled:
            forced = _force_eager_for_dump(resolved)
            if forced is not resolved:
                this_rank_log(
                    logger,
                    logging.WARNING,
                    "Tensor dump enabled (debug_tensor_dump_dir=%s): forcing "
                    "use_cuda_graph=False. Forward hooks cannot fire during a "
                    "captured CUDA-graph replay, so activation capture runs "
                    "eager-only (slower than the normal graph path).",
                    resolved.runtime.debug_tensor_dump_dir,
                )
            resolved = forced
        self.config: EngineConfig = init_engine_config(resolved)
        self._dumper: TensorDumper | None = None

        device_type = torch.device(self.config.device.target).type

        # 2. Per-concern bootstrap. Each ``init_*`` is independently
        #    callable; this method is the only orchestrator. Saved
        #    default dtype is restored in :meth:`close`.
        self._saved_default_dtype: torch.dtype = init_cuda(
            self.config.device.target, self.config.device.params_dtype
        )
        init_cublas()
        parallel = self.config.parallel
        self._owns_pg: bool = init_dist(
            world_size=parallel.world_size, device_type=device_type
        )

        # 3. phyai mesh + linear dispatcher. Both are process-level
        #    singletons; building them here means model constructors
        #    don't have to. The mesh is always 5-axis
        #    (dp / ep / sp / cp / tp); axes the user didn't size stay at
        #    ``1`` and short-circuit through the collective ops without
        #    any process-group traffic, while existing model code that
        #    addresses ``axis="tp"`` keeps working unchanged.
        P.init(
            layout=(
                parallel.dp_size,
                parallel.ep_size,
                parallel.sp_size,
                parallel.cp_size,
                parallel.tp_size,
            ),
            mesh_dim_names=("dp", "ep", "sp", "cp", "tp"),
            device=device_type,
        )
        L.init()

        # 4. Resolve the requested plugin and validate the args bundle.
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

        # 5. Instantiate the entry and run setup. The entry owns its
        #    model / scheduler / runners from this point on.
        self.args = args
        self.entry: Entry = entry_cls()
        self.entry.setup(args.plugin_args)

        # 6. If tensor dumping is on, attach the dumper to the modules the
        #    plugin exposes. Built after setup() so the model + weights are
        #    fully constructed; the runners were already forced eager in
        #    step 1, so the leaf forward hooks will actually fire.
        if self._dump_enabled:
            self._dumper = self._build_dumper()

    def _build_dumper(self) -> TensorDumper | None:
        """Construct the tensor dumper from the entry's dump targets.

        Returns ``None`` (and warns) when the plugin exposes no dump
        targets, so a mis-wired plugin surfaces loudly instead of
        silently recording nothing.
        """
        runtime = self.config.runtime
        targets = self.entry.dump_targets()
        if not targets:
            this_rank_log(
                logger,
                logging.WARNING,
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
        """Turn the two runtime dump-filter knobs into a single filter spec.

        ``debug_tensor_dump_filter_fn`` (a ``"module:func"`` path) wins and
        is resolved to a callable; otherwise the regex tuple
        ``debug_tensor_dump_filter`` (or ``None`` -> record everything) is
        passed through. The two are already validated mutually exclusive on
        :class:`~phyai.engine_config.RuntimeConfig`.
        """
        runtime = self.config.runtime
        if runtime.debug_tensor_dump_filter_fn is not None:
            return load_filter_fn(runtime.debug_tensor_dump_filter_fn)
        return runtime.debug_tensor_dump_filter

    def step(self, request: Any) -> Any:
        """Run one inference round; forwards to the registered entry.

        When tensor dumping is active, the activations recorded during
        this round are flushed to a single ``pass{N}.pt`` file once the
        entry returns.
        """
        result = self.entry.step(request)
        if self._dumper is not None:
            self._dumper.flush_pass()
        return result

    def close(self) -> None:
        """Release the plugin entry's resources, then tear down distributed
        state if the engine was the one to bring it up."""
        if self._dumper is not None:
            self._dumper.detach()
            self._dumper = None
        self.entry.close()
        if self._owns_pg and dist.is_initialized():
            dist.destroy_process_group()
            self._owns_pg = False
        torch.set_default_dtype(self._saved_default_dtype)


__all__ = [
    "EngineArgs",
    "Engine",
    "Entry",
    "EntryArgs",
]


# ---------------------------------------------------------------------- #
# Plugin discovery — explicit imports at module bottom.                  #
#                                                                        #
# Each ``main_*`` module decorates its Entry subclass with               #
# ``@Engine.register``; importing the module triggers registration.      #
# Add one import line per new model. Imports go at the bottom because    #
# plugin modules ``from phyai.engine import Engine, Entry, EntryArgs``;  #
# this module's symbols must be defined first.                           #
# ---------------------------------------------------------------------- #

from phyai.models.pi05 import main_pi05 as _main_pi05  # noqa: E402, F401
from phyai.models.gr00t_n17 import main_gr00t_n17 as _main_gr00t_n17  # noqa: E402, F401
from phyai.models.cosmos3 import main_cosmos3 as _main_cosmos3  # noqa: E402, F401
from phyai.models.cosmos3 import (  # noqa: E402, F401
    main_cosmos3_policy as _main_cosmos3_policy,
)
