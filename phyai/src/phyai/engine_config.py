"""Process-level configuration for engine defaults and runtime settings."""

from __future__ import annotations

import re
from threading import Lock
from dataclasses import dataclass, field, replace

import torch

from phyai.env import envs, removed_env_vars_in_use
from phyai.kernel.config import KernelConfig
from phyai.parallel.config import (
    AttentionParallelConfig,
    DenseParallelConfig,
    MoeParallelConfig,
    OuterParallelConfig,
    ParallelConfig,
    ResolvedParallelConfig,
)


def _canonical_backend_name(name: str) -> str:
    return name.lower().replace("_", "-")


# ---------------------------------------------------------------------- #
# Sub-configs                                                            #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackendConfig:
    """Configuration for the vGPU backend.

    ``None`` lets the vGPU layer choose a backend automatically.
    """

    vgpu: str | None = None

    def __post_init__(self) -> None:
        self._validate_vgpu()

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
    """Device target and default parameter dtype.

    Fields
    ------
    target:
        A value accepted by :func:`torch.device`.
    params_dtype:
        Dtype used for newly allocated parameters when a layer does not
        specify one.
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
class RuntimeConfig:
    """Runtime switches and process-level tunables.

    Fields
    ------
    use_cuda_graph:
        Whether to enable CUDA graph capture.
    freeze_kernel_choices:
        Whether to freeze kernel selections after warmup.
    flashinfer_workspace_bytes:
        Size in bytes of the FlashInfer workspace.
    debug_tensor_dump_dir:
        Directory for activation dumps. ``None`` disables dumping.
    debug_tensor_dump_filter:
        Regex patterns for selecting operators to dump.
    debug_tensor_dump_filter_fn:
        Path to a custom operator-selection function.
    dist_timeout_s:
        Distributed process-group timeout in seconds.
    num_threads:
        Torch intra-op thread count. ``None`` uses the default policy.
    seed:
        Optional process-level random seed.
    """

    use_cuda_graph: bool = True
    freeze_kernel_choices: bool = False
    flashinfer_workspace_bytes: int = 128 * 1024 * 1024
    debug_tensor_dump_dir: str | None = None
    debug_tensor_dump_filter: tuple[str, ...] | None = None
    debug_tensor_dump_filter_fn: str | None = None
    dist_timeout_s: int = 1800
    num_threads: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        v = self.flashinfer_workspace_bytes
        if not isinstance(v, int) or v <= 0:
            raise ValueError(
                f"RuntimeConfig.flashinfer_workspace_bytes must be a "
                f"positive int (bytes), got {v!r}."
            )
        if not isinstance(self.dist_timeout_s, int) or self.dist_timeout_s <= 0:
            raise ValueError(
                f"RuntimeConfig.dist_timeout_s must be a positive int "
                f"(seconds), got {self.dist_timeout_s!r}."
            )
        nt = self.num_threads
        if nt is not None and (not isinstance(nt, int) or nt < 1):
            raise ValueError(
                f"RuntimeConfig.num_threads must be None or an int >= 1, got {nt!r}."
            )
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(
                f"RuntimeConfig.seed must be None or an int, got {self.seed!r}."
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


# ---------------------------------------------------------------------- #
# Root EngineConfig                                                      #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngineConfig:
    """Process-level configuration composed of the engine sub-configs."""

    backends: BackendConfig = field(default_factory=BackendConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # Keep the kernel policy after the process-level fields.
    kernel: KernelConfig = field(default_factory=KernelConfig)

    @classmethod
    def auto(cls) -> "EngineConfig":
        """Build defaults based on CUDA availability."""
        cuda = torch.cuda.is_available()
        return cls(
            backends=BackendConfig(vgpu=None),
            # Let the kernel policy choose the profile when one is configured.
            kernel=KernelConfig(),
            device=DeviceConfig(
                target="cuda" if cuda else "cpu",
                params_dtype=torch.bfloat16 if cuda else torch.float32,
            ),
            parallel=ParallelConfig(),
            runtime=RuntimeConfig(use_cuda_graph=cuda),
        )

    @classmethod
    def from_env(cls, base: "EngineConfig | None" = None) -> "EngineConfig":
        """Build a config from ``base`` or auto defaults and apply ``PHYAI_*``
        overrides.
        """
        stale = removed_env_vars_in_use()
        if stale:
            lines = "\n".join(
                f"  {name}: use {advice}" for name, advice in stale.items()
            )
            raise ValueError(f"these environment variables no longer exist:\n{lines}")
        if base is None:
            base = cls.auto()

        backends_kw: dict[str, object] = {}
        if (v := envs.PHYAI_VGPU_BACKEND.get()) is not None:
            backends_kw["vgpu"] = v

        device_kw: dict[str, object] = {}
        if (v := envs.PHYAI_DEVICE.get()) is not None:
            device_kw["target"] = v
        if (v := envs.PHYAI_PARAMS_DTYPE.get()) is not None:
            device_kw["params_dtype"] = v

        runtime_kw: dict[str, object] = {}
        if (v := envs.PHYAI_USE_CUDA_GRAPH.get()) is not None:
            runtime_kw["use_cuda_graph"] = v
        if (v := envs.PHYAI_FREEZE_KERNEL_CHOICES.get()) is not None:
            runtime_kw["freeze_kernel_choices"] = v
        if (v := envs.PHYAI_FLASHINFER_WORKSPACE_BYTES.get()) is not None:
            runtime_kw["flashinfer_workspace_bytes"] = v
        if (v := envs.PHYAI_DEBUG_TENSOR_DUMP_DIR.get()) is not None:
            runtime_kw["debug_tensor_dump_dir"] = v
        if (v := envs.PHYAI_DEBUG_TENSOR_DUMP_FILTER.get()) is not None:
            runtime_kw["debug_tensor_dump_filter"] = v
        if (v := envs.PHYAI_DEBUG_TENSOR_DUMP_FILTER_FN.get()) is not None:
            runtime_kw["debug_tensor_dump_filter_fn"] = v
        if (v := envs.PHYAI_DIST_TIMEOUT_S.get()) is not None:
            runtime_kw["dist_timeout_s"] = v
        if (v := envs.PHYAI_NUM_THREADS.get()) is not None:
            runtime_kw["num_threads"] = v
        if (v := envs.PHYAI_SEED.get()) is not None:
            runtime_kw["seed"] = v

        kernel_kw: dict[str, object] = {}
        if (v := envs.PHYAI_KERNEL_CONFIG.get()) is not None:
            kernel_kw["config_path"] = v
            kernel_kw["policy_config"] = None
        if (v := envs.PHYAI_KERNEL_PROFILE.get()) is not None:
            kernel_kw["profile"] = v
        if (v := envs.PHYAI_KERNEL_AUTOTUNE_CACHE.get()) is not None:
            kernel_kw["autotune_cache"] = v

        return cls(
            backends=replace(base.backends, **backends_kw)
            if backends_kw
            else base.backends,
            kernel=replace(base.kernel, **kernel_kw) if kernel_kw else base.kernel,
            device=replace(base.device, **device_kw) if device_kw else base.device,
            parallel=base.parallel,
            runtime=replace(base.runtime, **runtime_kw) if runtime_kw else base.runtime,
        )

    def replace(
        self,
        *,
        backends: BackendConfig | None = None,
        kernel: KernelConfig | None = None,
        device: DeviceConfig | None = None,
        parallel: ParallelConfig | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> "EngineConfig":
        """Return a copy with the selected sub-configurations replaced."""
        return replace(
            self,
            backends=backends if backends is not None else self.backends,
            kernel=kernel if kernel is not None else self.kernel,
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
    """Return the process-level config, initializing it on first access."""
    global _config
    if _config is None:
        with _lock:
            if _config is None:
                _config = EngineConfig.from_env()
    return _config


def set_engine_config(cfg: EngineConfig) -> None:
    """Replace the process-level config."""
    global _config
    with _lock:
        _config = cfg


def init_engine_config(cfg: EngineConfig) -> EngineConfig:
    """Install and return the process-level config."""
    set_engine_config(cfg)
    return cfg


def resolve_params_dtype(params_dtype: "torch.dtype | None") -> "torch.dtype":
    """Return ``params_dtype`` when set; otherwise use the engine default."""

    if params_dtype is not None:
        return params_dtype
    return get_engine_config().device.params_dtype


__all__ = [
    "BackendConfig",
    "DeviceConfig",
    "AttentionParallelConfig",
    "DenseParallelConfig",
    "EngineConfig",
    "KernelConfig",
    "MoeParallelConfig",
    "OuterParallelConfig",
    "ParallelConfig",
    "ResolvedParallelConfig",
    "RuntimeConfig",
    "get_engine_config",
    "init_engine_config",
    "resolve_params_dtype",
    "set_engine_config",
]
