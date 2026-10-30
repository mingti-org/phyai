"""Typed registry for ``PHYAI_*`` environment variables."""

from __future__ import annotations

import os
from typing import Generic, TypeVar, Callable

T = TypeVar("T")


class EnvField(Generic[T]):
    """Typed environment-variable field with an optional default."""

    __slots__ = ("name", "default", "parser")

    def __init__(
        self,
        name: str,
        default: T | None,
        parser: Callable[[str], T],
    ) -> None:
        self.name = name
        self.default = default
        self.parser = parser

    def is_set(self) -> bool:
        """``True`` if the env var is present (even if empty)."""
        return self.name in os.environ

    def get(self) -> T | None:
        raw = os.environ.get(self.name)
        if raw is None:
            return self.default
        try:
            return self.parser(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{self.name}={raw!r}: {e}") from e


def _parse_bool(s: str) -> bool:
    """Parse a boolean environment value."""
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean (1/0/true/false/yes/no/on/off), got {s!r}")


def _parse_dtype(s: str):
    """Parse a PyTorch dtype name."""
    import torch

    table: dict[str, "torch.dtype"] = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "fp64": torch.float64,
        "float64": torch.float64,
        "double": torch.float64,
    }
    key = s.strip().lower()
    if key not in table:
        raise ValueError(
            f"expected one of {sorted(table)} (case-insensitive), got {s!r}"
        )
    return table[key]


def _parse_regex_list(s: str) -> tuple[str, ...]:
    """Parse a JSON regex list or a single bare pattern."""
    import json

    raw = s.strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return (raw,)
    if isinstance(parsed, str):
        return (parsed,)
    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        return tuple(parsed)
    raise ValueError(
        f"expected a JSON array of regex strings (or a single pattern), got {s!r}"
    )


class envs:
    """Process-level typed environment-variable registry."""

    # Backend and kernel selection.
    PHYAI_VGPU_BACKEND = EnvField("PHYAI_VGPU_BACKEND", None, str)
    # Prefer one collective (communication) backend by name: "nccl", "pynccl"
    # or "gloo". It wins whenever it can serve the call and falls back
    # otherwise (pynccl declines eager mode). Distinct from the vGPU backend
    # and from kernel policies.
    PHYAI_FORCE_COLLECTIVE_BACKEND = EnvField(
        "PHYAI_FORCE_COLLECTIVE_BACKEND", None, str
    )
    # Explicit libnccl path for the ctypes pynccl backend.
    PHYAI_NCCL_SO_PATH = EnvField("PHYAI_NCCL_SO_PATH", None, str)
    PHYAI_KERNEL_CONFIG = EnvField("PHYAI_KERNEL_CONFIG", None, str)
    PHYAI_KERNEL_PROFILE = EnvField("PHYAI_KERNEL_PROFILE", None, str)
    PHYAI_KERNEL_AUTOTUNE_CACHE = EnvField("PHYAI_KERNEL_AUTOTUNE_CACHE", None, str)
    # Recheck frozen kernel selections on every call.
    PHYAI_KERNEL_VERIFY_FROZEN = EnvField(
        "PHYAI_KERNEL_VERIFY_FROZEN", None, _parse_bool
    )

    # Device and dtype.
    PHYAI_DEVICE = EnvField("PHYAI_DEVICE", None, str)
    PHYAI_PARAMS_DTYPE = EnvField("PHYAI_PARAMS_DTYPE", None, _parse_dtype)

    # Runtime settings.
    PHYAI_USE_CUDA_GRAPH = EnvField("PHYAI_USE_CUDA_GRAPH", None, _parse_bool)
    # Experimental Pi0.5 dataflow optimizations. Unsupported instances stay native.
    PHYAI_PI05_THOR_OPTIMIZATIONS = EnvField(
        "PHYAI_PI05_THOR_OPTIMIZATIONS", False, _parse_bool
    )
    PHYAI_FREEZE_KERNEL_CHOICES = EnvField(
        "PHYAI_FREEZE_KERNEL_CHOICES", None, _parse_bool
    )
    # Distributed process-group timeout in seconds.
    PHYAI_DIST_TIMEOUT_S = EnvField("PHYAI_DIST_TIMEOUT_S", None, int)
    # Torch intra-op thread count.
    PHYAI_NUM_THREADS = EnvField("PHYAI_NUM_THREADS", None, int)
    # Global Python, NumPy, and Torch RNG seed.
    PHYAI_SEED = EnvField("PHYAI_SEED", None, int)

    # Bootstrap and diagnostics.
    # Default log-handler level.
    PHYAI_LOG_LEVEL = EnvField("PHYAI_LOG_LEVEL", None, str)
    # Skip PhyAI environment tuning.
    PHYAI_SKIP_ENV_SETUP = EnvField("PHYAI_SKIP_ENV_SETUP", None, _parse_bool)

    # Low-level tuning.
    PHYAI_FLASHINFER_WORKSPACE_BYTES = EnvField(
        "PHYAI_FLASHINFER_WORKSPACE_BYTES", None, int
    )

    # Tensor-dump output and operator filters.
    PHYAI_DEBUG_TENSOR_DUMP_DIR = EnvField("PHYAI_DEBUG_TENSOR_DUMP_DIR", None, str)
    PHYAI_DEBUG_TENSOR_DUMP_FILTER = EnvField(
        "PHYAI_DEBUG_TENSOR_DUMP_FILTER", None, _parse_regex_list
    )
    PHYAI_DEBUG_TENSOR_DUMP_FILTER_FN = EnvField(
        "PHYAI_DEBUG_TENSOR_DUMP_FILTER_FN", None, str
    )


#: Removed variables are rejected instead of silently changing topology.
REMOVED_ENV_VARS: dict[str, str] = {
    "PHYAI_REPLICA_COUNT": "DeploymentConfig(replica_count=...)",
    "PHYAI_CFG_SIZE": "EngineConfig.parallel.outer.cfg_size",
    "PHYAI_TP_SIZE": "EngineConfig.parallel.dense.tp_size",
    "PHYAI_EP_SIZE": "EngineConfig.parallel.moe.ep_size",
    "PHYAI_SP_SIZE": "EngineConfig.parallel.dense.sequence_parallel",
    "PHYAI_CP_SIZE": "EngineConfig.parallel.attention.cp_size",
    "PHYAI_DP_SIZE": "DeploymentConfig(replica_count=...)",
    "PHYAI_WORLD_SIZE": (
        "the launcher or local deployment determines the replica world size"
    ),
    "PHYAI_ATTN_BACKEND": (
        "a kernel policy rule -- e.g. "
        "`rules: [{match: {op: attention}, restrict_to: 'sdpa.attention'}]` "
        "in the YAML named by PHYAI_KERNEL_CONFIG -- or a layer's backend= "
        "argument. A policy rule can scope itself to one op and one role; this "
        "variable could not."
    ),
    "PHYAI_NORM_BACKEND": (
        "a kernel policy rule -- e.g. "
        "`rules: [{match: {op: rmsnorm}, restrict_to: 'phyai_kernel.*'}]` -- "
        "or a layer's backend= argument"
    ),
    "PHYAI_LINEAR_BACKEND": (
        "a kernel policy rule -- e.g. "
        "`rules: [{match: {op: gemm, role: mlp.down}, restrict_to: 'torch.gemm.*'}]`. "
        "This variable never had any effect at all: it was validated against the "
        "catalog but nothing read it."
    ),
    "PHYAI_FLASHINFER_PREFILL_BACKEND": (
        "a kernel policy rule naming the row -- e.g. "
        "`rules: [{match: {op: attention_paged}, restrict_to: "
        "'flashinfer.attention_paged.fa2'}]`. Each FlashInfer prefill "
        "kernel is now its own catalog row, so the choice can be scoped to one "
        "op or role and is capability-gated (FA3 is Hopper-only, trtllm-gen and "
        "cutlass are Blackwell-only). This variable was one name for every "
        "attention site in the process, and its single valid-name set was "
        "written for the paged wrapper while the no-cache stack uses the ragged "
        "one, which accepts a different set"
    ),
    "PHYAI_FORCE_BACKEND": (
        "PHYAI_FORCE_COLLECTIVE_BACKEND with the same value (nccl / pynccl / "
        "gloo); the old name did not say which backend it forced"
    ),
    "PHYAI_REPLICA_ID": (
        "the WorkerPlacement handed to the worker factory (replica_id); the "
        "supervisor no longer mirrors placement into the environment"
    ),
    "PHYAI_WORKER_ID": (
        "the WorkerPlacement handed to the worker factory (worker_id); the "
        "supervisor no longer mirrors placement into the environment"
    ),
    "PHYAI_REPLICA_NODE_COUNT": (
        "nothing: placement is probed at runtime by phyai.parallel.init (node "
        "identity gathered over the process group, NVLink checked with NVML)"
    ),
    "PHYAI_REPLICA_GPUS_PER_NODE": (
        "nothing: placement is probed at runtime by phyai.parallel.init (node "
        "identity gathered over the process group, NVLink checked with NVML)"
    ),
    "PHYAI_FORCE_LINEAR_KERNEL": (
        "a kernel policy rule -- e.g. "
        "`rules: [{match: {op: gemm}, restrict_to: 'torch.gemm.*'}]`, or add a "
        "`role:` to the match to change only one layer group"
    ),
}


def removed_env_vars_in_use() -> dict[str, str]:
    """Return removed environment variables currently in use."""

    import os

    return {
        name: advice
        for name, advice in REMOVED_ENV_VARS.items()
        if os.environ.get(name)
    }


__all__ = ["EnvField", "envs"]
