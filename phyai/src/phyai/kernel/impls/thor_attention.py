"""Explicit Thor compact attention backend; unsupported contracts use FlashInfer."""

from __future__ import annotations

import functools
import hashlib
import math
import threading
from collections import OrderedDict, Counter
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from phyai.utils import get_logger
from phyai.layers.attention.enums import AttnLayout, AttnMode
from phyai.layers.attention.nocache.backends.flashinfer import (
    FlashInferAttentionBackend,
)

logger = get_logger(__name__)


def unsupported(layer, q, k, v, ctx):
    if ctx is not None and ctx.attn_bias is not None:
        return "attention_bias"
    if not all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in (q, k, v)):
        return "rank"
    if not all(x.is_cuda and x.dtype == torch.bfloat16 for x in (q, k, v)):
        return "device_or_dtype"
    if q.device != k.device or q.device != v.device:
        return "device"
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        return "batch"
    if q.shape[-2:] != (32, 128) or k.shape[-2:] != (8, 128) or k.shape != v.shape:
        return "geometry"
    if min(q.shape[1], k.shape[1]) <= 0:
        return "empty"
    if any(
        x.stride(-1) != 1 or any(s <= 0 for s in x.stride()) or x.data_ptr() % 16
        for x in (q, k, v)
    ):
        return "layout"
    if torch.cuda.is_current_stream_capturing():
        return "capture"
    if (
        layer.causal
        or layer.sliding_window is not None
        or layer.logits_soft_cap is not None
        or layer.scale not in (128**-0.5, 1.0 / math.sqrt(128))
    ):
        return "semantics"
    if (
        ctx.mode != AttnMode.PREFILL
        or ctx.layout != AttnLayout.PADDED_4D
        or ctx.mask is not None
    ):
        return "context"
    return None


def verified_versions():
    try:
        return version("flashinfer-python") == "0.6.17" and str(
            torch.version.cuda
        ).startswith("13.0")
    except PackageNotFoundError:
        return False


@functools.lru_cache(maxsize=1)
def extension():
    import phyai_kernel
    import tvm_ffi.cpp
    from flashinfer.jit import env
    from phyai_kernel.jit_utils.cutlass import CUTLASS_REVISION, cutlass_include_dirs

    root = Path(phyai_kernel.__file__).resolve().parent / "cuda/csrc"
    sources = [
        root / "compact_attention.cu",
        *sorted((root / "compact_attention").rglob("*.cuh")),
        *sorted((root / "compact_attention").rglob("*.hpp")),
    ]
    digest = hashlib.sha256(
        b"".join(path.read_bytes() for path in sources)
        + CUTLASS_REVISION.encode()
        + version("flashinfer-python").encode()
        + str(torch.version.cuda).encode()
    ).hexdigest()[:16]
    includes = [
        root,
        *env.CCCL_INCLUDE_DIRS,
        *cutlass_include_dirs(),
        env.FLASHINFER_INCLUDE_DIR,
        env.FLASHINFER_CSRC_DIR,
        env.FLASHINFER_INCLUDE_DIR / "flashinfer/attention/blackwell",
        env.FLASHINFER_INCLUDE_DIR / "flashinfer/attention/blackwell/collective",
    ]
    from phyai_kernel.jit_utils.compile import PHYAI_KERNEL_CACHE_ROOT

    return tvm_ffi.cpp.load(
        name=f"phyai_thor_compact_attention_{digest}",
        build_directory=str(
            PHYAI_KERNEL_CACHE_ROOT / "thor_compact_attention" / digest
        ),
        sources=[str(root / "compact_attention.cu")],
        extra_include_paths=[str(p) for p in includes],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-arch=sm_110a",
            "-use_fast_math",
            "--expt-relaxed-constexpr",
            "-DFLASHINFER_ENABLE_BF16",
            "-DFLASHINFER_ENABLE_F16",
        ],
    )


@dataclass
class Plan:
    wrapper: object
    workspace: torch.Tensor
    qo: torch.Tensor
    kv: torch.Tensor
    done: object
    info: tuple


class CompactAttentionBackend(FlashInferAttentionBackend):
    """Runner contexts may share this backend across layers of identical geometry."""

    name = "thor_compact"
    share_across_layers = True

    def __init__(self, runner=None, *, max_plans=4, workspace_bytes=128 * 1024 * 1024):
        if max_plans < 1 or workspace_bytes < 1:
            raise ValueError("max_plans and workspace_bytes must be positive")
        # Unsupported contracts retain FlashInfer's default backend selection.
        # The compact fast path creates its own Cutlass wrapper below.
        super().__init__(runner, workspace_bytes=workspace_bytes)
        self.max_plans = max_plans
        self.workspace_bytes = workspace_bytes
        self.plans = OrderedDict()
        self.stats = Counter()
        self.lock = threading.RLock()
        self.abi_valid = verified_versions()

    @property
    def sharing_key(self):
        return (self.max_plans, self.workspace_bytes, self._prefill_backend)

    def release(self):
        with self.lock:
            for plan in self.plans.values():
                plan.done.synchronize()
            self.plans.clear()

    @staticmethod
    def plan_info(wrapper, device):
        info = getattr(wrapper, "_plan_info", None)
        if not isinstance(info, (tuple, list)) or len(info) != 4:
            return None
        if not all(
            isinstance(x, torch.Tensor)
            and x.device == device
            and x.dtype == torch.int32
            and x.ndim == 1
            and x.is_contiguous()
            for x in info
        ):
            return None
        return tuple(info)

    def forward(self, layer, q, k, v, ctx):
        reason = unsupported(layer, q, k, v, ctx)
        if reason is None and not self.abi_valid:
            reason = "version_or_plan_abi"
        if reason is not None:
            self.stats["fallback:" + reason] += 1
            return super().forward(layer, q, k, v, ctx)
        stream = torch.cuda.current_stream(q.device)
        key = (
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            tuple(q.stride()),
            tuple(k.stride()),
            tuple(v.stride()),
            q.device,
            q.dtype,
            layer.scale,
            stream.cuda_stream,
        )
        with self.lock:
            plan = self.plans.get(key)
            if plan is None:
                if len(self.plans) >= self.max_plans:
                    _, retired = self.plans.popitem(last=False)
                    retired.done.synchronize()
                import flashinfer

                workspace = torch.empty(
                    self.workspace_bytes, device=q.device, dtype=torch.uint8
                )
                wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace, kv_layout="NHD", backend="cutlass"
                )
                qo = torch.tensor([0, q.shape[1]], device=q.device, dtype=torch.int32)
                kv = torch.tensor([0, k.shape[1]], device=q.device, dtype=torch.int32)
                wrapper.plan(
                    qo,
                    kv,
                    32,
                    8,
                    128,
                    causal=False,
                    sm_scale=layer.scale,
                    q_data_type=q.dtype,
                    kv_data_type=k.dtype,
                )
                info = self.plan_info(wrapper, q.device)
                if info is None:
                    self.abi_valid = False
                    logger.warning_once(
                        "Thor compact attention plan ABI is unsupported; using FlashInfer."
                    )
                    return super().forward(layer, q, k, v, ctx)
                plan = Plan(wrapper, workspace, qo, kv, torch.cuda.Event(), info)
                self.plans[key] = plan
            self.plans.move_to_end(key)
            values = torch.ones((*v.shape[1:-1], 192), device=v.device, dtype=v.dtype)
            values[..., :128].copy_(v[0])
            output = torch.empty(q.shape[1:], device=q.device, dtype=q.dtype)
            try:
                extension().run(
                    plan.workspace,
                    q[0],
                    k[0],
                    values,
                    plan.qo,
                    plan.kv,
                    *plan.info,
                    output,
                    None,
                    0,
                    layer.scale,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    q.shape[1],
                )
                self.stats["dispatch"] += 1
                return output.unsqueeze(0)
            finally:
                # Keep workspace and plan tensors alive through every enqueued use.
                plan.done.record(stream)
