"""phyai.layers.linear — parallel linear layers with declarative kernel dispatch.

Quick start::

    import torch.distributed as dist
    import phyai.parallel as P
    import phyai.layers.linear as L

    dist.init_process_group("nccl")
    P.init(layout=(8,), mesh_dim_names=("tp",))
    L.init()

    qkv = L.QKVParallelLinear(
        hidden_size=4096, head_dim=128, num_heads=32, num_kv_heads=8,
        axis="tp", spec=L.Bf16Spec(),
    )
    o_proj = L.RowParallelLinear(
        in_features=4096, out_features=4096,
        axis="tp", sp_axis="sp",
        spec=L.Fp8Spec(granularity=L.Granularity.PER_CHANNEL),
    )
"""

from __future__ import annotations

from phyai.layers.linear.backend import Granularity, KernelProbe, LinearKernel
from phyai.layers.linear.backends import FlashInferKernel, TorchKernel
from phyai.layers.linear.dispatch import (
    KernelDispatcher,
    _set_linear_dispatcher,
    get_linear_dispatcher,
)
from phyai.layers.linear.layers import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.layers.linear.registry import (
    DefaultPolicy,
    ForcedPolicy,
    LinearKernelRegistry,
    Policy,
    list_registered_linear_kernels,
    register_linear_kernel,
)
from phyai.layers.linear.spec import ActivationView, Bf16Spec, Fp8Spec, Nvfp4Spec
from phyai.layers.quant import AllocationRequest, WeightSpec
from phyai.utils.cuda import sm_arch


def supported_specs_for_sm(sm: int) -> list[str]:
    """Specs ``validate()`` should *require* a kernel for on this hardware.

    ``validate()`` is a startup sanity check, not a feature gate: it should
    only assert coverage for specs the current GPU can actually run. fp8 has
    no backend below sm_89 (``torch._scaled_mm`` needs sm89+; flashinfer's
    block-fp8 GEMM is sm100+), so demanding fp8 coverage on Ampere (sm_86)
    or CPU (sm_arch → 0) would kill engine init for a pure-bf16 model that
    never touches fp8. bf16 is universal (it also covers the CPU fallback).

    A model that *does* request an fp8 spec on unsupported hardware still
    gets a clear, lazily-raised
    :class:`~phyai.parallel.exceptions.NoBackendError` at dispatch time
    (see :meth:`KernelDispatcher.select`) — the failure surfaces at the
    offending matmul, not at startup for everyone.
    """
    specs = ["bf16"]
    if sm >= 89:
        # torch._scaled_mm per-tensor / per-channel fp8.
        specs += ["fp8_per_tensor", "fp8_per_channel"]
    if sm >= 100:
        # flashinfer gemm_fp8_nt_groupwise (DeepSeek-V3 block-fp8).
        specs.append("fp8_block_128_128")
        # flashinfer mm_fp4 (Blackwell NVFP4) with 128x4 scale layout.
        specs.append("nvfp4_block_16_128x4")
    return specs


def init(
    *,
    register_flashinfer: bool = True,
    validate: bool = True,
    sample_specs: list[str] | None = None,
) -> KernelDispatcher:
    """Build the process-level :class:`KernelDispatcher` from the kernel
    declarations gathered by :func:`register_linear_kernel`.

    Call once after :func:`phyai.parallel.init`. Subsequent calls replace
    the dispatcher — useful in tests, harmless otherwise. Pass
    ``register_flashinfer=False`` to skip the flashinfer kernel (handy on
    CPU-only / flashinfer-unavailable hosts that still want validate to
    pass on torch alone).

    When ``sample_specs`` is left unset, the specs ``validate()`` requires
    are chosen by hardware capability via :func:`supported_specs_for_sm`
    so that pure-bf16 deployments on fp8-incapable GPUs (e.g. sm_86) start
    cleanly; unsupported specs only error if a layer actually requests one.
    """
    reg = LinearKernelRegistry()

    for cls, prefer_for in list_registered_linear_kernels():
        if not register_flashinfer and cls.name == "flashinfer":
            continue
        reg.register(cls(), prefer_for=set(prefer_for) if prefer_for else None)

    if validate:
        sm = sm_arch()
        reg.validate(
            sample_specs=(
                sample_specs if sample_specs is not None else supported_specs_for_sm(sm)
            ),
            sm=sm,
        )

    d = KernelDispatcher(reg)
    _set_linear_dispatcher(d)
    return d


def _reset_for_test() -> None:
    """Drop the dispatcher singleton. Tests only."""
    _set_linear_dispatcher(None)


__all__ = [
    "init",
    "supported_specs_for_sm",
    # layers
    "LinearBase",
    "ReplicatedLinear",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    # specs
    "Bf16Spec",
    "Fp8Spec",
    "Nvfp4Spec",
    "ActivationView",
    "AllocationRequest",
    "WeightSpec",
    "Granularity",
    # dispatcher / registry
    "KernelDispatcher",
    "get_linear_dispatcher",
    "LinearKernelRegistry",
    "DefaultPolicy",
    "ForcedPolicy",
    "Policy",
    "LinearKernel",
    "KernelProbe",
    "register_linear_kernel",
    "list_registered_linear_kernels",
    # backends
    "TorchKernel",
    "FlashInferKernel",
]
