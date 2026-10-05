"""Define RMSNorm, LayerNorm, AdaRMSNorm, and fused normalization operations."""

from __future__ import annotations

from phyai.kernel.facts import lib, attrs, dtype, shape, device
from phyai.kernel.opspec import (
    Impl,
    OpSpec,
    Priority,
    fixed,
    any_float,
    matches_activation,
)
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import all_of, implies

NVIDIA = device.vendor == "nvidia"
NVIDIA_FLASHINFER = lib.has("flashinfer") & NVIDIA
NVIDIA_TRITON = lib.has("phyai_kernel") & NVIDIA

#: The dtypes the Triton norm entry points accept.
TRITON_FLOATS = frozenset({"bf16", "fp16", "fp32"})


def _bench_tensor(facts, device, role: str, *shape_dims: int):
    """Synthesize one benchmark tensor from a call's dtype facts."""

    import torch

    from phyai.kernel.call import torch_dtype
    from phyai.kernel.facts import MISSING

    name = facts.lookup(f"dtype.{role}")
    if name in (MISSING, None):
        raise NotImplementedError(f"benchmark needs dtype.{role}")
    return torch.randn(*shape_dims, device=device, dtype=torch_dtype(str(name)))


def _bench_dims(facts) -> tuple[int, int]:
    from phyai.kernel.facts import MISSING

    tokens, hidden = facts.lookup("shape.tokens"), facts.lookup("shape.hidden")
    if MISSING in (tokens, hidden):
        raise NotImplementedError("benchmark needs the tokens and hidden dimensions")
    return int(tokens), int(hidden)


def _bench_rmsnorm(facts, device):
    tokens, hidden = _bench_dims(facts)
    x = _bench_tensor(facts, device, "input", tokens, hidden)
    weight = _bench_tensor(facts, device, "weight", hidden)
    return (x, weight, 1e-6)


def _bench_rmsnorm_add(facts, device):
    tokens, hidden = _bench_dims(facts)
    x = _bench_tensor(facts, device, "input", tokens, hidden)
    residual = _bench_tensor(facts, device, "residual", tokens, hidden)
    weight = _bench_tensor(facts, device, "weight", hidden)
    # The fused kernels mutate x and residual in place; values drifting
    # between timing iterations does not change the measured work.
    return (x, residual, weight, 1e-6)


def _bench_layernorm(facts, device):
    from phyai.kernel.facts import MISSING

    tokens, hidden = _bench_dims(facts)
    x = _bench_tensor(facts, device, "input", tokens, hidden)
    weight = _bench_tensor(facts, device, "weight", hidden)
    bias = None
    if facts.lookup("attrs.bias") not in (MISSING, None, False):
        bias = _bench_tensor(facts, device, "bias", hidden)
    return (x, weight, bias, 1e-6)


def _bench_rmsnorm_silu_mul(facts, device):
    tokens, hidden = _bench_dims(facts)
    x = _bench_tensor(facts, device, "input", tokens, hidden)
    gate = _bench_tensor(facts, device, "input", tokens, hidden)
    weight = _bench_tensor(facts, device, "weight", hidden)
    return (x, gate, weight, 1e-6)


RMSNORM = OpSpec(
    name="rmsnorm",
    dims=("tokens", "hidden"),
    dtypes=("input", "weight"),
    attributes=("variant",),
    params=("weight",),
    signature="(x, weight, eps) -> Tensor",
    bench_args=_bench_rmsnorm,
)

RMSNORM_ADD = OpSpec(
    name="rmsnorm_add",
    dims=("tokens", "hidden"),
    dtypes=("input", "weight", "residual"),
    attributes=("variant",),
    params=("weight",),
    signature="(x, residual, weight, eps) -> (Tensor, Tensor)",
    bench_args=_bench_rmsnorm_add,
    doc="Fused residual add followed by RMSNorm, in place.",
)

LAYERNORM = OpSpec(
    name="layernorm",
    dims=("tokens", "hidden"),
    dtypes=("input", "weight"),
    optional_dtypes=("bias",),
    attributes=("bias",),
    params=("weight", "bias"),
    signature="(x, weight, bias, eps) -> Tensor",
    bench_args=_bench_layernorm,
)

CHANNEL_L2_NORM = OpSpec(
    name="channel_l2_norm",
    dims=("tokens", "hidden"),
    dtypes=("input", "weight", "compute"),
    optional_dtypes=("bias",),
    attributes=("channel_first", "bias", "fused_compatible"),
    params=("weight", "bias"),
    signature="(x, weight, bias, eps, *, channel_first, compute_dtype) -> Tensor",
    doc="Channel L2 normalization with separately rounded scale and affine operations.",
)

ADARMSNORM = OpSpec(
    name="adarmsnorm",
    dims=("tokens", "hidden", "cond_dim"),
    dtypes=("input", "modulation"),
    attributes=(),
    signature="(x, modulation, eps) -> (Tensor, Tensor)",
    doc="RMSNorm modulated by an external (scale, shift, gate) triple.",
)

RMSNORM_SILU_MUL = OpSpec(
    name="rmsnorm_silu_mul",
    dims=("tokens", "hidden"),
    dtypes=("input", "weight"),
    attributes=(),
    signature="(x, gate, weight, eps) -> Tensor",
    bench_args=_bench_rmsnorm_silu_mul,
)


# FlashInfer RMSNorm requires bf16 input and weight tensors.
FLASHINFER_RMSNORM = all_of(
    NVIDIA_FLASHINFER, dtype.input == "bf16", dtype.weight == "bf16"
)

# The fused form also requires a bf16 residual tensor.
FLASHINFER_RMSNORM_ADD = all_of(FLASHINFER_RMSNORM, dtype.residual == "bf16")

# FlashInfer LayerNorm requires fp32 affine parameters when present.
FLASHINFER_LAYERNORM = all_of(
    NVIDIA_FLASHINFER,
    dtype.input == "bf16",
    dtype.weight == "fp32",
    implies(attrs.bias, dtype.bias == "fp32"),
)

TRITON_NORM = all_of(NVIDIA_TRITON, dtype.input.in_(TRITON_FLOATS))


def _gemma(facts) -> bool:
    return facts.lookup("attrs.variant") == "gemma"


def _flashinfer_rmsnorm(facts, params):
    from flashinfer.norm import rmsnorm, gemma_rmsnorm

    return gemma_rmsnorm if _gemma(facts) else rmsnorm


def _flashinfer_rmsnorm_add(facts, params):
    from flashinfer.norm import fused_add_rmsnorm, gemma_fused_add_rmsnorm

    fused = gemma_fused_add_rmsnorm if _gemma(facts) else fused_add_rmsnorm

    def wrapped(x, residual, weight, eps):
        # Normalize the in-place CUDA operation to the declared tuple result.
        result = fused(x, residual, weight, eps)
        return (x, residual) if result is None else result

    return wrapped


def _triton_rmsnorm(facts, params):
    from phyai_kernel import rmsnorm, gemma_rmsnorm

    return gemma_rmsnorm if _gemma(facts) else rmsnorm


def _triton_rmsnorm_add(facts, params):
    from phyai_kernel import fused_add_rmsnorm, gemma_fused_add_rmsnorm

    return gemma_fused_add_rmsnorm if _gemma(facts) else fused_add_rmsnorm


def _torch_rmsnorm_fn(gemma: bool):
    import torch

    def rmsnorm(x, weight, eps):
        promoted = x.float()
        normalized = promoted * torch.rsqrt(
            promoted.square().mean(dim=-1, keepdim=True) + eps
        )
        multiplier = 1.0 + weight.float() if gemma else weight.float()
        return (normalized * multiplier).to(x.dtype)

    return rmsnorm


def _torch_rmsnorm(facts, params):
    return _torch_rmsnorm_fn(_gemma(facts))


def _torch_rmsnorm_add(facts, params):
    inner = _torch_rmsnorm_fn(_gemma(facts))

    def fused(x, residual, weight, eps):
        residual.add_(x)
        return inner(residual, weight, eps), residual

    return fused


def _flashinfer_layernorm(facts, params):
    from flashinfer.norm import layernorm

    return layernorm


def _triton_layernorm(facts, params):
    from phyai_kernel import layernorm

    return layernorm


def _torch_layernorm(facts, params):
    import torch

    def layernorm(x, weight, bias, eps):
        # Cast affine tensors to the input dtype for the reference operator.
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)

    return layernorm


def _triton_channel_l2_norm(facts, params):
    from phyai_kernel import channel_l2_norm

    return channel_l2_norm


def _torch_channel_l2_norm(facts, params):
    import torch.nn.functional as F

    def channel_l2_norm(x, weight, bias, eps, *, channel_first, compute_dtype):
        value = x if compute_dtype is None else x.to(compute_dtype)
        dim = 1 if channel_first else -1
        normalized = F.normalize(value, dim=dim, eps=eps).to(x.dtype)
        output = normalized * weight.numel() ** 0.5 * weight
        return output if bias is None else output + bias

    return channel_l2_norm


def _triton_adarmsnorm(facts, params):
    from phyai_kernel import adarmsnorm

    return adarmsnorm


def _torch_adarmsnorm(facts, params):
    import torch

    def adarmsnorm(x, modulation, eps):
        promoted = x.float()
        normalized = promoted * torch.rsqrt(
            promoted.square().mean(dim=-1, keepdim=True) + eps
        )
        scale, shift, gate = modulation.chunk(3, dim=-1)
        out = (normalized * (1 + scale.float()) + shift.float()).to(x.dtype)
        return out, gate.to(x.dtype)

    return adarmsnorm


def _triton_rmsnorm_silu_mul(facts, params):
    from phyai_kernel import rmsnorm_silu_mul

    return rmsnorm_silu_mul


def _torch_rmsnorm_silu_mul(facts, params):
    import torch

    inner = _torch_rmsnorm_fn(False)

    def rmsnorm_silu_mul(x, gate, weight, eps):
        return inner(x, weight, eps) * torch.nn.functional.silu(gate.float()).to(
            x.dtype
        )

    return rmsnorm_silu_mul


def register(catalog: Catalog) -> None:
    for spec in (
        RMSNORM,
        RMSNORM_ADD,
        LAYERNORM,
        CHANNEL_L2_NORM,
        ADARMSNORM,
        RMSNORM_SILU_MUL,
    ):
        catalog.register_op(spec)

    catalog.register_many(
        (
            # RMSNorm.
            Impl(
                kernel_id="flashinfer.rmsnorm",
                op="rmsnorm",
                priority=Priority.OPTIMIZED + 2,
                when=FLASHINFER_RMSNORM,
                prepare=_flashinfer_rmsnorm,
                # Allocate gamma in the activation dtype.
                params={"weight": matches_activation()},
            ),
            Impl(
                kernel_id="phyai_kernel.rmsnorm",
                op="rmsnorm",
                priority=Priority.OPTIMIZED,
                when=TRITON_NORM,
                prepare=_triton_rmsnorm,
                params={"weight": any_float()},
            ),
            Impl(
                kernel_id="torch.rmsnorm",
                op="rmsnorm",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch_rmsnorm,
                params={"weight": any_float()},
            ),
            # Fused RMSNorm and residual add.
            Impl(
                kernel_id="flashinfer.rmsnorm_add",
                op="rmsnorm_add",
                priority=Priority.OPTIMIZED + 2,
                when=FLASHINFER_RMSNORM_ADD,
                prepare=_flashinfer_rmsnorm_add,
                params={"weight": matches_activation()},
            ),
            Impl(
                kernel_id="phyai_kernel.rmsnorm_add",
                op="rmsnorm_add",
                priority=Priority.OPTIMIZED,
                when=all_of(TRITON_NORM, dtype.residual.in_(TRITON_FLOATS)),
                prepare=_triton_rmsnorm_add,
                params={"weight": any_float()},
            ),
            Impl(
                kernel_id="torch.rmsnorm_add",
                op="rmsnorm_add",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch_rmsnorm_add,
                params={"weight": any_float()},
            ),
            # LayerNorm.
            Impl(
                kernel_id="flashinfer.layernorm",
                op="layernorm",
                priority=Priority.OPTIMIZED + 2,
                when=FLASHINFER_LAYERNORM,
                prepare=_flashinfer_layernorm,
                params={"weight": fixed("fp32"), "bias": fixed("fp32")},
            ),
            Impl(
                kernel_id="phyai_kernel.layernorm",
                op="layernorm",
                priority=Priority.OPTIMIZED,
                when=TRITON_NORM,
                prepare=_triton_layernorm,
                params={"weight": any_float(), "bias": any_float()},
            ),
            Impl(
                kernel_id="torch.layernorm",
                op="layernorm",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch_layernorm,
                # The reference path accepts any floating affine dtype.
                params={"weight": any_float(), "bias": any_float()},
            ),
            # Channel L2 normalization.
            Impl(
                kernel_id="phyai_kernel.channel_l2_norm",
                op="channel_l2_norm",
                priority=Priority.OPTIMIZED,
                when=all_of(TRITON_NORM, attrs.fused_compatible),
                prepare=_triton_channel_l2_norm,
                params={"weight": any_float(), "bias": any_float()},
            ),
            Impl(
                kernel_id="torch.channel_l2_norm",
                op="channel_l2_norm",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch_channel_l2_norm,
                params={"weight": any_float(), "bias": any_float()},
            ),
            # AdaRMSNorm.
            Impl(
                kernel_id="phyai_kernel.adarmsnorm",
                op="adarmsnorm",
                priority=Priority.OPTIMIZED,
                when=TRITON_NORM,
                prepare=_triton_adarmsnorm,
            ),
            Impl(
                kernel_id="torch.adarmsnorm",
                op="adarmsnorm",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch_adarmsnorm,
            ),
            # RMSNorm with SiLU gating.
            Impl(
                kernel_id="phyai_kernel.rmsnorm_silu_mul",
                op="rmsnorm_silu_mul",
                priority=Priority.OPTIMIZED,
                when=all_of(TRITON_NORM, shape.hidden <= 8192),
                prepare=_triton_rmsnorm_silu_mul,
            ),
            Impl(
                kernel_id="torch.rmsnorm_silu_mul",
                op="rmsnorm_silu_mul",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch_rmsnorm_silu_mul,
            ),
        )
    )


__all__ = [
    "ADARMSNORM",
    "LAYERNORM",
    "CHANNEL_L2_NORM",
    "RMSNORM",
    "RMSNORM_ADD",
    "RMSNORM_SILU_MUL",
    "register",
]
