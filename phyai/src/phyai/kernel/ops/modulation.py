"""Conditional affine modulation with explicit intermediate rounding."""

from __future__ import annotations

from phyai.kernel.facts import device, dtype, lib, shape
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.predicate import all_of
from phyai.kernel.registry import Catalog


MODULATE = OpSpec(
    name="modulate",
    dims=("batch", "tokens", "hidden"),
    dtypes=("input", "shift", "scale"),
    signature="(x, shift, scale) -> Tensor",
    doc="Compute x * (1 + scale[:, None]) + shift[:, None], rounding each operation.",
)


LAYERNORM_MODULATE = OpSpec(
    name="layernorm_modulate",
    dims=("batch", "tokens", "hidden"),
    dtypes=("input", "shift", "scale"),
    attributes=("residual",),
    signature="(x, shift, scale, eps, residual=None) -> (Tensor, Tensor)",
    doc="Non-affine LayerNorm and modulation with optional rounded residual add.",
)


def _triton_modulate(facts, params):
    from phyai_kernel import modulate

    return modulate


def _torch_modulate(facts, params):
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return modulate


def _triton_layernorm_modulate(facts, params):
    from phyai_kernel import layernorm_modulate

    return layernorm_modulate


def _torch_layernorm_modulate(facts, params):
    import torch.nn.functional as F

    def layernorm_modulate(x, shift, scale, eps, residual=None):
        updated = x if residual is None else x + residual
        normalized = F.layer_norm(updated.float(), (updated.shape[-1],), eps=eps).to(
            x.dtype
        )
        return normalized * (1 + scale[:, None]) + shift[:, None], updated

    return layernorm_modulate


def register(catalog: Catalog) -> None:
    catalog.register_op(MODULATE)
    catalog.register_op(LAYERNORM_MODULATE)
    catalog.register_many(
        (
            Impl(
                kernel_id="phyai_kernel.modulate",
                op="modulate",
                priority=Priority.OPTIMIZED,
                when=all_of(
                    lib.has("phyai_kernel"),
                    lib.has("triton"),
                    device.vendor == "nvidia",
                    dtype.input.in_({"fp16", "bf16", "fp32"}),
                    dtype.shift == dtype.input,
                    dtype.scale == dtype.input,
                ),
                prepare=_triton_modulate,
            ),
            Impl(
                kernel_id="torch.modulate",
                op="modulate",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch_modulate,
            ),
            Impl(
                kernel_id="phyai_kernel.layernorm_modulate",
                op="layernorm_modulate",
                priority=Priority.OPTIMIZED,
                when=all_of(
                    lib.has("phyai_kernel"),
                    lib.has("triton"),
                    device.vendor == "nvidia",
                    dtype.input.in_({"fp16", "bf16", "fp32"}),
                    dtype.shift == dtype.input,
                    dtype.scale == dtype.input,
                    shape.hidden <= 8192,
                ),
                prepare=_triton_layernorm_modulate,
            ),
            Impl(
                kernel_id="torch.layernorm_modulate",
                op="layernorm_modulate",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch_layernorm_modulate,
            ),
        )
    )


__all__ = ["MODULATE", "LAYERNORM_MODULATE", "register"]
