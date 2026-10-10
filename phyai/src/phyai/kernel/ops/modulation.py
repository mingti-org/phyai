"""Conditional affine modulation with explicit intermediate rounding."""

from __future__ import annotations

from phyai.kernel.facts import device, dtype, lib
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


def _triton_modulate(facts, params):
    from phyai_kernel import modulate

    return modulate


def _torch_modulate(facts, params):
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return modulate


def register(catalog: Catalog) -> None:
    catalog.register_op(MODULATE)
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
        )
    )


__all__ = ["MODULATE", "register"]
