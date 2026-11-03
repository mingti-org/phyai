"""Compute and combine selected SwiGLU experts.

Inputs are tokens [T, H], expert indices and routing weights [T, K],
FC1 weights [E, 2 * I, H] in [up, gate] order, and FC2 weights [E, H, I].
The output has shape [T, H] and the input dtype. Routing and TP reduction
belong to the caller; expert parallelism is unsupported.
Tensors must be contiguous on the same CUDA device, with H and I divisible by 8.
"""

from __future__ import annotations

from phyai.kernel.facts import lib, attrs, dtype, device
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import all_of

MOE = OpSpec(
    name="moe",
    dims=("tokens", "hidden"),
    dtypes=("input", "weight"),
    attributes=("activation",),
    signature=(
        "(x, selected_experts, routing_weights, fc1_weight, fc2_weight, *, "
        "tp_size, tp_rank) -> Tensor"
    ),
    requires_reference=False,
    doc=(
        "Routed expert computation for a token MoE. Consumes pre-computed "
        "expert indices and routing scales; routing stays in the model."
    ),
)

# Architectures accepted by FlashInfer's get_cutlass_fused_moe_module.
CUTLASS_MOE_ARCHES = frozenset(
    {"sm89", "sm90", "sm100", "sm103", "sm107", "sm110", "sm120", "sm121"}
)

# This adapter supports unquantized BF16 inputs and expert weights.
FLASHINFER_CUTLASS_MOE = all_of(
    lib.has("flashinfer"),
    device.vendor == "nvidia",
    device.arch.in_(CUTLASS_MOE_ARCHES),
    dtype.input == "bf16",
    dtype.weight == "bf16",
    attrs.activation == "swiglu",
)


def _flashinfer_cutlass_moe(facts, params):
    """Prepare the FlashInfer CUTLASS fused MoE callable."""

    import torch
    from flashinfer.fused_moe import cutlass_fused_moe

    def fused_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        fc1_expert_weights,
        fc2_expert_weights,
        *,
        tp_size,
        tp_rank,
    ):
        # FlashInfer requires FP32 routing scales and int32 expert indices.
        token_final_scales = routing_weights.float()
        token_selected_experts = selected_experts.to(torch.int32)
        outputs = cutlass_fused_moe(
            hidden_states,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc2_expert_weights,
            hidden_states.dtype,
            quant_scales=None,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        return outputs[0]

    return fused_moe


def register(catalog: Catalog) -> None:
    catalog.register_op(MOE)
    catalog.register(
        Impl(
            kernel_id="flashinfer.moe.cutlass_fused_moe",
            op="moe",
            priority=Priority.OPTIMIZED + 2,
            capture_safe=True,
            when=FLASHINFER_CUTLASS_MOE,
            prepare=_flashinfer_cutlass_moe,
            metadata={
                "package": "flashinfer",
                "backend": "cutlass",
                "activation": "swiglu",
            },
        )
    )


__all__ = ["CUTLASS_MOE_ARCHES", "MOE", "register"]
