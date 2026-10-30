"""Define the fused Mixture-of-Experts operation.

LingBot V2 runs only the selected SwiGLU experts, and the released
deployment binds that to FlashInfer's CUTLASS fused MoE. There is
deliberately no Torch reference implementation: a stand-in would not
reproduce the expert kernel, so a mis-selection would surface as plausible
but different numbers instead of an error. An unsupported configuration
therefore fails at selection time with the full candidate trace -- the same
choice ``attention_paged`` and ``attention_gdn`` make.

The router is *not* part of this operation. LingBot computes the FP32
sigmoid top-k router and rounds the routing weights to the model dtype
before calling here, and the tensor-parallel all-reduce that follows the
expert compute stays outside as well, because it is a collective over the
mesh rather than part of the kernel.

Only ``tp_size`` / ``tp_rank`` are declared. Expert parallelism is not
implemented: the underlying call keeps its default ``ep_size=1`` and no
all-to-all dispatch is configured.
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

# FlashInfer picks its CUTLASS module from ``f"{major * 10 + minor}"`` and
# accepts exactly these values (``flashinfer/fused_moe/core.py``,
# ``get_cutlass_fused_moe_module``); anything else raises ``ValueError``
# from inside the call, where no capability predicate can catch it. Naming
# them here turns an unbuildable architecture into a selection-time
# failure. ``sm110`` is Jetson Thor, which FlashInfer serves from its sm100
# module.
CUTLASS_MOE_ARCHES = frozenset(
    {"sm89", "sm90", "sm100", "sm103", "sm107", "sm110", "sm120", "sm121"}
)

# The released LingBot expert weights are bf16, the fused kernel consumes
# half-precision activations only, and the model already rejects any other
# parameter dtype at construction time.
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
        # LingBot rounds the routing weights to the model dtype before the
        # expert compute, while FlashInfer requires ``token_final_scales``
        # stored as FP32 and the expert indices as int32. Both conversions
        # belong to the kernel boundary, not to the model.
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
            # The model called this kernel unconditionally in every mode
            # before the integration -- including inside the captured Euler
            # loop -- so the default keeps that behaviour rather than adding
            # a gate the model never had. Capture safety is *unverified*
            # here: FlashInfer's cutlass path allocates and frees a
            # multi-GiB workspace per call unless ``workspace_buffer`` is
            # supplied, and its explicit graph-capture affordances sit in
            # the trtllm entry points, not this one.
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
