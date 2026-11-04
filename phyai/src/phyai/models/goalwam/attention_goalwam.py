"""GoalWAM's FlashAttention 3 backend through the public attention contract."""

from importlib.metadata import version

import torch

from phyai.kernel.facts import attrs, dtype, shape, device
from phyai.kernel.opspec import Impl, Priority
from phyai.kernel.policy import policy_from_mapping
from phyai.kernel.bootstrap import get_kernel_selector
from phyai.layers.attention.nocache.base import (
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
    AttentionBackend,
    AttentionLayerProto,
)


class GoalWAMAttentionBackend(AttentionBackend):
    """Preserve the checkpoint reference's attention arithmetic on Hopper."""

    name = "goalwam_fa3"

    def __init__(self, runner=None) -> None:
        del runner
        from flash_attn_3_nv.flash_attn_interface import flash_attn_func

        self.attention = flash_attn_func

    def supports_capture(self) -> bool:
        return True

    def init_forward_metadata(self, meta: AttnMetadata) -> AttnPlanHandle:
        if (
            not meta.layout.is_padded()
            or meta.mask is not None
            or meta.attn_bias is not None
        ):
            raise ValueError("GoalWAM attention requires unmasked padded tensors")
        return AttnPlanHandle()

    def forward(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        if ctx.mode.is_idle():
            return torch.zeros_like(q)
        output = self.attention(q, k, v, causal=layer.causal, softmax_scale=layer.scale)
        return output[0] if isinstance(output, tuple) else output


def prepare_goalwam_attention_backend(facts, params):
    del facts
    if params:
        raise ValueError("GoalWAM attention accepts no kernel parameters")
    return GoalWAMAttentionBackend


def configure_goalwam_kernels() -> None:
    """Configure this engine's selector without changing other model defaults."""
    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("GoalWAM's validated precision path requires a Hopper GPU")
    try:
        import flash_attn_3_nv.flash_attn_interface  # noqa: F401

        if version("flash-attn-3-nv").split("+")[0] != "1.0.3":
            raise ImportError(
                "GoalWAM attention requires FlashAttention 3 version 1.0.3"
            )
    except (ImportError, OSError) as error:
        raise ImportError(
            "GoalWAM requires flash-attn-3-nv==1.0.3 and its compatible CUDA runtime"
        ) from error
    selector = get_kernel_selector()
    kernel_id = "goalwam_fa3.attention"
    if selector.catalog.maybe_get(kernel_id) is None:
        selector.catalog.register(
            Impl(
                kernel_id=kernel_id,
                op="attention",
                priority=Priority.REFERENCE,
                when=(
                    (device.vendor == "nvidia")
                    & device.arch.family_in({"sm9"})
                    & (dtype.input == "bf16")
                    & (shape.head_dim == 128)
                    & (attrs.layout == "padded")
                    & attrs.mask_kind.is_none()
                    & (attrs.has_attn_bias != True)
                ),
                prepare=prepare_goalwam_attention_backend,
            )
        )
    # Pin the arithmetic whose end-to-end behavior defines this model's
    # inference contract. The selector belongs to this GoalWAM engine.
    selector.policy = policy_from_mapping(
        {
            "schema": "phyai.kernel/v1",
            "overrides": [
                {
                    "id": "goalwam-gemm",
                    "match": {"op": "gemm"},
                    "restrict_to": "torch.gemm.*",
                },
                {
                    "id": "goalwam-attention",
                    "match": {"op": "attention"},
                    "restrict_to": kernel_id,
                },
            ],
        },
        selector.catalog,
    )
    selector.clear_cache()
