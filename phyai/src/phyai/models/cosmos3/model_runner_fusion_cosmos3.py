"""Runner-owned compilation of exact Cosmos3 elementwise operations."""

from __future__ import annotations

import torch
from torch._dynamo.exc import (
    FailOnRecompileLimitHit,
    BackendCompilerFailed,
    Unsupported,
)

from phyai.utils import get_logger
from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3AttentionPreparation,
    prepare_cosmos3_attention,
)

logger = get_logger(__name__)


def prepare_attention_static_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Head layout is fixed by the model; sequence lengths vary by request.
    # Specialize only inside compilation, without annotating caller tensors.
    for value in (q, k, v, cos, sin, k_und, v_und):
        torch._dynamo.mark_static(value, -2)
        torch._dynamo.mark_static(value, -1)
    return prepare_cosmos3_attention(q, k, v, cos, sin, k_und, v_und)


class Cosmos3AttentionFusion:
    """Fuse GEN RoPE and UND/GEN KV concatenation with BF16 rounding intact.

    Compilation state belongs to the runner. The model receives only a callable
    with the same stateless contract as its eager helper. Warm this callable
    before CUDA graph capture, and call ``reset`` when releasing runner state.
    """

    def __init__(self, *, enabled: bool = True, eager_enabled: bool = False) -> None:
        self.enabled = enabled
        self.eager_enabled = eager_enabled
        self._compiled: Cosmos3AttentionPreparation | None = None
        self._recompile_limit_hit = False

    def reset(self) -> None:
        self._compiled = None
        self._recompile_limit_hit = False

    def prepare(
        self, *, device: torch.device | str, use_cuda_graph: bool
    ) -> Cosmos3AttentionPreparation | None:
        if (
            not self.enabled
            or not (use_cuda_graph or self.eager_enabled)
            or torch.device(device).type != "cuda"
            or self._recompile_limit_hit
        ):
            return None
        if self._compiled is None:
            self._compiled = torch.compile(
                prepare_attention_static_heads,
                fullgraph=True,
                dynamic=True,
                options={
                    "emulate_precision_casts": True,
                    "triton.cudagraphs": False,
                },
            )
        return self.forward

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._compiled is None or q.device.type != "cuda":
            return prepare_cosmos3_attention(q, k, v, cos, sin, k_und, v_und)
        try:
            return self._compiled(q, k, v, cos, sin, k_und, v_und)
        except (FailOnRecompileLimitHit, BackendCompilerFailed, Unsupported):
            # Unusual stride/dtype contracts can exceed Dynamo's bounded cache
            # even with symbolic dimensions. Preserve inference in that case.
            self._compiled = None
            self._recompile_limit_hit = True
            logger.warning_once(
                "Cosmos3 attention preparation compilation failed; "
                "using the eager helper for this runner."
            )
            return prepare_cosmos3_attention(q, k, v, cos, sin, k_und, v_und)
