"""Cosmos3 action/policy runner"""

from __future__ import annotations

import torch

from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Transformer
from phyai.models.cosmos3.model_runner_cosmos3 import Cosmos3T2VRunner


class Cosmos3ActionRunner(Cosmos3T2VRunner):
    """Action runner with optional capture of each complete Generator step."""

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
        use_fused_rope: bool = True,
        fuse_eager_attention: bool = False,
        rounded_residual_norm: bool = False,
        torch_compile: bool = False,
        compile_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            transformer,
            device=device,
            use_cuda_graph=use_cuda_graph,
            use_fused_rope=use_fused_rope,
            fuse_eager_attention=fuse_eager_attention,
            rounded_residual_norm=rounded_residual_norm,
            torch_compile=torch_compile,
            compile_kwargs=compile_kwargs,
        )


__all__ = ["Cosmos3ActionRunner"]
