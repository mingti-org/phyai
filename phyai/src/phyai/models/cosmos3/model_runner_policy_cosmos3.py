"""Cosmos3 action/policy runner"""

from __future__ import annotations

import torch

from phyai.models.cosmos3.model_runner_cosmos3 import Cosmos3T2VRunner
from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Transformer


class Cosmos3ActionRunner(Cosmos3T2VRunner):
    """Thin eager action runner"""

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
        torch_compile: bool = False,
        compile_kwargs: dict | None = None,
    ) -> None:
        super().__init__(
            transformer,
            device=device,
            torch_compile=torch_compile,
            compile_kwargs=compile_kwargs,
        )
        # TODO(wch): make this cuda graph friendly later.
        self.use_cuda_graph = False


__all__ = ["Cosmos3ActionRunner"]
