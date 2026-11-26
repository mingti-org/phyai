"""Fixed-shape RL-token encoder runner; decoder stays available on the model."""

from __future__ import annotations

import torch

from phyai.models.pi05.modeling_rlt_pi05 import PI05RLTTokenAutoencoder
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner


class PI05RLTRunner(ModelRunner):
    def __init__(
        self,
        model: PI05RLTTokenAutoencoder,
        *,
        batch_size: int,
        visual_tokens: int,
        device: torch.device | str,
        use_cuda_graph: bool,
    ) -> None:
        if not 0 < visual_tokens <= model.max_visual_tokens:
            raise ValueError("camera token count exceeds RLT visual-token capacity")
        self.model = model
        self.batch_size = batch_size
        self.visual_tokens = visual_tokens
        self.device = torch.device(device)
        self.use_cuda_graph = use_cuda_graph
        self.graph: CudaGraph | None = None

    @torch.no_grad()
    def setup(self) -> None:
        if not self.use_cuda_graph or self.device.type != "cuda":
            return
        self.graph = CudaGraph()
        self.graph.capture(
            self._encode,
            {
                "features": torch.zeros(
                    self.batch_size,
                    self.visual_tokens,
                    self.model.input_dim,
                    device=self.device,
                    dtype=self.model.rl_token.dtype,
                ),
                "mask": torch.ones(
                    self.batch_size,
                    self.visual_tokens,
                    device=self.device,
                    dtype=torch.bool,
                ),
            },
        )

    def _encode(self, *, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.encode(features, mask)

    @torch.no_grad()
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expected = (self.batch_size, self.visual_tokens, self.model.input_dim)
        if features.shape != expected or mask.shape != expected[:2]:
            raise ValueError(
                "RLT runner requires its configured batch and visual shape"
            )
        features = features.to(device=self.device, dtype=self.model.rl_token.dtype)
        mask = mask.to(device=self.device)
        if self.graph is None:
            return self._encode(features=features, mask=mask)
        return self.graph.replay({"features": features, "mask": mask})

    def close(self) -> None:
        self.graph = None
