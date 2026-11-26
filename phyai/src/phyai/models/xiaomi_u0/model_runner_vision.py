"""Runner for native U0 image encoding and decoding."""

import torch

from phyai.models.xiaomi_u0.modeling_vision import (
    U0VisionImageEncoder,
    U0VisionTokenizer,
)
from phyai.runtime.model_runner import ModelRunner


class U0VisionRunner(ModelRunner):
    """Move image or code tensors to the loaded model's device."""

    def __init__(self, model: U0VisionImageEncoder | U0VisionTokenizer) -> None:
        """Retain the loaded model without owning checkpoint loading."""
        self.model = model
        self.ready = False

    def setup(self) -> None:
        """Require an eval-mode model before serving requests."""
        if self.ready:
            raise RuntimeError("The vision runner is already set up.")
        if self.model.training:
            raise ValueError("The vision runner requires an eval-mode model.")
        self.ready = True

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Run the configured vision model on its device."""
        if not self.ready:
            raise RuntimeError("Set up the vision runner before running inference.")
        device = self.model.quantize.embedding.weight.device
        return self.model(codes.to(device))

    def close(self) -> None:
        """Stop accepting forward calls until the runner is set up again."""
        self.ready = False
