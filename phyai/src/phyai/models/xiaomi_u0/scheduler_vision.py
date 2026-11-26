"""Single-image scheduling for the native U0 image tokenizer."""

import torch

from phyai.models.xiaomi_u0.model_runner_vision import U0VisionRunner
from phyai.runtime.schedule import Scheduler


class _U0VisionScheduler(Scheduler):
    """Manage the runner lifecycle for either direction of image tokenization."""

    def __init__(self, runner: U0VisionRunner) -> None:
        """Retain the runner for image requests."""
        self.runner = runner

    def setup(self) -> None:
        """Prepare the vision runner."""
        self.runner.setup()

    def close(self) -> None:
        """Release the runner's request lifecycle state."""
        self.runner.close()


class U0VisionScheduler(_U0VisionScheduler):
    """Validate one released 64-by-64 code grid and decode it through the runner."""

    @torch.inference_mode()
    def step(self, codes: torch.Tensor) -> torch.Tensor:
        """Return finite FP32 RGB pixels for a single grid of valid codebook IDs."""
        if not isinstance(codes, torch.Tensor) or codes.shape != (1, 64, 64):
            raise ValueError("The U0 image decoder expects a (1, 64, 64) code grid.")
        if (
            codes.dtype != torch.int64
            or (codes < 0).any().item()
            or (codes >= 131072).any().item()
        ):
            raise ValueError("Image code grids must contain int64 IDs in [0, 131072).")
        pixels = self.runner.forward(codes)
        if (
            pixels.shape != (1, 3, 1024, 1024)
            or not torch.isfinite(pixels).all().item()
        ):
            raise RuntimeError("The native image decoder returned invalid RGB pixels.")
        return pixels.float()


class U0VisionEncodeScheduler(_U0VisionScheduler):
    """Encode one prepared RGB image into a grid of visual codebook IDs."""

    @torch.inference_mode()
    def step(self, pixels: torch.Tensor) -> torch.Tensor:
        """Validate and encode one normalized FP32 RGB image."""
        factor = self.runner.model.config.spatial_factor
        if (
            not isinstance(pixels, torch.Tensor)
            or pixels.ndim != 4
            or pixels.shape[:2] != (1, 3)
            or any(size == 0 or size % factor for size in pixels.shape[2:])
            or pixels.dtype != torch.float32
        ):
            raise ValueError(
                f"The U0 image encoder expects FP32 (1, 3, H, W) with positive "
                f"dimensions divisible by {factor}."
            )
        if not torch.isfinite(pixels).all().item():
            raise ValueError("The U0 image encoder expects finite RGB pixels.")
        return self.runner.forward(pixels)
