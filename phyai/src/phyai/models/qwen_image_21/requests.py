"""Requests and outputs for Qwen-Image 2.1 generation and editing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image

from phyai_utils_tools.models.qwen_image_21 import QwenImage21ProcessedInputs


@dataclass
class QwenImage21Request:
    prompt: str | list[str] | None = None
    image: Image.Image | np.ndarray | list[Image.Image | np.ndarray] | None = None
    negative_prompt: str | list[str] | None = None
    height: int | None = None
    width: int | None = None
    output_resolution: int = 1024
    num_inference_steps: int = 40
    true_cfg_scale: float = 1.0
    num_images_per_prompt: int = 1
    seed: int = 42
    generator: torch.Generator | list[torch.Generator] | None = None
    latents: torch.Tensor | None = None
    sigmas: list[float] | tuple[float, ...] | None = None
    prompt_embeds: torch.Tensor | None = None
    prompt_embeds_mask: torch.Tensor | None = None
    image_pad_mask: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds_mask: torch.Tensor | None = None
    negative_image_pad_mask: torch.Tensor | None = None
    processed: QwenImage21ProcessedInputs | None = None
    negative_processed: QwenImage21ProcessedInputs | None = None
    condition_latents: torch.Tensor | None = None
    condition_image_shapes: tuple[tuple[int, int, int], ...] = ()
    output_type: Literal["pt", "np", "pil", "latent"] = "pt"

    def __post_init__(self) -> None:
        if self.num_inference_steps < 1 or self.num_images_per_prompt < 1:
            raise ValueError("Inference steps and images per prompt must be positive.")
        if self.output_resolution < 32:
            raise ValueError("output_resolution must be at least 32 pixels.")
        if any(value is not None and value < 32 for value in (self.height, self.width)):
            raise ValueError("height and width must be at least 32 pixels.")
        if self.output_type not in ("pt", "np", "pil", "latent"):
            raise ValueError("output_type must be pt, np, pil, or latent.")
        if self.condition_latents is None and self.condition_image_shapes:
            raise ValueError("condition_image_shapes requires condition_latents.")
        if self.condition_latents is not None and not self.condition_image_shapes:
            raise ValueError("condition_latents requires condition_image_shapes.")


@dataclass(frozen=True)
class QwenImage21Output:
    images: torch.Tensor | np.ndarray | list[Image.Image]
    latents: torch.Tensor


__all__ = ["QwenImage21Request", "QwenImage21Output"]
