"""Generation inputs and outputs for MOSS-Image."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image


@dataclass
class MOSSImageRequest:
    prompt: str | list[str] | None = None
    negative_prompt: str | list[str] = ""
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    num_images_per_prompt: int = 1
    seed: int = 42
    generator: torch.Generator | list[torch.Generator] | None = None
    latents: torch.Tensor | None = None
    prompt_embeds: torch.Tensor | None = None
    prompt_attention_mask: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None
    negative_prompt_attention_mask: torch.Tensor | None = None
    max_sequence_length: int = 4096
    uncond_skip_layers: tuple[int, ...] | None = (22,)
    slg_layers: tuple[int, ...] | None = None
    slg_scale: float = 0.0
    slg_start: float = 0.0
    slg_stop: float = 1.0
    output_type: Literal["pt", "np", "pil", "latent"] = "pil"

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0 or value % 16
            for value in (self.height, self.width)
        ):
            raise ValueError("height and width must be positive multiples of 16")
        if any(
            type(value) is not int or value < 1
            for value in (self.num_inference_steps, self.num_images_per_prompt)
        ):
            raise ValueError(
                "Inference steps and images per prompt must be positive integers"
            )
        if (
            type(self.max_sequence_length) is not int
            or not 1 <= self.max_sequence_length <= 4096
        ):
            raise ValueError("max_sequence_length must be between 1 and 4096")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative")
        if not math.isfinite(self.slg_scale) or self.slg_scale < 0:
            raise ValueError("slg_scale must be finite and nonnegative")
        if not 0 <= self.slg_start <= self.slg_stop <= 1:
            raise ValueError("Require 0 <= slg_start <= slg_stop <= 1")
        if (self.prompt is None) == (self.prompt_embeds is None):
            raise ValueError("Provide exactly one of prompt and prompt_embeds")
        if self.output_type not in ("pt", "np", "pil", "latent"):
            raise ValueError("output_type must be pt, np, pil, or latent")


@dataclass(frozen=True)
class MOSSImageOutput:
    images: torch.Tensor | np.ndarray | list[Image.Image]
    latents: torch.Tensor


__all__ = ["MOSSImageRequest", "MOSSImageOutput"]
