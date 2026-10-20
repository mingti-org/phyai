"""Unpadded Qwen chat prompts and RGB output conversion for MOSS-Image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline
from phyai_utils_tools.tokenizer import get_tokenizer


class MOSSImageProcessor(BaseModelProcessor):
    def __init__(
        self, tokenizer_path: str | Path | None = None, *, tokenizer: Any = None
    ) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer is None and tokenizer_path is not None:
            self.tokenizer = get_tokenizer(str(tokenizer_path), local_files_only=True)
        super().__init__()

    def tokenize(
        self, prompt: str | list[str], max_sequence_length: int = 4096
    ) -> list[torch.Tensor]:
        if self.tokenizer is None:
            raise ValueError("String prompts require a tokenizer")
        if not 1 <= max_sequence_length <= 4096:
            raise ValueError("max_sequence_length must be between 1 and 4096")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or any(not isinstance(value, str) for value in prompts):
            raise ValueError("prompt must be a string or a nonempty list of strings")
        rows = []
        for value in prompts:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": value}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_sequence_length,
            )
            rows.append(inputs["input_ids"])
        return rows

    @staticmethod
    def format_images(
        pixels: torch.Tensor, output_type: str = "pil"
    ) -> torch.Tensor | np.ndarray | list[Image.Image]:
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("Decoded pixels must have shape (batch, 3, height, width)")
        if output_type not in ("pt", "np", "pil"):
            raise ValueError("Decoded output_type must be pt, np, or pil")
        pixels = (pixels.detach().float().clamp(-1, 1) / 2 + 0.5).clamp(0, 1)
        if output_type == "pt":
            return pixels
        array = pixels.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return array
        return [
            Image.fromarray((sample * 255).round().astype(np.uint8)) for sample in array
        ]

    def build_preprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            name="moss_image_preprocessor",
            to_output=lambda value: self.tokenize(
                value["prompt"], value.get("max_sequence_length", 4096)
            ),
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            name="moss_image_postprocessor",
            to_output=lambda value: self.format_images(
                value["pixels"], value.get("output_type", "pil")
            ),
        )


__all__ = ["MOSSImageProcessor"]
