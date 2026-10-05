"""Prompt tokenization and RGBA image processing for Qwen-Image 2.1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLProcessor

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline


SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


@dataclass(frozen=True)
class QwenImage21ImageInputs:
    images: tuple[Image.Image, ...] = ()
    vae_images: tuple[torch.Tensor, ...] = ()
    image_sizes: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class QwenImage21TokenizedPrompt:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    drop_idx: int
    image_token_id: int
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None


@dataclass(frozen=True)
class QwenImage21ProcessedInputs:
    prompt: QwenImage21TokenizedPrompt
    images: QwenImage21ImageInputs

    @property
    def vae_images(self) -> tuple[torch.Tensor, ...]:
        return self.images.vae_images

    @property
    def image_sizes(self) -> tuple[tuple[int, int], ...]:
        return self.images.image_sizes


def as_rgba_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGBA")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("Condition images must be PIL images, NumPy arrays, or paths.")
    return image.convert("RGBA")


class QwenImage21Processor(BaseModelProcessor):
    """Build matching VLM and VAE inputs while preserving the alpha channel."""

    def __init__(self, processor_path: str | Path) -> None:
        self.processor = Qwen3VLProcessor.from_pretrained(str(processor_path))
        self.processor.tokenizer.padding_side = "left"
        tokens = self.processor.apply_chat_template(
            [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}],
            tokenize=True,
            return_dict=False,
        )
        first = tokens[0]
        self.drop_idx = len(first) if isinstance(first, (list, tuple)) else len(tokens)
        self.image_token_id = self.processor.tokenizer.encode("<|image_pad|>")[0]
        super().__init__()

    def prepare_images(
        self,
        image: Image.Image | np.ndarray | str | Path | list | tuple | None,
        *,
        output_resolution: int = 1024,
    ) -> QwenImage21ImageInputs:
        if image is None:
            return QwenImage21ImageInputs()
        if output_resolution <= 0:
            raise ValueError("output_resolution must be positive.")
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        if not images or len(images) > 10:
            raise ValueError("Pass between one and ten condition images.")
        resized_images = []
        vae_images = []
        image_sizes = []
        for value in images:
            pil_image = as_rgba_image(value)
            ratio = pil_image.width / pil_image.height
            width = round(math.sqrt(output_resolution**2 * ratio) / 32) * 32
            height = round(math.sqrt(output_resolution**2 / ratio) / 32) * 32
            if width == 0 or height == 0:
                raise ValueError("The condition image aspect ratio is too extreme.")
            resized = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.array(resized).astype(np.float32) / 255.0)
            pixels = pixels.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
            resized_images.append(resized)
            vae_images.append(pixels * 2.0 - 1.0)
            image_sizes.append((width, height))
        return QwenImage21ImageInputs(
            tuple(resized_images), tuple(vae_images), tuple(image_sizes)
        )

    def tokenize(
        self,
        prompt: str | list[str],
        images: tuple[Image.Image, ...] | list[Image.Image] = (),
    ) -> QwenImage21TokenizedPrompt:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or any(not isinstance(value, str) for value in prompts):
            raise ValueError("prompt must contain at least one string.")
        prefix = " ".join(
            f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
            for i in range(len(images))
        )
        text = [PROMPT_TEMPLATE.format(prefix + (value or " ")) for value in prompts]
        kwargs: dict[str, Any] = {
            "text": text,
            "padding": True,
            "padding_side": "left",
            "return_tensors": "pt",
        }
        if images:
            rgb_images = []
            for image in images:
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, (255, 255, 255))
                rgb.paste(rgba, mask=rgba.getchannel("A"))
                rgb_images.append(rgb)
            kwargs["images"] = rgb_images * len(prompts)
        inputs = self.processor(**kwargs)
        return QwenImage21TokenizedPrompt(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            drop_idx=self.drop_idx,
            image_token_id=self.image_token_id,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )

    def process_request(self, raw: dict[str, Any]) -> QwenImage21ProcessedInputs:
        images = self.prepare_images(
            raw.get("image"), output_resolution=raw.get("output_resolution", 1024)
        )
        return QwenImage21ProcessedInputs(
            self.tokenize(raw["prompt"], images.images), images
        )

    @staticmethod
    def to_pil(pixels: torch.Tensor) -> list[Image.Image]:
        if pixels.ndim == 5:
            if pixels.shape[2] != 1:
                raise ValueError("Qwen-Image 2.1 returns a single frame.")
            pixels = pixels[:, :, 0]
        if pixels.ndim != 4 or pixels.shape[1] != 4:
            raise ValueError(
                "Decoded pixels must have shape (batch, 4, height, width)."
            )
        pixels = (pixels.detach().float() / 2 + 0.5).clamp(0, 1)
        array = (
            (pixels.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        )
        return [Image.fromarray(sample) for sample in array]

    def build_preprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[],
            name="qwen_image_21_preprocessor",
            to_transition=lambda value: value,
            to_output=self.process_request,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[],
            name="qwen_image_21_postprocessor",
            to_transition=lambda value: value,
            to_output=self.to_pil,
        )


__all__ = [
    "QwenImage21ImageInputs",
    "QwenImage21ProcessedInputs",
    "QwenImage21Processor",
    "QwenImage21TokenizedPrompt",
]
