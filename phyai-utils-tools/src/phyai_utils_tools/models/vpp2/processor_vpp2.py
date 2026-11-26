"""VPP2 image composition, UMT5 tokenization, and action normalization."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TVF
from transformers import AutoTokenizer

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline


VPP2_PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


def clean_vpp2_prompt(text: str) -> str:
    try:
        import ftfy
    except ImportError:
        pass
    else:
        text = ftfy.fix_text(text)
    return re.sub(r"\s+", " ", html.unescape(html.unescape(text))).strip()


def rgb_tensor(image: Any) -> torch.Tensor:
    """Convert an RGB image to a CPU [3,H,W] tensor in [0,1]."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), copy=True)
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Images must be PIL RGB, HWC NumPy arrays, or CHW tensors.")
    if image.dtype == torch.uint8:
        return image.float().cpu() / 255.0
    image = image.float().cpu()
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Floating RGB images must be finite and in [0,1].")
    return image


def resize_policy_image(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return TVF.resize(
        image, list(size), interpolation=TVF.InterpolationMode.BILINEAR, antialias=True
    )


def normalization_affine(
    stats: dict[str, Any], mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    def value(name: str) -> torch.Tensor:
        return torch.as_tensor(stats[f"global_{name}"], dtype=torch.float32)

    if mode == "z-score":
        scale = 1.0 / (value("std") + 1e-8)
        return scale, -value("mean") / (value("std") + 1e-8)
    if mode != "min/max":
        raise ValueError("VPP2 normalization must be 'z-score' or 'min/max'.")
    minimum, maximum = value("min"), value("max")
    span = maximum - minimum
    constant = span < 1e-4
    span = torch.where(constant, 2.0, span)
    scale = 2.0 / span
    offset = torch.where(constant, -minimum, -1.0 - scale * minimum)
    return scale, offset


class VPP2Processor(BaseModelProcessor):
    def __init__(
        self,
        tokenizer: Any,
        *,
        mode: str = "action",
        layout: str = "robodojo_tshape",
        image_size: tuple[int, int] = (240, 416),
        max_length: int = 128,
        dataset_stats: dict[str, Any] | None = None,
        normalization: str = "z-score",
    ) -> None:
        if mode not in {"action", "video"}:
            raise ValueError("mode must be 'action' or 'video'.")
        if layout not in {"robodojo_tshape", "horizontal", "single"}:
            raise ValueError("Unknown camera layout.")
        if min(image_size) <= 0 or any(size % 16 for size in image_size):
            raise ValueError("Image height and width must be positive multiples of 16.")
        self.tokenizer = tokenizer
        self.mode = mode
        self.layout = layout
        self.image_size = image_size
        self.max_length = max_length
        self.affine = {}
        if dataset_stats is not None:
            for kind in ("state", "action"):
                self.affine[kind] = normalization_affine(
                    dataset_stats[kind]["default"], normalization
                )
        super().__init__()

    @classmethod
    def from_pretrained(
        cls, checkpoint_dir: str | Path, *, variant: str = "robodojo_joint2b_100k"
    ) -> VPP2Processor:
        root = Path(checkpoint_dir)
        release = json.loads((root / "config.json").read_text())
        variant_config = release["variants"][variant]
        assets = root / release["shared_assets"]["wan_model_dir"]
        tokenizer = AutoTokenizer.from_pretrained(
            assets / "google/umt5-xxl", local_files_only=True
        )
        policy = "action_checkpoint" in variant_config
        libero = variant.startswith("libero")
        stats_path = variant_config.get("dataset_stats")
        stats = json.loads((root / stats_path).read_text()) if stats_path else None
        inference = variant_config.get("inference", {})
        return cls(
            tokenizer,
            mode="action" if policy else "video",
            layout="horizontal"
            if libero
            else "robodojo_tshape"
            if policy
            else "single",
            image_size=(224, 448)
            if libero
            else (inference.get("height", 240), inference.get("width", 416)),
            max_length=128 if policy else 512,
            dataset_stats=stats,
            normalization="min/max" if libero else "z-score",
        )

    def build_preprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(name="vpp2_preprocessor", to_transition=self.prepare)

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(name="vpp2_postprocessor", to_transition=self.finish)

    def compose_images(self, images: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        views = [rgb_tensor(image) for image in images]
        if self.mode == "video":
            if len(views) != 1:
                raise ValueError("Standalone video generation expects one image.")
            native = views[0].unsqueeze(0).sub(0.5).div(0.5)
            resized = F.interpolate(
                native, size=self.image_size, mode="bicubic", align_corners=False
            )
            return resized, native
        if self.layout == "robodojo_tshape" and len(views) == 3:
            head = resize_policy_image(views[0], (480, 640))
            left = resize_policy_image(views[1], (240, 240))
            right = resize_policy_image(views[2], (240, 240))
            composed = torch.cat([head, torch.cat([left, right], dim=1)], dim=2)
            composed = resize_policy_image(composed, self.image_size)
        elif self.layout == "horizontal" and len(views) == 2:
            height, width = self.image_size
            composed = torch.cat(
                [resize_policy_image(view, (height, width // 2)) for view in views],
                dim=2,
            )
        elif len(views) == 1:
            composed = resize_policy_image(views[0], self.image_size)
        else:
            raise ValueError(
                "Supply head/left/right RoboDojo views, two LIBERO views, or one composed image."
            )
        pixels = composed.mul(2).sub(1).unsqueeze(0).contiguous()
        return pixels, pixels

    def tokenize(self, prompt: str) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            [clean_vpp2_prompt(prompt)],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )

    def prepare(self, raw: dict[str, Any]) -> dict[str, Any]:
        images = raw["images"]
        if not isinstance(images, (list, tuple)):
            images = [images]
        pixels, clip_pixels = self.compose_images(images)
        prompt = raw.get("prompt")
        if prompt is None:
            task = raw["task"]
            prompt = (
                VPP2_PROMPT_TEMPLATE.format(task=task)
                if self.mode == "action"
                else task
            )
        tokens = self.tokenize(prompt)
        result = {
            "pixel_values": pixels,
            "clip_pixel_values": clip_pixels,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "mode": self.mode,
        }
        if self.mode == "video":
            negative = self.tokenize(raw.get("negative_prompt", ""))
            result.update(
                negative_input_ids=negative["input_ids"],
                negative_attention_mask=negative["attention_mask"],
            )
        if "proprio" in raw:
            state = torch.as_tensor(raw["proprio"], dtype=torch.float32)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            if "state" in self.affine:
                scale, offset = self.affine["state"]
                if state.shape[-1] != scale.numel():
                    raise ValueError(
                        "Proprioception dimensions do not match checkpoint statistics."
                    )
                state = (state * scale + offset).clamp(-5, 5)
            result["proprio"] = state
        for key in (
            "condition_video",
            "condition_observation_video",
            "inference",
            "tiled",
            "decode_video",
        ):
            if key in raw:
                result[key] = raw[key]
        return result

    def finish(self, output: dict[str, Any]) -> dict[str, Any]:
        result = dict(output)
        if "action" in result and "action" in self.affine:
            scale, offset = self.affine["action"]
            action = result["action"].float().cpu()
            if action.shape[-1] != scale.numel():
                raise ValueError(
                    "Action dimensions do not match checkpoint statistics."
                )
            result["normalized_action"] = action
            result["action"] = (action - offset) / scale
        return result
