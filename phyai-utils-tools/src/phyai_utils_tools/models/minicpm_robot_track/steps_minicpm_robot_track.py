"""Reference-compatible image and text steps for MiniCPM-RobotTrack."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    IMAGES,
    INPUT_IDS,
    LANG_LENS,
    PIXEL_VALUES,
    TASK,
    Transition,
)


def to_uint8_rgb(frame: Any) -> np.ndarray:
    """Convert one HWC/CHW image to contiguous uint8 RGB."""

    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"), dtype=np.uint8, copy=True)
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().to(device="cpu").numpy()
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"Each frame must be 3-D, got shape {array.shape}.")
    if array.shape[-1] != 3:
        if array.shape[0] != 3:
            raise ValueError(
                f"Each frame must contain three RGB channels, got {array.shape}."
            )
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Floating-point frames must contain only finite values.")
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                "Floating-point frames must use the [0, 1] range; "
                f"got [{minimum}, {maximum}]."
            )
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def split_frames(images: Any) -> list[Any]:
    """Split a supported frame container without changing frame values."""

    if isinstance(images, (torch.Tensor, np.ndarray)):
        if images.ndim == 5:
            if images.shape[0] != 1:
                raise ValueError("RobotTrack preprocessing supports batch_size=1 only.")
            images = images[0]
        if images.ndim == 3:
            images = images[None]
        if images.ndim != 4:
            raise ValueError(
                "images must be [T,H,W,3], [T,3,H,W], or have one leading batch "
                f"dimension; got {tuple(images.shape)}."
            )
        return [images[index] for index in range(images.shape[0])]
    if isinstance(images, Image.Image):
        return [images]
    return list(images)


def prepare_frame(frame: Any, image_size: int) -> np.ndarray:
    """Convert and resize one frame with the reference PIL implementation."""

    rgb = to_uint8_rgb(frame)
    image = Image.fromarray(rgb, mode="RGB")
    if image.size != (image_size, image_size):
        bicubic = getattr(Image, "Resampling", Image).BICUBIC
        image = image.resize((image_size, image_size), bicubic)
    return np.asarray(image, dtype=np.uint8)


@ProcessorStepRegistry.register("minicpm_robot_track_image_prepare_step")
@dataclass
class MiniCPMRobotTrackImagePrepareStep(ProcessorStep):
    image_size: int = 384
    history_frames: int = 31
    resize_workers: int | None = None
    _resize_executor: ThreadPoolExecutor | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.resize_workers is not None and self.resize_workers <= 0:
            raise ValueError("resize_workers must be positive or None.")
        workers = self.resize_workers
        if workers is None:
            workers = min(8, os.cpu_count() or 1)
        if workers > 1:
            self._resize_executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="robottrack-resize",
            )

    def __call__(self, transition: Transition) -> Transition:
        if IMAGES not in transition:
            raise ValueError("MiniCPM-RobotTrack preprocessing requires `images`.")
        frames = split_frames(transition[IMAGES])
        expected_full_window = self.history_frames + 1
        if len(frames) not in (1, expected_full_window):
            raise ValueError(
                f"Expected 1 incremental frame or {expected_full_window} complete-window "
                f"frames, got {len(frames)}."
            )
        if len(frames) == 1 or self._resize_executor is None:
            prepared = [prepare_frame(frame, self.image_size) for frame in frames]
        else:
            prepared = list(
                self._resize_executor.map(
                    prepare_frame,
                    frames,
                    [self.image_size] * len(frames),
                )
            )
        out = transition.copy()
        out[PIXEL_VALUES] = torch.from_numpy(np.stack(prepared, axis=0))
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "history_frames": self.history_frames,
            "resize_workers": self.resize_workers,
        }

    def close(self) -> None:
        """Release resize worker threads; safe to call more than once."""

        if self._resize_executor is not None:
            self._resize_executor.shutdown(wait=True)
            self._resize_executor = None


@ProcessorStepRegistry.register("minicpm_robot_track_tokenizer_step")
@dataclass
class MiniCPMRobotTrackTokenizerStep(ProcessorStep):
    tokenizer: Any = field(repr=False, default=None)
    text_capacity: int = 35
    tokenizer_name: str | None = None

    def __call__(self, transition: Transition) -> Transition:
        if self.tokenizer is None:
            raise ValueError("MiniCPMRobotTrackTokenizerStep requires a tokenizer.")
        tasks = transition.get(TASK)
        if tasks is None:
            raise ValueError("MiniCPM-RobotTrack preprocessing requires `task`.")
        if isinstance(tasks, str):
            tasks = [tasks]
        tasks = list(tasks)
        if len(tasks) != 1:
            raise ValueError(
                "MiniCPM-RobotTrack preprocessing supports one task at a time."
            )
        encoded = self.tokenizer(
            tasks,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_capacity,
        )
        input_ids = encoded["input_ids"].to(torch.long)
        attention_mask = encoded["attention_mask"].to(torch.bool)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")
        compact_ids = torch.full(
            (1, self.text_capacity), int(pad_token_id), dtype=torch.long
        )
        valid_ids = input_ids[0, attention_mask[0]]
        if valid_ids.numel() == 0:
            raise ValueError("The RobotTrack task produced zero valid tokens.")
        compact_ids[0, : valid_ids.numel()] = valid_ids
        out = transition.copy()
        out[INPUT_IDS] = compact_ids
        out[LANG_LENS] = torch.tensor([valid_ids.numel()], dtype=torch.long)
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "text_capacity": self.text_capacity,
            "tokenizer_name": self.tokenizer_name,
        }


__all__ = [
    "MiniCPMRobotTrackImagePrepareStep",
    "MiniCPMRobotTrackTokenizerStep",
    "prepare_frame",
    "split_frames",
    "to_uint8_rgb",
]
