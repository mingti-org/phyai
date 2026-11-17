"""Image processing steps — stack cameras, resize-with-pad, normalize pixels.

Each is a registered :class:`~phyai_utils_tools.processing.pipeline.ProcessorStep`
wrapping a pure op from :mod:`phyai_utils_tools.processing.ops.image_ops`. They
read :data:`~phyai_utils_tools.processing.transition.IMAGES` (a list of
per-camera ``(B, C, H, W)`` tensors or a stacked ``(B, n, C, H, W)`` tensor) and
write the canonical :data:`~phyai_utils_tools.processing.transition.PIXEL_VALUES`
``(B, n, C, target, target)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from phyai_utils_tools.processing.ops.image_ops import (
    normalize_pixels,
    resize_with_pad,
)
from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import IMAGES, PIXEL_VALUES, Transition


@ProcessorStepRegistry.register("resize_with_pad_step")
@dataclass
class ResizeWithPadStep(ProcessorStep):
    """Resize each camera to ``target_size`` x ``target_size`` (aspect-preserving + pad).

    Reads ``IMAGES`` (list of ``(B, C, H, W)`` or stacked ``(B, n, C, H, W)``),
    resize-with-pads each camera, and writes the stacked canonical
    ``PIXEL_VALUES`` ``(B, num_images, C, target, target)``. Validates the camera
    count and channels (raising ``ValueError`` on mismatch). Already-square
    ``target_size`` inputs hit the op's fast path (no copy).
    """

    target_size: int
    num_images: int
    num_channels: int
    pad_value: float = 0.0
    backend: str = "torch"

    def __call__(self, transition: Transition) -> Transition:
        images = transition[IMAGES]
        if isinstance(images, torch.Tensor):
            if images.dim() != 5:
                raise ValueError(
                    f"stacked images must be 5-D (B, num_images, C, H, W); got "
                    f"shape {tuple(images.shape)}."
                )
            if images.shape[1] != self.num_images:
                raise ValueError(
                    f"stacked images camera dim {images.shape[1]} != "
                    f"num_images={self.num_images}."
                )
            per_camera = [images[:, i] for i in range(self.num_images)]
        else:
            per_camera = list(images)
            if len(per_camera) != self.num_images:
                raise ValueError(
                    f"expected {self.num_images} camera tensors, got {len(per_camera)}."
                )

        processed: list[torch.Tensor] = []
        for i, cam in enumerate(per_camera):
            if cam.dim() != 4:
                raise ValueError(
                    f"camera {i} images must be 4-D (B, C, H, W); got shape "
                    f"{tuple(cam.shape)}."
                )
            if cam.shape[1] != self.num_channels:
                raise ValueError(
                    f"camera {i} has {cam.shape[1]} channels; expected "
                    f"num_channels={self.num_channels}."
                )
            processed.append(
                resize_with_pad(
                    cam,
                    self.target_size,
                    self.target_size,
                    pad_value=self.pad_value,
                    backend=self.backend,
                )
            )

        # (num_images, B, C, T, T) -> (B, num_images, C, T, T)
        out = transition.copy()
        out[PIXEL_VALUES] = torch.stack(processed, dim=0).transpose(0, 1).contiguous()
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "target_size": self.target_size,
            "num_images": self.num_images,
            "num_channels": self.num_channels,
            "pad_value": self.pad_value,
            **({"backend": self.backend} if self.backend != "torch" else {}),
        }


@ProcessorStepRegistry.register("normalize_image_step")
@dataclass
class NormalizeImageStep(ProcessorStep):
    """Map float ``[0, 1]`` or uint8 ``[0, 255]`` pixels to SigLIP ``[-1, 1]``.

    Optional: include this step for raw uint8 or float ``[0, 1]`` pixels.
    Operates in place on the canonical ``PIXEL_VALUES`` produced by
    :class:`ResizeWithPadStep`.
    """

    def __call__(self, transition: Transition) -> Transition:
        out = transition.copy()
        out[PIXEL_VALUES] = normalize_pixels(out[PIXEL_VALUES])
        return out


__all__ = ["NormalizeImageStep", "ResizeWithPadStep"]
