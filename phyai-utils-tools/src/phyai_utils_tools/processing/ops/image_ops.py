"""Image preprocessing ops — resize-with-pad and pixel normalization.

``resize_with_pad`` is the openpi / lerobot ``resize_with_pad_torch`` port
(channels-first ``(B, C, H, W)`` only), moved out of ``phyai`` so all image
preprocessing lives in one place. ``normalize_pixels`` is the SigLIP
``[0, 1] -> [-1, 1]`` map that the model expects but ``phyai`` never applied
(callers used to pre-normalize); it is provided here so the processor can own
the full raw-image -> model-ready path.

These are pure tensor functions; the :class:`~phyai_utils_tools.processing.pipeline.ProcessorStep`
wrappers live in :mod:`phyai_utils_tools.processing.steps.image_steps`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_with_pad(
    images: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    mode: str = "bilinear",
    pad_value: float = 0.0,
    backend: str = "torch",
) -> torch.Tensor:
    """Aspect-preserving resize of ``(B, C, H, W)`` images, padded to target.

    The default torch backend follows openpi / lerobot's tensor operation.
    The optional PIL backend preserves PIL's uint8 bilinear rounding for RGB
    checkpoints trained with that transform. Both use channels-first inputs.
    The image is downscaled by the larger of the two axis ratios so it fits
    inside ``target_h`` x ``target_w`` without distortion, then symmetrically
    padded with ``pad_value`` to exactly the target size. Inputs already at the
    target size are returned unchanged (fast path).

    The float path does **not** clamp to ``[0, 1]`` (pixels may already be in
    the model's ``[-1, 1]`` range); bilinear interpolation is a convex
    combination so it cannot overshoot the input range. ``uint8`` inputs are
    rounded and clamped to ``[0, 255]`` (range-agnostic).
    """
    if images.dim() != 4:
        raise ValueError(
            f"resize_with_pad expects 4-D (B, C, H, W); got shape "
            f"{tuple(images.shape)}."
        )
    _, _, cur_h, cur_w = images.shape
    if backend not in {"torch", "pil"}:
        raise ValueError(f"Unknown resize backend: {backend!r}")
    if min(cur_h, cur_w, target_h, target_w) <= 0:
        raise ValueError("Image dimensions must be positive")
    if backend == "pil" and (images.dtype != torch.uint8 or images.shape[1] != 3):
        raise ValueError("PIL resize requires RGB uint8 images")
    if cur_h == target_h and cur_w == target_w:
        return images

    ratio = max(cur_w / target_w, cur_h / target_h)
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)
    if min(resized_h, resized_w) < 1:
        raise ValueError("Image aspect ratio is too extreme for the target size")

    if backend == "pil":
        import numpy as np
        from PIL import Image

        if mode != "bilinear":
            raise ValueError("PIL resize currently supports bilinear interpolation")
        processed = []
        for image in images.detach().cpu().permute(0, 2, 3, 1).numpy():
            resized = Image.fromarray(image).resize(
                (resized_w, resized_h), Image.Resampling.BILINEAR
            )
            canvas = Image.new("RGB", (target_w, target_h), (int(pad_value),) * 3)
            canvas.paste(
                resized, ((target_w - resized_w) // 2, (target_h - resized_h) // 2)
            )
            processed.append(torch.from_numpy(np.array(canvas)).permute(2, 0, 1))
        return torch.stack(processed).to(images.device)

    resized = F.interpolate(
        images,
        size=(resized_h, resized_w),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized = torch.round(resized).clamp(0, 255).to(torch.uint8)
    elif not images.dtype.is_floating_point:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, rem_h = divmod(target_h - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(target_w - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    # F.pad order for the last two dims: (left, right, top, bottom).
    return F.pad(
        resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=pad_value
    )


def normalize_pixels(images: torch.Tensor) -> torch.Tensor:
    """Map float ``[0, 1]`` or uint8 ``[0, 255]`` pixels to ``[-1, 1]``.

    ``images * 2 - 1``. Mirrors lerobot's ``img * 2.0 - 1.0`` step. Apply only
    to raw uint8 or float ``[0, 1]`` images; images already in ``[-1, 1]``
    should skip this (the processor exposes it as an optional step).
    """
    if images.dtype == torch.uint8:
        images = images.to(torch.float32) / 255.0
    return images * 2.0 - 1.0


__all__ = ["normalize_pixels", "resize_with_pad"]
