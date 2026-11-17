"""Select the causal pixel prefix consumed by a policy request."""

from __future__ import annotations


def conditioning_pixel_count(
    total_pixels: int,
    latent_frames: int,
    frame_indexes: tuple[int, ...] | None,
    temporal_scale: int,
) -> int:
    """Keep unusual selections on the full encode path and its original errors."""
    indexes = (0,) if frame_indexes is None else tuple(frame_indexes)
    if temporal_scale != 4 or type(latent_frames) is not int or latent_frames <= 0:
        return total_pixels
    if not indexes or any(
        type(i) is not int or not 0 <= i < latent_frames for i in indexes
    ):
        return total_pixels
    required = 1 + temporal_scale * max(indexes)
    return required if 0 < required < total_pixels else total_pixels
