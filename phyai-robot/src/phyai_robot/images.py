"""Model-size RGB images for camera decoders; requires the 'images' extra (Pillow).

Design Rationale:
    Decoders invoke these resizing utilities directly on their receive worker threads.
    As a consequence:
    1. Memory Footprint: Stored observation cache samples are already small
       (e.g. 224x224 RGB = ~150 KB) rather than massive uncompressed camera frames
       (e.g. 2560x1984 NV12 = 7.6 MB).
    2. Zero Main-Thread Jitter: Image downscaling and padding occur off the control thread,
       preventing CPU-heavy bilinear interpolation from stalling actuator command dispatch.
    3. Numerical Parity: The letterboxing implementation exactly matches the PIL backend
       of PhyAI's PI05Processor, ensuring that inputs pre-resized at the client arrive at
       the model server without further numerical alteration or redundant scaling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _pil():
    """Dynamically import and return PIL.Image; raise helpful error if missing."""
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise ImportError(
            "image resizing requires Pillow; install phyai-robot[images]"
        ) from error
    return Image


def _check_rgb(image: NDArray[np.uint8]) -> None:
    """Validate that input array is an 8-bit unsigned integer HWC RGB image."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected an RGB uint8 HWC image")


def resize(image: NDArray[np.uint8], height: int, width: int) -> NDArray[np.uint8]:
    """Resize an RGB image to exact dimensions (height, width) with PIL bilinear filtering.

    Notice:
        Aspect ratio is intentionally not preserved. This is used specifically to reproduce
        training dataset recording pipelines that stretched native camera frames into fixed
        tile dimensions (such as Tianji Marvin Pro stretching 960x744 mosaic tiles to 960x540).

    Args:
        image: Source RGB uint8 array shaped (H, W, 3).
        height: Target height in positive pixels.
        width: Target width in positive pixels.

    Returns:
        New contiguous RGB uint8 array shaped (height, width, 3) owning its memory.
    """
    _check_rgb(image)
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("height and width must be positive integers")
    if image.shape[:2] == (height, width):
        return np.array(image)
    Image = _pil()
    resized = Image.fromarray(np.ascontiguousarray(image)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    return np.asarray(resized).copy()


def resize_with_pad(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    """Aspect-preserving resize with PIL bilinear filtering and centered black padding.

    Scaling & Padding Geometry:
        - Downscales the input by the larger of the two axis ratios: ratio = max(W/size, H/size).
        - The longest dimension matches `size`; the shorter dimension retains its aspect ratio
          under integer truncation.
        - Centered in a black (zeros) square canvas of shape (size, size, 3).
        - Matches the PIL image_resize_backend of PhyAI's PI05Processor, ensuring identical
          pixel values between client-side pre-resizing and server-side model inputs.

    Args:
        image: Source RGB uint8 array shaped (H, W, 3).
        size: Target square dimension in positive pixels (e.g. 224).

    Returns:
        Contiguous RGB uint8 array shaped (size, size, 3) owning its memory.
    """
    _check_rgb(image)
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer")
    height, width = image.shape[:2]
    if (height, width) == (size, size):
        return np.array(image)
    Image = _pil()
    ratio = max(width / size, height / size)
    resized = Image.fromarray(np.ascontiguousarray(image)).resize(
        (int(width / ratio), int(height / ratio)), Image.Resampling.BILINEAR
    )
    padded = Image.new("RGB", (size, size))
    # Paste centered within the square canvas; empty borders remain 0 (black).
    padded.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.array(padded)
