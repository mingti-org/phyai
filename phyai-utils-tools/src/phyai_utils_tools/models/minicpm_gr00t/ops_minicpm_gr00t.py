"""Pure MiniCPM-GR00T preprocessing ops (image / prompt / state).

These mirror the reference EmbodyEvalKit adapter byte-for-byte so the phyai
inputs stay numerically identical to the original inference stack:

* :func:`to_uint8_image` — accept torch / numpy / PIL-compatible HWC images,
  convert to contiguous ``uint8`` RGB and PIL-resize to the model input size.
* :func:`format_prompt` — insert the instruction into the checkpoint's prompt
  template, passing through instructions that are already fully formatted.
* :func:`state_to_batched_tensor` — canonicalize the proprioceptive state to
  the ``(1, 1, state_dim)`` float32 tensor the action head expects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE = (
    "The robot is LIBERO Franka, a simulated single-arm Franka manipulator. "
    "Its action control method is absolute single-arm end-effector pose in the "
    "unified 80D layout with gripper closed command, and its action FPS is 20 Hz. "
    "Task: {instruction}"
)


def to_uint8_image(image: Any, image_size: tuple[int, int] | None = None) -> np.ndarray:
    """Convert one HWC RGB image to contiguous uint8, optionally PIL-resized.

    Float inputs in ``[0, 1]`` are scaled by 255 before clipping; everything is
    clipped to ``[0, 255]``. The optional resize uses ``PIL.Image.resize`` so
    interpolation matches the checkpoint's training-time preprocessing exactly.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "numpy"):
        image = image.numpy()
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 image, got shape {arr.shape}.")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if image_size is not None:
        arr = np.asarray(Image.fromarray(arr).resize(tuple(image_size)))
    # PIL-backed arrays are read-only; hand out a writable contiguous copy so
    # downstream torch.from_numpy never sees a non-writable buffer.
    return np.array(arr, copy=True, order="C")


def format_prompt(instruction: str, template: str) -> str:
    """Fill ``template`` with ``instruction`` unless it is already a full prompt.

    Fully formatted prompts (produced by callers that render the template
    themselves) are detected by the unified-80D marker and passed through, the
    same rule the reference adapter applies.
    """
    if "unified 80D layout" in instruction and "Task:" in instruction:
        return instruction
    return template.format(instruction=instruction)


def state_to_batched_tensor(state: Any, state_dim: int) -> torch.Tensor:
    """Canonicalize the state to a ``(1, 1, state_dim)`` float32 tensor.

    Accepts ``(D,)``, ``(T, D)``, or ``(B, T, D)`` array-likes; the batch and
    time dims must already be 1 (the phyai MiniCPM-GR00T plugin is batch-one).
    """
    if state is None:
        raise ValueError("MiniCPM-GR00T inference requires a state vector.")
    if isinstance(state, torch.Tensor):
        arr = state.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(state, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        arr = arr.reshape(1, *arr.shape)
    elif arr.ndim != 3:
        raise ValueError(
            f"Expected state shape (D,), (T, D), or (B, T, D); got {arr.shape}."
        )
    if arr.shape[0] != 1 or arr.shape[1] != 1:
        raise ValueError(
            f"MiniCPM-GR00T expects a single (batch=1, T=1) state, got {arr.shape}."
        )
    if arr.shape[-1] != state_dim:
        raise ValueError(
            f"MiniCPM-GR00T state must be {state_dim}D, got {arr.shape[-1]}D."
        )
    # MessagePack and PIL-backed arrays can be contiguous but read-only.
    # torch.from_numpy requires writable storage for defined mutation semantics.
    return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True, order="C"))


__all__ = [
    "MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE",
    "format_prompt",
    "state_to_batched_tensor",
    "to_uint8_image",
]
