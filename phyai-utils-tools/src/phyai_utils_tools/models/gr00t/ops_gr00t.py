"""GR00T-N1.7 processor ops — pure, dependency-light transforms.

Verbatim port of the deterministic eval image transform, the native Qwen3-VL
image preprocessing (``smart_resize`` + patchify), and the state/action
normalization math from phyai's in-engine ``processor_gr00t_n17``. These are
plain functions over numpy / torch / PIL with **no ``phyai`` dependency**, so the
package stays a workspace leaf.

The image-transform helpers take explicit size/crop primitives instead of a
processor object, so they are reusable and trivially testable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


# Checkpoint embodiment-tag aliases -> canonical lowercase value.
EMBODIMENT_NAME_TO_VALUE: dict[str, str] = {
    "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT": "oxe_droid_relative_eef_relative_joint",
    "XDOF": "xdof_relative_eef_relative_joint",
    "XDOF_SUBTASK": "xdof_relative_eef_relative_joint_subtask",
    "REAL_G1": "real_g1_relative_eef_relative_joints",
    "REAL_R1_PRO_SHARPA": "real_r1_pro_sharpa_relative_eef",
    "REAL_R1_PRO_SHARPA_HUMAN": "real_r1_pro_sharpa_relative_eef_human",
    "REAL_R1_PRO_SHARPA_MAXINSIGHTS": "real_r1_pro_sharpa_relative_eef_maxinsights",
    "REAL_R1_PRO_SHARPA_MECKA": "real_r1_pro_sharpa_relative_eef_mecka",
    "UNITREE_G1": "unitree_g1_full_body_with_waist_height_nav_cmd",
    "UNITREE_G1_SONIC": "unitree_g1_sonic",
    "SIMPLER_ENV_GOOGLE": "simpler_env_google",
    "SIMPLER_ENV_WIDOWX": "simpler_env_widowx",
    "LIBERO_PANDA": "libero_sim",
    "NEW_EMBODIMENT": "new_embodiment",
    "ROBOCASA_GR1_TABLETOP": "robocasa_gr1_tabletop",
    "ROBOCASA_PANDA_OMRON": "robocasa_panda_omron",
}


# Canonical embodiment value -> action-head projector slot.
DEFAULT_EMBODIMENT_ID_MAPPING: dict[str, int] = {
    "oxe_droid_relative_eef_relative_joint": 24,
    "xdof_relative_eef_relative_joint": 27,
    "xdof_relative_eef_relative_joint_subtask": 27,
    "real_g1_relative_eef_relative_joints": 25,
    "real_r1_pro_sharpa_relative_eef": 26,
    "real_r1_pro_sharpa_relative_eef_human": 26,
    "real_r1_pro_sharpa_relative_eef_maxinsights": 26,
    "real_r1_pro_sharpa_relative_eef_mecka": 26,
    "unitree_g1_full_body_with_waist_height_nav_cmd": 25,
    "unitree_g1_sonic": 11,
    "simpler_env_google": 0,
    "simpler_env_widowx": 1,
    "libero_sim": 2,
    "new_embodiment": 10,
    "robocasa_panda_omron": 10,
    "robocasa_gr1_tabletop": 10,
}


# --------------------------------------------------------------------------- #
# Small parsing / loading helpers                                             #
# --------------------------------------------------------------------------- #


def resolve_embodiment_tag(tag: str) -> str:
    """Map a user embodiment tag to its canonical lowercase value."""
    if not tag:
        raise ValueError("embodiment_tag must be a non-empty string.")
    stripped = tag.strip()
    upper = stripped.upper()
    if upper in EMBODIMENT_NAME_TO_VALUE:
        return EMBODIMENT_NAME_TO_VALUE[upper]
    lower = stripped.lower()
    for value in EMBODIMENT_NAME_TO_VALUE.values():
        if value.lower() == lower:
            return value
    return lower


def tuple2(v: list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    """Coerce a length-2 sequence to an ``(int, int)`` tuple (or ``None``)."""
    if v is None:
        return None
    if len(v) != 2:
        raise ValueError(f"expected 2 values, got {v!r}.")
    return (int(v[0]), int(v[1]))


def enum_value(value: Any) -> str:
    """Lowercase the trailing token of an enum-ish string (``A.B`` -> ``b``)."""
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.lower()


def load_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    with path.open("r") as f:
        return json.load(f)


def load_preprocessor_config(
    model_name_or_path: str, *, local_files_only: bool = False
) -> dict[str, Any] | None:
    """Load ``preprocessor_config.json`` from a local dir or the hub cache.

    Returns ``None`` when it cannot be resolved (callers fall back to
    Cosmos-Reason2 defaults). Uses ``huggingface_hub`` for cache resolution —
    no ``transformers`` processor machinery.
    """
    local = Path(model_name_or_path)
    if local.is_dir():
        candidate = local / "preprocessor_config.json"
        return load_json(candidate) if candidate.is_file() else None
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            model_name_or_path,
            "preprocessor_config.json",
            local_files_only=local_files_only,
        )
    except Exception:
        return None
    return load_json(Path(path))


# --------------------------------------------------------------------------- #
# Deterministic eval image transform (-> CHW uint8)                           #
# --------------------------------------------------------------------------- #


def eval_transform_image(
    image: np.ndarray,
    *,
    shortest_image_edge: int | None,
    image_target_size: tuple[int, int] | None,
    image_crop_size: tuple[int, int] | None,
    crop_fraction: float | None,
) -> np.ndarray:
    """Deterministic GR00T eval image transform returning CHW uint8.

    Prefers the cv2 path (matches the official transform); falls back to PIL
    when OpenCV is unavailable.
    """
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8, got {image.dtype}.")
    transformed = _eval_transform_image_cv2(
        image,
        shortest_image_edge=shortest_image_edge,
        image_target_size=image_target_size,
        image_crop_size=image_crop_size,
        crop_fraction=crop_fraction,
    )
    if transformed is not None:
        return transformed
    pil = Image.fromarray(image)
    pil = _letterbox_pad_pil(pil)
    max_size = _resolve_max_size(shortest_image_edge, image_target_size)
    pil = _resize_shortest_edge(pil, max_size)
    fraction = _resolve_crop_fraction(crop_fraction, image_crop_size, image_target_size)
    pil = _fractional_center_crop(pil, float(fraction))
    pil = _resize_shortest_edge(pil, max_size)
    arr = np.asarray(pil, dtype=np.uint8)
    return np.transpose(arr, (2, 0, 1))


def _resolve_max_size(
    shortest_image_edge: int | None, image_target_size: tuple[int, int] | None
) -> int:
    if shortest_image_edge is not None:
        return int(shortest_image_edge)
    if image_target_size is None:
        raise ValueError(
            "image_target_size is required when shortest_image_edge is None."
        )
    return int(image_target_size[0])


def _resolve_crop_fraction(
    crop_fraction: float | None,
    image_crop_size: tuple[int, int] | None,
    image_target_size: tuple[int, int] | None,
) -> float:
    if crop_fraction is not None:
        return float(crop_fraction)
    if image_crop_size is None or image_target_size is None:
        raise ValueError(
            "image_crop_size/image_target_size required when crop_fraction is None."
        )
    return image_crop_size[0] / image_target_size[0]


def _eval_transform_image_cv2(
    image: np.ndarray,
    *,
    shortest_image_edge: int | None,
    image_target_size: tuple[int, int] | None,
    image_crop_size: tuple[int, int] | None,
    crop_fraction: float | None,
) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    image = _letterbox_pad_cv2(image, cv2)
    max_size = _resolve_max_size(shortest_image_edge, image_target_size)
    image = _resize_shortest_edge_cv2(image, int(max_size), cv2)
    fraction = _resolve_crop_fraction(crop_fraction, image_crop_size, image_target_size)
    image = _fractional_center_crop_np(image, float(fraction))
    image = _resize_shortest_edge_cv2(image, int(max_size), cv2)
    return np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))


def _letterbox_pad_cv2(image: np.ndarray, cv2: Any) -> np.ndarray:
    height, width = image.shape[:2]
    if height == width:
        return image
    max_dim = max(height, width)
    pad_h = max_dim - height
    pad_w = max_dim - width
    return cv2.copyMakeBorder(
        image,
        pad_h // 2,
        pad_h - pad_h // 2,
        pad_w // 2,
        pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT,
        value=0,
    )


def _resize_shortest_edge_cv2(image: np.ndarray, max_size: int, cv2: Any) -> np.ndarray:
    height, width = image.shape[:2]
    shortest = min(height, width)
    if shortest == max_size:
        return image
    scale = max_size / shortest
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def _fractional_center_crop_np(image: np.ndarray, crop_fraction: float) -> np.ndarray:
    if not 0.0 < crop_fraction <= 1.0:
        raise ValueError("crop_fraction must be in (0, 1].")
    height, width = image.shape[:2]
    crop_height = max(1, int(height * crop_fraction))
    crop_width = max(1, int(width * crop_fraction))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return image[top : top + crop_height, left : left + crop_width]


def _letterbox_pad_pil(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    max_dim = max(width, height)
    out = Image.new("RGB", (max_dim, max_dim), color=(0, 0, 0))
    out.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
    return out


def _resize_shortest_edge(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    shortest = min(width, height)
    if shortest == max_size:
        return image
    scale = max_size / shortest
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return image.resize(new_size, resample=Image.Resampling.BOX)


def _fractional_center_crop(image: Image.Image, crop_fraction: float) -> Image.Image:
    if not 0.0 < crop_fraction <= 1.0:
        raise ValueError("crop_fraction must be in (0, 1].")
    width, height = image.size
    crop_width = max(1, int(width * crop_fraction))
    crop_height = max(1, int(height * crop_fraction))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


# --------------------------------------------------------------------------- #
# Native Qwen3-VL image preprocessing                                         #
# --------------------------------------------------------------------------- #


def qwen3vl_smart_resize(
    height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Qwen2/3-VL ``smart_resize`` — verbatim from the transformers source.

    Rounds H/W to multiples of ``factor`` while keeping the pixel count in
    ``[min_pixels, max_pixels]`` and the aspect ratio as close as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}."
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen3vl_process_image(
    image_chw: np.ndarray, params: dict[str, Any]
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Native Qwen2-VL image processor for one CHW ``uint8`` image.

    Reproduces ``Qwen2VLImageProcessor._preprocess`` for a single image
    (``grid_t = 1``, temporal axis filled by repeating the frame): resize to a
    ``smart_resize`` grid (BICUBIC, no-op when already aligned), rescale to
    ``[0, 1]``, normalize with ``image_mean``/``image_std``, then reshape /
    permute into flattened patches.

    Returns ``(pixel_values, (grid_t, grid_h, grid_w))`` where ``pixel_values``
    is ``(grid_t*grid_h*grid_w, C*temporal*patch*patch)``.
    """
    patch_size = int(params["patch_size"])
    temporal = int(params["temporal_patch_size"])
    merge = int(params["merge_size"])
    factor = patch_size * merge

    tensor = torch.from_numpy(np.ascontiguousarray(image_chw)).to(torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {tuple(tensor.shape)}.")
    channels, height, width = tensor.shape
    resized_h, resized_w = qwen3vl_smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=int(params["min_pixels"]),
        max_pixels=int(params["max_pixels"]),
    )
    if (resized_h, resized_w) != (height, width):
        from torchvision.transforms import functional as tvf

        tensor = tvf.resize(
            tensor,
            [resized_h, resized_w],
            interpolation=tvf.InterpolationMode.BICUBIC,
            antialias=True,
        )

    tensor = tensor / 255.0
    mean = torch.tensor(params["image_mean"], dtype=torch.float32).view(channels, 1, 1)
    std = torch.tensor(params["image_std"], dtype=torch.float32).view(channels, 1, 1)
    tensor = (tensor - mean) / std

    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    patches = tensor.reshape(
        1,
        channels,
        grid_h // merge,
        merge,
        patch_size,
        grid_w // merge,
        merge,
        patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flatten_patches = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal, -1, -1)
        .reshape(grid_h * grid_w, channels * temporal * patch_size * patch_size)
    )
    return flatten_patches.contiguous(), (1, int(grid_h), int(grid_w))


# --------------------------------------------------------------------------- #
# State / action normalization math                                          #
# --------------------------------------------------------------------------- #


def apply_sin_cos_encoding(values: np.ndarray) -> np.ndarray:
    """Concatenate ``[sin(values), cos(values)]`` along the last axis."""
    return np.concatenate([np.sin(values), np.cos(values)], axis=-1)


def normalize_values_minmax(
    values: np.ndarray, params: dict[str, np.ndarray]
) -> np.ndarray:
    """Min-max normalize to ``[-1, 1]`` (degenerate dims map to 0)."""
    min_vals = params["min"]
    max_vals = params["max"]
    normalized = np.zeros_like(values)
    mask = ~np.isclose(max_vals, min_vals)
    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask]
    )
    normalized[..., mask] = 2 * normalized[..., mask] - 1
    return normalized


def unnormalize_values_minmax(
    normalized_values: np.ndarray, params: dict[str, np.ndarray]
) -> np.ndarray:
    """Invert :func:`normalize_values_minmax` (clips to ``[-1, 1]`` first)."""
    min_vals = params["min"]
    max_vals = params["max"]
    return (np.clip(normalized_values, -1.0, 1.0) + 1.0) / 2.0 * (
        max_vals - min_vals
    ) + min_vals


def normalize_values_meanstd(
    values: np.ndarray, params: dict[str, np.ndarray]
) -> np.ndarray:
    """Mean-std normalize (zero-std dims pass through unchanged)."""
    mean_vals = params["mean"]
    std_vals = params["std"]
    normalized = np.zeros_like(values)
    mask = std_vals != 0
    normalized[..., mask] = (values[..., mask] - mean_vals[..., mask]) / std_vals[
        ..., mask
    ]
    normalized[..., ~mask] = values[..., ~mask]
    return normalized


def unnormalize_values_meanstd(
    normalized_values: np.ndarray, params: dict[str, np.ndarray]
) -> np.ndarray:
    """Invert :func:`normalize_values_meanstd`."""
    mean_vals = params["mean"]
    std_vals = params["std"]
    unnormalized = np.zeros_like(normalized_values)
    mask = std_vals != 0
    unnormalized[..., mask] = (
        normalized_values[..., mask] * std_vals[..., mask] + mean_vals[..., mask]
    )
    unnormalized[..., ~mask] = normalized_values[..., ~mask]
    return unnormalized


def relative_non_eef_to_absolute(
    relative_action: np.ndarray, reference_state: np.ndarray
) -> np.ndarray:
    """Add the last reference state to a relative (non-EEF) action chunk."""
    is_batched = relative_action.ndim == 3
    rel = relative_action if is_batched else relative_action[None]
    state = reference_state
    if state.ndim == 2:
        state = state[None]
    refs = state[:, -1, :]
    absolute = rel + refs[:, None, :]
    return absolute if is_batched else absolute[0]


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero rotation vector.")
    return vector / norm


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert GR00T's row-major 6D rotation representation to a matrix.

    Matches Isaac-GR00T's ``EndEffectorPose._rot6d_to_matrix`` convention: the
    6D vector stores the first two **rows** of the rotation matrix, then uses
    Gram-Schmidt to recover an orthonormal frame.
    """
    rows = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    row1 = _normalize_vector(rows[0])
    row2 = rows[1] - np.dot(row1, rows[1]) * row1
    row2 = _normalize_vector(row2)
    row3 = np.cross(row1, row2)
    return np.vstack([row1, row2, row3])


def matrix_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to GR00T's row-major 6D representation."""
    return np.asarray(rotation_matrix, dtype=np.float64)[:2, :].reshape(6)


def xyz_rot6d_to_homogeneous(action: np.ndarray) -> np.ndarray:
    """Convert one ``XYZ_ROT6D`` action row to a homogeneous transform."""
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (9,):
        raise ValueError(f"XYZ_ROT6D action must have shape (9,), got {values.shape}.")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot6d_to_matrix(values[3:])
    transform[:3, 3] = values[:3]
    return transform


def homogeneous_to_xyz_rot6d(transform: np.ndarray) -> np.ndarray:
    """Convert one homogeneous transform to an ``XYZ_ROT6D`` action row."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"homogeneous transform must have shape (4, 4), got {matrix.shape}."
        )
    return np.concatenate([matrix[:3, 3], matrix_to_rot6d(matrix[:3, :3])])


def relative_eef_to_absolute(
    relative_action: np.ndarray,
    reference_state: np.ndarray,
    *,
    action_format: str,
) -> np.ndarray:
    """Compose relative EEF actions with the last reference EEF state.

    For ``XYZ_ROT6D``, this mirrors Isaac-GR00T's
    ``EndEffectorActionChunk.to_absolute_chunking``:
    ``T_absolute = T_reference @ T_relative`` for every action in the chunk.
    """
    if action_format.lower() != "xyz_rot6d":
        raise NotImplementedError(
            f"Relative EEF decode currently supports XYZ_ROT6D, got {action_format!r}."
        )

    is_batched = relative_action.ndim == 3
    rel = relative_action if is_batched else relative_action[None]
    state = reference_state
    if state.ndim == 2:
        state = state[None]
    if rel.ndim != 3 or state.ndim != 3:
        raise ValueError(
            "relative EEF action/state must have shapes (B,T,D)/(B,S,D) "
            f"or (T,D)/(S,D), got {relative_action.shape} and {reference_state.shape}."
        )
    if rel.shape[-1] != 9 or state.shape[-1] != 9:
        raise ValueError(
            "XYZ_ROT6D relative EEF action/state require last dimension 9, got "
            f"{rel.shape[-1]} and {state.shape[-1]}."
        )

    outputs = []
    for rel_batch, state_batch in zip(rel, state):
        reference = xyz_rot6d_to_homogeneous(state_batch[-1])
        rows = []
        for rel_row in rel_batch:
            absolute = reference @ xyz_rot6d_to_homogeneous(rel_row)
            rows.append(homogeneous_to_xyz_rot6d(absolute))
        outputs.append(np.stack(rows, axis=0))
    absolute = np.stack(outputs, axis=0).astype(np.float32, copy=False)
    return absolute if is_batched else absolute[0]


__all__ = [
    "DEFAULT_EMBODIMENT_ID_MAPPING",
    "EMBODIMENT_NAME_TO_VALUE",
    "apply_sin_cos_encoding",
    "enum_value",
    "eval_transform_image",
    "load_json",
    "load_preprocessor_config",
    "normalize_values_meanstd",
    "normalize_values_minmax",
    "homogeneous_to_xyz_rot6d",
    "matrix_to_rot6d",
    "qwen3vl_process_image",
    "qwen3vl_smart_resize",
    "relative_eef_to_absolute",
    "relative_non_eef_to_absolute",
    "resolve_embodiment_tag",
    "rot6d_to_matrix",
    "tuple2",
    "unnormalize_values_meanstd",
    "unnormalize_values_minmax",
    "xyz_rot6d_to_homogeneous",
]
