"""Cosmos3 policy pipeline steps — image, text, action, and domain processing."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    IMAGES,
    PIXEL_VALUES,
    TASK,
    Transition,
)

# Cosmos3 policy-specific transition keys (local, not shared).
TEXT_IDS = "text_ids"
TEXT_MASK = "text_mask"
NEG_TEXT_IDS = "neg_text_ids"
NEG_TEXT_MASK = "neg_text_mask"
COND_ACTION = "cond_action"
COND_ACTION_INDEXES = "cond_action_indexes"
ACTION_START_FRAME_OFFSET = "action_start_frame_offset"
CAPTION = "caption"
DOMAIN_ID = "domain_id"
MODE = "mode"
ACTION_CHUNK = "action_chunk"
RAW_ACTION_DIM = "raw_action_dim"
VIDEO_SHAPE = "video_shape"
# Scaled pre-pad observation dims (height==image_size), used only to build the
# prompt-metadata resolution sentence (which reports the pre-pad final_h/final_w,
# not the padded target).
META_HEIGHT = "meta_height"
META_WIDTH = "meta_width"


EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "galbot": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "agibot_gear_gripper": 15,
    "agibot_gear_gripper_ext": 15,
    "fractal": 20,
}


def resolve_domain_id(domain: str | int) -> int:
    if isinstance(domain, int):
        if domain < 0:
            raise ValueError(f"domain_id must be non-negative, got {domain}.")
        return domain
    key = str(domain).strip().lower()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise ValueError(
            f"Unknown domain_name={domain!r}; expected one of "
            f"{sorted(EMBODIMENT_TO_DOMAIN_ID)} or pass an int domain_id."
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


# Per-embodiment raw (physical) action width — the true action dimension before
# zero-padding to the model's ``action_dim``.
EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "agibotworld": 29,
    "fractal": 10,
}


def resolve_raw_action_dim(domain: str | int) -> int:
    """Resolve the embodiment's raw action width from its name."""
    if isinstance(domain, int):
        raise ValueError(
            "raw_action_dim cannot be inferred from an integer domain_id; pass "
            "raw_action_dim explicitly or use the embodiment name."
        )
    key = str(domain).strip().lower()
    if key not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise ValueError(
            f"Unknown domain_name={domain!r} for raw_action_dim; expected one of "
            f"{sorted(EMBODIMENT_TO_RAW_ACTION_DIM)} or pass raw_action_dim."
        )
    return EMBODIMENT_TO_RAW_ACTION_DIM[key]


# Predefined per-tier target sizes ``{tier: {aspect: (width, height)}}``. The
# observation frame is snapped to the closest aspect ratio within its tier so the
# grid the model sees matches training.
VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    },
    "704": {
        "1,1": (960, 960),
        "4,3": (1088, 832),
        "3,4": (832, 1088),
        "16,9": (1280, 704),
        "9,16": (704, 1280),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
    "768": {
        "1,1": (1024, 1024),
        "4,3": (1184, 880),
        "3,4": (880, 1184),
        "16,9": (1360, 768),
        "9,16": (768, 1360),
    },
    "1080": {
        "1,1": (1440, 1440),
        "4,3": (1664, 1248),
        "3,4": (1248, 1664),
        "16,9": (1920, 1080),
        "9,16": (1080, 1920),
    },
    "1280": {
        "1,1": (1712, 1712),
        "4,3": (1968, 1472),
        "3,4": (1472, 1968),
        "16,9": (2272, 1280),
        "9,16": (1280, 2272),
    },
    "2048": {
        "1,1": (2728, 2728),
        "4,3": (3160, 2368),
        "3,4": (2368, 3160),
        "16,9": (3640, 2048),
        "9,16": (2048, 3640),
    },
    "gt_2048": {
        "1,1": (5464, 5464),
        "4,3": (6304, 4728),
        "3,4": (4728, 6304),
        "16,9": (7280, 4096),
        "9,16": (4096, 7280),
    },
}

# Exact 768-tier shapes, matched before the ``min_dim`` fallback.
_RESOLUTION_768_SHAPES: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1184, 880),
    (880, 1184),
    (1360, 768),
    (768, 1360),
)


def get_vision_data_resolution(spatial_shape: tuple[int, int]) -> str:
    """Map ``(height, width)`` to a resolution tier key."""
    if spatial_shape in _RESOLUTION_768_SHAPES:
        return "768"
    min_dim = min(spatial_shape[0], spatial_shape[1])
    if min_dim <= 256:
        return "256"
    if min_dim <= 640:
        return "480"
    if min_dim <= 960:
        return "720"
    raise ValueError(f"Unsupported resolution: {spatial_shape}")


def find_closest_target_size(h: int, w: int, resolution: str | int) -> tuple[int, int]:
    """Closest predefined ``(target_w, target_h)`` for input ``(h, w)`` in a tier.

    Selects the aspect ratio whose ``H/W`` is closest to the input.
    """
    resolution = str(resolution)
    if resolution not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"Resolution {resolution!r} not in VIDEO_RES_SIZE_INFO; "
            f"available: {list(VIDEO_RES_SIZE_INFO)}."
        )
    candidates = VIDEO_RES_SIZE_INFO[resolution]
    input_ratio = h / w
    best_key, best_diff = None, float("inf")
    for aspect_key, (cand_w, cand_h) in candidates.items():
        diff = abs(input_ratio - cand_h / cand_w)
        if diff < best_diff:
            best_diff, best_key = diff, aspect_key
    assert best_key is not None
    target_w, target_h = candidates[best_key]
    return target_w, target_h


def resolve_target_size(
    native_h: int, native_w: int, image_size: int
) -> tuple[int, int]:
    """Resolve the snapped ``(target_h, target_w)`` for an observation frame.

    The native frame is first scaled so its height equals ``image_size`` (aspect
    preserved), the resolution tier is chosen from those dims, then the closest
    aspect-ratio target in that tier is returned. The actual pixel resize/pad to
    this target is done downstream by :func:`_resize_and_pad_action_image`
    (scale-down BICUBIC + reflect/edge pad).
    """
    scale = image_size / native_h
    h1, w1 = image_size, max(1, int(round(native_w * scale)))
    tier = get_vision_data_resolution((h1, w1))
    target_w, target_h = find_closest_target_size(h1, w1, tier)
    return target_h, target_w


# Viewpoint -> framing sentence.
DEFAULT_VIEWPOINT_TEMPLATES: dict[str, str] = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}


def _aspect_ratio_str(width: int, height: int) -> str:
    """Canonical ``"w,h"`` aspect-ratio string for the JSON caption."""
    for tier in VIDEO_RES_SIZE_INFO.values():
        for ar, (cand_w, cand_h) in tier.items():
            if width == cand_w and height == cand_h:
                return ar
    divisor = math.gcd(width, height)
    if divisor == 0:
        raise ValueError(f"width/height must be non-zero, got {width}x{height}.")
    return f"{width // divisor},{height // divisor}"


def cosmos3_action_json_caption(
    prompt: str,
    *,
    view_point: str,
    additional_view_description: str | None = None,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
) -> str:
    """Build the structured JSON action caption for the single-observation case
    (no idle-frame metadata).

    ``height``/``width`` are the **padded** target dims reported in ``resolution``;
    ``num_frames`` is the observation/rollout pixel-frame count
    (``action_chunk_size + 1``); duration = ``int(num_frames/fps)`` seconds.
    """
    dur = num_frames / float(fps)
    duration = int(dur) if (dur >= 0 and math.isfinite(dur)) else 0
    end = round(dur) if (dur >= 0 and math.isfinite(dur)) else 0
    minutes, seconds = divmod(end, 60)
    desc = prompt.strip()
    if not desc.endswith((".", "!", "?")):
        desc = f"{desc}."
    out: dict[str, Any] = {}
    framing = DEFAULT_VIEWPOINT_TEMPLATES.get(view_point)
    if additional_view_description:
        if framing:
            separator = " " if framing.endswith(".") else ". "
            framing = framing + separator + additional_view_description.rstrip()
        else:
            framing = additional_view_description.rstrip()
    if framing:
        out["cinematography"] = {"framing": framing}
    out["actions"] = [{"time": f"0:00-{minutes}:{seconds:02d}", "description": desc}]
    out["duration"] = f"{duration}s"
    out["fps"] = float(fps)
    out["resolution"] = {"H": int(height), "W": int(width)}
    out["aspect_ratio"] = _aspect_ratio_str(int(width), int(height))
    return json.dumps(out)


# Native generation prompt metadata templates (cosmos-framework text2video).
GENERATION_DURATION_FPS_TEMPLATE = (
    "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
)
GENERATION_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."


def _parse_json_object_prompt(prompt: str) -> dict | None:
    """Return the parsed dict iff ``prompt`` is a JSON-object string, else ``None``.

    JSON arrays / numbers / strings / nulls stay on the plain-text path.
    """
    try:
        obj = json.loads(prompt)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def cosmos3_generation_caption(
    prompt: str,
    *,
    fps: float,
    num_frames: int,
    height: int,
    width: int,
    aspect_ratio: str | None = None,
) -> str:
    """Append duration/resolution metadata to a generation (T2V/I2V) prompt.

    A JSON-object prompt gets ``duration``/``fps``/``resolution``/``aspect_ratio``
    injected (overwriting existing values); a plain-text prompt gets a duration
    sentence (only when ``num_frames > 1``) and a resolution sentence appended.
    Plain-path duration is ``num_frames / fps`` seconds (``{:.1f}``); JSON-path
    duration is the integer ``int(num_frames / fps)`` seconds.
    """
    obj = _parse_json_object_prompt(prompt)
    if obj is not None:
        meta: dict[str, Any] = {}
        if num_frames > 1:
            meta["duration"] = f"{int(num_frames / fps) if fps > 0 else 0}s"
            meta["fps"] = float(fps)
        else:
            obj.pop("duration", None)
            obj.pop("fps", None)
        meta["resolution"] = {"H": int(height), "W": int(width)}
        if aspect_ratio is not None:
            meta["aspect_ratio"] = aspect_ratio
        obj.update(meta)
        return json.dumps(obj)

    text = prompt.strip()
    if num_frames > 1:
        duration = num_frames / float(fps)
        text = (
            text.rstrip(".")
            + ". "
            + GENERATION_DURATION_FPS_TEMPLATE.format(duration=duration, fps=fps)
        )
    text = text.strip()
    text = (
        text.rstrip(".")
        + ". "
        + GENERATION_RESOLUTION_TEMPLATE.format(height=height, width=width)
    )
    return text


_DEFAULT_NEGATIVE_PROMPT_CACHE: str | None = None


def cosmos3_default_negative_prompt() -> str:
    """Native generation default negative prompt.

    Returns ``json.dumps(json.loads(neg_prompts.json))`` for the bundled
    ``neg_prompts.json`` (the structured "bad video" negative), byte-matching how
    the native sampler loads its ``negative_prompt_file``. Cached after first read.
    """
    global _DEFAULT_NEGATIVE_PROMPT_CACHE
    if _DEFAULT_NEGATIVE_PROMPT_CACHE is None:
        from importlib.resources import files

        raw = (
            files("phyai_utils_tools.models.cosmos3")
            .joinpath("neg_prompts.json")
            .read_text()
        )
        _DEFAULT_NEGATIVE_PROMPT_CACHE = json.dumps(json.loads(raw))
    return _DEFAULT_NEGATIVE_PROMPT_CACHE


def _to_pil_rgb(value: Any):
    """Convert various image formats to PIL RGB."""
    import PIL.Image

    if isinstance(value, str):
        return PIL.Image.open(value).convert("RGB")
    if isinstance(value, PIL.Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = value
        if (
            array.ndim == 3
            and array.shape[0] in (3, 4)
            and array.shape[-1] not in (3, 4)
        ):
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            if array.min() < 0.0 or array.max() > 1.0:
                array = np.clip(array, -1.0, 1.0) * 0.5 + 0.5
            array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return PIL.Image.fromarray(array).convert("RGB")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 0)
        if tensor.is_floating_point():
            if tensor.min().item() < 0.0 or tensor.max().item() > 1.0:
                tensor = tensor.clamp(-1.0, 1.0) * 0.5 + 0.5
            tensor = (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return PIL.Image.fromarray(tensor.numpy()).convert("RGB")
    raise TypeError(
        f"Expected PIL image, numpy array, torch tensor, or path, got {type(value)!r}."
    )


def _resize_and_pad_action_image(
    image, target_h: int, target_w: int
) -> tuple[np.ndarray, int, int]:
    """Resize and pad an action frame, returning pixels and valid content size."""
    import PIL.Image

    img = _to_pil_rgb(image)
    scale = min(target_w / img.width, target_h / img.height, 1.0)
    resize_w = max(1, int(scale * img.width + 0.5))
    resize_h = max(1, int(scale * img.height + 0.5))
    if (resize_w, resize_h) != img.size:
        img = img.resize((resize_w, resize_h), PIL.Image.Resampling.BICUBIC)

    array = np.asarray(img)
    pad_h = target_h - resize_h
    pad_w = target_w - resize_w
    if pad_h == 0 and pad_w == 0:
        return array, resize_h, resize_w
    pad_mode = "reflect" if pad_h < resize_h and pad_w < resize_w else "edge"
    padded = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode=pad_mode)
    return padded, resize_h, resize_w


@ProcessorStepRegistry.register("cosmos3_image_preprocess_step")
class Cosmos3ImagePreprocessStep(ProcessorStep):
    """Load/resize/normalize observation image(s) to pixel tensor.

    For action modes (policy/forward_dynamics): single frame,
    scale-down + reflect-pad.
    For inverse_dynamics: all provided frames.
    Output: [1, 3, T, H, W] in [-1, 1].

    When ``image_size`` is set the target ``(H, W)`` is not taken from the
    explicit ``height``/``width`` but resolved by snapping to the closest
    predefined aspect ratio in the matching resolution tier (height ->
    ``image_size``, then :func:`find_closest_target_size`). The snapped target is
    computed once from the first frame and applied to every frame so the model sees
    one grid. When ``image_size`` is ``None`` the explicit ``height``/``width`` is
    used.
    """

    def __init__(
        self, *, height: int, width: int, mode: str, image_size: int | None = None
    ) -> None:
        self.height = int(height)
        self.width = int(width)
        self.mode = mode
        self.image_size = int(image_size) if image_size is not None else None

    def __call__(self, transition: Transition) -> Transition:
        raw = transition.get(IMAGES)
        if raw is None:
            raise ValueError("Cosmos3ImagePreprocessStep requires an IMAGES entry.")

        if isinstance(raw, list):
            frames = raw
        else:
            frames = [raw]

        # note(chenghua): Resolve one canvas and preserve the valid content size
        # so the transformer can remove reflection padding after VAE encoding.
        # With image_size set, snap the first
        # frame's native dims to the closest predefined aspect ratio; otherwise
        # use the explicit height/width.
        if self.image_size is not None:
            first = _to_pil_rgb(frames[0])
            target_h, target_w = resolve_target_size(
                first.height, first.width, self.image_size
            )
        else:
            target_h, target_w = self.height, self.width

        # Process every frame the caller provides (single image -> 1 frame ->
        # t_lat=1; a video observation -> N frames -> correct t_lat). Only the first
        # 1-2 latent frames are conditioned clean downstream, but the full clip is
        # VAE-encoded, so all provided frames are kept here.
        processed = [
            _resize_and_pad_action_image(frame, target_h, target_w) for frame in frames
        ]
        processed_frames = [item[0] for item in processed]
        content_sizes = [(item[1], item[2]) for item in processed]
        if any(size != content_sizes[0] for size in content_sizes[1:]):
            raise ValueError(
                f"All observation frames must share one content size, got {content_sizes}."
            )
        meta_h, meta_w = content_sizes[0]

        tensors = []
        for arr in processed_frames:
            t = torch.from_numpy(arr).permute(2, 0, 1).float()
            t = t / 127.5 - 1.0
            tensors.append(t)

        pixel_values = torch.stack(tensors, dim=1).unsqueeze(0)

        out = transition.copy()
        out[PIXEL_VALUES] = pixel_values
        out[VIDEO_SHAPE] = (len(processed_frames), target_h, target_w)
        out[META_HEIGHT] = meta_h
        out[META_WIDTH] = meta_w
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "mode": self.mode,
            "image_size": self.image_size,
        }


@ProcessorStepRegistry.register("cosmos3_text_tokenize_step")
class Cosmos3TextTokenizeStep(ProcessorStep):
    """Tokenize prompt + negative prompt via Cosmos3Processor.

    ``prompt_format`` selects how the task text is built before tokenizing:

    * ``"json"`` — the structured JSON action caption
      (:func:`cosmos3_action_json_caption`). Reads the padded target ``(H, W)`` from
      ``video_shape`` and uses ``view_point``/``num_frames``/``fps``.
    * ``"plain"`` — append the viewpoint and optional camera-layout description,
      then duration/FPS + padded resolution when ``append_metadata`` is enabled.

    The negative prompt is left un-augmented (the unconditional branch is the bare
    empty/negative string).
    """

    _DURATION_FPS_TEMPLATE = (
        "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
    )
    _RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."

    def __init__(
        self,
        *,
        tokenizer_name_or_path: str,
        negative_prompt: str = "",
        append_metadata: bool = True,
        prompt_format: str = "plain",
        view_point: str = "ego_view",
        additional_view_description: str | None = None,
        fps: float = 24.0,
        num_frames: int = 17,
    ) -> None:
        from phyai_utils_tools.models.cosmos3.processor_cosmos3 import Cosmos3Processor

        self._proc = Cosmos3Processor(tokenizer_name_or_path)
        self._negative_prompt = negative_prompt
        self._append_metadata = append_metadata
        self._prompt_format = prompt_format
        self._view_point = view_point
        self._additional_view_description = additional_view_description
        self._fps = float(fps)
        self._num_frames = int(num_frames)

    def _json_caption(self, prompt: str, transition: Transition) -> str:
        """Structured JSON caption from the padded target dims + fps/view_point."""
        shape = transition.get(VIDEO_SHAPE)
        height, width = (shape[1], shape[2]) if shape is not None else (None, None)
        return cosmos3_action_json_caption(
            prompt,
            view_point=self._view_point,
            additional_view_description=self._additional_view_description,
            num_frames=self._num_frames,
            fps=self._fps,
            height=height,
            width=width,
        )

    def _augment(self, prompt: str, transition: Transition) -> str:
        """Append native action viewpoint, duration, and resolution text."""
        viewpoint = DEFAULT_VIEWPOINT_TEMPLATES.get(self._view_point)
        if self._additional_view_description:
            if viewpoint:
                separator = " " if viewpoint.endswith(".") else ". "
                viewpoint = (
                    viewpoint + separator + self._additional_view_description.rstrip()
                )
            else:
                viewpoint = self._additional_view_description.rstrip()
        if viewpoint:
            sep = " " if prompt.rstrip().endswith(".") else ". "
            prompt = prompt + sep + viewpoint
        if not self._append_metadata:
            return prompt

        shape = transition.get(VIDEO_SHAPE)
        height, width = (shape[1], shape[2]) if shape is not None else (None, None)
        duration = int(self._num_frames / self._fps) if self._fps > 0 else 0
        sep = " " if prompt.rstrip().endswith(".") else ". "
        prompt = (
            prompt
            + sep
            + self._DURATION_FPS_TEMPLATE.format(duration=duration, fps=self._fps)
        )
        if height is not None and width is not None:
            sep = " " if prompt.rstrip().endswith(".") else ". "
            prompt = (
                prompt
                + sep
                + self._RESOLUTION_TEMPLATE.format(height=height, width=width)
            )
        return prompt

    def __call__(self, transition: Transition) -> Transition:
        prompt = transition.get(TASK)
        if prompt is None:
            raise ValueError("Cosmos3TextTokenizeStep requires a TASK entry.")
        if isinstance(prompt, list):
            prompt = prompt[0]
        if self._prompt_format == "json":
            prompt = self._json_caption(prompt, transition)
        elif self._prompt_format == "plain":
            prompt = self._augment(prompt, transition)
        else:
            raise ValueError(
                f"prompt_format must be 'plain' or 'json', got {self._prompt_format!r}."
            )

        cond, uncond = self._proc.tokenize_pair(
            prompt, self._negative_prompt, device="cpu"
        )
        out = transition.copy()
        out[CAPTION] = prompt
        out[TEXT_IDS] = cond.text_ids
        out[TEXT_MASK] = cond.text_mask
        out[NEG_TEXT_IDS] = uncond.text_ids
        out[NEG_TEXT_MASK] = uncond.text_mask
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "negative_prompt": self._negative_prompt,
            "append_metadata": self._append_metadata,
            "prompt_format": self._prompt_format,
            "view_point": self._view_point,
            "additional_view_description": self._additional_view_description,
            "fps": self._fps,
            "num_frames": self._num_frames,
        }


@ProcessorStepRegistry.register("cosmos3_action_pad_step")
class Cosmos3ActionPadStep(ProcessorStep):
    """Build the padded model action stream and its clean prefix."""

    def __init__(
        self,
        *,
        action_chunk_size: int,
        raw_action_dim: int,
        action_dim: int = 64,
        mode: str = "policy",
        action_history_length: int = 0,
    ) -> None:
        self.action_chunk_size = int(action_chunk_size)
        self.raw_action_dim = int(raw_action_dim)
        self.action_dim = int(action_dim)
        self.mode = mode
        self.action_history_length = int(action_history_length)
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive.")
        if self.action_history_length < 0:
            raise ValueError("action_history_length must be non-negative.")

    def __call__(self, transition: Transition) -> Transition:
        out = transition.copy()
        total_action_length = self.action_chunk_size + self.action_history_length
        out[ACTION_CHUNK] = total_action_length
        out[RAW_ACTION_DIM] = self.raw_action_dim
        out[ACTION_START_FRAME_OFFSET] = 1 - self.action_history_length

        if self.mode == "forward_dynamics":
            clean_length = total_action_length
        else:
            clean_length = self.action_history_length
        out[COND_ACTION_INDEXES] = tuple(range(clean_length))

        if clean_length == 0:
            out[COND_ACTION] = None
            return out

        raw_action = transition.get(COND_ACTION)
        if raw_action is None:
            raise ValueError(
                f"{self.mode} mode with {clean_length} conditioned action rows "
                "requires a 'cond_action' entry."
            )
        if isinstance(raw_action, (list, np.ndarray)):
            raw_action = torch.as_tensor(raw_action, dtype=torch.float32)
        if not isinstance(raw_action, torch.Tensor):
            raise TypeError(
                "cond_action must be a torch tensor, numpy array, or nested list."
            )
        if raw_action.ndim == 3 and raw_action.shape[0] == 1:
            raw_action = raw_action.squeeze(0)
        if raw_action.ndim != 2:
            raise ValueError(
                f"cond_action must have shape [T,D] or [1,T,D], got {tuple(raw_action.shape)}."
            )

        if raw_action.shape[0] < clean_length:
            if self.mode != "forward_dynamics":
                raise ValueError(
                    f"cond_action has {raw_action.shape[0]} rows, but "
                    f"action_history_length={clean_length}."
                )
            pad = raw_action[-1:].repeat(clean_length - raw_action.shape[0], 1)
            raw_action = torch.cat([raw_action, pad], dim=0)
        elif raw_action.shape[0] > clean_length:
            raw_action = raw_action[:clean_length]

        padded = torch.zeros(total_action_length, self.action_dim, dtype=torch.float32)
        dim = min(raw_action.shape[-1], self.action_dim)
        padded[:clean_length, :dim] = raw_action[:clean_length, :dim]

        out[COND_ACTION] = padded.unsqueeze(0)
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "action_chunk_size": self.action_chunk_size,
            "raw_action_dim": self.raw_action_dim,
            "action_dim": self.action_dim,
            "mode": self.mode,
            "action_history_length": self.action_history_length,
        }


@ProcessorStepRegistry.register("cosmos3_domain_resolve_step")
class Cosmos3DomainResolveStep(ProcessorStep):
    """Resolve domain_name string to integer domain_id."""

    def __call__(self, transition: Transition) -> Transition:
        domain = transition.get(DOMAIN_ID)
        if domain is None:
            raise ValueError("Cosmos3DomainResolveStep requires a DOMAIN_ID entry.")
        out = transition.copy()
        out[DOMAIN_ID] = resolve_domain_id(domain)
        return out


__all__ = [
    "ACTION_CHUNK",
    "ACTION_START_FRAME_OFFSET",
    "CAPTION",
    "COND_ACTION",
    "COND_ACTION_INDEXES",
    "Cosmos3ActionPadStep",
    "Cosmos3DomainResolveStep",
    "Cosmos3ImagePreprocessStep",
    "Cosmos3TextTokenizeStep",
    "DOMAIN_ID",
    "EMBODIMENT_TO_DOMAIN_ID",
    "EMBODIMENT_TO_RAW_ACTION_DIM",
    "MODE",
    "NEG_TEXT_IDS",
    "NEG_TEXT_MASK",
    "RAW_ACTION_DIM",
    "TEXT_IDS",
    "TEXT_MASK",
    "VIDEO_SHAPE",
    "cosmos3_action_json_caption",
    "resolve_domain_id",
    "resolve_raw_action_dim",
]
