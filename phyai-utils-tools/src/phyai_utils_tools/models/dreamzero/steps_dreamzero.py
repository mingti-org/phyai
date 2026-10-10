"""DreamZero processor steps."""

from __future__ import annotations

import ast
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as vision_transforms

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    ACTION,
    PROMPT,
    STATE,
    TASK,
    Transition,
)
from phyai_utils_tools.processing.types import NormalizationMode

VIDEO = "video"
IMAGES = "images"
STATE_MASK = "state_mask"
ACTION_MASK = "action_mask"
EMBODIMENT_ID = "embodiment_id"
EMBODIMENT_TAG = "embodiment_tag"
OBS = "obs"
TEXT = "text"
TEXT_ATTENTION_MASK = "text_attention_mask"
TEXT_NEGATIVE = "text_negative"
TEXT_ATTENTION_MASK_NEGATIVE = "text_attention_mask_negative"

DREAMZERO_STATE_FEATURE = "state"
DREAMZERO_ACTION_FEATURE = "action"
DREAMZERO_DEFAULT_STATE_KEYS: tuple[str, ...] = ("joint_position", "gripper_position")
DREAMZERO_DEFAULT_ACTION_KEYS: tuple[str, ...] = ("joint_position", "gripper_position")
DREAMZERO_DEFAULT_ACTION_NORMALIZATION = "q99"

DREAMZERO_DEFAULT_NEGATIVE_PROMPT = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, "
    "style, artwork, painting, image, still, grayscale, dull, worst quality, "
    "low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, "
    "bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant "
    "image, cluttered background, three legs, many people in the background, "
    "walking backwards."
)

DREAMZERO_DEFAULT_EMBODIMENT_TAG_MAPPING: dict[str, int] = {
    "real_gr1_arms_only": 0,
    "real_gr1_arms_only_annotated": 1,
    "real_gr1_arms_waist": 2,
    "real_gr1_arms_waist_annotated": 3,
    "dexmg_gr1_arms_only_inspire": 4,
    "dexmg_gr1_arms_only_fourier": 5,
    "dexmg_gr1_arms_waist_fourier": 6,
    "robocasa_single_arm": 7,
    "onex_eve_gripper": 8,
    "robocasa_gr1_arms_only_inspire_hands": 9,
    "robocasa_gr1_arms_only_fourier_hands": 10,
    "robocasa_gr1_fixed_lower_body_inspire_hands": 11,
    "robocasa_gr1_fixed_lower_body_fourier_hands": 12,
    "robocasa_panda_omron": 13,
    "gr1_unified_segmentation": 14,
    "robocasa_bimanual_panda_parallel_gripper": 15,
    "robocasa_bimanual_panda_inspire_hand": 16,
    "oxe_droid": 17,
    "oxe_fractal": 18,
    "oxe_language_table": 19,
    "oxe_bridge": 20,
    "real_panda_single_arm": 21,
    "xdof": 22,
    "hot3d_hands_only": 23,
    "gr1_unified": 24,
    "robocasa_gr1_arms_waist_fourier_hands": 25,
    "agibot": 26,
    "lapa": 27,
    "oxe_mutex": 28,
    "oxe_roboset": 29,
    "oxe_plex": 30,
    "dream": 31,
    "language_table_sim": 7,
    "gr1_isaac": 0,
    "sim_behavior_r1_pro": 31,
    "mecka_hands": 27,
    "real_r1_pro_sharpa": 28,
}


def whitespace_clean(text: str) -> str:
    """Match DreamZero's whitespace-clean tokenizer input path."""
    try:
        import ftfy
    except ImportError:
        ftfy = None
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


def normalize_prompt_item(item: Any) -> str:
    """Match DreamZero collate's permissive prompt scalar/list handling."""
    if isinstance(item, np.ndarray):
        item = item.item() if item.size == 1 else item[0]
    if isinstance(item, list):
        item = item[0]
    if isinstance(item, tuple):
        item = item[0]
    if isinstance(item, str):
        try:
            parsed_item = ast.literal_eval(item)
        except (ValueError, SyntaxError, TypeError):
            return item
        if isinstance(parsed_item, (list, tuple)):
            return str(parsed_item[0])
        return str(parsed_item)
    return str(item)


def format_dreamzero_prompt(
    prompt: Any,
    *,
    embodiment_id: int | None,
    embodiment_tag_mapping: dict[str, int],
    num_views: int,
) -> str:
    """Apply DreamZero's embodiment-specific language prompt template."""
    item = normalize_prompt_item(prompt)
    item_lower = str(item).lower()
    if embodiment_id is None:
        return str(item)

    if (
        num_views > 1 and embodiment_id == embodiment_tag_mapping.get("agibot")
    ) or embodiment_id == embodiment_tag_mapping.get("xdof"):
        return (
            "A multi-view video shows that a robot "
            + item_lower
            + " The video is split into four views: The top-left view shows "
            "the camera view from the robot's head, the top-right view shows "
            "the camera view from the right hand, the bottom-left view shows "
            "the camera view from the left hand, and the bottom-right view is "
            "a black screen (inactive view). The robot " + item_lower
        )
    if embodiment_id == embodiment_tag_mapping.get("oxe_droid"):
        return (
            "A multi-view video shows that a robot "
            + item_lower
            + " The video is split into three views: The top view shows the "
            "camera view from the robot's wrist, the bottom-left view shows "
            "the camera view from the left exterior camera, and the "
            "bottom-right view shows the camera view from the right exterior "
            "camera. During training, one of the two bottom exterior views may "
            "be a black screen (dropped view). The robot " + item_lower
        )
    if embodiment_id in {
        embodiment_tag_mapping.get("gr1_unified"),
        embodiment_tag_mapping.get("mecka_hands"),
    }:
        return "A single view video shows that a human " + item_lower
    if embodiment_id == embodiment_tag_mapping.get("yam"):
        return (
            "A multi-view video shows that a robot "
            + item_lower
            + " The video is split into four views: The top-left view shows "
            "the top camera, the top-right view shows the right camera, the "
            "bottom-left view shows the left camera, and the bottom-right view "
            "is a black screen. The robot " + item_lower
        )
    raise ValueError(f"Embodiment ID {embodiment_id} is not supported.")


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def select_batch_value(value: Any, index: int, batch_size: int) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index]
        return value
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index]
        return value
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def resolve_language(transition: Transition, index: int, batch_size: int) -> Any:
    for key in (
        PROMPT,
        TASK,
        TEXT,
        "language",
        "annotation.language.action_text",
        "annotation.language",
    ):
        if key in transition:
            return select_batch_value(transition[key], index, batch_size)
    for key in transition:
        if "annotation" in key:
            return select_batch_value(transition[key], index, batch_size)
    raise ValueError("DreamZero processor requires prompt/task/text/language.")


def prepare_video_grid(video: Any, *, embodiment_tag: str | None) -> np.ndarray:
    """Prepare one sample's video to DreamZero `images` layout."""
    video_arr = as_numpy(video)
    if video_arr.ndim != 5:
        raise ValueError(
            "DreamZero video sample must be [T, V, H, W, C], got "
            f"{tuple(video_arr.shape)}."
        )
    images = np.transpose(video_arr, (1, 0, 4, 2, 3))
    if embodiment_tag == "oxe_droid":
        views, frames, channels, height, width = images.shape
        tensor = torch.from_numpy(images).reshape(
            views * frames, channels, height, width
        )
        tensor = tensor.to(torch.float32).div_(255.0)
        crop_height = int(180 * 0.95)
        crop_width = int(320 * 0.95)
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        tensor = tensor[..., top : top + crop_height, left : left + crop_width]
        tensor = vision_transforms.functional.resize(
            tensor,
            (176, 320),
            interpolation=vision_transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
        images = (
            tensor.reshape(views, frames, channels, 176, 320)
            .mul_(255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
    if images.shape[0] <= 1:
        return images

    v, t, c, h, w = images.shape
    if embodiment_tag == "oxe_droid" and v >= 3:
        concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
        wrist_wide = np.repeat(images[2], 2, axis=-1)
        concat_images[0, :, :, :h, :] = wrist_wide
        concat_images[0, :, :, h:, :w] = images[0]
        concat_images[0, :, :, h:, w:] = images[1]
        return concat_images

    concat_images = np.zeros((1, t, c, 2 * h, 2 * w), dtype=images.dtype)
    if v > 0:
        concat_images[0, :, :, :h, :w] = images[0]
    if v > 1:
        concat_images[0, :, :, h:, :w] = images[1]
    if v > 2:
        concat_images[0, :, :, :h, w:] = images[2]
    return concat_images


def vlm_images(images: np.ndarray) -> np.ndarray:
    return np.transpose(images, (1, 0, 3, 4, 2)).reshape(
        images.shape[1] * images.shape[0],
        images.shape[3],
        images.shape[4],
        images.shape[2],
    )


def prepare_state(
    state: Any | None,
    *,
    state_horizon: int,
    max_state_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if state is None:
        values = np.zeros((state_horizon, max_state_dim), dtype=np.float32)
        return values, np.zeros_like(values, dtype=bool)

    values = as_numpy(state)
    if values.shape[0] % state_horizon != 0:
        raise ValueError(f"{values.shape=}, {state_horizon=}")
    n_state_dims = values.shape[-1]
    if n_state_dims > max_state_dim:
        values = values[:, :max_state_dim]
        n_state_dims = max_state_dim
    else:
        values = np.pad(values, ((0, 0), (0, max_state_dim - n_state_dims)), "constant")
    mask = np.zeros_like(values, dtype=bool)
    mask[:, :n_state_dims] = True
    return values, mask


def prepare_action(
    action: Any | None,
    *,
    action_horizon: int,
    max_action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if action is None:
        values = np.zeros((action_horizon, max_action_dim), dtype=np.float32)
        return values, np.zeros_like(values, dtype=bool)

    values = as_numpy(action)
    if values.shape[0] % action_horizon != 0:
        raise ValueError(f"{values.shape=}, {action_horizon=}")
    n_action_dims = values.shape[-1]
    if n_action_dims > max_action_dim:
        raise ValueError(
            f"Action dim {n_action_dims} exceeds max allowed {max_action_dim}."
        )
    values = np.pad(values, ((0, 0), (0, max_action_dim - n_action_dims)), "constant")
    mask = np.zeros((values.shape[0], max_action_dim), dtype=bool)
    mask[:, :n_action_dims] = True
    return values, mask


def load_dreamzero_metadata(path: str | Path) -> dict[str, Any]:
    """Load DreamZero metadata from a checkpoint dir or metadata json file."""
    metadata_path = Path(path)
    if metadata_path.is_dir():
        metadata_path = metadata_path / "experiment_cfg" / "metadata.json"
    with open(metadata_path) as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"DreamZero metadata must be a dict: {metadata_path}")
    return data


def load_dreamzero_conf(path: str | Path) -> dict[str, Any]:
    """Load DreamZero experiment config from a checkpoint dir or yaml file."""
    conf_path = Path(path)
    if conf_path.is_dir():
        conf_path = conf_path / "experiment_cfg" / "conf.yaml"
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            f"DreamZeroProcessor.from_pretrained requires PyYAML to parse {conf_path}."
        ) from exc
    with open(conf_path) as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"DreamZero conf must be a dict: {conf_path}")
    return data


def dreamzero_modality_keys_from_conf(
    conf: dict[str, Any],
    *,
    embodiment_tag: str,
    modality: str,
) -> tuple[str, ...] | None:
    """Return modality keys such as ``joint_position`` from a DreamZero config."""
    candidates = [
        conf.get(f"modality_config_{embodiment_tag}"),
        (conf.get("modality_configs") or {}).get(embodiment_tag),
        ((conf.get("train_dataset") or {}).get("all_modality_configs") or {}).get(
            embodiment_tag
        ),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        section = candidate.get(modality)
        if not isinstance(section, dict):
            continue
        keys = section.get("modality_keys")
        if keys:
            return tuple(str(key).split(".", 1)[-1] for key in keys)
    return None


def dreamzero_relative_action_keys_from_conf(conf: dict[str, Any]) -> tuple[str, ...]:
    """Return relative-action keys configured by the official policy config."""
    enabled = bool(conf.get("relative_action")) or bool(
        conf.get("relative_action_per_horizon")
    )
    if not enabled:
        return ()
    keys = conf.get("relative_action_keys") or ()
    return tuple(str(key).split(".", 1)[-1] for key in keys)


def dreamzero_normalization_mode(name: str) -> NormalizationMode:
    """Map DreamZero transform names to the shared normalizer modes."""
    normalized = name.lower()
    if normalized in ("q99", "quantile", "quantiles"):
        return NormalizationMode.QUANTILES
    if normalized in ("minmax", "min_max"):
        return NormalizationMode.MIN_MAX
    if normalized in ("meanstd", "mean_std"):
        return NormalizationMode.MEAN_STD
    if normalized == "identity":
        return NormalizationMode.IDENTITY
    raise ValueError(f"Unsupported DreamZero normalization mode {name!r}.")


def dreamzero_metadata_stats(
    metadata: dict[str, Any],
    *,
    embodiment_tag: str,
    state_keys: tuple[str, ...] = DREAMZERO_DEFAULT_STATE_KEYS,
    action_keys: tuple[str, ...] = DREAMZERO_DEFAULT_ACTION_KEYS,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build concat state/action stats from DreamZero metadata.

    The official metadata stores stats per modality key. DreamTransform consumes
    one concatenated state/action vector, so phyai's shared normalizer needs the
    same stats concatenated in the modality order.
    """
    entry = metadata.get(embodiment_tag)
    if not isinstance(entry, dict):
        raise ValueError(f"Metadata has no embodiment tag {embodiment_tag!r}.")
    statistics = entry.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError(f"Metadata for {embodiment_tag!r} has no statistics.")

    def concat_stats(modality: str, keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
        source = statistics.get(modality)
        if not isinstance(source, dict):
            raise ValueError(f"Metadata statistics has no {modality!r} block.")
        values: dict[str, list[torch.Tensor]] = {}
        for key in keys:
            if key not in source:
                raise ValueError(
                    f"Metadata {modality!r} block has no key {key!r}; "
                    f"available keys: {sorted(source)}."
                )
            for stat_name, stat_value in source[key].items():
                values.setdefault(stat_name, []).append(
                    torch.as_tensor(stat_value, dtype=torch.float32).reshape(-1)
                )
        return {
            stat_name: torch.cat(parts, dim=0) for stat_name, parts in values.items()
        }

    return {
        DREAMZERO_STATE_FEATURE: concat_stats("state", state_keys),
        DREAMZERO_ACTION_FEATURE: concat_stats("action", action_keys),
    }


def dreamzero_feature_shapes_from_stats(
    stats: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Create shared normalizer feature descriptors for available stats."""
    features: dict[str, dict[str, Any]] = {}
    if not stats:
        return features
    if DREAMZERO_STATE_FEATURE in stats:
        features[DREAMZERO_STATE_FEATURE] = {
            "type": "STATE",
            "shape": [],
        }
    if DREAMZERO_ACTION_FEATURE in stats:
        features[DREAMZERO_ACTION_FEATURE] = {
            "type": "ACTION",
            "shape": [],
        }
    return features


def dreamzero_key_slices(
    keys: tuple[str, ...],
    stats: dict[str, dict[str, Any]],
) -> dict[str, tuple[int, int]]:
    """Return ``{key: (start, end)}`` slices from key-ordered stats tensors."""
    slices: dict[str, tuple[int, int]] = {}
    start = 0
    for key in keys:
        width = int(torch.as_tensor(stats[key]["mean"]).numel())
        slices[key] = (start, start + width)
        start += width
    return slices


def select_last_state_for_key(
    obs: Any, key: str, state_slice: tuple[int, int]
) -> torch.Tensor | None:
    """Find the last raw state value for a relative action key."""
    state_key = f"state.{key}"
    value = None
    if isinstance(obs, dict):
        if state_key in obs:
            value = obs[state_key]
        elif key in obs:
            value = obs[key]
        elif STATE in obs:
            value = obs[STATE]
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim >= 2:
        tensor = tensor[..., -1, :]
    start, end = state_slice
    if tensor.shape[-1] >= end:
        tensor = tensor[..., start:end]
    return tensor


@ProcessorStepRegistry.register("dreamzero_relative_action_step")
@dataclass
class DreamZeroRelativeActionStep(ProcessorStep):
    """Convert selected relative action channels back to absolute coordinates."""

    relative_action_keys: tuple[str, ...] = field(default_factory=tuple)
    action_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    state_slices: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __call__(self, transition: Transition) -> Transition:
        if not self.relative_action_keys:
            return transition
        action = transition.get(ACTION)
        if action is None:
            raise ValueError("DreamZeroRelativeActionStep requires an ACTION entry.")
        obs = transition.get(OBS)
        if obs is None:
            return transition

        out = transition.copy()
        action_out = action.clone()
        for key in self.relative_action_keys:
            action_slice = self.action_slices.get(key)
            state_slice = self.state_slices.get(key)
            if action_slice is None or state_slice is None:
                continue
            last_state = select_last_state_for_key(obs, key, state_slice)
            if last_state is None:
                continue
            last_state = last_state.to(device=action_out.device, dtype=action_out.dtype)
            if last_state.ndim == action_out.ndim - 1:
                last_state = last_state.unsqueeze(-2)
            start, end = action_slice
            action_out[..., start:end] = action_out[..., start:end] + last_state
        out[ACTION] = action_out
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "relative_action_keys": list(self.relative_action_keys),
            "action_slices": {
                key: list(value) for key, value in self.action_slices.items()
            },
            "state_slices": {
                key: list(value) for key, value in self.state_slices.items()
            },
        }


@ProcessorStepRegistry.register("dreamzero_prepare_step")
@dataclass
class DreamZeroPrepareStep(ProcessorStep):
    """Prepare DreamZero video/state/action fields before tokenization."""

    max_state_dim: int = 64
    max_action_dim: int = 32
    state_horizon: int = 1
    action_horizon: int = 24
    num_views: int = 3
    embodiment_tag: str = "oxe_droid"
    embodiment_tag_mapping: dict[str, int] = field(
        default_factory=lambda: DREAMZERO_DEFAULT_EMBODIMENT_TAG_MAPPING.copy()
    )
    default_instruction: str = "Perform the default behavior."
    training: bool = False
    include_action_if_present: bool = True
    negative_prompt: str = DREAMZERO_DEFAULT_NEGATIVE_PROMPT

    def resolve_embodiment_tag(
        self, transition: Transition, index: int, batch_size: int
    ) -> str:
        value = select_batch_value(transition.get(EMBODIMENT_TAG), index, batch_size)
        if value is None:
            value = self.embodiment_tag
        if hasattr(value, "value"):
            value = value.value
        return str(value)

    def resolve_embodiment_id(
        self,
        transition: Transition,
        index: int,
        batch_size: int,
        tag: str,
    ) -> int:
        value = select_batch_value(transition.get(EMBODIMENT_ID), index, batch_size)
        if value is not None:
            if isinstance(value, torch.Tensor):
                return int(value.item())
            if isinstance(value, np.ndarray):
                return int(value.item())
            return int(value)
        if tag not in self.embodiment_tag_mapping:
            raise ValueError(f"Unknown DreamZero embodiment tag {tag!r}.")
        return int(self.embodiment_tag_mapping[tag])

    def __call__(self, transition: Transition) -> Transition:
        out = transition.copy()
        video = transition.get(VIDEO)
        if video is None:
            return out

        video_arr = as_numpy(video)
        if video_arr.ndim == 5:
            video_arr = video_arr[None, ...]
        if video_arr.ndim != 6:
            raise ValueError(
                "DreamZero video must be [T, V, H, W, C] or [B, T, V, H, W, C], "
                f"got {tuple(video_arr.shape)}."
            )
        batch_size = video_arr.shape[0]

        images_list = []
        prompts = []
        negative_prompts = []
        states = []
        state_masks = []
        actions = []
        action_masks = []
        embodiment_ids = []

        for index in range(batch_size):
            tag = self.resolve_embodiment_tag(transition, index, batch_size)
            embodiment_id = self.resolve_embodiment_id(
                transition,
                index,
                batch_size,
                tag,
            )
            images = prepare_video_grid(video_arr[index], embodiment_tag=tag).astype(
                np.uint8
            )
            images_list.append(vlm_images(images))
            raw_prompt = resolve_language(transition, index, batch_size)
            prompts.append(
                raw_prompt if raw_prompt is not None else self.default_instruction
            )
            negative_prompts.append(self.negative_prompt)
            state = select_batch_value(transition.get(STATE), index, batch_size)
            state_value, state_mask = prepare_state(
                state,
                state_horizon=self.state_horizon,
                max_state_dim=self.max_state_dim,
            )
            state_value = np.clip(state_value, -1.0, 1.0)
            states.append(state_value)
            state_masks.append(state_mask)
            action = select_batch_value(transition.get(ACTION), index, batch_size)
            if self.training or (self.include_action_if_present and action is not None):
                action_value, action_mask = prepare_action(
                    action,
                    action_horizon=self.action_horizon,
                    max_action_dim=self.max_action_dim,
                )
                actions.append(action_value)
                action_masks.append(action_mask)
            embodiment_ids.append(np.asarray(embodiment_id, dtype=np.int64))

        out[IMAGES] = torch.from_numpy(np.stack(images_list))
        out[PROMPT] = prompts
        out[TEXT_NEGATIVE] = negative_prompts
        out[STATE] = torch.from_numpy(np.stack(states))
        out[STATE_MASK] = torch.from_numpy(np.stack(state_masks))
        if actions:
            out[ACTION] = torch.from_numpy(np.stack(actions))
            out[ACTION_MASK] = torch.from_numpy(np.stack(action_masks))
        out[EMBODIMENT_ID] = torch.from_numpy(np.stack(embodiment_ids))
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "state_horizon": self.state_horizon,
            "action_horizon": self.action_horizon,
            "num_views": self.num_views,
            "embodiment_tag": self.embodiment_tag,
            "embodiment_tag_mapping": self.embodiment_tag_mapping,
            "default_instruction": self.default_instruction,
            "training": self.training,
            "include_action_if_present": self.include_action_if_present,
            "negative_prompt": self.negative_prompt,
        }


@ProcessorStepRegistry.register("dreamzero_text_tokenize_step")
@dataclass
class DreamZeroTextTokenizeStep(ProcessorStep):
    """Tokenize DreamZero prompt and negative prompt with a HuggingFace tokenizer."""

    tokenizer: Any = field(repr=False, default=None)
    max_length: int = 512
    tokenizer_name: str | None = None
    negative_prompt: str = ""
    clean: bool = True
    padding: str = "max_length"
    truncation: bool = True
    num_views: int = 3
    apply_prompt_template: bool = True
    embodiment_tag_mapping: dict[str, int] = field(
        default_factory=lambda: DREAMZERO_DEFAULT_EMBODIMENT_TAG_MAPPING.copy()
    )

    def __call__(self, transition: Transition) -> Transition:
        if self.tokenizer is None:
            raise ValueError("DreamZeroTextTokenizeStep requires a tokenizer object.")
        prompts = transition.get(PROMPT)
        if prompts is None:
            prompts = transition.get(TASK)
        if prompts is None:
            raise ValueError("DreamZeroTextTokenizeStep requires PROMPT or TASK.")
        if isinstance(prompts, str):
            prompts = [prompts]
        embodiment_ids = transition.get(EMBODIMENT_ID)
        if isinstance(embodiment_ids, torch.Tensor):
            embodiment_ids = [int(x.item()) for x in embodiment_ids.reshape(-1)]
        elif isinstance(embodiment_ids, np.ndarray):
            embodiment_ids = [int(x.item()) for x in embodiment_ids.reshape(-1)]
        elif embodiment_ids is not None and not isinstance(
            embodiment_ids, (list, tuple)
        ):
            embodiment_ids = [int(embodiment_ids)]
        if self.apply_prompt_template and embodiment_ids is not None:
            prompts = [
                format_dreamzero_prompt(
                    prompt,
                    embodiment_id=embodiment_ids[index],
                    embodiment_tag_mapping=self.embodiment_tag_mapping,
                    num_views=self.num_views,
                )
                for index, prompt in enumerate(prompts)
            ]
        prompts = [
            whitespace_clean(str(prompt)) if self.clean else str(prompt)
            for prompt in prompts
        ]
        negative_prompts = transition.get(TEXT_NEGATIVE)
        if negative_prompts is None or isinstance(negative_prompts, torch.Tensor):
            negative_prompts = [self.negative_prompt] * len(prompts)
        elif isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)
        else:
            negative_prompts = list(negative_prompts)
        if self.clean:
            negative_prompts = [whitespace_clean(prompt) for prompt in negative_prompts]

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        encoded_negative = self.tokenizer(
            negative_prompts,
            return_tensors="pt",
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        out = transition.copy()
        out[TEXT] = encoded["input_ids"].to(torch.int64)
        out[TEXT_ATTENTION_MASK] = encoded["attention_mask"].to(torch.int64)
        out[TEXT_NEGATIVE] = encoded_negative["input_ids"].to(torch.int64)
        out[TEXT_ATTENTION_MASK_NEGATIVE] = encoded_negative["attention_mask"].to(
            torch.int64
        )
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "max_length": self.max_length,
            "tokenizer_name": self.tokenizer_name,
            "negative_prompt": self.negative_prompt,
            "clean": self.clean,
            "padding": self.padding,
            "truncation": self.truncation,
            "num_views": self.num_views,
            "apply_prompt_template": self.apply_prompt_template,
            "embodiment_tag_mapping": self.embodiment_tag_mapping,
        }


__all__ = [
    "ACTION_MASK",
    "DREAMZERO_ACTION_FEATURE",
    "DREAMZERO_DEFAULT_ACTION_KEYS",
    "DREAMZERO_DEFAULT_ACTION_NORMALIZATION",
    "DREAMZERO_DEFAULT_EMBODIMENT_TAG_MAPPING",
    "DREAMZERO_DEFAULT_NEGATIVE_PROMPT",
    "DREAMZERO_DEFAULT_STATE_KEYS",
    "DREAMZERO_STATE_FEATURE",
    "EMBODIMENT_ID",
    "EMBODIMENT_TAG",
    "IMAGES",
    "OBS",
    "STATE_MASK",
    "TEXT",
    "TEXT_ATTENTION_MASK",
    "TEXT_ATTENTION_MASK_NEGATIVE",
    "TEXT_NEGATIVE",
    "VIDEO",
    "DreamZeroPrepareStep",
    "DreamZeroRelativeActionStep",
    "DreamZeroTextTokenizeStep",
    "dreamzero_feature_shapes_from_stats",
    "dreamzero_key_slices",
    "dreamzero_metadata_stats",
    "dreamzero_modality_keys_from_conf",
    "dreamzero_normalization_mode",
    "dreamzero_relative_action_keys_from_conf",
    "format_dreamzero_prompt",
    "load_dreamzero_conf",
    "load_dreamzero_metadata",
    "prepare_action",
    "prepare_state",
    "prepare_video_grid",
    "whitespace_clean",
]
