"""MolmoAct2 robot prompts, checkpoint preprocessing, and action decoding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline, ProcessorStep
from phyai_utils_tools.processing.transition import ACTION, Transition


QUESTION_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)
MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
)


def as_float_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def feature_dim(stats: Mapping[str, Any] | None) -> int | None:
    for key in (
        "mean",
        "std",
        "min",
        "max",
        "q01",
        "q99",
        "q10",
        "q90",
        "mask",
        "names",
    ):
        value = (stats or {}).get(key)
        if value is not None and np.asarray(value).ndim:
            return int(np.asarray(value).shape[-1])
    return None


class MolmoAct2FeatureNormalizer:
    """Match the checkpoint's masked float32 normalization, including clipping."""

    def __init__(self, stats: Mapping[str, Any] | None, mode: str) -> None:
        self.mode = mode if stats is not None else "none"
        stats = stats or {}
        raw_mask = stats.get("mask")
        self.mask = (
            np.asarray(raw_mask, dtype=np.bool_) if raw_mask is not None else None
        )
        self.zero_mask = None
        self.offset = np.float32(0)
        self.scale = np.float32(1)
        if self.mode == "none":
            return
        if self.mode == "mean_std":
            self.offset = as_float_array(stats["mean"])
            self.scale = as_float_array(stats["std"])
        elif self.mode in {"min_max", "q01_q99", "q10_q90"}:
            low_key, high_key = {
                "min_max": ("min", "max"),
                "q01_q99": ("q01", "q99"),
                "q10_q90": ("q10", "q90"),
            }[self.mode]
            self.offset = as_float_array(stats[low_key])
            self.scale = as_float_array(stats[high_key]) - self.offset
            if "min" in stats and "max" in stats:
                self.zero_mask = as_float_array(stats["min"]) == as_float_array(
                    stats["max"]
                )
        else:
            raise ValueError(f"Unsupported MolmoAct2 normalization mode {mode!r}.")

    def normalize(self, value: Any) -> np.ndarray:
        values = as_float_array(value)
        if self.mode == "none":
            normalized = values
        elif self.mode == "mean_std":
            normalized = (values - self.offset) / np.maximum(self.scale, 1e-6)
        else:
            normalized = np.clip(
                2.0 * (values - self.offset) / np.maximum(self.scale, 1e-6) - 1.0,
                -1.0,
                1.0,
            )
        if self.mask is not None:
            normalized = np.where(self.mask, normalized, values)
        if self.zero_mask is not None:
            normalized = np.where(self.zero_mask, 0.0, normalized)
        return normalized

    def unnormalize(self, value: Any) -> np.ndarray:
        values = as_float_array(value)
        if self.mode == "none":
            result = values
        elif self.mode == "mean_std":
            result = values * self.scale + self.offset
        else:
            values = np.clip(values, -1.0, 1.0)
            result = (values + 1.0) * self.scale / 2.0 + self.offset
        if self.mask is not None:
            result = np.where(self.mask, result, values)
        return result


def normalize_task_text(task: str) -> str:
    text = re.sub(r"\s+", " ", task).strip()
    previous = None
    while text and text != previous:
        previous = text
        text = text.strip().strip("\"'`“”‘’[](){}").strip()
        for pattern in QUESTION_PREFIX_PATTERNS:
            text = pattern.sub("", text, count=1).strip()
        text = text.rstrip(".,!?;:,…").rstrip()
        text = text.rstrip("\"'”’)]}").rstrip()
        text = text.rstrip(".,!?;:,…").rstrip()
    chunks = [chunk.strip() for chunk in re.split(r"[.!?]+", text) if chunk.strip()]
    if len(chunks) > 1:
        text = "; ".join(chunks)
    return text.lower()


def wrap_robot_metadata(text: str, kind: str, enabled: bool) -> str:
    start, end = f"<{kind}_start>", f"<{kind}_end>"
    if not text or not enabled or (text.startswith(start) and text.endswith(end)):
        return text
    return f"{start}{text}{end}"


def build_robot_prompt(
    *,
    task: str,
    normalized_state: np.ndarray,
    metadata: Mapping[str, Any],
    num_state_tokens: int,
    num_images: int,
    add_setup_tokens: bool,
    add_control_tokens: bool,
) -> str:
    state = np.nan_to_num(normalized_state, nan=0.0, posinf=1.0, neginf=-1.0)
    scaled = (np.clip(state, -1.0, 1.0) + 1.0) / 2.0 * float(num_state_tokens - 1)
    bins = np.clip(np.rint(scaled).astype(np.int64), 0, num_state_tokens - 1)
    state_text = (
        "<state_start>"
        + "".join(f"<state_{int(index)}>" for index in bins.reshape(-1))
        + "<state_end>"
    )
    setup = wrap_robot_metadata(
        str(metadata.get("setup_type") or ""), "setup", add_setup_tokens
    )
    control = wrap_robot_metadata(
        str(metadata.get("control_mode") or ""), "control", add_control_tokens
    )
    prompt = (
        f"The task is to {task}. The setup is {setup}. "
        f"The current state of the robot is {state_text}. "
        f"The expected control mode is {control}. Given these, "
        "what action should the robot take to complete the task?"
    )
    if num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(
            f"Image {index + 1}<|image|>" for index in range(num_images)
        )
    return (
        f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<action_output>"
    )


@dataclass(frozen=True)
class MolmoAct2ProcessedInputs:
    """Single-observation model inputs; tensors retain checkpoint processor dtypes."""

    tensors: dict[str, torch.Tensor]
    action_horizon: int
    prompt: str


@dataclass
class MolmoAct2PrepareStep(ProcessorStep):
    normalizer: MolmoAct2FeatureNormalizer
    metadata: dict[str, Any]
    num_state_tokens: int
    n_obs_steps: int
    normalize_language: bool
    add_setup_tokens: bool
    add_control_tokens: bool

    def __call__(self, transition: Transition) -> Transition:
        if transition.get("state") is None:
            raise ValueError("MolmoAct2 requires state for discrete state prompting.")
        state = as_float_array(transition["state"])
        state_dim = feature_dim(self.metadata.get("state_stats"))
        if state.ndim == 0 or (state_dim is not None and state.shape[-1] != state_dim):
            raise ValueError(
                f"Expected state with last dimension {state_dim}, got {state.shape}."
            )
        if state.size != self.n_obs_steps * state.shape[-1]:
            raise ValueError("Provide one observation with n_obs_steps state vectors.")
        images = transition.get("images")
        if isinstance(images, Mapping):
            camera_keys = self.metadata.get("camera_keys")
            if not camera_keys:
                raise ValueError(
                    "This tag has no camera order; provide images as an ordered list."
                )
            images = [images[key] for key in camera_keys]
        if images is None:
            num_images = 0
        elif isinstance(images, (list, tuple)):
            num_images = len(images)
            images = list(images) or None
        else:
            shape = getattr(images, "shape", ())
            num_images = int(shape[0]) if len(shape) == 4 else 1
        task = str(transition.get("task") or "")
        if self.normalize_language:
            task = normalize_task_text(task)
        prompt = build_robot_prompt(
            task=task,
            normalized_state=self.normalizer.normalize(state),
            metadata=self.metadata,
            num_state_tokens=self.num_state_tokens,
            num_images=num_images,
            add_setup_tokens=self.add_setup_tokens,
            add_control_tokens=self.add_control_tokens,
        )
        return {**transition, "images": images, "prompt": prompt}


@dataclass
class MolmoAct2TokenizeStep(ProcessorStep):
    processor: Any
    action_dim: int
    max_action_dim: int

    def __call__(self, transition: Transition) -> Transition:
        encoded = self.processor(
            text=transition["prompt"], images=transition["images"], return_tensors="pt"
        )
        tensors = {key: encoded[key] for key in MODEL_INPUT_KEYS if key in encoded}
        attention_mask = tensors.get("attention_mask")
        if attention_mask is not None and bool(attention_mask.bool().all()):
            tensors.pop("attention_mask")
        if self.action_dim < self.max_action_dim:
            tensors["action_dim_is_pad"] = (
                torch.arange(self.max_action_dim)[None, :] >= self.action_dim
            ).expand(tensors["input_ids"].shape[0], -1)
        return {**transition, "tensors": tensors}


@dataclass
class MolmoAct2DecodeStep(ProcessorStep):
    normalizer: MolmoAct2FeatureNormalizer
    action_dim: int
    n_obs_steps: int
    n_action_steps: int

    def __call__(self, transition: Transition) -> Transition:
        actions = as_float_array(transition[ACTION])
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        if (
            actions.ndim != 3
            or actions.shape[-1] < self.action_dim
            or actions.shape[1] < end
        ):
            raise ValueError(
                f"Expected (batch, horizon >= {end}, dim >= {self.action_dim}), got {actions.shape}."
            )
        decoded = self.normalizer.unnormalize(actions[:, start:end, : self.action_dim])
        output = torch.from_numpy(decoded.copy())
        if isinstance(transition[ACTION], torch.Tensor):
            output = output.to(dtype=transition[ACTION].dtype)
        return {**transition, ACTION: output.float()}


class MolmoAct2Processor(BaseModelProcessor):
    """Prepare one robot observation and decode its continuous action chunk."""

    def __init__(
        self,
        *,
        processor: Any,
        norm_stats: Mapping[str, Any],
        norm_tag: str,
        max_action_dim: int = 32,
        max_action_horizon: int = 30,
        num_state_tokens: int = 256,
        n_obs_steps: int = 1,
        n_action_steps: int | None = None,
        add_setup_tokens: bool = True,
        add_control_tokens: bool = True,
        normalize_language: bool = True,
    ) -> None:
        metadata_by_tag = norm_stats.get("metadata_by_tag", {})
        norm_tag = str(norm_tag).strip()
        if norm_tag not in metadata_by_tag:
            raise ValueError(
                f"Unknown normalization tag {norm_tag!r}; choose from {sorted(metadata_by_tag)}."
            )
        if num_state_tokens < 1 or n_obs_steps < 1:
            raise ValueError("num_state_tokens and n_obs_steps must be positive.")
        self.processor = processor
        self.norm_tag = norm_tag
        self.metadata = dict(metadata_by_tag[norm_tag])
        self.action_dim = (
            feature_dim(self.metadata.get("action_stats")) or max_action_dim
        )
        self.state_dim = feature_dim(self.metadata.get("state_stats"))
        self.action_horizon = int(
            self.metadata.get("action_horizon") or max_action_horizon
        )
        self.n_action_steps = int(
            n_action_steps
            if n_action_steps is not None
            else self.metadata.get("n_action_steps") or self.action_horizon
        )
        if not 1 <= self.action_dim <= max_action_dim:
            raise ValueError(
                "Tag action dimension exceeds the checkpoint action dimension."
            )
        if not 1 <= self.action_horizon <= max_action_horizon:
            raise ValueError(
                "Tag action horizon exceeds the checkpoint action horizon."
            )
        if (
            self.n_action_steps < 1
            or n_obs_steps - 1 + self.n_action_steps > self.action_horizon
        ):
            raise ValueError(
                "Requested action steps exceed the available action horizon."
            )
        mode = str(norm_stats.get("norm_mode", "min_max"))
        self.state_normalizer = MolmoAct2FeatureNormalizer(
            self.metadata.get("state_stats"), mode
        )
        self.action_normalizer = MolmoAct2FeatureNormalizer(
            self.metadata.get("action_stats"), mode
        )
        self.max_action_dim = max_action_dim
        self.num_state_tokens = num_state_tokens
        self.n_obs_steps = n_obs_steps
        self.add_setup_tokens = add_setup_tokens
        self.add_control_tokens = add_control_tokens
        self.normalize_language = normalize_language
        super().__init__()

    def build_preprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            name="molmoact2_preprocessor",
            steps=[
                MolmoAct2PrepareStep(
                    self.state_normalizer,
                    self.metadata,
                    self.num_state_tokens,
                    self.n_obs_steps,
                    self.normalize_language,
                    self.add_setup_tokens,
                    self.add_control_tokens,
                ),
                MolmoAct2TokenizeStep(
                    self.processor, self.action_dim, self.max_action_dim
                ),
            ],
            to_output=lambda transition: MolmoAct2ProcessedInputs(
                tensors=transition["tensors"],
                action_horizon=self.action_horizon,
                prompt=transition["prompt"],
            ),
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            name="molmoact2_postprocessor",
            steps=[
                MolmoAct2DecodeStep(
                    self.action_normalizer,
                    self.action_dim,
                    self.n_obs_steps,
                    self.n_action_steps,
                )
            ],
            to_transition=lambda action: {ACTION: action},
            to_output=lambda transition: transition[ACTION],
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        norm_tag: str,
        processor: Any = None,
        **overrides: Any,
    ) -> MolmoAct2Processor:
        """Load a local checkpoint's processor and robot normalization metadata."""
        root = Path(checkpoint)
        config = json.loads((root / "config.json").read_text())
        stats = json.loads(
            (root / config.get("norm_stats_filename", "norm_stats.json")).read_text()
        )
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(root, trust_remote_code=True)
        keys = (
            "max_action_dim",
            "max_action_horizon",
            "num_state_tokens",
            "n_obs_steps",
            "add_setup_tokens",
            "add_control_tokens",
        )
        options = {key: config[key] for key in keys if key in config}
        options.update(overrides)
        return cls(processor=processor, norm_stats=stats, norm_tag=norm_tag, **options)


__all__ = ["MolmoAct2ProcessedInputs", "MolmoAct2Processor"]
