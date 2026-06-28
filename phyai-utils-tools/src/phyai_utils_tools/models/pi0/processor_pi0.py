"""pi0 pre/post processor built on the shared pipeline framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from phyai_utils_tools.models.pi0.steps_pi0 import PadStateStep, Pi0PromptPrepareStep
from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline
from phyai_utils_tools.processing.steps import (
    DeviceStep,
    FeatureType,
    NormalizationMode,
    NormalizeImageStep,
    NormalizerStep,
    ResizeWithPadStep,
    SliceActionStep,
    TokenizerStep,
    UnnormalizerStep,
)
from phyai_utils_tools.processing.transition import (
    ACTION,
    INPUT_IDS,
    LANG_LENS,
    PIXEL_VALUES,
    STATE,
    Transition,
)
from phyai_utils_tools.tokenizer import get_tokenizer

PI0_DEFAULT_TOKENIZER_NAME = "google/paligemma-3b-pt-224"

STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"

PI0_NORM_MAP: dict[str, str] = {
    FeatureType.VISUAL.value: NormalizationMode.IDENTITY.value,
    FeatureType.STATE.value: NormalizationMode.MEAN_STD.value,
    FeatureType.ACTION.value: NormalizationMode.MEAN_STD.value,
}

PRE_CONFIG_FILENAME = "policy_preprocessor.json"
POST_CONFIG_FILENAME = "policy_postprocessor.json"


@dataclass
class PI0ProcessedInputs:
    """Canonical preprocessed pi0 inputs for ``phyai.models.pi0.PI0Request``."""

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    lang_lens: torch.Tensor
    state: torch.Tensor


def _features_for_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    if not dataset_stats:
        return features
    if STATE_FEATURE in dataset_stats:
        features[STATE_FEATURE] = {"type": FeatureType.STATE.value, "shape": []}
    if ACTION_FEATURE in dataset_stats:
        features[ACTION_FEATURE] = {"type": FeatureType.ACTION.value, "shape": []}
    return features


def _validate_num_images(num_images: int) -> int:
    num = int(num_images)
    if num not in (2, 3):
        raise ValueError(f"pi0 supports exactly 2 or 3 images, got {num}.")
    return num


class PI0Processor(BaseModelProcessor):
    """pi0 pre/post processor matching LeRobot's pi0 input conventions."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        num_channels: int = 3,
        num_images: int = 3,
        tokenizer_max_length: int = 48,
        max_state_dim: int = 32,
        action_dim: int | None = None,
        tokenizer_name: str = PI0_DEFAULT_TOKENIZER_NAME,
        tokenizer: Any = None,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
        normalize_pixels: bool = True,
        image_pad_value: float = 0.0,
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.image_size = int(image_size)
        self.num_channels = int(num_channels)
        self.num_images = _validate_num_images(num_images)
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.max_state_dim = int(max_state_dim)
        self.action_dim = action_dim
        self.tokenizer_name = tokenizer_name
        self.dataset_stats = dataset_stats
        self.normalize_pixels = bool(normalize_pixels)
        self.image_pad_value = float(image_pad_value)
        self.device = device
        self.params_dtype = params_dtype
        self.tokenizer = (
            tokenizer if tokenizer is not None else get_tokenizer(tokenizer_name)
        )
        super().__init__()

    @staticmethod
    def _to_inputs(transition: Transition) -> PI0ProcessedInputs:
        return PI0ProcessedInputs(
            pixel_values=transition[PIXEL_VALUES],
            input_ids=transition[INPUT_IDS],
            lang_lens=transition[LANG_LENS],
            state=transition[STATE],
        )

    @staticmethod
    def _action_to_transition(action: torch.Tensor) -> Transition:
        return {ACTION: action}

    @staticmethod
    def _transition_to_action(transition: Transition) -> torch.Tensor:
        return transition[ACTION]

    def _vision_steps(self) -> list:
        steps: list = [
            ResizeWithPadStep(
                target_size=self.image_size,
                num_images=self.num_images,
                num_channels=self.num_channels,
                pad_value=self.image_pad_value,
            )
        ]
        if self.normalize_pixels:
            steps.append(NormalizeImageStep())
        return steps

    def build_preprocessor(self) -> ProcessorPipeline:
        steps = self._vision_steps()
        steps += [
            NormalizerStep(
                features=_features_for_stats(self.dataset_stats),
                norm_map=PI0_NORM_MAP,
                stats=self.dataset_stats,
                device=self.device,
            ),
            PadStateStep(max_state_dim=self.max_state_dim),
            Pi0PromptPrepareStep(),
            TokenizerStep(
                tokenizer=self.tokenizer,
                max_length=self.tokenizer_max_length,
                tokenizer_name=self.tokenizer_name,
            ),
            DeviceStep(device=self.device, float_dtype=self.params_dtype),
        ]
        return ProcessorPipeline(
            steps=steps,
            name="pi0_preprocessor",
            to_output=self._to_inputs,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        steps = [
            SliceActionStep(action_dim=self.action_dim),
            UnnormalizerStep(
                features=_features_for_stats(self.dataset_stats),
                norm_map=PI0_NORM_MAP,
                stats=self.dataset_stats,
                device=self.device,
            ),
            DeviceStep(device="cpu"),
        ]
        return ProcessorPipeline(
            steps=steps,
            name="pi0_postprocessor",
            to_transition=self._action_to_transition,
            to_output=self._transition_to_action,
        )

    @classmethod
    def from_pretrained(
        cls,
        ckpt: str | Path,
        *,
        tokenizer: Any = None,
        tokenizer_name: str = PI0_DEFAULT_TOKENIZER_NAME,
        image_size: int = 224,
        num_channels: int = 3,
        num_images: int = 3,
        max_state_dim: int = 32,
        action_dim: int | None = None,
        normalize_pixels: bool = True,
        image_pad_value: float = 0.0,
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
        **hub_kwargs: Any,
    ) -> PI0Processor:
        """Build a processor from phyai-saved pi0 processor files."""

        tok = tokenizer if tokenizer is not None else get_tokenizer(tokenizer_name)
        pre = ProcessorPipeline.from_pretrained(
            ckpt,
            PRE_CONFIG_FILENAME,
            step_kwargs={
                "tokenizer_processor": {"tokenizer": tok},
                "device_processor": {"device": device, "float_dtype": params_dtype},
                "normalizer_processor": {"device": device},
            },
            **hub_kwargs,
        )
        post_step_kwargs: dict[str, dict[str, Any]] = {
            "unnormalizer_processor": {"device": device},
        }
        if action_dim is not None:
            post_step_kwargs["slice_action_step"] = {"action_dim": action_dim}
        post = ProcessorPipeline.from_pretrained(
            ckpt,
            POST_CONFIG_FILENAME,
            step_kwargs=post_step_kwargs,
            **hub_kwargs,
        )

        obj = cls.__new__(cls)
        obj.image_size = int(image_size)
        obj.num_channels = int(num_channels)
        obj.num_images = _validate_num_images(num_images)
        obj.tokenizer_max_length = next(
            (s.max_length for s in pre.steps if isinstance(s, TokenizerStep)),
            48,
        )
        obj.max_state_dim = next(
            (s.max_state_dim for s in pre.steps if isinstance(s, PadStateStep)),
            int(max_state_dim),
        )
        obj.action_dim = action_dim
        obj.tokenizer_name = tokenizer_name
        obj.dataset_stats = None
        obj.normalize_pixels = bool(normalize_pixels)
        obj.image_pad_value = float(image_pad_value)
        obj.device = device
        obj.params_dtype = params_dtype
        obj.tokenizer = tok

        pre_steps = [
            *obj._vision_steps(),
            *[s for s in pre.steps if not _is_vision_glue(s)],
        ]
        if not any(isinstance(s, PadStateStep) for s in pre_steps):
            insert_at = next(
                (
                    i + 1
                    for i in range(len(pre_steps) - 1, -1, -1)
                    if isinstance(pre_steps[i], NormalizerStep)
                ),
                None,
            )
            if insert_at is None:
                insert_at = next(
                    (
                        i
                        for i, s in enumerate(pre_steps)
                        if isinstance(s, (Pi0PromptPrepareStep, TokenizerStep))
                    ),
                    len(pre_steps),
                )
            pre_steps.insert(insert_at, PadStateStep(max_state_dim=obj.max_state_dim))
        pre.steps = pre_steps
        pre.name = "pi0_preprocessor"
        pre.to_output = cls._to_inputs

        if not any(isinstance(s, SliceActionStep) for s in post.steps):
            post.steps = [SliceActionStep(action_dim=action_dim), *list(post.steps)]
        post.name = "pi0_postprocessor"
        post.to_transition = cls._action_to_transition
        post.to_output = cls._transition_to_action

        obj._preprocessor = pre
        obj._postprocessor = post
        return obj

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write phyai pi0 processor json and stats sidecars."""

        save_dir = Path(save_directory)
        pre_core = ProcessorPipeline(
            steps=[s for s in self._preprocessor.steps if not _is_vision_glue(s)],
            name="policy_preprocessor",
        )
        post_core = ProcessorPipeline(
            steps=list(self._postprocessor.steps),
            name="policy_postprocessor",
        )
        pre_core.save_pretrained(save_dir, config_filename=PRE_CONFIG_FILENAME)
        post_core.save_pretrained(save_dir, config_filename=POST_CONFIG_FILENAME)


def _is_vision_glue(step: Any) -> bool:
    return isinstance(step, (ResizeWithPadStep, NormalizeImageStep))


def make_pi0_processors(**kwargs: Any) -> tuple[ProcessorPipeline, ProcessorPipeline]:
    proc = PI0Processor(**kwargs)
    return proc.preprocessor, proc.postprocessor


__all__ = [
    "PI0_DEFAULT_TOKENIZER_NAME",
    "PI0ProcessedInputs",
    "PI0Processor",
    "make_pi0_processors",
]
