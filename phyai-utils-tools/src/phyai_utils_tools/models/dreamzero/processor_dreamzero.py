"""DreamZero processor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from phyai_utils_tools.models.dreamzero.steps_dreamzero import (
    ACTION_MASK,
    DREAMZERO_DEFAULT_EMBODIMENT_TAG_MAPPING,
    DREAMZERO_DEFAULT_ACTION_KEYS,
    DREAMZERO_DEFAULT_ACTION_NORMALIZATION,
    DREAMZERO_DEFAULT_STATE_KEYS,
    DREAMZERO_ACTION_FEATURE,
    EMBODIMENT_ID,
    IMAGES,
    OBS,
    STATE_MASK,
    TEXT,
    TEXT_ATTENTION_MASK,
    TEXT_ATTENTION_MASK_NEGATIVE,
    TEXT_NEGATIVE,
    DreamZeroPrepareStep,
    DreamZeroRelativeActionStep,
    DreamZeroTextTokenizeStep,
    dreamzero_feature_shapes_from_stats,
    dreamzero_key_slices,
    dreamzero_metadata_stats,
    dreamzero_modality_keys_from_conf,
    dreamzero_normalization_mode,
    dreamzero_relative_action_keys_from_conf,
    load_dreamzero_conf,
    load_dreamzero_metadata,
)
from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline
from phyai_utils_tools.processing.steps import (
    DeviceStep,
    NormalizerStep,
    SliceActionStep,
    UnnormalizerStep,
)
from phyai_utils_tools.processing.transition import ACTION, STATE, Transition
from phyai_utils_tools.tokenizer import get_tokenizer

DREAMZERO_DEFAULT_TOKENIZER_NAME = "google/umt5-xxl"
DREAMZERO_PRE_CONFIG_FILENAME = "policy_preprocessor.json"
DREAMZERO_POST_CONFIG_FILENAME = "policy_postprocessor.json"


@dataclass
class DreamZeroTextInputs:
    """Text tensors consumed by DreamZero text encoder runner."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    negative_input_ids: torch.Tensor
    negative_attention_mask: torch.Tensor


@dataclass
class DreamZeroProcessedInputs(DreamZeroTextInputs):
    """Full DreamZero processor output using official transform field names."""

    images: torch.Tensor | None = None
    state: torch.Tensor | None = None
    state_mask: torch.Tensor | None = None
    action: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    embodiment_id: torch.Tensor | None = None


@dataclass
class DreamZeroActionOutput:
    """CPU-ready DreamZero policy action output."""

    action: torch.Tensor
    normalized_action: torch.Tensor | None = None
    raw_output: Any = None


class DreamZeroProcessor(BaseModelProcessor):
    """DreamZero processor matching the upstream transform/collate path."""

    def __init__(
        self,
        *,
        tokenizer_name: str = DREAMZERO_DEFAULT_TOKENIZER_NAME,
        tokenizer: Any = None,
        max_length: int = 512,
        negative_prompt: str = "",
        max_state_dim: int = 64,
        max_action_dim: int = 32,
        state_horizon: int = 1,
        action_horizon: int = 24,
        num_views: int = 3,
        embodiment_tag: str = "oxe_droid",
        embodiment_tag_mapping: dict[str, int] | None = None,
        default_instruction: str = "Perform the default behavior.",
        training: bool = False,
        include_action_if_present: bool = True,
        apply_prompt_template: bool = True,
        raw_action_dim: int | None = None,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
        state_keys: tuple[str, ...] = DREAMZERO_DEFAULT_STATE_KEYS,
        action_keys: tuple[str, ...] = DREAMZERO_DEFAULT_ACTION_KEYS,
        action_normalization: str = DREAMZERO_DEFAULT_ACTION_NORMALIZATION,
        relative_action_keys: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.tokenizer = (
            tokenizer if tokenizer is not None else get_tokenizer(tokenizer_name)
        )
        self.max_length = int(max_length)
        self.negative_prompt = negative_prompt
        self.max_state_dim = int(max_state_dim)
        self.max_action_dim = int(max_action_dim)
        self.state_horizon = int(state_horizon)
        self.action_horizon = int(action_horizon)
        self.num_views = int(num_views)
        self.embodiment_tag = embodiment_tag
        self.embodiment_tag_mapping = (
            DREAMZERO_DEFAULT_EMBODIMENT_TAG_MAPPING.copy()
            if embodiment_tag_mapping is None
            else dict(embodiment_tag_mapping)
        )
        self.default_instruction = default_instruction
        self.training = bool(training)
        self.include_action_if_present = bool(include_action_if_present)
        self.apply_prompt_template = bool(apply_prompt_template)
        self.raw_action_dim = (
            int(raw_action_dim) if raw_action_dim is not None else None
        )
        self.dataset_stats = dataset_stats
        self.state_keys = tuple(state_keys)
        self.action_keys = tuple(action_keys)
        self.action_normalization = action_normalization
        self.relative_action_keys = tuple(relative_action_keys)
        self._metadata = metadata
        self.device = device
        self.params_dtype = params_dtype
        super().__init__()

    @staticmethod
    def to_text_inputs(transition: Transition) -> DreamZeroProcessedInputs:
        return DreamZeroProcessedInputs(
            input_ids=transition[TEXT],
            attention_mask=transition[TEXT_ATTENTION_MASK],
            negative_input_ids=transition[TEXT_NEGATIVE],
            negative_attention_mask=transition[TEXT_ATTENTION_MASK_NEGATIVE],
            images=transition.get(IMAGES),
            state=transition.get(STATE),
            state_mask=transition.get(STATE_MASK),
            action=transition.get(ACTION),
            action_mask=transition.get(ACTION_MASK),
            embodiment_id=transition.get(EMBODIMENT_ID),
        )

    @staticmethod
    def action_to_transition(output: Any) -> Transition:
        if isinstance(output, torch.Tensor):
            return {ACTION: output}
        if isinstance(output, dict):
            transition: Transition = {ACTION: output[ACTION]}
            if OBS in output:
                transition[OBS] = output[OBS]
            elif "observation" in output:
                transition[OBS] = output["observation"]
            return transition
        action = getattr(output, ACTION)
        transition = {ACTION: action}
        obs = getattr(output, OBS, None)
        if obs is not None:
            transition[OBS] = obs
        return transition

    @staticmethod
    def transition_to_action_output(transition: Transition) -> DreamZeroActionOutput:
        return DreamZeroActionOutput(action=transition[ACTION])

    def _normalizer_features(self) -> dict[str, dict[str, Any]]:
        return dreamzero_feature_shapes_from_stats(self.dataset_stats)

    def _normalizer_norm_map(self) -> dict[str, str]:
        mode = dreamzero_normalization_mode(self.action_normalization).value
        return {"STATE": mode, "ACTION": mode}

    def _raw_action_dim(self) -> int | None:
        if self.raw_action_dim is not None:
            return self.raw_action_dim
        stats = self.dataset_stats or {}
        action_stats = stats.get(DREAMZERO_ACTION_FEATURE)
        if action_stats:
            first = next(iter(action_stats.values()))
            return int(torch.as_tensor(first).numel())
        return None

    def build_preprocessor(self) -> ProcessorPipeline:
        steps = []
        if self.dataset_stats:
            steps.append(
                NormalizerStep(
                    features=self._normalizer_features(),
                    norm_map=self._normalizer_norm_map(),
                    stats=self.dataset_stats,
                    device=self.device,
                )
            )
        steps.extend(
            [
                DreamZeroPrepareStep(
                    max_state_dim=self.max_state_dim,
                    max_action_dim=self.max_action_dim,
                    state_horizon=self.state_horizon,
                    action_horizon=self.action_horizon,
                    num_views=self.num_views,
                    embodiment_tag=self.embodiment_tag,
                    embodiment_tag_mapping=self.embodiment_tag_mapping,
                    default_instruction=self.default_instruction,
                    training=self.training,
                    include_action_if_present=self.include_action_if_present,
                ),
                DreamZeroTextTokenizeStep(
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    tokenizer_name=self.tokenizer_name,
                    negative_prompt=self.negative_prompt,
                    num_views=self.num_views,
                    apply_prompt_template=self.apply_prompt_template,
                    embodiment_tag_mapping=self.embodiment_tag_mapping,
                ),
            ],
        )
        return ProcessorPipeline(
            steps=steps,
            name="dreamzero_preprocessor",
            to_output=self.to_text_inputs,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        steps = [
            SliceActionStep(action_dim=self._raw_action_dim()),
            # The official policy promotes predicted actions before unnormalizing.
            DeviceStep(device="cpu", float_dtype="float32"),
            UnnormalizerStep(
                features=self._normalizer_features(),
                norm_map=self._normalizer_norm_map(),
                stats=self.dataset_stats,
                device="cpu",
                eps=0.0,
            ),
        ]
        if self.relative_action_keys:
            metadata = getattr(self, "_metadata", None)
            if isinstance(metadata, dict) and self.embodiment_tag in metadata:
                statistics = metadata[self.embodiment_tag]["statistics"]
                action_slices = dreamzero_key_slices(
                    self.action_keys, statistics["action"]
                )
                state_slices = dreamzero_key_slices(
                    self.state_keys, statistics["state"]
                )
            else:
                action_slices = {}
                state_slices = {}
            steps.append(
                DreamZeroRelativeActionStep(
                    relative_action_keys=self.relative_action_keys,
                    action_slices=action_slices,
                    state_slices=state_slices,
                )
            )
        return ProcessorPipeline(
            steps=steps,
            name="dreamzero_postprocessor",
            to_transition=self.action_to_transition,
            to_output=self.transition_to_action_output,
        )

    def postprocess(
        self,
        output: Any,
        *,
        obs: Any | None = None,
    ) -> DreamZeroActionOutput:
        action = (
            output[ACTION]
            if isinstance(output, dict)
            else getattr(output, ACTION, output)
        )
        transition: Transition | Any
        if obs is not None:
            transition = {ACTION: action, OBS: obs}
        else:
            transition = output
        result = self.postprocessor(transition)
        result.normalized_action = (
            action.detach().cpu() if isinstance(action, torch.Tensor) else None
        )
        result.raw_output = output
        return result

    @classmethod
    def from_pretrained(
        cls,
        ckpt: str | Path,
        *,
        tokenizer_name: str = DREAMZERO_DEFAULT_TOKENIZER_NAME,
        tokenizer: Any = None,
        embodiment_tag: str = "oxe_droid",
        action_normalization: str = DREAMZERO_DEFAULT_ACTION_NORMALIZATION,
        relative_action: bool | None = None,
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ) -> DreamZeroProcessor:
        """Build a DreamZero processor from a checkpoint's experiment metadata."""
        metadata = load_dreamzero_metadata(ckpt)
        try:
            conf = load_dreamzero_conf(ckpt)
        except FileNotFoundError:
            conf = {}
        state_keys = (
            dreamzero_modality_keys_from_conf(
                conf,
                embodiment_tag=embodiment_tag,
                modality="state",
            )
            or DREAMZERO_DEFAULT_STATE_KEYS
        )
        action_keys = (
            dreamzero_modality_keys_from_conf(
                conf,
                embodiment_tag=embodiment_tag,
                modality="action",
            )
            or DREAMZERO_DEFAULT_ACTION_KEYS
        )
        dataset_stats = dreamzero_metadata_stats(
            metadata,
            embodiment_tag=embodiment_tag,
            state_keys=state_keys,
            action_keys=action_keys,
        )
        if relative_action is None:
            relative_keys = dreamzero_relative_action_keys_from_conf(conf)
        elif relative_action:
            relative_keys = ("joint_position",)
        else:
            relative_keys = ()
        processor = cls(
            tokenizer_name=tokenizer_name,
            tokenizer=tokenizer,
            embodiment_tag=embodiment_tag,
            dataset_stats=dataset_stats,
            state_keys=state_keys,
            action_keys=action_keys,
            action_normalization=action_normalization,
            relative_action_keys=relative_keys,
            metadata=metadata,
            device=device,
            params_dtype=params_dtype,
            **kwargs,
        )
        return processor


def make_dreamzero_processors(
    **kwargs: Any,
) -> tuple[ProcessorPipeline, ProcessorPipeline]:
    proc = DreamZeroProcessor(**kwargs)
    return proc.preprocessor, proc.postprocessor


__all__ = [
    "DREAMZERO_DEFAULT_TOKENIZER_NAME",
    "DreamZeroProcessedInputs",
    "DreamZeroProcessor",
    "DreamZeroTextInputs",
    "make_dreamzero_processors",
]
