"""LingBot-VLA 2.0 pre/post processor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from phyai_utils_tools.models.lingbot_v2.steps_lingbotv2 import (
    IMAGE_GRID_THW,
    IMAGE_MASKS,
    NOISE,
    LingBotV2DeviceStep,
    LingBotV2PadStateStep,
    LingBotV2PromptPrepareStep,
    Qwen3VLImagePackStep,
)
from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline
from phyai_utils_tools.processing.steps import (
    FeatureType,
    NormalizationMode,
    NormalizerStep,
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

LINGBOT_V2_DEFAULT_PROCESSOR_NAME = "Qwen/Qwen3-VL-4B-Instruct"
STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
PRE_CONFIG_FILENAME = "policy_preprocessor.json"
POST_CONFIG_FILENAME = "policy_postprocessor.json"


@dataclass
class LingBotV2ProcessedInputs:
    """Model-ready fields matching ``LingBotV2Request`` one-to-one."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    image_masks: torch.Tensor
    input_ids: torch.Tensor
    lang_lens: torch.Tensor
    state: torch.Tensor
    noise: torch.Tensor | None = None


def load_hf_processor(name_or_path: str, *, trust_remote_code: bool) -> Any:
    """Load the Qwen3-VL processor without adding a dependency on ``phyai``."""

    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        name_or_path,
        padding_side="right",
        trust_remote_code=trust_remote_code,
    )


def resolve_processor_components(
    *,
    processor_name: str,
    processor: Any,
    tokenizer: Any,
    image_processor: Any,
    trust_remote_code: bool,
) -> tuple[Any, Any, Any]:
    """Resolve injected or HuggingFace-loaded processor components."""

    hf_processor = processor
    if hf_processor is None and (tokenizer is None or image_processor is None):
        hf_processor = load_hf_processor(
            processor_name,
            trust_remote_code=trust_remote_code,
        )
    if tokenizer is None and hf_processor is not None:
        tokenizer = getattr(hf_processor, "tokenizer", None)
    if image_processor is None and hf_processor is not None:
        image_processor = getattr(hf_processor, "image_processor", None)
    if tokenizer is None:
        raise ValueError("LingBotV2Processor requires a tokenizer.")
    if image_processor is None:
        raise ValueError("LingBotV2Processor requires a Qwen3-VL image processor.")
    return hf_processor, tokenizer, image_processor


def canonical_stats(
    dataset_stats: dict[str, Any] | None,
) -> dict[str, dict[str, Any]] | None:
    """Accept either a direct stats dict or ``{"norm_stats": ...}``."""

    if not dataset_stats:
        return None
    nested = dataset_stats.get("norm_stats")
    if isinstance(nested, dict):
        return nested
    return dataset_stats


def stats_from_normalizer(
    normalizer: NormalizerStep | UnnormalizerStep | None,
) -> dict[str, dict[str, torch.Tensor]] | None:
    """Reconstruct nested dataset statistics from a loaded sidecar state."""

    if normalizer is None:
        return None
    nested: dict[str, dict[str, torch.Tensor]] = {}
    for flat_key, tensor in normalizer.state_dict().items():
        feature, statistic = flat_key.rsplit(".", 1)
        nested.setdefault(feature, {})[statistic] = tensor.detach().cpu().clone()
    return nested or None


def features_for_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Declare canonical state/action features present in ``dataset_stats``."""

    features: dict[str, dict[str, Any]] = {}
    if not dataset_stats:
        return features
    if STATE_FEATURE in dataset_stats:
        features[STATE_FEATURE] = {
            "type": FeatureType.STATE.value,
            "shape": [],
        }
    if ACTION_FEATURE in dataset_stats:
        features[ACTION_FEATURE] = {
            "type": FeatureType.ACTION.value,
            "shape": [],
        }
    return features


class LingBotV2Processor(BaseModelProcessor):
    """Convert canonical robot observations to LingBot-VLA 2.0 request tensors.

    Robot-specific feature mapping and relative-pose conversion stay outside
    this model processor. ``state`` is expected to be an already ordered flat
    vector; optional canonical ``observation.state`` and ``action`` statistics
    provide quantile normalization and inverse action normalization.
    """

    def __init__(
        self,
        *,
        processor_name: str = LINGBOT_V2_DEFAULT_PROCESSOR_NAME,
        processor: Any = None,
        tokenizer: Any = None,
        image_processor: Any = None,
        num_images: int = 3,
        num_channels: int = 3,
        patch_vector_dim: int = 1536,
        max_patches_per_image: int | None = None,
        tokenizer_max_length: int = 72,
        max_state_dim: int = 55,
        action_dim: int | None = 55,
        dataset_stats: dict[str, Any] | None = None,
        normalization_mode: str | NormalizationMode = NormalizationMode.QUANTILES,
        normalization_eps: float = 1e-6,
        use_chat_template: bool = True,
        convert_unit_float_to_uint8: bool = True,
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ) -> None:
        if num_images <= 0:
            raise ValueError("num_images must be positive.")
        if tokenizer_max_length <= 0:
            raise ValueError("tokenizer_max_length must be positive.")
        if max_state_dim <= 0:
            raise ValueError("max_state_dim must be positive.")
        if action_dim is not None and action_dim <= 0:
            raise ValueError("action_dim must be positive when provided.")
        if normalization_eps <= 0:
            raise ValueError("normalization_eps must be positive.")

        (
            self.processor,
            self.tokenizer,
            self.image_processor,
        ) = resolve_processor_components(
            processor_name=processor_name,
            processor=processor,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=trust_remote_code,
        )
        self.processor_name = processor_name
        self.num_images = int(num_images)
        self.num_channels = int(num_channels)
        self.patch_vector_dim = int(patch_vector_dim)
        self.max_patches_per_image = max_patches_per_image
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.max_state_dim = int(max_state_dim)
        self.action_dim = action_dim
        self.dataset_stats = canonical_stats(dataset_stats)
        self.normalization_mode = NormalizationMode(normalization_mode)
        self.normalization_eps = float(normalization_eps)
        self.use_chat_template = bool(use_chat_template)
        self.convert_unit_float_to_uint8 = bool(convert_unit_float_to_uint8)
        self.device = device
        self.params_dtype = params_dtype
        self.trust_remote_code = bool(trust_remote_code)
        super().__init__()

    @staticmethod
    def to_inputs(transition: Transition) -> LingBotV2ProcessedInputs:
        """Validate and extract the typed scheduler handoff."""

        result = LingBotV2ProcessedInputs(
            pixel_values=transition[PIXEL_VALUES],
            image_grid_thw=transition[IMAGE_GRID_THW],
            image_masks=transition[IMAGE_MASKS],
            input_ids=transition[INPUT_IDS],
            lang_lens=transition[LANG_LENS],
            state=transition[STATE],
            noise=transition.get(NOISE),
        )
        batch_size = int(result.pixel_values.shape[0])
        batch_fields = {
            "image_grid_thw": result.image_grid_thw,
            "image_masks": result.image_masks,
            "input_ids": result.input_ids,
            "lang_lens": result.lang_lens,
            "state": result.state,
        }
        if result.noise is not None:
            batch_fields["noise"] = result.noise
        for name, tensor in batch_fields.items():
            if int(tensor.shape[0]) != batch_size:
                raise ValueError(
                    f"{name} batch size {tensor.shape[0]} does not match "
                    f"pixel_values batch size {batch_size}."
                )
        return result

    @staticmethod
    def action_to_transition(action: torch.Tensor) -> Transition:
        return {ACTION: action}

    @staticmethod
    def transition_to_action(transition: Transition) -> torch.Tensor:
        return transition[ACTION]

    def norm_map(self) -> dict[str, str]:
        return {
            FeatureType.VISUAL.value: NormalizationMode.IDENTITY.value,
            FeatureType.STATE.value: self.normalization_mode.value,
            FeatureType.ACTION.value: self.normalization_mode.value,
        }

    def build_preprocessor(self) -> ProcessorPipeline:
        steps = [
            Qwen3VLImagePackStep(
                image_processor=self.image_processor,
                processor_name=self.processor_name,
                num_images=self.num_images,
                num_channels=self.num_channels,
                patch_vector_dim=self.patch_vector_dim,
                max_patches_per_image=self.max_patches_per_image,
                convert_unit_float_to_uint8=(self.convert_unit_float_to_uint8),
            ),
            NormalizerStep(
                features=features_for_stats(self.dataset_stats),
                norm_map=self.norm_map(),
                stats=self.dataset_stats,
                device=self.device,
                eps=self.normalization_eps,
            ),
            LingBotV2PadStateStep(max_state_dim=self.max_state_dim),
            LingBotV2PromptPrepareStep(
                tokenizer=self.tokenizer,
                use_chat_template=self.use_chat_template,
            ),
            TokenizerStep(
                tokenizer=self.tokenizer,
                max_length=self.tokenizer_max_length,
                tokenizer_name=self.processor_name,
            ),
            LingBotV2DeviceStep(
                device=self.device,
                float_dtype=self.params_dtype,
            ),
        ]
        return ProcessorPipeline(
            steps=steps,
            name="lingbot_v2_preprocessor",
            to_output=self.to_inputs,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        steps = [
            SliceActionStep(action_dim=self.action_dim),
            UnnormalizerStep(
                features=features_for_stats(self.dataset_stats),
                norm_map=self.norm_map(),
                stats=self.dataset_stats,
                device=self.device,
                eps=self.normalization_eps,
            ),
            LingBotV2DeviceStep(
                device="cpu",
                float_dtype=torch.float32,
            ),
        ]
        return ProcessorPipeline(
            steps=steps,
            name="lingbot_v2_postprocessor",
            to_transition=self.action_to_transition,
            to_output=self.transition_to_action,
        )

    @classmethod
    def from_pretrained(
        cls,
        ckpt: str | Path,
        *,
        processor_name: str = LINGBOT_V2_DEFAULT_PROCESSOR_NAME,
        processor: Any = None,
        tokenizer: Any = None,
        image_processor: Any = None,
        num_images: int | None = None,
        max_patches_per_image: int | None = None,
        tokenizer_max_length: int | None = None,
        max_state_dim: int | None = None,
        action_dim: int | None = None,
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        **hub_kwargs: Any,
    ) -> LingBotV2Processor:
        """Load saved pipelines while injecting live Qwen processor objects."""

        hf_processor, tokenizer, image_processor = resolve_processor_components(
            processor_name=processor_name,
            processor=processor,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=trust_remote_code,
        )
        pre_overrides: dict[str, dict[str, Any]] = {}
        if num_images is not None:
            pre_overrides.setdefault(
                "lingbot_v2_qwen3vl_image_pack_step",
                {},
            )["num_images"] = num_images
        if max_patches_per_image is not None:
            pre_overrides.setdefault(
                "lingbot_v2_qwen3vl_image_pack_step",
                {},
            )["max_patches_per_image"] = max_patches_per_image
        if tokenizer_max_length is not None:
            pre_overrides["tokenizer_processor"] = {"max_length": tokenizer_max_length}
        if max_state_dim is not None:
            pre_overrides["lingbot_v2_pad_state_step"] = {
                "max_state_dim": max_state_dim
            }

        pre = ProcessorPipeline.from_pretrained(
            ckpt,
            PRE_CONFIG_FILENAME,
            overrides=pre_overrides,
            step_kwargs={
                "lingbot_v2_qwen3vl_image_pack_step": {
                    "image_processor": image_processor,
                    "processor_name": processor_name,
                },
                "lingbot_v2_prompt_prepare_step": {"tokenizer": tokenizer},
                "tokenizer_processor": {
                    "tokenizer": tokenizer,
                    "tokenizer_name": processor_name,
                },
                "normalizer_processor": {"device": device},
                "lingbot_v2_device_step": {
                    "device": device,
                    "float_dtype": params_dtype,
                },
            },
            **hub_kwargs,
        )
        post_step_kwargs: dict[str, dict[str, Any]] = {
            "unnormalizer_processor": {"device": device},
            "lingbot_v2_device_step": {
                "device": "cpu",
                "float_dtype": torch.float32,
            },
        }
        if action_dim is not None:
            post_step_kwargs["slice_action_step"] = {"action_dim": action_dim}
        post = ProcessorPipeline.from_pretrained(
            ckpt,
            POST_CONFIG_FILENAME,
            step_kwargs=post_step_kwargs,
            **hub_kwargs,
        )

        image_step = next(
            step for step in pre.steps if isinstance(step, Qwen3VLImagePackStep)
        )
        state_step = next(
            step for step in pre.steps if isinstance(step, LingBotV2PadStateStep)
        )
        text_step = next(step for step in pre.steps if isinstance(step, TokenizerStep))
        prompt_step = next(
            step for step in pre.steps if isinstance(step, LingBotV2PromptPrepareStep)
        )
        slice_step = next(
            step for step in post.steps if isinstance(step, SliceActionStep)
        )
        normalizer = next(
            (step for step in pre.steps if isinstance(step, NormalizerStep)),
            None,
        )
        norm_config = normalizer.get_config() if normalizer is not None else {}
        norm_map = norm_config.get("norm_map", {})

        obj = cls.__new__(cls)
        obj.processor = hf_processor
        obj.tokenizer = tokenizer
        obj.image_processor = image_processor
        obj.processor_name = processor_name
        obj.num_images = image_step.num_images
        obj.num_channels = image_step.num_channels
        obj.patch_vector_dim = image_step.patch_vector_dim
        obj.max_patches_per_image = image_step.max_patches_per_image
        obj.tokenizer_max_length = text_step.max_length
        obj.max_state_dim = state_step.max_state_dim
        obj.action_dim = slice_step.action_dim
        obj.dataset_stats = stats_from_normalizer(normalizer)
        obj.normalization_mode = NormalizationMode(
            norm_map.get(
                FeatureType.STATE.value,
                NormalizationMode.QUANTILES.value,
            )
        )
        obj.normalization_eps = float(norm_config.get("eps", 1e-6))
        obj.use_chat_template = prompt_step.use_chat_template
        obj.convert_unit_float_to_uint8 = image_step.convert_unit_float_to_uint8
        obj.device = device
        obj.params_dtype = params_dtype
        obj.trust_remote_code = bool(trust_remote_code)

        pre.name = "lingbot_v2_preprocessor"
        pre.to_output = cls.to_inputs
        post.name = "lingbot_v2_postprocessor"
        post.to_transition = cls.action_to_transition
        post.to_output = cls.transition_to_action
        obj._preprocessor = pre
        obj._postprocessor = post
        return obj

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save serializable pipelines and normalization sidecars."""

        self.preprocessor.save_pretrained(
            save_directory,
            config_filename=PRE_CONFIG_FILENAME,
        )
        self.postprocessor.save_pretrained(
            save_directory,
            config_filename=POST_CONFIG_FILENAME,
        )


def make_lingbot_v2_processors(
    **kwargs: Any,
) -> tuple[ProcessorPipeline, ProcessorPipeline]:
    """Return the LingBot V2 preprocessing and postprocessing pipelines."""

    processor = LingBotV2Processor(**kwargs)
    return processor.preprocessor, processor.postprocessor


__all__ = [
    "LINGBOT_V2_DEFAULT_PROCESSOR_NAME",
    "LingBotV2ProcessedInputs",
    "LingBotV2Processor",
    "make_lingbot_v2_processors",
]
