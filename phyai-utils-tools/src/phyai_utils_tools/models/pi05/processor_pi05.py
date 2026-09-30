"""pi0.5 processor — the first :class:`BaseModelProcessor` subclass.

Two construction paths:

* **Programmatic** (``PI05Processor(...)``): composes the steps directly. Used
  for the default / no-checkpoint case. With no ``dataset_stats`` the normalize
  steps are identity (the pi05_base default).
* **From a lerobot checkpoint** (``PI05Processor.from_pretrained(ckpt_dir)``):
  loads the serializable core (normalize / tokenize / device, and
  unnormalize / device) straight from the checkpoint's ``policy_*processor.json``
  + stats sidecars, then **prepends** phyai's vision glue (resize, optional
  pixel-normalize) and **appends** the action slice — those are model-side in
  lerobot, so they're not in the json.

The pipelines flow over the canonical transition dict; the preprocess pipeline
outputs :class:`PI05ProcessedInputs` (fields line up 1:1 with phyai's
``PI05Request``). The processor takes only primitives + an injected tokenizer
(no ``phyai`` import), keeping the package a workspace leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from phyai_utils_tools.models.pi05.steps_pi05 import StateTokenizerPrepareStep
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
    Transition,
)
from phyai_utils_tools.tokenizer import get_tokenizer

PI05_DEFAULT_TOKENIZER_NAME = "google/paligemma-3b-pt-224"

# Canonical lerobot feature names pi05 normalizes (stats dicts are keyed by them).
STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"

# pi05_base's norm_map: state + action quantile-normalized, vision identity.
PI05_NORM_MAP: dict[str, str] = {
    FeatureType.VISUAL.value: NormalizationMode.IDENTITY.value,
    FeatureType.STATE.value: NormalizationMode.QUANTILES.value,
    FeatureType.ACTION.value: NormalizationMode.QUANTILES.value,
}

PRE_CONFIG_FILENAME = "policy_preprocessor.json"
POST_CONFIG_FILENAME = "policy_postprocessor.json"


@dataclass
class PI05ProcessedInputs:
    """Canonical preprocessed pi0.5 inputs — the handoff to the engine.

    Field names line up 1:1 with phyai's ``PI05Request`` so the caller can
    build the request directly: ``PI05Request(pixel_values=..., input_ids=...,
    lang_lens=...)``.
    """

    pixel_values: torch.Tensor  # (B, num_images, C, image_size, image_size)
    input_ids: torch.Tensor  # (B, tokenizer_max_length) int64
    lang_lens: torch.Tensor  # (B,) int64


def _features_for_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Declare ``features`` entries for whichever of state/action have stats.

    Empty when ``dataset_stats`` is ``None`` (⇒ identity normalize). Shapes are
    recorded but unused by the transform; they exist for json fidelity.
    """
    features: dict[str, dict[str, Any]] = {}
    if not dataset_stats:
        return features
    if STATE_FEATURE in dataset_stats:
        features[STATE_FEATURE] = {"type": FeatureType.STATE.value, "shape": []}
    if ACTION_FEATURE in dataset_stats:
        features[ACTION_FEATURE] = {"type": FeatureType.ACTION.value, "shape": []}
    return features


class PI05Processor(BaseModelProcessor):
    """pi0.5 pre/post processor.

    Parameters are primitives mirroring the pi0.5 config (no ``phyai``
    dependency). ``dataset_stats`` is an optional ``{feature_name: {stat: ...}}``
    dict keyed by lerobot feature names (:data:`STATE_FEATURE`,
    :data:`ACTION_FEATURE`); absent ⇒ identity normalization (pi05_base default).
    ``normalize_pixels`` maps float pixels in ``[0, 1]`` or uint8 pixels in
    ``[0, 255]`` to ``[-1, 1]``. ``image_resize_backend="pil"`` selects PIL
    bilinear resizing of RGB uint8 inputs; the default remains ``"torch"``.
    Postprocessing also accepts an action/state transition for checkpoints
    whose serialized pipeline converts deltas to absolute targets.
    """

    def __init__(
        self,
        *,
        image_size: int = 224,
        num_channels: int = 3,
        num_images: int = 3,
        tokenizer_max_length: int = 200,
        action_dim: int | None = None,
        tokenizer_name: str = PI05_DEFAULT_TOKENIZER_NAME,
        tokenizer: Any = None,
        dataset_stats: dict[str, dict[str, Any]] | None = None,
        normalize_pixels: bool = False,
        image_pad_value: float = 0.0,
        image_resize_backend: str = "torch",
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.image_size = int(image_size)
        self.num_channels = int(num_channels)
        self.num_images = int(num_images)
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.action_dim = action_dim
        self.tokenizer_name = tokenizer_name
        self.dataset_stats = dataset_stats
        self.normalize_pixels = bool(normalize_pixels)
        self.image_pad_value = float(image_pad_value)
        self.image_resize_backend = image_resize_backend
        self.device = device
        self.params_dtype = params_dtype
        self.tokenizer = (
            tokenizer if tokenizer is not None else get_tokenizer(tokenizer_name)
        )
        super().__init__()

    # -- adapters ------------------------------------------------------- #

    @staticmethod
    def _to_inputs(transition: Transition) -> PI05ProcessedInputs:
        """Extract the typed :class:`PI05ProcessedInputs` from the final transition."""
        return PI05ProcessedInputs(
            pixel_values=transition[PIXEL_VALUES],
            input_ids=transition[INPUT_IDS],
            lang_lens=transition[LANG_LENS],
        )

    @staticmethod
    def _action_to_transition(action: torch.Tensor | Transition) -> Transition:
        return action.copy() if isinstance(action, dict) else {ACTION: action}

    @staticmethod
    def _transition_to_action(transition: Transition) -> torch.Tensor:
        return transition[ACTION]

    # -- vision glue (phyai-only; not in the lerobot json) -------------- #

    def _vision_steps(self) -> list:
        steps: list = [
            ResizeWithPadStep(
                target_size=self.image_size,
                num_images=self.num_images,
                num_channels=self.num_channels,
                pad_value=self.image_pad_value,
                backend=self.image_resize_backend,
            )
        ]
        if self.normalize_pixels:
            steps.append(NormalizeImageStep())
        return steps

    # -- programmatic build (default / no-ckpt) ------------------------- #

    def build_preprocessor(self) -> ProcessorPipeline:
        steps = self._vision_steps()
        steps += [
            NormalizerStep(
                features=_features_for_stats(self.dataset_stats),
                norm_map=PI05_NORM_MAP,
                stats=self.dataset_stats,
                device=self.device,
            ),
            StateTokenizerPrepareStep(),
            TokenizerStep(
                tokenizer=self.tokenizer,
                max_length=self.tokenizer_max_length,
                tokenizer_name=self.tokenizer_name,
            ),
            DeviceStep(device=self.device, float_dtype=self.params_dtype),
        ]
        return ProcessorPipeline(
            steps=steps,
            name="pi05_preprocessor",
            to_output=self._to_inputs,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        steps = [
            UnnormalizerStep(
                features=_features_for_stats(self.dataset_stats),
                norm_map=PI05_NORM_MAP,
                stats=self.dataset_stats,
                device=self.device,
            ),
            SliceActionStep(action_dim=self.action_dim),
            DeviceStep(device="cpu"),
        ]
        return ProcessorPipeline(
            steps=steps,
            name="pi05_postprocessor",
            to_transition=self._action_to_transition,
            to_output=self._transition_to_action,
        )

    # -- lerobot-checkpoint construction -------------------------------- #

    @classmethod
    def from_pretrained(
        cls,
        ckpt: str | Path,
        *,
        tokenizer: Any = None,
        tokenizer_name: str = PI05_DEFAULT_TOKENIZER_NAME,
        image_size: int = 224,
        num_channels: int = 3,
        num_images: int = 3,
        action_dim: int | None = None,
        normalize_pixels: bool = False,
        image_pad_value: float = 0.0,
        image_resize_backend: str = "torch",
        device: torch.device | str = "cpu",
        params_dtype: torch.dtype = torch.bfloat16,
        **hub_kwargs: Any,
    ) -> PI05Processor:
        """Build a processor from a lerobot-format checkpoint dir / HF repo id.

        Loads ``policy_preprocessor.json`` + ``policy_postprocessor.json`` (and
        any stats sidecars) via :meth:`ProcessorPipeline.from_pretrained`,
        injecting the tokenizer object + the model ``device`` into the steps the
        json can't carry, then prepends phyai's vision glue (resize / optional
        pixel-normalize) and appends the action slice. The result behaves like a
        programmatically-built :class:`PI05Processor` but with the checkpoint's
        exact normalize stats / tokenizer config.

        The preprocess ``device_processor`` is steered to ``device`` (so model
        inputs land on the model device); the postprocess ``device_processor``
        keeps the checkpoint's own device (``cpu`` for pi05_base — actions
        return to CPU), so it is **not** overridden here. The unnormalizer's
        stats still go to ``device`` via its own ``device`` kwarg for the math.
        """
        tok = tokenizer if tokenizer is not None else get_tokenizer(tokenizer_name)

        pre = ProcessorPipeline.from_pretrained(
            ckpt,
            PRE_CONFIG_FILENAME,
            step_kwargs={
                "tokenizer_processor": {"tokenizer": tok},
                "device_processor": {"device": device},
            },
            **hub_kwargs,
        )
        post = ProcessorPipeline.from_pretrained(
            ckpt,
            POST_CONFIG_FILENAME,
            # Do NOT override the postprocess device_processor — the checkpoint
            # sets it to cpu (return actions to host). Only put the
            # unnormalizer's stat tensors on the model device for the math.
            step_kwargs={"unnormalizer_processor": {"device": device}},
            **hub_kwargs,
        )

        # Build the instance without re-running build_preprocessor/postprocessor
        # (those are the programmatic path); we splice the loaded pipelines below.
        obj = cls.__new__(cls)
        obj.image_size = int(image_size)
        obj.num_channels = int(num_channels)
        obj.num_images = int(num_images)
        obj.tokenizer_max_length = next(
            (s.max_length for s in pre.steps if isinstance(s, TokenizerStep)),
            200,
        )
        obj.action_dim = action_dim
        obj.tokenizer_name = tokenizer_name
        obj.dataset_stats = None
        obj.normalize_pixels = bool(normalize_pixels)
        obj.image_pad_value = float(image_pad_value)
        obj.image_resize_backend = image_resize_backend
        obj.device = device
        obj.params_dtype = params_dtype
        obj.tokenizer = tok

        # Prepend phyai vision glue to the loaded preprocess chain.
        pre.steps = [*obj._vision_steps(), *list(pre.steps)]
        pre.name = "pi05_preprocessor"
        pre.to_output = cls._to_inputs

        # Append the action slice to the loaded postprocess chain.
        post.steps = [*list(post.steps), SliceActionStep(action_dim=action_dim)]
        post.name = "pi05_postprocessor"
        post.to_transition = cls._action_to_transition
        post.to_output = cls._transition_to_action

        obj._preprocessor = pre
        obj._postprocessor = post
        return obj

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Write the serializable core as lerobot-format json (+ stats sidecars).

        Emits ``policy_preprocessor.json`` / ``policy_postprocessor.json`` for
        the normalize / tokenize / device (and unnormalize / device) sub-chains
        — the lerobot-portable part. phyai's vision-glue (resize / pixel-norm)
        and the action slice are reconstructed from constructor args on load, so
        they are excluded from the json (matching lerobot, which does image
        resize model-side).
        """
        save_dir = Path(save_directory)
        pre_core = ProcessorPipeline(
            steps=[s for s in self._preprocessor.steps if not _is_vision_glue(s)],
            name="policy_preprocessor",
        )
        post_core = ProcessorPipeline(
            steps=[
                s
                for s in self._postprocessor.steps
                if not isinstance(s, SliceActionStep)
            ],
            name="policy_postprocessor",
        )
        pre_core.save_pretrained(save_dir, config_filename=PRE_CONFIG_FILENAME)
        post_core.save_pretrained(save_dir, config_filename=POST_CONFIG_FILENAME)


def _is_vision_glue(step: Any) -> bool:
    """True for phyai-only vision steps that aren't part of the lerobot json."""
    return isinstance(step, (ResizeWithPadStep, NormalizeImageStep))


def make_pi05_processors(**kwargs: Any) -> tuple[ProcessorPipeline, ProcessorPipeline]:
    """Build a :class:`PI05Processor` and return its ``(preprocessor, postprocessor)``."""
    proc = PI05Processor(**kwargs)
    return proc.preprocessor, proc.postprocessor


__all__ = [
    "PI05_DEFAULT_TOKENIZER_NAME",
    "PI05ProcessedInputs",
    "PI05Processor",
    "make_pi05_processors",
]
