"""MiniCPM-GR00T model-specific processor steps.

Four registered steps compose the checkpoint's exact preprocessing:

* :class:`MiniCPMGR00TImagePrepareStep` — raw per-camera images to uint8 RGB at
  the model input size (:func:`~phyai_utils_tools.models.minicpm_gr00t.ops_minicpm_gr00t.to_uint8_image`).
* :class:`MiniCPMGR00TPromptStep` — ``TASK`` instruction to the checkpoint's
  ``PROMPT`` string.
* :class:`MiniCPMGR00TChatTemplateStep` — MiniCPM-V 4.6 ``AutoProcessor``
  chat-template encode (dependency-injected object, like the generic
  :class:`~phyai_utils_tools.processing.steps.text_steps.TokenizerStep`),
  producing ``INPUT_IDS`` / ``ATTENTION_MASK`` / ``PIXEL_VALUES`` /
  ``TARGET_SIZES``.
* :class:`MiniCPMGR00TStateStep` — proprioceptive ``STATE`` to the canonical
  ``(1, 1, state_dim)`` float32 tensor.

The MiniCPM-V processor emits two extra tensors the generic transition keys do
not cover, so this module defines :data:`ATTENTION_MASK` and
:data:`TARGET_SIZES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from phyai_utils_tools.models.minicpm_gr00t.ops_minicpm_gr00t import (
    MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE,
    format_prompt,
    state_to_batched_tensor,
    to_uint8_image,
)
from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    IMAGES,
    INPUT_IDS,
    PIXEL_VALUES,
    PROMPT,
    STATE,
    TASK,
    Transition,
)

# MiniCPM-V-specific processed-input keys (alongside the generic ones).
ATTENTION_MASK = "attention_mask"  # (1, S) int64
TARGET_SIZES = "target_sizes"  # (num_images, 2) int32 patch grids


@ProcessorStepRegistry.register("minicpm_gr00t_image_prepare_step")
@dataclass
class MiniCPMGR00TImagePrepareStep(ProcessorStep):
    """Convert raw camera images to uint8 RGB at the model input size.

    Reads ``IMAGES`` (a list of ``num_images`` HWC arrays / tensors for one
    sample) and rewrites it with resized contiguous uint8 arrays, matching the
    reference adapter's PIL resize exactly.
    """

    image_size: tuple[int, int] = (448, 448)
    num_images: int = 2

    def __call__(self, transition: Transition) -> Transition:
        images = transition.get(IMAGES)
        if not images:
            raise ValueError("MiniCPMGR00TImagePrepareStep requires an IMAGES list.")
        images = list(images)
        if len(images) != self.num_images:
            raise ValueError(
                f"MiniCPM-GR00T expects exactly {self.num_images} images, "
                f"got {len(images)}."
            )
        out = transition.copy()
        out[IMAGES] = [
            to_uint8_image(image, tuple(self.image_size)) for image in images
        ]
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "image_size": list(self.image_size),
            "num_images": self.num_images,
        }


@ProcessorStepRegistry.register("minicpm_gr00t_prompt_step")
@dataclass
class MiniCPMGR00TPromptStep(ProcessorStep):
    """Render the checkpoint's prompt template from the ``TASK`` instruction."""

    prompt_template: str = MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE

    def __call__(self, transition: Transition) -> Transition:
        task = transition.get(TASK)
        if task is None:
            raise ValueError("MiniCPMGR00TPromptStep requires a TASK entry.")
        if isinstance(task, (list, tuple)):
            if len(task) != 1:
                raise ValueError(
                    f"MiniCPM-GR00T is batch-one; got {len(task)} task strings."
                )
            task = task[0]
        out = transition.copy()
        out[PROMPT] = format_prompt(str(task), self.prompt_template)
        return out

    def get_config(self) -> dict[str, Any]:
        return {"prompt_template": self.prompt_template}


@ProcessorStepRegistry.register("minicpm_gr00t_chat_template_step")
@dataclass
class MiniCPMGR00TChatTemplateStep(ProcessorStep):
    """Encode images + prompt with the MiniCPM-V 4.6 ``AutoProcessor``.

    ``processor`` is the live HF processor object (dependency-injected; the
    config json carries only ``processor_name``). Produces the canonical
    ``INPUT_IDS`` / ``ATTENTION_MASK`` / ``PIXEL_VALUES`` / ``TARGET_SIZES``
    tensors, using the identical ``apply_chat_template`` call as the reference
    inference stack.
    """

    processor: Any = field(repr=False, default=None)
    processor_name: str | None = None

    def __call__(self, transition: Transition) -> Transition:
        if self.processor is None:
            raise ValueError(
                "MiniCPMGR00TChatTemplateStep requires a `processor` object."
            )
        images = transition.get(IMAGES)
        prompt = transition.get(PROMPT)
        if not images:
            raise ValueError("MiniCPMGR00TChatTemplateStep requires IMAGES.")
        if prompt is None:
            raise ValueError("MiniCPMGR00TChatTemplateStep requires a PROMPT entry.")
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": str(prompt)})
        encoded = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": False},
        )
        if "target_sizes" not in encoded:
            raise KeyError("MiniCPM-V processor output is missing target_sizes.")
        out = transition.copy()
        out[INPUT_IDS] = encoded["input_ids"].to(torch.int64)
        out[ATTENTION_MASK] = encoded["attention_mask"].to(torch.int64)
        out[PIXEL_VALUES] = encoded["pixel_values"]
        out[TARGET_SIZES] = encoded["target_sizes"]
        return out

    def get_config(self) -> dict[str, Any]:
        # The processor object is never serialized; processor_name lets a
        # loader re-fetch it (mirroring tokenizer_processor's tokenizer_name).
        return {"processor_name": self.processor_name}


@ProcessorStepRegistry.register("minicpm_gr00t_state_step")
@dataclass
class MiniCPMGR00TStateStep(ProcessorStep):
    """Canonicalize ``STATE`` to the ``(1, 1, state_dim)`` float32 tensor."""

    state_dim: int = 80

    def __call__(self, transition: Transition) -> Transition:
        out = transition.copy()
        out[STATE] = state_to_batched_tensor(transition.get(STATE), self.state_dim)
        return out

    def get_config(self) -> dict[str, Any]:
        return {"state_dim": self.state_dim}


__all__ = [
    "ATTENTION_MASK",
    "TARGET_SIZES",
    "MiniCPMGR00TChatTemplateStep",
    "MiniCPMGR00TImagePrepareStep",
    "MiniCPMGR00TPromptStep",
    "MiniCPMGR00TStateStep",
]
