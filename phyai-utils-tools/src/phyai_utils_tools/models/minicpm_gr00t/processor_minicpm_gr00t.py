"""MiniCPM-GR00T processor — pre/post pipelines mirroring the reference stack.

Follows the :class:`~phyai_utils_tools.models.pi05.processor_pi05.PI05Processor`
pattern: a :class:`~phyai_utils_tools.processing.base_processor.BaseModelProcessor`
subclass composes registered steps into a preprocess pipeline (raw cameras +
instruction + state -> canonical model inputs) and a postprocess pipeline
(raw engine action chunk -> CPU float32 actions).

The preprocess output :class:`MiniCPMGR00TProcessedInputs` fields line up 1:1
with phyai's ``MiniCPMGR00TRequest``, so the caller builds the request
directly::

    processed = processor.preprocess(
        {"images": [base_cam, wrist_cam], "task": instruction, "state": state}
    )
    request = MiniCPMGR00TRequest(
        input_ids=processed.input_ids,
        attention_mask=processed.attention_mask,
        pixel_values=processed.pixel_values,
        target_sizes=processed.target_sizes,
        state=processed.state,
    )

The heavy tokenize/vision-slice work is the MiniCPM-V 4.6 ``AutoProcessor``
itself (dependency-injected, like pi0.5's tokenizer), so the produced tensors
are byte-identical to the original EmbodyEvalKit inference path. The
checkpoint stores actions unnormalized, so the postprocess is device/dtype
canonicalization plus an optional action slice — there are no dataset stats to
unnormalize with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from phyai_utils_tools.models.minicpm_gr00t.ops_minicpm_gr00t import (
    MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE,
)
from phyai_utils_tools.models.minicpm_gr00t.steps_minicpm_gr00t import (
    ATTENTION_MASK,
    TARGET_SIZES,
    MiniCPMGR00TChatTemplateStep,
    MiniCPMGR00TImagePrepareStep,
    MiniCPMGR00TPromptStep,
    MiniCPMGR00TStateStep,
)
from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline
from phyai_utils_tools.processing.steps import DeviceStep, SliceActionStep
from phyai_utils_tools.processing.transition import (
    ACTION,
    INPUT_IDS,
    PIXEL_VALUES,
    STATE,
    Transition,
)


@dataclass
class MiniCPMGR00TProcessedInputs:
    """Canonical preprocessed MiniCPM-GR00T inputs — the engine handoff.

    Field names line up 1:1 with phyai's ``MiniCPMGR00TRequest``. All tensors
    stay on CPU; the phyai runner owns the device transfer (unlike pi0.5,
    whose request carries device tensors).
    """

    input_ids: torch.Tensor  # (1, S) int64
    attention_mask: torch.Tensor  # (1, S) int64
    pixel_values: torch.Tensor  # MiniCPM-V NaViT patch tensor
    target_sizes: torch.Tensor  # (num_images, 2) int32 patch grids
    state: torch.Tensor  # (1, 1, state_dim) float32


class MiniCPMGR00TProcessor(BaseModelProcessor):
    """MiniCPM-GR00T pre/post processor.

    ``processor`` is the MiniCPM-V 4.6 ``AutoProcessor`` object (injected;
    this package never hardcodes a checkpoint path). All other parameters are
    primitives mirroring the checkpoint's runtime contract: two cameras
    PIL-resized to ``image_size``, the unified-80D prompt template, an 80-D
    state, and a 30x80 unnormalized action chunk.
    """

    def __init__(
        self,
        *,
        processor: Any,
        processor_name: str | None = None,
        image_size: tuple[int, int] | list[int] = (448, 448),
        num_images: int = 2,
        prompt_template: str = MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE,
        state_dim: int = 80,
        action_dim: int | None = None,
    ) -> None:
        if processor is None:
            raise ValueError("MiniCPMGR00TProcessor requires a processor object.")
        self.processor = processor
        self.processor_name = processor_name
        self.image_size = tuple(int(side) for side in image_size)
        self.num_images = int(num_images)
        self.prompt_template = prompt_template
        self.state_dim = int(state_dim)
        self.action_dim = action_dim
        super().__init__()

    # -- adapters ------------------------------------------------------- #

    @staticmethod
    def _to_inputs(transition: Transition) -> MiniCPMGR00TProcessedInputs:
        return MiniCPMGR00TProcessedInputs(
            input_ids=transition[INPUT_IDS],
            attention_mask=transition[ATTENTION_MASK],
            pixel_values=transition[PIXEL_VALUES],
            target_sizes=transition[TARGET_SIZES],
            state=transition[STATE],
        )

    @staticmethod
    def _action_to_transition(action: torch.Tensor) -> Transition:
        return {ACTION: action}

    @staticmethod
    def _transition_to_action(transition: Transition) -> torch.Tensor:
        return transition[ACTION].to(torch.float32)

    # -- pipelines ------------------------------------------------------- #

    def build_preprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[
                MiniCPMGR00TImagePrepareStep(
                    image_size=self.image_size,
                    num_images=self.num_images,
                ),
                MiniCPMGR00TPromptStep(prompt_template=self.prompt_template),
                MiniCPMGR00TChatTemplateStep(
                    processor=self.processor,
                    processor_name=self.processor_name,
                ),
                MiniCPMGR00TStateStep(state_dim=self.state_dim),
            ],
            name="minicpm_gr00t_preprocessor",
            to_output=self._to_inputs,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[
                SliceActionStep(action_dim=self.action_dim),
                DeviceStep(device="cpu"),
            ],
            name="minicpm_gr00t_postprocessor",
            to_transition=self._action_to_transition,
            to_output=self._transition_to_action,
        )


def make_minicpm_gr00t_processors(
    **kwargs: Any,
) -> tuple[ProcessorPipeline, ProcessorPipeline]:
    """Build a :class:`MiniCPMGR00TProcessor` and return its two pipelines."""
    proc = MiniCPMGR00TProcessor(**kwargs)
    return proc.preprocessor, proc.postprocessor


__all__ = [
    "MiniCPMGR00TProcessedInputs",
    "MiniCPMGR00TProcessor",
    "make_minicpm_gr00t_processors",
]
