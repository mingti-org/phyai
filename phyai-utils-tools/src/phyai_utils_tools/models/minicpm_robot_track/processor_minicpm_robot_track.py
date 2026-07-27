"""Raw image/text processor for MiniCPM-RobotTrack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from phyai_utils_tools.models.minicpm_robot_track.steps_minicpm_robot_track import (
    MiniCPMRobotTrackImagePrepareStep,
    MiniCPMRobotTrackTokenizerStep,
)
from phyai_utils_tools.processing.base_processor import BaseModelProcessor
from phyai_utils_tools.processing.pipeline import ProcessorPipeline
from phyai_utils_tools.processing.steps import DeviceStep
from phyai_utils_tools.processing.transition import (
    ACTION,
    INPUT_IDS,
    LANG_LENS,
    PIXEL_VALUES,
    Transition,
)


@dataclass
class MiniCPMRobotTrackProcessedInputs:
    frames: torch.Tensor
    input_ids: torch.Tensor
    text_lengths: torch.Tensor


class MiniCPMRobotTrackProcessor(BaseModelProcessor):
    """Convert raw RGB frames and an instruction to the engine request contract."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        tokenizer_name: str | None = None,
        image_size: int = 384,
        history_frames: int = 31,
        text_capacity: int = 35,
        resize_workers: int | None = None,
    ) -> None:
        if tokenizer is None:
            raise ValueError("MiniCPMRobotTrackProcessor requires a tokenizer.")
        if image_size <= 0 or history_frames <= 0 or text_capacity <= 0:
            raise ValueError("Processor sizes must be positive.")
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name
        self.image_size = int(image_size)
        self.history_frames = int(history_frames)
        self.text_capacity = int(text_capacity)
        self.resize_workers = resize_workers
        super().__init__()

    @staticmethod
    def to_inputs(transition: Transition) -> MiniCPMRobotTrackProcessedInputs:
        return MiniCPMRobotTrackProcessedInputs(
            frames=transition[PIXEL_VALUES],
            input_ids=transition[INPUT_IDS],
            text_lengths=transition[LANG_LENS],
        )

    @staticmethod
    def action_to_transition(action: torch.Tensor) -> Transition:
        return {ACTION: action}

    @staticmethod
    def transition_to_action(transition: Transition) -> torch.Tensor:
        return transition[ACTION].to(torch.float32)

    def build_preprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[
                MiniCPMRobotTrackImagePrepareStep(
                    image_size=self.image_size,
                    history_frames=self.history_frames,
                    resize_workers=self.resize_workers,
                ),
                MiniCPMRobotTrackTokenizerStep(
                    tokenizer=self.tokenizer,
                    text_capacity=self.text_capacity,
                    tokenizer_name=self.tokenizer_name,
                ),
            ],
            name="minicpm_robot_track_preprocessor",
            to_output=self.to_inputs,
        )

    def build_postprocessor(self) -> ProcessorPipeline:
        return ProcessorPipeline(
            steps=[DeviceStep(device="cpu", float_dtype=torch.float32)],
            name="minicpm_robot_track_postprocessor",
            to_transition=self.action_to_transition,
            to_output=self.transition_to_action,
        )

    def close(self) -> None:
        """Release resources owned by preprocessing steps."""

        for step in self.preprocessor.steps:
            if isinstance(step, MiniCPMRobotTrackImagePrepareStep):
                step.close()


def make_minicpm_robot_track_processors(
    **kwargs: Any,
) -> tuple[ProcessorPipeline, ProcessorPipeline]:
    processor = MiniCPMRobotTrackProcessor(**kwargs)
    return processor.preprocessor, processor.postprocessor


__all__ = [
    "MiniCPMRobotTrackProcessedInputs",
    "MiniCPMRobotTrackProcessor",
    "make_minicpm_robot_track_processors",
]
