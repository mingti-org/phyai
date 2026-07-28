"""Single-device MiniCPM-RobotTrack scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.models.minicpm_robot_track.model_runner_minicpm_robot_track import (
    MiniCPMRobotTrackForwardBatch,
    MiniCPMRobotTrackImageForwardBatch,
    MiniCPMRobotTrackImageForwardOutput,
    MiniCPMRobotTrackModelRunner,
)
from phyai.runtime.schedule import Scheduler


@dataclass
class MiniCPMRobotTrackRequest:
    input_ids: torch.Tensor
    text_lengths: torch.Tensor
    coarse_tokens: torch.Tensor
    coarse_time_indices: torch.Tensor
    fine_tokens: torch.Tensor
    fine_time_indices: torch.Tensor


@dataclass
class MiniCPMRobotTrackImageRequest:
    """Reference-resized RGB frames for one explicitly identified stream.

    A complete 32-frame request always replaces that stream's history. After a
    complete request with ``frame_index``, callers may send one new frame at a
    time with consecutive frame indices. Repeating the latest frame index is an
    idempotent retry and reuses the committed visual features.
    """

    frames: torch.Tensor
    input_ids: torch.Tensor
    text_lengths: torch.Tensor
    stream_id: str = "default"
    frame_index: int | None = None
    collect_timing: bool = False


MiniCPMRobotTrackImageOutput = MiniCPMRobotTrackImageForwardOutput


class MiniCPMRobotTrackWS1Scheduler(Scheduler):
    def __init__(self, runner: MiniCPMRobotTrackModelRunner) -> None:
        self.runner = runner
        self.config = runner.config
        self.batch_size = runner.batch_size

    def setup(self) -> None:
        self.runner.setup()

    def validate_text_inputs(
        self, input_ids: torch.Tensor, text_lengths: torch.Tensor
    ) -> None:
        expected_ids = (self.batch_size, self.config.text_capacity)
        if tuple(input_ids.shape) != expected_ids:
            raise ValueError(
                f"input_ids must have shape {expected_ids}, got {tuple(input_ids.shape)}."
            )
        self.validate_integer_tensor("input_ids", input_ids)
        expected_lengths = (self.batch_size,)
        if tuple(text_lengths.shape) != expected_lengths:
            raise ValueError(
                "text_lengths must have shape "
                f"{expected_lengths}, got {tuple(text_lengths.shape)}."
            )
        self.validate_integer_tensor("text_lengths", text_lengths)
        # Raising a synchronous ValueError requires a host sync. At batch size 1,
        # copying this scalar is faster than launching device-side range kernels.
        lengths_cpu = text_lengths.detach().to(device="cpu", dtype=torch.long)
        lengths_valid = (lengths_cpu >= 1) & (lengths_cpu <= self.config.text_capacity)
        if not bool(lengths_valid.all()):
            raise ValueError(
                f"text_lengths must be in [1, {self.config.text_capacity}]."
            )

    @staticmethod
    def validate_integer_tensor(name: str, tensor: torch.Tensor) -> None:
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if tensor.dtype not in integer_dtypes:
            raise TypeError(f"{name} must use an integer dtype, got {tensor.dtype}.")

    def validate_token_request(self, request: MiniCPMRobotTrackRequest) -> None:
        config = self.config
        batch_size = self.batch_size
        expected = {
            "input_ids": (batch_size, config.text_capacity),
            "text_lengths": (batch_size,),
            "coarse_tokens": (
                batch_size,
                config.coarse_token_count,
                config.vision_feature_dim,
            ),
            "coarse_time_indices": (batch_size, config.coarse_token_count),
            "fine_tokens": (
                batch_size,
                config.fine_tokens_current_frame,
                config.vision_feature_dim,
            ),
            "fine_time_indices": (
                batch_size,
                config.fine_tokens_current_frame,
            ),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(request, name).shape)
            if actual != shape:
                raise ValueError(f"{name} must have shape {shape}, got {actual}.")
        self.validate_text_inputs(request.input_ids, request.text_lengths)
        self.validate_integer_tensor("coarse_time_indices", request.coarse_time_indices)
        self.validate_integer_tensor("fine_time_indices", request.fine_time_indices)
        for name in ("coarse_tokens", "fine_tokens"):
            tensor = getattr(request, name)
            if not tensor.is_floating_point():
                raise TypeError(
                    f"{name} must use a floating dtype, got {tensor.dtype}."
                )

    def validate_image_request(self, request: MiniCPMRobotTrackImageRequest) -> None:
        self.validate_text_inputs(request.input_ids, request.text_lengths)
        if not isinstance(request.stream_id, str) or not request.stream_id.strip():
            raise ValueError("stream_id must be a non-empty string.")
        if request.frame_index is not None and (
            isinstance(request.frame_index, bool)
            or not isinstance(request.frame_index, int)
        ):
            raise TypeError("frame_index must be an integer or None.")

    @torch.inference_mode()
    def step(
        self, request: MiniCPMRobotTrackRequest | MiniCPMRobotTrackImageRequest
    ) -> torch.Tensor | MiniCPMRobotTrackImageOutput:
        if isinstance(request, MiniCPMRobotTrackImageRequest):
            self.validate_image_request(request)
            output = self.runner.forward(
                MiniCPMRobotTrackImageForwardBatch(
                    frames=request.frames,
                    input_ids=request.input_ids,
                    text_lengths=request.text_lengths,
                    stream_id=request.stream_id,
                    frame_index=request.frame_index,
                    collect_timing=request.collect_timing,
                )
            )
            if not isinstance(output, MiniCPMRobotTrackImageForwardOutput):
                raise TypeError(
                    f"Image runner returned an unexpected type: {type(output).__name__}."
                )
            return output
        if isinstance(request, MiniCPMRobotTrackRequest):
            self.validate_token_request(request)
            output = self.runner.forward(
                MiniCPMRobotTrackForwardBatch(
                    input_ids=request.input_ids,
                    text_lengths=request.text_lengths,
                    coarse_tokens=request.coarse_tokens,
                    coarse_time_indices=request.coarse_time_indices,
                    fine_tokens=request.fine_tokens,
                    fine_time_indices=request.fine_time_indices,
                )
            )
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"Policy runner returned an unexpected type: {type(output).__name__}."
                )
            return output
        raise TypeError(
            f"Unsupported RobotTrack request type: {type(request).__name__}."
        )

    def close(self) -> None:
        self.runner.close()


__all__ = [
    "MiniCPMRobotTrackImageOutput",
    "MiniCPMRobotTrackImageRequest",
    "MiniCPMRobotTrackRequest",
    "MiniCPMRobotTrackWS1Scheduler",
]
