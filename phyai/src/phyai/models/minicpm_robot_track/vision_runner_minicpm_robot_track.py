"""TensorRT vision towers and sliding-window packing for RobotTrack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner

_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD = (0.5, 0.5, 0.5)


@dataclass
class MiniCPMRobotTrackVisionBatch:
    coarse_tokens: torch.Tensor
    fine_tokens: torch.Tensor
    encoded_frames: int
    cached_frames: int
    next_state: MiniCPMRobotTrackVisionState
    cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class MiniCPMRobotTrackVisionState:
    """Committed visual history for one explicitly named client stream."""

    coarse_history: tuple[torch.Tensor, ...]
    fine_tokens: torch.Tensor
    frame_index: int | None


def classify_vision_request(
    *,
    frame_count: int,
    history_frames: int,
    previous_state: MiniCPMRobotTrackVisionState | None,
    frame_index: int | None,
) -> Literal["replace", "append", "reuse"]:
    """Validate an image request and state how it changes stream history."""

    if frame_count == history_frames + 1:
        if frame_index is not None and frame_index < 0:
            raise ValueError("frame_index must be non-negative.")
        return "replace"
    if frame_count != 1:
        raise ValueError(
            f"frames must contain 1 incremental frame or {history_frames + 1} "
            f"complete-window frames, got {frame_count}."
        )
    if previous_state is None:
        raise ValueError(
            "A single-frame request requires an existing stream. Send a complete "
            f"{history_frames + 1}-frame window first."
        )
    if frame_index is None or previous_state.frame_index is None:
        raise ValueError(
            "Single-frame requests require frame_index, and the preceding complete "
            "window must also specify its final frame_index."
        )
    if frame_index == previous_state.frame_index:
        return "reuse"
    expected = previous_state.frame_index + 1
    if frame_index != expected:
        raise ValueError(
            f"Out-of-order frame_index={frame_index}; expected {expected} for this stream."
        )
    return "append"


class _StaticTensorRTEngine:
    """Bind one static TensorRT engine directly to torch CUDA tensors."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        expected_input_shape: tuple[int, ...],
        expected_output_shape: tuple[int, ...],
        device: torch.device,
    ) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "Raw-image RobotTrack inference requires the TensorRT Python package."
            ) from exc

        if device.type != "cuda":
            raise ValueError("RobotTrack TensorRT vision inference requires CUDA.")
        self.device = device
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        path = Path(engine_path)
        with path.open("rb") as handle:
            self._engine = self._runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {path}")

        input_names: list[str] = []
        output_names: list[str] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                output_names.append(name)
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError(
                f"Expected one input and one output in {path}, got "
                f"inputs={input_names}, outputs={output_names}."
            )
        self._input_name = input_names[0]
        self._output_name = output_names[0]
        input_shape = tuple(
            int(value) for value in self._engine.get_tensor_shape(self._input_name)
        )
        output_shape = tuple(
            int(value) for value in self._engine.get_tensor_shape(self._output_name)
        )
        if input_shape != expected_input_shape:
            raise ValueError(
                f"{path.name} input shape must be {expected_input_shape}, "
                f"got {input_shape}."
            )
        if output_shape != expected_output_shape:
            raise ValueError(
                f"{path.name} output shape must be {expected_output_shape}, "
                f"got {output_shape}."
            )
        if self._engine.get_tensor_dtype(self._input_name) != trt.float32:
            raise ValueError(f"{path.name} must accept float32 input tensors.")
        if self._engine.get_tensor_dtype(self._output_name) != trt.float32:
            raise ValueError(f"{path.name} must return float32 output tensors.")

        self.input_shape = input_shape
        self.output_shape = output_shape
        self._output = torch.empty(
            output_shape, dtype=torch.float32, device=self.device
        )
        self._stream = torch.cuda.Stream(device=self.device)

    @torch.inference_mode()
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape) != self.input_shape:
            raise ValueError(
                f"TensorRT input must have shape {self.input_shape}, "
                f"got {tuple(inputs.shape)}."
            )
        if inputs.device != self.device:
            raise ValueError(
                f"TensorRT input must be on {self.device}, got {inputs.device}."
            )
        if inputs.dtype != torch.float32 or not inputs.is_contiguous():
            inputs = inputs.to(dtype=torch.float32).contiguous()

        current_stream = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current_stream)
        self._context.set_tensor_address(self._input_name, int(inputs.data_ptr()))
        self._context.set_tensor_address(
            self._output_name, int(self._output.data_ptr())
        )
        ok = self._context.execute_async_v3(self._stream.cuda_stream)
        if ok is False:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        current_stream.wait_stream(self._stream)
        return self._output

    def close(self) -> None:
        self._output = torch.empty(0, device=self.device)
        self._context = None
        self._engine = None
        self._runtime = None


class MiniCPMRobotTrackVisionRunner(ModelRunner):
    """Encode RGB frames and produce a candidate 31-frame visual state."""

    def __init__(
        self,
        *,
        dino_engine_path: str | Path,
        siglip_engine_path: str | Path,
        history_frames: int,
        coarse_tokens_per_frame: int,
        fine_tokens_current_frame: int,
        vision_feature_dim: int,
        image_size: int = 384,
        device: torch.device | str = "cuda",
        use_cuda_graph: bool = True,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("RobotTrack TensorRT vision inference requires CUDA.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.image_size = int(image_size)
        self.history_frames = int(history_frames)
        self.coarse_tokens_per_frame = int(coarse_tokens_per_frame)
        self.fine_tokens_current_frame = int(fine_tokens_current_frame)
        self.vision_feature_dim = int(vision_feature_dim)
        self.use_cuda_graph = bool(use_cuda_graph)
        if self.image_size != 384:
            raise ValueError(
                "Released RobotTrack TensorRT engines require 384x384 input."
            )
        if (
            self.coarse_tokens_per_frame != 4
            or self.fine_tokens_current_frame != 64
            or self.vision_feature_dim != 1536
        ):
            raise ValueError(
                "Released RobotTrack vision towers require coarse=4, fine=64, "
                "and vision_feature_dim=1536."
            )

        input_shape = (1, 3, self.image_size, self.image_size)
        self.dino = _StaticTensorRTEngine(
            dino_engine_path,
            expected_input_shape=input_shape,
            expected_output_shape=(1, 576, 384),
            device=self.device,
        )
        self.siglip = _StaticTensorRTEngine(
            siglip_engine_path,
            expected_input_shape=input_shape,
            expected_output_shape=(1, 576, 1152),
            device=self.device,
        )
        self._dino_mean = self._channel_tensor(_DINO_MEAN)
        self._dino_std = self._channel_tensor(_DINO_STD)
        self._siglip_mean = self._channel_tensor(_SIGLIP_MEAN)
        self._siglip_std = self._channel_tensor(_SIGLIP_STD)
        self.graph: CudaGraph | None = None

    def _channel_tensor(self, values: tuple[float, float, float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

    @staticmethod
    def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def setup(self) -> None:
        dummy = torch.zeros(
            1,
            3,
            self.image_size,
            self.image_size,
            dtype=torch.float32,
            device=self.device,
        )
        for _ in range(3):
            self.dino((dummy - self._dino_mean) / self._dino_std)
            self.siglip((dummy - self._siglip_mean) / self._siglip_std)
        torch.cuda.current_stream(self.device).synchronize()
        if self.use_cuda_graph:
            self.graph = CudaGraph()
            self.graph.capture(self._forward_single_frame, {"frame": dummy})

    def _prepare_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 5:
            if frames.shape[0] != 1:
                raise ValueError("RobotTrack image requests support batch_size=1 only.")
            frames = frames[0]
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4:
            raise ValueError(
                "frames must be [T,H,W,3], [T,3,H,W], or include a leading "
                f"batch dimension; got {tuple(frames.shape)}."
            )
        if frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)
        elif frames.shape[1] != 3:
            raise ValueError(
                f"frames must contain three RGB channels, got {tuple(frames.shape)}."
            )
        if frames.dtype == torch.uint8:
            frames = frames.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            ).div_(255.0)
        elif frames.is_floating_point():
            frames = frames.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
        else:
            raise TypeError(
                f"frames must be uint8 RGB or floating point in [0,1], got {frames.dtype}."
            )
        if tuple(frames.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"frames must be PIL-BICUBIC resized to {self.image_size}x{self.image_size} "
                "before inference; use MiniCPMRobotTrackProcessor for reference-parity "
                f"preprocessing, got {tuple(frames.shape[-2:])}."
            )
        return frames.contiguous()

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor, output_side: int) -> torch.Tensor:
        batch_size, token_count, hidden_size = tokens.shape
        if token_count != 24 * 24:
            raise ValueError(f"Expected a 24x24 token grid, got {token_count} tokens.")
        features = tokens.transpose(1, 2).reshape(batch_size, hidden_size, 24, 24)
        features = F.adaptive_avg_pool2d(features, (output_side, output_side))
        return features.flatten(2).transpose(1, 2).contiguous()

    def _forward_single_frame(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dino_tokens = self.dino(
            ((frame - self._dino_mean) / self._dino_std).contiguous()
        )
        siglip_tokens = self.siglip(
            ((frame - self._siglip_mean) / self._siglip_std).contiguous()
        )
        combined = torch.cat((dino_tokens, siglip_tokens), dim=-1)
        return self._pool_tokens(combined, 2), self._pool_tokens(combined, 8)

    @torch.inference_mode()
    def forward(
        self,
        frames: torch.Tensor,
        *,
        previous_state: MiniCPMRobotTrackVisionState | None,
        frame_index: int | None,
        collect_timing: bool = False,
    ) -> MiniCPMRobotTrackVisionBatch:
        if frames.ndim == 5 and frames.shape[0] == 1:
            received_frame_count = int(frames.shape[1])
        elif frames.ndim == 4:
            received_frame_count = int(frames.shape[0])
        elif frames.ndim == 3:
            received_frame_count = 1
        else:
            received_frame_count = -1
        mode = classify_vision_request(
            frame_count=received_frame_count,
            history_frames=self.history_frames,
            previous_state=previous_state,
            frame_index=frame_index,
        )
        events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

        if mode == "reuse":
            if previous_state is None:
                raise RuntimeError("Vision state disappeared after request validation.")
            history = list(previous_state.coarse_history)
            coarse_tokens = torch.cat(history, dim=0).unsqueeze(0)
            return MiniCPMRobotTrackVisionBatch(
                coarse_tokens=coarse_tokens,
                fine_tokens=previous_state.fine_tokens,
                encoded_frames=0,
                cached_frames=len(history),
                next_state=previous_state,
            )

        preprocess_pair = self._event_pair() if collect_timing else None
        if preprocess_pair is not None:
            preprocess_pair[0].record()
        prepared = self._prepare_frames(frames)
        if preprocess_pair is not None:
            preprocess_pair[1].record()
            events["vision_preprocess_ms"] = [preprocess_pair]

        fine_current: torch.Tensor | None = None
        encoded_coarse: list[torch.Tensor] = []
        if self.graph is not None and prepared.shape[0] == 1:
            graph_pair = self._event_pair() if collect_timing else None
            if graph_pair is not None:
                graph_pair[0].record()
            coarse_batch, fine_current = self.graph.replay({"frame": prepared})
            # Graph outputs reuse storage, so cached coarse features must own
            # their data before the next replay overwrites that storage. Fine
            # features are cloned for the same reason before returning them.
            encoded_coarse.append(coarse_batch[0].detach().clone())
            fine_current = fine_current.clone()
            if graph_pair is not None:
                graph_pair[1].record()
                events["vision_graph_ms"] = [graph_pair]
        else:
            for frame in prepared:
                shared = frame.unsqueeze(0)
                dino_pair = self._event_pair() if collect_timing else None
                if dino_pair is not None:
                    dino_pair[0].record()
                dino_tokens = self.dino(
                    ((shared - self._dino_mean) / self._dino_std).contiguous()
                )
                if dino_pair is not None:
                    dino_pair[1].record()
                    events.setdefault("vision_dino_ms", []).append(dino_pair)

                siglip_pair = self._event_pair() if collect_timing else None
                if siglip_pair is not None:
                    siglip_pair[0].record()
                siglip_tokens = self.siglip(
                    ((shared - self._siglip_mean) / self._siglip_std).contiguous()
                )
                if siglip_pair is not None:
                    siglip_pair[1].record()
                    events.setdefault("vision_siglip_ms", []).append(siglip_pair)

                pool_pair = self._event_pair() if collect_timing else None
                if pool_pair is not None:
                    pool_pair[0].record()
                combined = torch.cat((dino_tokens, siglip_tokens), dim=-1)
                coarse = self._pool_tokens(combined, output_side=2)[0]
                fine_current = self._pool_tokens(combined, output_side=8)
                encoded_coarse.append(coarse.detach().clone())
                if pool_pair is not None:
                    pool_pair[1].record()
                    events.setdefault("vision_pool_ms", []).append(pool_pair)

        if fine_current is None:
            raise RuntimeError("No frames were encoded.")
        if mode == "replace":
            history = encoded_coarse[-self.history_frames :]
        else:
            if previous_state is None:
                raise RuntimeError(
                    "Incremental vision state disappeared during encoding."
                )
            history = [*previous_state.coarse_history, *encoded_coarse][
                -self.history_frames :
            ]
        if not history:
            raise RuntimeError("RobotTrack vision history is empty after encoding.")
        history = [history[0]] * (self.history_frames - len(history)) + history
        coarse_tokens = torch.cat(history, dim=0).unsqueeze(0)
        next_state = MiniCPMRobotTrackVisionState(
            coarse_history=tuple(history),
            fine_tokens=fine_current,
            frame_index=frame_index,
        )
        return MiniCPMRobotTrackVisionBatch(
            coarse_tokens=coarse_tokens,
            fine_tokens=fine_current,
            encoded_frames=int(prepared.shape[0]),
            cached_frames=len(history),
            next_state=next_state,
            cuda_events=events,
        )

    def close(self) -> None:
        self.graph = None
        self.dino.close()
        self.siglip.close()


def resolve_cuda_event_timings(
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> dict[str, float]:
    return {
        name: float(sum(start.elapsed_time(end) for start, end in pairs))
        for name, pairs in events.items()
    }


__all__ = [
    "MiniCPMRobotTrackVisionBatch",
    "MiniCPMRobotTrackVisionRunner",
    "MiniCPMRobotTrackVisionState",
    "classify_vision_request",
    "resolve_cuda_event_timings",
]
