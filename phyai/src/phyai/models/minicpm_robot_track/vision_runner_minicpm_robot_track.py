"""PhyAI vision towers and sliding-window packing for RobotTrack."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

from phyai.models.minicpm_robot_track.modeling_vision_minicpm_robot_track import (
    MiniCPMRobotTrackPhyAIVisionEncoder,
)
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


class _PhyAISiglipPooled:
    """Adapt the 27x27 PhyAI SigLIP output to RobotTrack's 24x24 grid."""

    def __init__(self, encoder: MiniCPMRobotTrackPhyAIVisionEncoder) -> None:
        self.encoder = encoder

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder.pool_siglip(self.encoder.siglip(inputs))


class MiniCPMRobotTrackVisionRunner(ModelRunner):
    """Encode RGB frames and produce a candidate 31-frame visual state."""

    def __init__(
        self,
        *,
        dino_checkpoint_dir: str | Path,
        siglip_checkpoint_dir: str | Path,
        vision_attention_backend: str = "sdpa",
        vision_norm_backend: str | None = "phyai-kernel",
        vision_params_dtype: torch.dtype = torch.float16,
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
            raise ValueError("RobotTrack PhyAI vision inference requires CUDA.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.image_size = int(image_size)
        self.history_frames = int(history_frames)
        self.coarse_tokens_per_frame = int(coarse_tokens_per_frame)
        self.fine_tokens_current_frame = int(fine_tokens_current_frame)
        self.vision_feature_dim = int(vision_feature_dim)
        self.use_cuda_graph = bool(use_cuda_graph)
        if vision_params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "RobotTrack PhyAI vision supports float16 or bfloat16 parameters."
            )
        self.vision_params_dtype = vision_params_dtype
        if self.image_size != 384:
            raise ValueError("RobotTrack vision towers require 384x384 input.")
        if (
            self.coarse_tokens_per_frame != 4
            or self.fine_tokens_current_frame != 64
            or self.vision_feature_dim != 1536
        ):
            raise ValueError(
                "Released RobotTrack vision towers require coarse=4, fine=64, "
                "and vision_feature_dim=1536."
            )

        self._phyai_encoder = MiniCPMRobotTrackPhyAIVisionEncoder(
            dino_checkpoint_dir=dino_checkpoint_dir,
            siglip_checkpoint_dir=siglip_checkpoint_dir,
            attn_backend=vision_attention_backend,
            norm_backend=vision_norm_backend,
            params_dtype=self.vision_params_dtype,
        )
        self.dino = self._phyai_encoder.dino
        self.siglip = _PhyAISiglipPooled(self._phyai_encoder)
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
        if 24 % output_side:
            raise ValueError(f"output_side={output_side} must divide the 24x24 grid.")
        features = tokens.transpose(1, 2).reshape(batch_size, hidden_size, 24, 24)
        block_size = 24 // output_side
        features = features.reshape(
            batch_size,
            hidden_size,
            output_side,
            block_size,
            output_side,
            block_size,
        ).mean(dim=(3, 5))
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
        combined = torch.cat((dino_tokens, siglip_tokens), dim=-1).float()
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
                combined = torch.cat((dino_tokens, siglip_tokens), dim=-1).float()
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
        if len(history) != self.history_frames:
            raise RuntimeError(
                f"RobotTrack vision history must contain {self.history_frames} "
                f"frames, got {len(history)}."
            )
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
        self.dino = None
        self.siglip = None
        self._phyai_encoder.close()


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
