"""Runtime ownership for fixed-shape MiniCPM-RobotTrack inference."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch

from phyai.models.minicpm_robot_track.modeling_minicpm_robot_track import (
    MiniCPMRobotTrackModel,
)
from phyai.models.minicpm_robot_track.vision_runner_minicpm_robot_track import (
    MiniCPMRobotTrackVisionRunner,
    MiniCPMRobotTrackVisionState,
    resolve_cuda_event_timings,
)
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner


@dataclass
class MiniCPMRobotTrackForwardBatch:
    input_ids: torch.Tensor
    text_lengths: torch.Tensor
    coarse_tokens: torch.Tensor
    coarse_time_indices: torch.Tensor
    fine_tokens: torch.Tensor
    fine_time_indices: torch.Tensor


@dataclass
class MiniCPMRobotTrackImageForwardBatch:
    frames: torch.Tensor
    input_ids: torch.Tensor
    text_lengths: torch.Tensor
    stream_id: str
    frame_index: int | None
    collect_timing: bool = False


@dataclass
class MiniCPMRobotTrackImageForwardOutput:
    waypoints: torch.Tensor
    encoded_frames: int
    cached_frames: int
    stream_id: str
    frame_index: int | None
    cache_reused: bool
    timing_ms: dict[str, float]


class MiniCPMRobotTrackPolicyRunner(ModelRunner):
    """Own the policy CUDA Graph and execute model-ready token batches."""

    def __init__(
        self,
        model: MiniCPMRobotTrackModel,
        *,
        batch_size: int,
        device: torch.device | str,
        use_cuda_graph: bool = True,
    ) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.use_cuda_graph = bool(use_cuda_graph) and self.device.type == "cuda"
        self.graph: CudaGraph | None = None

    def example_inputs(self) -> dict[str, torch.Tensor]:
        config = self.model.config
        batch_size = self.batch_size
        coarse_times = torch.arange(
            config.history_frames, dtype=torch.long, device=self.device
        ).repeat_interleave(config.coarse_tokens_per_frame)
        coarse_times = coarse_times[None].expand(batch_size, -1).contiguous()
        fine_times = torch.full(
            (batch_size, config.fine_tokens_current_frame),
            config.history_frames,
            dtype=torch.long,
            device=self.device,
        )
        return {
            "input_ids": torch.ones(
                batch_size,
                config.text_capacity,
                dtype=torch.long,
                device=self.device,
            ),
            "text_lengths": torch.ones(
                batch_size, dtype=torch.long, device=self.device
            ),
            "coarse_tokens": torch.zeros(
                batch_size,
                config.coarse_token_count,
                config.vision_feature_dim,
                dtype=torch.float32,
                device=self.device,
            ),
            "coarse_time_indices": coarse_times,
            "fine_tokens": torch.zeros(
                batch_size,
                config.fine_tokens_current_frame,
                config.vision_feature_dim,
                dtype=torch.float32,
                device=self.device,
            ),
            "fine_time_indices": fine_times,
        }

    def setup(self) -> None:
        if self.use_cuda_graph:
            self.graph = CudaGraph()
            self.graph.capture(self._forward, self.example_inputs())

    def _forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        return self.model(**inputs)

    @torch.inference_mode()
    def forward(self, batch: MiniCPMRobotTrackForwardBatch) -> torch.Tensor:
        inputs = {
            "input_ids": batch.input_ids,
            "text_lengths": batch.text_lengths,
            "coarse_tokens": batch.coarse_tokens,
            "coarse_time_indices": batch.coarse_time_indices,
            "fine_tokens": batch.fine_tokens,
            "fine_time_indices": batch.fine_time_indices,
        }
        if self.graph is not None:
            return self.graph.replay(inputs).clone()
        return self._forward(**inputs)

    def close(self) -> None:
        self.graph = None


class MiniCPMRobotTrackModelRunner(ModelRunner):
    """Own policy and vision runners, CUDA Graphs, and explicit stream state."""

    def __init__(
        self,
        model: MiniCPMRobotTrackModel,
        *,
        batch_size: int,
        device: torch.device | str,
        use_cuda_graph: bool = True,
        dino_checkpoint_dir: str | Path | None = None,
        siglip_checkpoint_dir: str | Path | None = None,
        vision_attention_backend: str = "sdpa",
        vision_norm_backend: str | None = "phyai-kernel",
        vision_params_dtype: torch.dtype = torch.float16,
        use_vision_cuda_graph: bool = True,
        max_cached_streams: int = 8,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_cached_streams <= 0:
            raise ValueError("max_cached_streams must be positive.")
        if (dino_checkpoint_dir is None) != (siglip_checkpoint_dir is None):
            raise ValueError(
                "dino_checkpoint_dir and siglip_checkpoint_dir must be configured "
                "together."
            )
        vision_configured = dino_checkpoint_dir is not None
        self.model = model
        self.config = model.config
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.max_cached_streams = int(max_cached_streams)
        self.inference_lock = Lock()
        self.policy_runner = MiniCPMRobotTrackPolicyRunner(
            model,
            batch_size=self.batch_size,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
        )
        self.vision_runner: MiniCPMRobotTrackVisionRunner | None = None
        if vision_configured:
            if self.batch_size != 1:
                raise ValueError(
                    "Raw-image RobotTrack inference currently requires batch_size=1."
                )
            self.vision_runner = MiniCPMRobotTrackVisionRunner(
                dino_checkpoint_dir=dino_checkpoint_dir,
                siglip_checkpoint_dir=siglip_checkpoint_dir,
                vision_attention_backend=vision_attention_backend,
                vision_norm_backend=vision_norm_backend,
                vision_params_dtype=vision_params_dtype,
                history_frames=self.config.history_frames,
                coarse_tokens_per_frame=self.config.coarse_tokens_per_frame,
                fine_tokens_current_frame=self.config.fine_tokens_current_frame,
                vision_feature_dim=self.config.vision_feature_dim,
                device=self.device,
                use_cuda_graph=use_vision_cuda_graph,
            )
        self.stream_states: OrderedDict[str, MiniCPMRobotTrackVisionState] = (
            OrderedDict()
        )
        self.coarse_time_indices = torch.arange(
            self.config.history_frames, dtype=torch.long, device=self.device
        ).repeat_interleave(self.config.coarse_tokens_per_frame)[None]
        self.fine_time_indices = torch.full(
            (1, self.config.fine_tokens_current_frame),
            self.config.history_frames,
            dtype=torch.long,
            device=self.device,
        )

    @property
    def vision_enabled(self) -> bool:
        return self.vision_runner is not None

    def setup(self) -> None:
        self.policy_runner.setup()
        if self.vision_runner is not None:
            self.vision_runner.setup()

    def reset_stream(self, stream_id: str | None = None) -> None:
        with self.inference_lock:
            if stream_id is None:
                self.stream_states.clear()
            else:
                self.stream_states.pop(stream_id, None)

    def to_device_batch(
        self, batch: MiniCPMRobotTrackForwardBatch
    ) -> MiniCPMRobotTrackForwardBatch:
        return MiniCPMRobotTrackForwardBatch(
            input_ids=batch.input_ids.to(
                device=self.device, dtype=torch.long, non_blocking=True
            ),
            text_lengths=batch.text_lengths.to(
                device=self.device, dtype=torch.long, non_blocking=True
            ),
            coarse_tokens=batch.coarse_tokens.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            ),
            coarse_time_indices=batch.coarse_time_indices.to(
                device=self.device, dtype=torch.long, non_blocking=True
            ),
            fine_tokens=batch.fine_tokens.to(
                device=self.device, dtype=torch.float32, non_blocking=True
            ),
            fine_time_indices=batch.fine_time_indices.to(
                device=self.device, dtype=torch.long, non_blocking=True
            ),
        )

    def commit_stream(
        self, stream_id: str, state: MiniCPMRobotTrackVisionState
    ) -> None:
        self.stream_states[stream_id] = state
        self.stream_states.move_to_end(stream_id)
        while len(self.stream_states) > self.max_cached_streams:
            self.stream_states.popitem(last=False)

    @torch.inference_mode()
    def _forward_images(
        self, batch: MiniCPMRobotTrackImageForwardBatch
    ) -> MiniCPMRobotTrackImageForwardOutput:
        if self.vision_runner is None:
            raise RuntimeError(
                "Raw-image inference is disabled; configure both PhyAI vision "
                "checkpoint paths."
            )
        total_pair = None
        if batch.collect_timing:
            total_pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            total_pair[0].record()

        previous_state = self.stream_states.get(batch.stream_id)
        vision = self.vision_runner.forward(
            batch.frames,
            previous_state=previous_state,
            frame_index=batch.frame_index,
            collect_timing=batch.collect_timing,
        )
        policy_pair = None
        if batch.collect_timing:
            policy_pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            policy_pair[0].record()
        policy_batch = self.to_device_batch(
            MiniCPMRobotTrackForwardBatch(
                input_ids=batch.input_ids,
                text_lengths=batch.text_lengths,
                coarse_tokens=vision.coarse_tokens,
                coarse_time_indices=self.coarse_time_indices,
                fine_tokens=vision.fine_tokens,
                fine_time_indices=self.fine_time_indices,
            )
        )
        waypoints = self.policy_runner.forward(policy_batch)

        # Commit only after policy succeeds, so failed requests remain retryable.
        self.commit_stream(batch.stream_id, vision.next_state)

        timing_ms: dict[str, float] = {}
        if policy_pair is not None and total_pair is not None:
            policy_pair[1].record()
            total_pair[1].record()
            total_pair[1].synchronize()
            timing_ms = resolve_cuda_event_timings(vision.cuda_events)
            timing_ms["vision_encode_ms"] = sum(timing_ms.values())
            timing_ms["policy_forward_ms"] = float(
                policy_pair[0].elapsed_time(policy_pair[1])
            )
            timing_ms["total_ms"] = float(total_pair[0].elapsed_time(total_pair[1]))
        return MiniCPMRobotTrackImageForwardOutput(
            waypoints=waypoints,
            encoded_frames=vision.encoded_frames,
            cached_frames=vision.cached_frames,
            stream_id=batch.stream_id,
            frame_index=batch.frame_index,
            cache_reused=vision.encoded_frames == 0,
            timing_ms=timing_ms,
        )

    @torch.inference_mode()
    def forward(
        self,
        batch: MiniCPMRobotTrackForwardBatch | MiniCPMRobotTrackImageForwardBatch,
    ) -> torch.Tensor | MiniCPMRobotTrackImageForwardOutput:
        with self.inference_lock:
            if isinstance(batch, MiniCPMRobotTrackImageForwardBatch):
                return self._forward_images(batch)
            if isinstance(batch, MiniCPMRobotTrackForwardBatch):
                return self.policy_runner.forward(self.to_device_batch(batch))
            raise TypeError(
                f"Unsupported RobotTrack runner batch: {type(batch).__name__}."
            )

    def close(self) -> None:
        self.reset_stream()
        if self.vision_runner is not None:
            self.vision_runner.close()
            self.vision_runner = None
        self.policy_runner.close()


__all__ = [
    "MiniCPMRobotTrackForwardBatch",
    "MiniCPMRobotTrackImageForwardBatch",
    "MiniCPMRobotTrackImageForwardOutput",
    "MiniCPMRobotTrackModelRunner",
    "MiniCPMRobotTrackPolicyRunner",
]
