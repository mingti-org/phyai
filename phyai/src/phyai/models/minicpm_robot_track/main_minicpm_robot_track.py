"""MiniCPM-RobotTrack Engine plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.minicpm_robot_track.configuration_minicpm_robot_track import (
    MiniCPMRobotTrackConfig,
)
from phyai.models.minicpm_robot_track.model_runner_minicpm_robot_track import (
    MiniCPMRobotTrackModelRunner,
)
from phyai.models.minicpm_robot_track.modeling_minicpm_robot_track import (
    MiniCPMRobotTrackModel,
    minicpm_robot_track_weight_remap,
)
from phyai.models.minicpm_robot_track.scheduler_ws1_minicpm_robot_track import (
    MiniCPMRobotTrackImageOutput,
    MiniCPMRobotTrackImageRequest,
    MiniCPMRobotTrackRequest,
    MiniCPMRobotTrackWS1Scheduler,
)
from phyai.utils import load_config
from phyai.weights import load_pretrained


@dataclass
class MiniCPMRobotTrackArgs(EntryArgs):
    checkpoint_dir: str | Path
    config: MiniCPMRobotTrackConfig | None = None
    batch_size: int = 1
    dino_checkpoint_dir: str | Path | None = None
    siglip_checkpoint_dir: str | Path | None = None
    vision_attention_backend: str = "sdpa"
    vision_norm_backend: str | None = "phyai-kernel"
    vision_params_dtype: Literal["float16", "bfloat16"] = "float16"
    use_vision_cuda_graph: bool = True
    max_cached_streams: int = 8
    float32_norm_backend: str = "phyai-kernel"


@Engine.register
class MiniCPMRobotTrackEntry(Entry):
    name: ClassVar[str] = "minicpm_robot_track"
    args_cls: ClassVar[type[EntryArgs]] = MiniCPMRobotTrackArgs

    def __init__(self) -> None:
        self.model: MiniCPMRobotTrackModel | None = None
        self.scheduler: MiniCPMRobotTrackWS1Scheduler | None = None

    def setup(self, args: MiniCPMRobotTrackArgs) -> None:
        engine_config = get_engine_config()
        config = args.config or load_config(
            args.checkpoint_dir, MiniCPMRobotTrackConfig
        )
        self.model = MiniCPMRobotTrackModel(
            config,
            float32_norm_backend=args.float32_norm_backend,
            device=engine_config.device.target,
        )
        report = load_pretrained(
            self.model,
            args.checkpoint_dir,
            remap=minicpm_robot_track_weight_remap,
            strict=True,
        )
        if report.missing:
            raise RuntimeError(
                f"MiniCPM-RobotTrack has missing weights: {report.missing[:8]}"
            )
        self.model.eval()
        runner = MiniCPMRobotTrackModelRunner(
            self.model,
            batch_size=args.batch_size,
            device=engine_config.device.target,
            use_cuda_graph=engine_config.runtime.use_cuda_graph,
            dino_checkpoint_dir=args.dino_checkpoint_dir,
            siglip_checkpoint_dir=args.siglip_checkpoint_dir,
            vision_attention_backend=args.vision_attention_backend,
            vision_norm_backend=args.vision_norm_backend,
            vision_params_dtype={
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[args.vision_params_dtype],
            use_vision_cuda_graph=(
                engine_config.runtime.use_cuda_graph and args.use_vision_cuda_graph
            ),
            max_cached_streams=args.max_cached_streams,
        )
        self.scheduler = MiniCPMRobotTrackWS1Scheduler(runner)
        self.scheduler.setup()

    def step(
        self, request: MiniCPMRobotTrackRequest | MiniCPMRobotTrackImageRequest
    ) -> torch.Tensor | MiniCPMRobotTrackImageOutput:
        if self.scheduler is None:
            raise RuntimeError("MiniCPMRobotTrackEntry.step called before setup.")
        return self.scheduler.step(request)

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        return {"model": self.model} if self.model is not None else {}

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        self.model = None


__all__ = ["MiniCPMRobotTrackArgs", "MiniCPMRobotTrackEntry"]
