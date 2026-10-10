"""Engine registration and checkpoint loading for MolmoAct2."""

from typing import ClassVar
from pathlib import Path
from dataclasses import dataclass

import torch

from phyai.utils import load_config
from phyai.engine import Entry, Engine, EntryArgs
from phyai.weights import load_pretrained
from phyai.engine_config import get_engine_config
from phyai.models.molmoact2.policy_molmoact2 import (
    MolmoAct2ForConditionalGeneration,
    molmoact2_weight_remap,
)
from phyai.models.molmoact2.scheduler_molmoact2 import (
    MolmoAct2Request,
    MolmoAct2Scheduler,
    MolmoAct2GenerationRequest,
)
from phyai.models.molmoact2.model_runner_molmoact2 import MolmoAct2Runner
from phyai.models.molmoact2.configuration_molmoact2 import MolmoAct2Config


@dataclass
class MolmoAct2Args(EntryArgs):
    checkpoint_dir: str | Path | None = None
    config: MolmoAct2Config | None = None
    vision_params_dtype: torch.dtype | None = None


@Engine.register
class MolmoAct2Entry(Entry):
    name: ClassVar[str] = "molmoact2"
    args_cls: ClassVar[type[EntryArgs]] = MolmoAct2Args

    def __init__(self) -> None:
        self.model: MolmoAct2ForConditionalGeneration | None = None
        self.scheduler: MolmoAct2Scheduler | None = None

    def setup(self, args: MolmoAct2Args) -> None:
        if args.checkpoint_dir is None:
            raise ValueError("MolmoAct2 requires checkpoint_dir.")
        config = args.config or load_config(args.checkpoint_dir, MolmoAct2Config)
        engine = get_engine_config()
        self.model = MolmoAct2ForConditionalGeneration(
            config,
            params_dtype=engine.device.params_dtype,
            vision_params_dtype=args.vision_params_dtype,
            device=engine.device.target,
        ).eval()
        report = load_pretrained(
            self.model, args.checkpoint_dir, remap=molmoact2_weight_remap, strict=True
        )
        if report.missing:
            raise RuntimeError(f"Missing MolmoAct2 weights: {report.missing}")
        self.scheduler = MolmoAct2Scheduler(
            MolmoAct2Runner(self.model, use_cuda_graph=engine.runtime.use_cuda_graph)
        )
        self.scheduler.setup()

    def step(
        self, request: MolmoAct2Request | MolmoAct2GenerationRequest
    ) -> torch.Tensor:
        if self.scheduler is None:
            raise RuntimeError("MolmoAct2Entry has not been set up.")
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.model = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        return {} if self.model is None else {"model": self.model}
