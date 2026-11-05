"""Inference-only HumanWAM-Goal engine plugin."""

from typing import ClassVar
from pathlib import Path
from dataclasses import replace, dataclass

import torch

from phyai.utils import get_logger, load_config
from phyai.engine import Entry, Engine, EntryArgs
from phyai.weights import load_pretrained
from phyai.engine_config import ParallelConfig, get_engine_config, set_engine_config
from phyai.layers.quant.active import resolve_quant_plan
from phyai.models.humanwam_goal.vae_humanwam_goal import (
    HumanWAMGoalVAE,
    humanwam_goal_vae_weight_remap,
)
from phyai.models.humanwam_goal.weights_humanwam_goal import humanwam_goal_weight_remap
from phyai.models.cosmos3.modeling_cosmos3 import TimestepEmbedder
from phyai.models.cosmos3.parallel_cosmos3 import validate_cosmos3_parallel
from phyai.models.humanwam_goal.modeling_humanwam_goal import HumanWAMGoalTransformer
from phyai.models.humanwam_goal.requests_humanwam_goal import HumanWAMGoalStageRequest
from phyai.models.humanwam_goal.attention_humanwam_goal import (
    configure_humanwam_goal_kernels,
)
from phyai.models.humanwam_goal.scheduler_humanwam_goal import HumanWAMGoalScheduler
from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3WanVAEConfig
from phyai.models.humanwam_goal.configuration_humanwam_goal import (
    HumanWAMGoalConfig,
    HumanWAMGoalSamplingConfig,
)

logger = get_logger(__name__)


@dataclass
class HumanWAMGoalArgs(EntryArgs):
    checkpoint_dir: str | Path | None = None
    config: HumanWAMGoalConfig | None = None
    sampling: HumanWAMGoalSamplingConfig | None = None
    load_vae: bool = True


@Engine.register
class HumanWAMGoalEntry(Entry):
    name: ClassVar[str] = "humanwam_goal"
    args_cls: ClassVar[type[EntryArgs]] = HumanWAMGoalArgs
    parallel_domains: ClassVar[frozenset[str]] = frozenset({"dense", "attention"})

    def __init__(self) -> None:
        self.transformer: HumanWAMGoalTransformer | None = None
        self.vae: HumanWAMGoalVAE | None = None
        self.scheduler: HumanWAMGoalScheduler | None = None

    @classmethod
    def validate_parallel(
        cls, parallel: ParallelConfig, replica_world_size: int | None = None
    ) -> None:
        super().validate_parallel(parallel, replica_world_size)
        validate_cosmos3_parallel(parallel, replica_world_size)
        resolved = parallel.resolve(
            replica_world_size or parallel.infer_replica_world_size()
        )
        if resolved.dense.tp_size not in (1, 2, 4) or resolved.outer.cfg_size != 1:
            raise ValueError("HumanWAM-Goal supports TP=1/2/4 with CFG size 1")

    def setup(self, args: HumanWAMGoalArgs) -> None:
        if args.checkpoint_dir is None:
            raise ValueError("HumanWAMGoalArgs.checkpoint_dir is required")
        checkpoint = Path(args.checkpoint_dir)
        engine = get_engine_config()
        if torch.device(engine.device.target).type != "cuda":
            raise ValueError("HumanWAM-Goal inference requires a CUDA device")
        if engine.device.params_dtype != torch.bfloat16:
            raise ValueError("HumanWAM-Goal requires BF16 model parameters")
        if engine.runtime.allow_tf32:
            raise ValueError("HumanWAM-Goal requires runtime.allow_tf32=False")
        configure_humanwam_goal_kernels()
        if resolve_quant_plan(checkpoint / "transformer") is not None:
            raise ValueError("HumanWAM-Goal requires unquantized weights")
        config = args.config or load_config(
            checkpoint / "transformer", HumanWAMGoalConfig
        )
        sampling = args.sampling or load_config(
            checkpoint, HumanWAMGoalSamplingConfig, filename="sampling.json"
        )
        set_engine_config(engine.replace(device=replace(engine.device, target="cpu")))
        try:
            self.transformer = (
                HumanWAMGoalTransformer(
                    config,
                    params_dtype=engine.device.params_dtype,
                    device="cpu",
                    attn_backend="humanwam_goal_fa3",
                )
                .eval()
                .requires_grad_(False)
            )
            report = load_pretrained(
                self.transformer,
                checkpoint / "transformer",
                remap=humanwam_goal_weight_remap,
                strict=True,
            )
            # The source runtime stores BF16 parameters, then promotes this MLP
            # for FP32 execution. Promote after loading to preserve that rounding.
            fp32_time = TimestepEmbedder(
                config.hidden_size,
                params_dtype=torch.float32,
                device="cpu",
                prefix="time_embedder",
            )
            fp32_time.load_state_dict(self.transformer.time_embedder.state_dict())
            self.transformer.time_embedder = fp32_time.eval().requires_grad_(False)
            logger.info_rank0("HumanWAM-Goal transformer: %s", report.summary())
            if args.load_vae:
                self.vae = (
                    HumanWAMGoalVAE(
                        load_config(checkpoint / "vae", Cosmos3WanVAEConfig),
                        params_dtype=engine.device.params_dtype,
                    )
                    .to(device="cpu", dtype=engine.device.params_dtype)
                    .eval()
                    .requires_grad_(False)
                )
                report = load_pretrained(
                    self.vae,
                    checkpoint / "vae",
                    remap=humanwam_goal_vae_weight_remap,
                    strict=True,
                )
                logger.info_rank0("HumanWAM-Goal VAE: %s", report.summary())
        finally:
            set_engine_config(engine)
        self.scheduler = HumanWAMGoalScheduler(
            self.transformer,
            device=engine.device.target,
            sampling=sampling,
            vae=self.vae,
            use_cuda_graph=engine.runtime.use_cuda_graph,
        )
        self.scheduler.setup()

    def step(self, request: HumanWAMGoalStageRequest):
        if self.scheduler is None:
            raise RuntimeError("Set up HumanWAM-Goal before submitting requests")
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.transformer = None
        self.vae = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        return {} if self.transformer is None else {"transformer": self.transformer}
