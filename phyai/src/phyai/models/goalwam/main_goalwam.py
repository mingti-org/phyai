"""Inference-only GoalWAM engine plugin."""

from typing import ClassVar
from pathlib import Path
from dataclasses import replace, dataclass

import torch

from phyai.utils import get_logger, load_config
from phyai.engine import Entry, Engine, EntryArgs
from phyai.weights import load_pretrained
from phyai.engine_config import ParallelConfig, get_engine_config, set_engine_config
from phyai.layers.quant.active import resolve_quant_plan
from phyai.models.cosmos3.vae_wan import cosmos3_vae_weight_remap
from phyai.models.goalwam.vae_goalwam import GoalWAMVAE
from phyai.models.goalwam.weights_goalwam import goalwam_weight_remap
from phyai.models.cosmos3.modeling_cosmos3 import TimestepEmbedder
from phyai.models.cosmos3.parallel_cosmos3 import validate_cosmos3_parallel
from phyai.models.goalwam.modeling_goalwam import GoalWAMTransformer
from phyai.models.goalwam.requests_goalwam import GoalWAMStageRequest
from phyai.models.goalwam.attention_goalwam import configure_goalwam_kernels
from phyai.models.goalwam.scheduler_goalwam import GoalWAMScheduler
from phyai.models.goalwam.collective_goalwam import install_goalwam_collective
from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3WanVAEConfig
from phyai.models.goalwam.configuration_goalwam import (
    GoalWAMConfig,
    GoalWAMSamplingConfig,
)

logger = get_logger(__name__)


@dataclass
class GoalWAMArgs(EntryArgs):
    checkpoint_dir: str | Path | None = None
    config: GoalWAMConfig | None = None
    sampling: GoalWAMSamplingConfig | None = None
    load_vae: bool = True


@Engine.register
class GoalWAMEntry(Entry):
    name: ClassVar[str] = "goalwam"
    args_cls: ClassVar[type[EntryArgs]] = GoalWAMArgs
    parallel_domains: ClassVar[frozenset[str]] = frozenset({"dense", "attention"})

    def __init__(self) -> None:
        self.transformer: GoalWAMTransformer | None = None
        self.vae: GoalWAMVAE | None = None
        self.scheduler: GoalWAMScheduler | None = None
        self.collective = None

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
            raise ValueError("GoalWAM supports TP=1/2/4 with CFG size 1")

    def setup(self, args: GoalWAMArgs) -> None:
        if args.checkpoint_dir is None:
            raise ValueError("GoalWAMArgs.checkpoint_dir is required")
        checkpoint = Path(args.checkpoint_dir)
        engine = get_engine_config()
        if torch.device(engine.device.target).type != "cuda":
            raise ValueError("GoalWAM inference requires a CUDA device")
        if engine.device.params_dtype != torch.bfloat16:
            raise ValueError("GoalWAM requires BF16 model parameters")
        if engine.runtime.allow_tf32:
            raise ValueError("GoalWAM requires runtime.allow_tf32=False")
        configure_goalwam_kernels()
        if resolve_quant_plan(checkpoint / "transformer") is not None:
            raise ValueError("GoalWAM requires unquantized weights")
        config = args.config or load_config(checkpoint / "transformer", GoalWAMConfig)
        sampling = args.sampling or load_config(
            checkpoint, GoalWAMSamplingConfig, filename="sampling.json"
        )
        set_engine_config(engine.replace(device=replace(engine.device, target="cpu")))
        try:
            self.transformer = (
                GoalWAMTransformer(
                    config,
                    params_dtype=engine.device.params_dtype,
                    device="cpu",
                    attn_backend="goalwam_fa3",
                )
                .eval()
                .requires_grad_(False)
            )
            report = load_pretrained(
                self.transformer,
                checkpoint / "transformer",
                remap=goalwam_weight_remap,
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
            logger.info_rank0("GoalWAM transformer: %s", report.summary())
            if args.load_vae:
                self.vae = (
                    GoalWAMVAE(
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
                    remap=cosmos3_vae_weight_remap,
                    strict=True,
                )
                logger.info_rank0("GoalWAM VAE: %s", report.summary())
        finally:
            set_engine_config(engine)
        self.scheduler = GoalWAMScheduler(
            self.transformer,
            device=engine.device.target,
            sampling=sampling,
            vae=self.vae,
            use_cuda_graph=engine.runtime.use_cuda_graph,
        )
        self.collective = install_goalwam_collective()
        self.scheduler.setup()

    def step(self, request: GoalWAMStageRequest):
        if self.scheduler is None:
            raise RuntimeError("Set up GoalWAM before submitting requests")
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        if self.collective is not None:
            self.collective.close()
            self.collective = None
        self.transformer = None
        self.vae = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        return {} if self.transformer is None else {"transformer": self.transformer}
