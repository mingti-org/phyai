"""Engine plugin for native Qwen-Image 2.1 generation and editing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from phyai_utils_tools.models.qwen_image_21 import QwenImage21Processor

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import ParallelConfig, get_engine_config
from phyai.layers.backbones.qwen3_vl.configuration import Qwen3VLConfig
from phyai.parallel import default_mesh
from phyai.utils import get_logger, load_config
from phyai.weights import LoadReport, load_pretrained

from phyai.models.qwen_image_21.condition_encoder import (
    QwenImage21ConditionEncoder,
    qwen_image_21_condition_weight_remap,
)
from phyai.models.qwen_image_21.configuration_qwen_image_21 import QwenImage21Config
from phyai.models.qwen_image_21.configuration_vae import QwenImage21VAEConfig
from phyai.models.qwen_image_21.model_runner_condition import QwenImage21ConditionRunner
from phyai.models.qwen_image_21.model_runner_qwen_image_21 import QwenImage21Runner
from phyai.models.qwen_image_21.model_runner_vae import QwenImage21VAERunner
from phyai.models.qwen_image_21.modeling_qwen_image_21 import (
    QwenImage21Transformer,
    qwen_image_21_weight_remap,
)
from phyai.models.qwen_image_21.requests import QwenImage21Output, QwenImage21Request
from phyai.models.qwen_image_21.sampler_flow_match import FlowMatchEulerConfig
from phyai.models.qwen_image_21.scheduler_qwen_image_21 import QwenImage21Scheduler
from phyai.models.qwen_image_21.vae import (
    QwenImage21VAE,
    qwen_image_21_vae_weight_remap,
)


logger = get_logger(__name__)


@dataclass
class QwenImage21Args(EntryArgs):
    checkpoint_dir: str | Path | None = None
    load_condition_encoder: bool = True
    load_vae: bool = True
    use_kv_cache: bool = True
    torch_compile: bool = False
    vae_tiling: bool = False
    vae_slicing: bool = False


@Engine.register
class QwenImage21Entry(Entry):
    name: ClassVar[str] = "qwen_image_21"
    args_cls: ClassVar[type[EntryArgs]] = QwenImage21Args
    parallel_domains: ClassVar[frozenset[str]] = frozenset(
        {"dense", "attention", "cfg"}
    )

    def __init__(self) -> None:
        self.transformer: QwenImage21Transformer | None = None
        self.condition_encoder: QwenImage21ConditionEncoder | None = None
        self.vae: QwenImage21VAE | None = None
        self.scheduler: QwenImage21Scheduler | None = None
        self.load_reports: dict[str, LoadReport] = {}

    @classmethod
    def validate_parallel(
        cls, parallel: ParallelConfig, replica_world_size: int | None = None
    ) -> None:
        super().validate_parallel(parallel, replica_world_size)
        resolved = parallel.resolve(
            parallel.infer_replica_world_size()
            if replica_world_size is None
            else replica_world_size
        )
        if (
            resolved.outer.cfg_size not in (1, 2)
            or resolved.dense.dp_size != 1
            or resolved.dense.sequence_parallel
            or resolved.attention.cp_size != 1
            or resolved.attention.decode_cp_size != 1
            or resolved.attention.dp_size != 1
            or resolved.attention.tp_size != resolved.dense.tp_size
        ):
            raise ValueError(
                "Qwen-Image 2.1 supports matching dense/attention TP, CFG size 1 or 2, "
                "and no sequence/context parallelism inside one model replica."
            )

    def setup(self, args: EntryArgs) -> None:
        if not isinstance(args, QwenImage21Args) or args.checkpoint_dir is None:
            raise ValueError("QwenImage21Args.checkpoint_dir is required")
        checkpoint = Path(args.checkpoint_dir)
        engine = get_engine_config()
        device, dtype = engine.device.target, engine.device.params_dtype
        config = load_config(checkpoint / "transformer", QwenImage21Config)
        self.transformer = QwenImage21Transformer(
            config, params_dtype=dtype, device=device
        ).eval()
        self.load_reports["transformer"] = load_pretrained(
            self.transformer,
            checkpoint / "transformer",
            remap=qwen_image_21_weight_remap,
            strict=True,
        )
        condition_runner, processor = None, None
        if args.load_condition_encoder:
            text_config = load_config(checkpoint / "text_encoder", Qwen3VLConfig)
            self.condition_encoder = QwenImage21ConditionEncoder(
                text_config, params_dtype=dtype, device=device, prefix="model"
            ).eval()
            self.load_reports["text_encoder"] = load_pretrained(
                self.condition_encoder,
                checkpoint / "text_encoder",
                remap=qwen_image_21_condition_weight_remap,
                strict=True,
            )
            condition_runner = QwenImage21ConditionRunner(self.condition_encoder)
            processor = QwenImage21Processor(checkpoint / "processor")
        vae_runner = None
        if args.load_vae:
            vae_config = load_config(checkpoint / "vae", QwenImage21VAEConfig)
            self.vae = QwenImage21VAE(
                vae_config, params_dtype=dtype, device=device
            ).eval()
            self.load_reports["vae"] = load_pretrained(
                self.vae,
                checkpoint / "vae",
                remap=qwen_image_21_vae_weight_remap,
                strict=True,
            )
            vae_runner = QwenImage21VAERunner(
                self.vae,
                device=device,
                dtype=dtype,
                use_tiling=args.vae_tiling,
                use_slicing=args.vae_slicing,
            )
        sampler_config = FlowMatchEulerConfig.from_json(
            checkpoint / "scheduler" / "scheduler_config.json"
        )
        mesh = default_mesh()
        self.scheduler = QwenImage21Scheduler(
            QwenImage21Runner(
                self.transformer,
                use_kv_cache=args.use_kv_cache,
                torch_compile=args.torch_compile,
                use_cuda_graph=engine.runtime.use_cuda_graph,
            ),
            sampler_config=sampler_config,
            condition_runner=condition_runner,
            vae_runner=vae_runner,
            processor=processor,
            device=device,
            dtype=dtype,
            latent_channels=config.in_channels,
            cfg_rank=mesh.group_rank("cfg"),
            cfg_size=mesh.group_size("cfg"),
        )
        self.scheduler.setup()
        logger.info_rank0(
            "Qwen-Image 2.1 ready (TP=%d, CFG=%d)",
            mesh.group_size("dense_tp"),
            mesh.group_size("cfg"),
        )

    def step(self, request: QwenImage21Request) -> QwenImage21Output:
        if self.scheduler is None:
            raise RuntimeError("call setup() before step()")
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.transformer = None
        self.condition_encoder = None
        self.vae = None
        self.load_reports.clear()

    def dump_targets(self) -> dict:
        return {
            name: model
            for name, model in (
                ("transformer", self.transformer),
                ("text_encoder", self.condition_encoder),
                ("vae", self.vae),
            )
            if model is not None
        }
