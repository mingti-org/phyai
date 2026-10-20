"""Engine entry for native MOSS-Image text-to-image inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from phyai_utils_tools.models.moss_image import MOSSImageProcessor

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.moss_image.configuration_moss_image import MOSSImageConfig
from phyai.models.moss_image.configuration_text_encoder import MOSSImageTextConfig
from phyai.models.moss_image.configuration_vae import MOSSImageVAEConfig
from phyai.models.moss_image.model_runner_moss_image import (
    MOSSImageRunner,
    normalize_moss_image_graph_buckets,
)
from phyai.models.moss_image.model_runner_text_encoder import MOSSImageTextRunner
from phyai.models.moss_image.model_runner_vae import MOSSImageVAERunner
from phyai.models.moss_image.modeling_moss_image import (
    MOSSImageTransformer,
    moss_image_weight_remap,
)
from phyai.models.moss_image.requests import MOSSImageOutput, MOSSImageRequest
from phyai.models.moss_image.sampler_moss_image import MOSSImageSamplerConfig
from phyai.models.moss_image.scheduler_moss_image import MOSSImageScheduler
from phyai.models.moss_image.text_encoder import (
    MOSSImageTextEncoder,
    moss_image_text_weight_remap,
)
from phyai.models.moss_image.vae import MOSSImageVAE, moss_image_vae_weight_remap
from phyai.utils import get_logger, load_config
from phyai.weights import LoadReport, load_pretrained


logger = get_logger(__name__)


@dataclass
class MOSSImageArgs(EntryArgs):
    checkpoint_dir: str | Path | None = None
    load_text_encoder: bool = True
    load_vae: bool = True
    cpu_offload: bool = False
    torch_compile: bool = False
    vae_slicing: bool = False
    graph_cache_size: int = 4
    graph_warmup_buckets: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        if type(self.graph_cache_size) is not int or self.graph_cache_size < 1:
            raise ValueError("graph_cache_size must be a positive integer")
        for bucket in self.graph_warmup_buckets:
            if len(bucket) != 3 or any(type(value) is not int for value in bucket):
                raise ValueError(
                    "graph_warmup_buckets must contain (height, width, text_tokens) integer triples"
                )
            height, width, tokens = bucket
            if (
                height <= 0
                or width <= 0
                or height % 16
                or width % 16
                or not 1 <= tokens <= 4096
            ):
                raise ValueError(
                    "Graph warmup requires image sizes divisible by 16 and 1..4096 text tokens"
                )
        self.graph_warmup_buckets = normalize_moss_image_graph_buckets(
            self.graph_warmup_buckets
        )
        if len(self.graph_warmup_buckets) * 2 > self.graph_cache_size:
            raise ValueError(
                "graph_cache_size must hold both CFG branches of each warmup bucket"
            )


@Engine.register
class MOSSImageEntry(Entry):
    name: ClassVar[str] = "moss_image"
    args_cls: ClassVar[type[EntryArgs]] = MOSSImageArgs

    def __init__(self) -> None:
        self.transformer: MOSSImageTransformer | None = None
        self.text_encoder: MOSSImageTextEncoder | None = None
        self.vae: MOSSImageVAE | None = None
        self.scheduler: MOSSImageScheduler | None = None
        self.load_reports: dict[str, LoadReport] = {}

    def setup(self, args: EntryArgs) -> None:
        if not isinstance(args, MOSSImageArgs) or args.checkpoint_dir is None:
            raise ValueError("MOSSImageArgs.checkpoint_dir is required")
        checkpoint = Path(args.checkpoint_dir)
        engine = get_engine_config()
        if args.graph_warmup_buckets and not engine.runtime.use_cuda_graph:
            raise ValueError("graph_warmup_buckets requires runtime.use_cuda_graph")
        device, dtype = engine.device.target, engine.device.params_dtype
        construction_device = "cpu" if args.cpu_offload else device
        config = load_config(checkpoint / "transformer", MOSSImageConfig)
        self.transformer = MOSSImageTransformer(
            config, params_dtype=dtype, device=construction_device
        ).eval()
        self.load_reports["transformer"] = load_pretrained(
            self.transformer,
            checkpoint / "transformer",
            remap=moss_image_weight_remap,
            strict=True,
        )
        text_runner, processor = None, None
        if args.load_text_encoder:
            text_config = load_config(checkpoint / "text_encoder", MOSSImageTextConfig)
            self.text_encoder = MOSSImageTextEncoder(
                text_config, params_dtype=dtype, device=construction_device
            ).eval()
            self.load_reports["text_encoder"] = load_pretrained(
                self.text_encoder,
                checkpoint / "text_encoder",
                remap=moss_image_text_weight_remap,
                strict=True,
            )
            text_runner = MOSSImageTextRunner(self.text_encoder)
            processor = MOSSImageProcessor(checkpoint / "tokenizer")
        vae_runner = None
        if args.load_vae:
            vae_config = load_config(checkpoint / "vae", MOSSImageVAEConfig)
            if vae_config.packed_latent_channels != config.latent_channels:
                raise ValueError(
                    "VAE packed channels must match transformer latent_channels"
                )
            self.vae = MOSSImageVAE(
                vae_config, params_dtype=torch.float32, device=construction_device
            ).eval()
            self.load_reports["vae"] = load_pretrained(
                self.vae,
                checkpoint / "vae",
                remap=moss_image_vae_weight_remap,
                strict=True,
            )
            vae_runner = MOSSImageVAERunner(self.vae, use_slicing=args.vae_slicing)
        sampler_config = MOSSImageSamplerConfig.from_json(
            checkpoint / "scheduler" / "scheduler_config.json"
        )
        runner = MOSSImageRunner(
            self.transformer,
            text_runner=text_runner,
            vae_runner=vae_runner,
            device=device,
            cpu_offload=args.cpu_offload,
            torch_compile=args.torch_compile,
            use_cuda_graph=engine.runtime.use_cuda_graph,
            graph_cache_size=args.graph_cache_size,
        )
        self.scheduler = MOSSImageScheduler(
            runner,
            sampler_config=sampler_config,
            processor=processor,
            device=device,
            dtype=dtype,
            latent_channels=config.latent_channels,
        )
        self.scheduler.setup()
        runner.warmup_graphs(args.graph_warmup_buckets)
        logger.info_rank0(
            "MOSS-Image ready (CPU offload=%s, CUDA graphs=%s)",
            args.cpu_offload,
            engine.runtime.use_cuda_graph,
        )

    def step(self, request: MOSSImageRequest) -> MOSSImageOutput:
        if self.scheduler is None:
            raise RuntimeError("call setup() before step()")
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.transformer = None
        self.text_encoder = None
        self.vae = None
        self.load_reports.clear()

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        return {
            name: model
            for name, model in (
                ("transformer", self.transformer),
                ("text_encoder", self.text_encoder),
                ("vae", self.vae),
            )
            if model is not None
        }


__all__ = ["MOSSImageArgs", "MOSSImageEntry"]
