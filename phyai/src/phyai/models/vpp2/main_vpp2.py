"""VPP2 engine entry for released video and action checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import ParallelConfig, get_engine_config
from phyai.models.vpp2.configuration_encoders import VPP2ImageEncoderConfig
from phyai.models.vpp2.configuration_vpp2 import VPP2Config, VPP2VideoConfig
from phyai.models.vpp2.image_encoder_wan import (
    VPP2WanImageEncoder,
    vpp2_image_encoder_weight_remap,
)
from phyai.models.vpp2.model_runner_encoders import VPP2EncoderRunner
from phyai.models.vpp2.model_runner_vpp2 import VPP2Runner
from phyai.models.vpp2.modeling_vpp2 import VPP2Model, vpp2_weight_remap
from phyai.models.vpp2.pipeline_vpp2 import VPP2Pipeline
from phyai.models.vpp2.scheduler_vpp2 import VPP2Scheduler
from phyai.models.vpp2.text_encoder_wan import (
    VPP2WanTextEncoder,
    vpp2_text_encoder_weight_remap,
)
from phyai.models.vpp2.vae_wan import VPP2WanVAE, vpp2_vae_weight_remap
from phyai.models.vpp2.weights_vpp2 import load_vpp2_weights, read_vpp2_checkpoint
from phyai.utils import get_logger, load_config
from phyai.weights import LoadReport, load_pretrained


logger = get_logger(__name__)


@dataclass
class VPP2Args(EntryArgs):
    checkpoint_dir: str | Path | None = None
    variant: str = "robodojo_joint2b_100k"
    config: VPP2Config | None = None
    video_checkpoint: str | Path | None = None
    action_checkpoint: str | Path | None = None
    cache_dir: str | Path = ".cache/vpp2/weights"
    load_encoders: bool = True
    torch_compile: bool = False
    use_cuda_graph: bool = True
    progress: bool = False


def vpp2_inference_defaults(variant: dict[str, Any], *, action: bool) -> dict[str, Any]:
    inference = variant.get("inference", {})
    if action:
        return {
            "num_inference_steps": inference.get(
                "steps", inference.get("euler_steps", 10)
            ),
            "sigma_shift": inference.get("shift", inference.get("action_shift", 1.0)),
            "action_horizon": inference.get(
                "horizon", inference.get("action_horizon", 32)
            ),
            "seed": inference.get("seed", 1),
        }
    return {
        "num_video_frames": variant.get("num_frames", 17),
        "num_inference_steps": inference.get("sampling_steps", 30),
        "sigma_shift": inference.get("sigma_shift", 3.0),
        "text_cfg_scale": inference.get("guidance_scale", 4.0),
    }


@Engine.register
class VPP2Entry(Entry):
    name: ClassVar[str] = "vpp2"
    args_cls: ClassVar[type[EntryArgs]] = VPP2Args
    parallel_domains: ClassVar[frozenset[str]] = frozenset(
        {"cfg", "dense", "attention"}
    )

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
        if resolved.outer.cfg_size not in (1, 2):
            raise ValueError("VPP2 supports CFG size 1 or 2.")
        if (
            resolved.dense.dp_size != 1
            or resolved.dense.sequence_parallel
            or resolved.attention.dp_size != 1
            or resolved.attention.cp_size != 1
            or resolved.attention.decode_cp_size != 1
            or resolved.attention.tp_size != resolved.dense.tp_size
        ):
            raise ValueError(
                "VPP2 requires matching dense/attention TP, DP=CP=1, and no sequence parallelism."
            )

    def __init__(self) -> None:
        self.model: VPP2Model | None = None
        self.pipeline: VPP2Pipeline | None = None
        self.load_reports: dict[str, LoadReport] = {}

    def setup(self, args: VPP2Args) -> None:
        if args.checkpoint_dir is None:
            raise ValueError("VPP2Args.checkpoint_dir is required.")
        root = Path(args.checkpoint_dir)
        release = json.loads((root / "config.json").read_text())
        variants = release.get("variants", {})
        if args.variant not in variants:
            raise ValueError(
                f"Unknown VPP2 variant {args.variant!r}; available: {sorted(variants)}"
            )
        variant = variants[args.variant]
        video_path = (
            Path(args.video_checkpoint)
            if args.video_checkpoint
            else root / variant["video_checkpoint"]
        )
        action_path = (
            Path(args.action_checkpoint)
            if args.action_checkpoint
            else root / variant["action_checkpoint"]
            if "action_checkpoint" in variant
            else None
        )
        for source in (video_path, action_path):
            if source is not None and not source.exists():
                raise FileNotFoundError(source)
        assets = root / release["shared_assets"]["wan_model_dir"]
        config = args.config
        if config is None:
            if action_path is None:
                video_config = load_config(assets, VPP2VideoConfig)
                video_dict = video_config.to_dict()
                video_dict["video_attention_mask_mode"] = "bidirectional"
                config = VPP2Config(video=VPP2VideoConfig.from_dict(video_dict))
            else:
                if action_path.is_dir() or action_path.suffix == ".safetensors":
                    raise ValueError(
                        "VPP2Args.config is required for an action checkpoint without release metadata."
                    )
                metadata = read_vpp2_checkpoint(action_path)
                config = VPP2Config.from_checkpoint_metadata(dict(metadata))
                del metadata
        engine = get_engine_config()
        self.model = VPP2Model(
            config,
            params_dtype=engine.device.params_dtype,
            device=engine.device.target,
            with_action=action_path is not None,
        ).eval()
        self.load_reports["model"] = load_vpp2_weights(
            self.model,
            video_path,
            action_path,
            remap=vpp2_weight_remap,
            cache_dir=args.cache_dir,
            progress=args.progress,
        )
        encoders = None
        if args.load_encoders:
            text = VPP2WanTextEncoder(
                params_dtype=engine.device.params_dtype, device=engine.device.target
            ).eval()
            image_config = VPP2ImageEncoderConfig()
            image = VPP2WanImageEncoder(
                image_config, params_dtype=torch.float16, device=engine.device.target
            ).eval()
            vae = VPP2WanVAE(device=engine.device.target).float().eval()
            for name, module, filename, remap in (
                (
                    "text_encoder",
                    text,
                    "models_t5_umt5-xxl-enc-bf16.pth",
                    vpp2_text_encoder_weight_remap,
                ),
                (
                    "image_encoder",
                    image,
                    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                    partial(vpp2_image_encoder_weight_remap, config=image_config),
                ),
                ("vae", vae, "Wan2.1_VAE.pth", vpp2_vae_weight_remap),
            ):
                self.load_reports[name] = load_pretrained(
                    module,
                    assets / filename,
                    remap=remap,
                    strict=True,
                    progress=args.progress,
                )
            encoders = VPP2EncoderRunner(
                text,
                image,
                vae,
                device=engine.device.target,
                dtype=engine.device.params_dtype,
                use_cuda_graph=args.use_cuda_graph and engine.runtime.use_cuda_graph,
            )
        runner = VPP2Runner(
            self.model,
            torch_compile=args.torch_compile,
            use_cuda_graph=args.use_cuda_graph and engine.runtime.use_cuda_graph,
        )
        self.pipeline = VPP2Pipeline(
            VPP2Scheduler(runner),
            encoders,
            defaults={
                "action": vpp2_inference_defaults(variant, action=True),
                "video": vpp2_inference_defaults(variant, action=False),
            },
            video_seed_offset=1 if args.variant.startswith("libero") else 1_000_003,
        )
        self.pipeline.setup()
        logger.info_rank0(
            "VPP2 %s loaded; %s", args.variant, self.load_reports["model"].summary()
        )

    def step(self, request: Any) -> dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("VPP2Entry.setup() must run before inference.")
        return self.pipeline.step(request)

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.close()
        self.pipeline = None
        self.model = None
        self.load_reports.clear()

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        targets = {} if self.model is None else {"model": self.model}
        if self.pipeline is not None and self.pipeline.encoders is not None:
            encoders = self.pipeline.encoders
            targets.update(
                text_encoder=encoders.text_encoder,
                image_encoder=encoders.image_encoder,
                vae=encoders.vae,
            )
        return targets
