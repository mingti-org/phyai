"""LingBot-VLA 2.0 engine plugin entry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.layers.quant.active import load_quant_plan, use_quant_plan
from phyai.models.lingbot_v2.configuration_lingbotv2 import LingBotVLA2Config
from phyai.models.lingbot_v2.modeling_lingbotv2 import (
    LingBotV2Model,
    lingbot_v2_weight_remap,
)
from phyai.models.lingbot_v2.scheduler_ws1_lingbotv2 import (
    LingBotV2Request,
    LingBotV2WS1Scheduler,
)
from phyai.utils import load_config
from phyai.weights import load_pretrained


_WeightRemap = Callable[[str], str | None] | dict[str, str] | None


def compose_lingbot_v2_weight_remap(
    user_remap: _WeightRemap,
) -> Callable[[str], str | None]:
    """Apply LingBot's inference-only rules before an optional user remap."""

    if user_remap is None:
        return lingbot_v2_weight_remap

    if callable(user_remap):

        def chained(key: str) -> str | None:
            mapped = lingbot_v2_weight_remap(key)
            if mapped is None:
                return None
            return user_remap(mapped)

        return chained

    if isinstance(user_remap, dict):
        rules = tuple(user_remap.items())

        def chained_dict(key: str) -> str | None:
            mapped = lingbot_v2_weight_remap(key)
            if mapped is None:
                return None
            for source, target in rules:
                if source in mapped:
                    mapped = mapped.replace(source, target)
            return mapped

        return chained_dict

    raise TypeError(
        "weight_remap must be callable, dict, or None; "
        f"got {type(user_remap).__name__}."
    )


@dataclass
class LingBotV2Args(EntryArgs):
    """Arguments used to construct the LingBot V2 inference plugin."""

    checkpoint_dir: str | Path | None = None
    config: LingBotVLA2Config | None = None
    max_batch_size: int = 1
    num_images: int = 3
    max_vision_tokens_per_image: int | None = None
    weight_remap: _WeightRemap = None
    weight_strict: bool = True
    vision_params_dtype: torch.dtype | None = None
    vision_patch_embed_backend: str = "gemm"

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError(
                f"max_batch_size must be positive, got {self.max_batch_size}."
            )
        if self.num_images <= 0:
            raise ValueError(f"num_images must be positive, got {self.num_images}.")
        if (
            self.max_vision_tokens_per_image is not None
            and self.max_vision_tokens_per_image <= 0
        ):
            raise ValueError(
                "max_vision_tokens_per_image must be positive when provided, "
                f"got {self.max_vision_tokens_per_image}."
            )
        if self.vision_patch_embed_backend not in {"conv3d", "gemm"}:
            raise ValueError(
                "vision_patch_embed_backend must be 'conv3d' or 'gemm', "
                f"got {self.vision_patch_embed_backend!r}."
            )


@Engine.register
class LingBotV2Entry(Entry):
    """Build, run, and release a LingBot-VLA 2.0 inference instance."""

    name: ClassVar[str] = "lingbot_v2"
    args_cls: ClassVar[type[EntryArgs]] = LingBotV2Args

    def __init__(self) -> None:
        self.model: LingBotV2Model | None = None
        self.scheduler: LingBotV2WS1Scheduler | None = None
        self._previous_matmul_precision: str | None = None

    def setup(self, args: LingBotV2Args) -> None:
        """Resolve config, load weights, and prepare the WS1 scheduler."""

        if self.model is not None or self.scheduler is not None:
            raise RuntimeError("LingBotV2Entry.setup may only be called once.")

        # The released FP32 router uses PyTorch matmul precision "high". Keep
        # this process-wide policy at the engine boundary and restore it when
        # this entry is closed or setup fails.
        self._previous_matmul_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        try:
            self._setup(args)
        except BaseException:
            self._restore_matmul_precision()
            raise

    def _setup(self, args: LingBotV2Args) -> None:
        """Construct the model and scheduler after process policy is installed."""

        engine_config = get_engine_config()
        if args.config is not None:
            config = args.config
        elif args.checkpoint_dir is not None:
            config = load_config(args.checkpoint_dir, LingBotVLA2Config)
        else:
            config = LingBotVLA2Config()

        with use_quant_plan(load_quant_plan(args.checkpoint_dir)):
            model = LingBotV2Model(
                config,
                vision_params_dtype=args.vision_params_dtype,
                vision_patch_embed_backend=args.vision_patch_embed_backend,
                device=engine_config.device.target,
            )

        if args.checkpoint_dir is not None:
            load_pretrained(
                model,
                args.checkpoint_dir,
                remap=compose_lingbot_v2_weight_remap(args.weight_remap),
                strict=args.weight_strict,
            )

        scheduler = LingBotV2WS1Scheduler(
            model,
            max_batch_size=args.max_batch_size,
            num_images=args.num_images,
            max_vision_tokens_per_image=args.max_vision_tokens_per_image,
            device=engine_config.device.target,
            use_cuda_graph=engine_config.runtime.use_cuda_graph,
        )
        scheduler.setup()
        self.model = model
        self.scheduler = scheduler

    def step(self, request: LingBotV2Request) -> torch.Tensor:
        """Run one request and return ``(B, chunk_size, action_dim)`` actions."""

        if self.scheduler is None:
            raise RuntimeError(
                "LingBotV2Entry.step called before setup; scheduler is None."
            )
        return self.scheduler.step(request)

    def close(self) -> None:
        try:
            if self.scheduler is not None:
                self.scheduler.close()
        finally:
            self.scheduler = None
            self.model = None
            self._restore_matmul_precision()

    def _restore_matmul_precision(self) -> None:
        previous = self._previous_matmul_precision
        if previous is not None:
            torch.set_float32_matmul_precision(previous)
            self._previous_matmul_precision = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        """Expose model leaves to the engine's optional tensor dumper."""

        if self.model is None:
            return {}
        return {"model": self.model}


__all__ = [
    "LingBotV2Args",
    "LingBotV2Entry",
    "compose_lingbot_v2_weight_remap",
]
