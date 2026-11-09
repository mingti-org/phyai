"""Engine plugins for native Emu3.5 image encoding and decoding."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.xiaomi_u0.configuration_vision import U0VisionConfig
from phyai.models.xiaomi_u0.modeling_vision import (
    U0VisionImageEncoder,
    U0VisionTokenizer,
    vision_decode_weight_remap,
    vision_encode_weight_remap,
)
from phyai.models.xiaomi_u0.model_runner_vision import U0VisionRunner
from phyai.models.xiaomi_u0.scheduler_vision import (
    U0VisionEncodeScheduler,
    U0VisionScheduler,
)
from phyai.weights.loader import load_pretrained


@dataclass
class U0VisionArgs(EntryArgs):
    """Local VisionTokenizer checkpoint directory."""

    checkpoint_dir: str = ""


class _U0VisionEntry(Entry):
    """Load the requested vision path and its codebook from the local checkpoint."""

    args_cls: ClassVar[type[EntryArgs]] = U0VisionArgs
    model_cls: ClassVar[type[U0VisionImageEncoder] | type[U0VisionTokenizer]]
    scheduler_cls: ClassVar[type[U0VisionEncodeScheduler] | type[U0VisionScheduler]]
    weight_remap: ClassVar[Callable[[str], str | None]]

    def __init__(self) -> None:
        """Create an entry without allocating model parameters."""
        self.model: U0VisionImageEncoder | U0VisionTokenizer | None = None
        self.scheduler: U0VisionEncodeScheduler | U0VisionScheduler | None = None

    def setup(self, args: U0VisionArgs) -> None:
        """Build the FP32 vision model and load its required tensors strictly."""
        if self.model is not None:
            raise RuntimeError("The vision entry is already set up.")
        checkpoint = Path(args.checkpoint_dir)
        if not args.checkpoint_dir or not checkpoint.is_dir():
            raise ValueError(
                "A local VisionTokenizer checkpoint directory is required."
            )
        engine = get_engine_config()
        if engine.runtime.use_cuda_graph:
            raise ValueError("Native U0 vision inference requires CUDA Graph disabled.")
        if engine.device.params_dtype != torch.float32:
            raise ValueError("The released VisionTokenizer requires FP32 parameters.")
        config = U0VisionConfig.from_json(checkpoint / "config.json")
        if (
            config.codebook_size != 131072
            or config.embed_dim != 256
            or config.spatial_factor != 16
        ):
            raise ValueError(
                "The U0 example requires the released Emu3.5 codebook and 16x spatial scale."
            )
        try:
            self.model = self.model_cls(config, device=engine.device.target).eval()
            report = load_pretrained(
                self.model, checkpoint, remap=self.weight_remap, strict=True
            )
            if report.casts:
                raise ValueError(
                    "VisionTokenizer weights must load in FP32 without precision casts."
                )
            self.scheduler = self.scheduler_cls(U0VisionRunner(self.model))
            self.scheduler.setup()
        except Exception:
            self.close()
            raise

    def step(self, codes: torch.Tensor) -> torch.Tensor:
        """Delegate an image or a code grid to the configured scheduler."""
        if self.scheduler is None:
            raise RuntimeError("Set up the vision entry before running inference.")
        return self.scheduler.step(codes)

    def close(self) -> None:
        """Release the scheduler and native vision model."""
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.model = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        """Expose the loaded model to engine inspection tools."""
        return {} if self.model is None else {"vision": self.model}


@Engine.register
class U0VisionEntry(_U0VisionEntry):
    """Load the codebook and decoder used for single-image generation."""

    name: ClassVar[str] = "xiaomi_u0_decode"
    model_cls = U0VisionTokenizer
    scheduler_cls = U0VisionScheduler
    weight_remap = staticmethod(vision_decode_weight_remap)


@Engine.register
class U0VisionEncodeEntry(_U0VisionEntry):
    """Load the encoder and codebook used for reference-image conditioning."""

    name: ClassVar[str] = "xiaomi_u0_encode"
    model_cls = U0VisionImageEncoder
    scheduler_cls = U0VisionEncodeScheduler
    weight_remap = staticmethod(vision_encode_weight_remap)
