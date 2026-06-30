"""Cosmos3 world-size-N (tensor-parallel) generation plugin entry.

Sibling of :mod:`phyai.models.cosmos3.main_cosmos3` for the multi-GPU
tensor-parallel ("wn") path. Identical build (transformer + VAE [+ AVAE], weight
load, scheduler warmup) except it constructs the tensor-parallel
:class:`~phyai.models.cosmos3.scheduler_wn_cosmos3.Cosmos3T2VWNScheduler`.

The engine bootstrap (``init_dist`` -> ``P.init`` 5-axis mesh -> ``L.init``) runs
before :meth:`setup`, so the model constructors find the TP mesh ready and the
parallel layers shard themselves on load. Drive TP via ``ParallelConfig(world_size=N,
tp_size=N)`` and launch one process per rank under ``torchrun`` (see
``examples/cosmos3/run_cosmos3_wn.py``). Every rank runs the identical denoise loop;
only rank 0 should persist the returned media.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.cosmos3.avae_sound import (
    Cosmos3AVAESoundDecoder,
    cosmos3_avae_weight_remap,
)
from phyai.models.cosmos3.configuration_cosmos3 import (
    Cosmos3AVAESoundConfig,
    Cosmos3Config,
    Cosmos3WanVAEConfig,
)
from phyai.models.cosmos3.main_cosmos3 import _should_load_sound
from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Transformer,
    cosmos3_weight_remap,
)
from phyai.models.cosmos3.sampler_unipc import resolve_use_karras_sigmas
from phyai.models.cosmos3.scheduler_ws1_cosmos3 import Cosmos3T2VRequest
from phyai.models.cosmos3.scheduler_wn_cosmos3 import Cosmos3T2VWNScheduler
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE, cosmos3_vae_weight_remap
from phyai.utils import load_config, this_rank_log
from phyai.weights import load_pretrained


logger = logging.getLogger(__name__)


@dataclass
class Cosmos3WNArgs(EntryArgs):
    """Args bundle for the cosmos3 tensor-parallel generation plugin."""

    checkpoint_dir: str | Path | None = None
    config: Cosmos3Config | None = None
    flow_shift: float = 10.0
    use_karras_sigmas: bool | None = False
    load_sound: bool | None = None
    weight_strict: bool = False
    torch_compile: bool = False
    compile_kwargs: dict | None = None


@Engine.register
class Cosmos3WNEntry(Entry):
    """Cosmos3 tensor-parallel generation plugin entry (T2V / I2V / T2AV / I2AV)."""

    name: ClassVar[str] = "cosmos3_wn"
    args_cls: ClassVar[type[EntryArgs]] = Cosmos3WNArgs

    def __init__(self) -> None:
        self.transformer: Cosmos3Transformer | None = None
        self.vae: Cosmos3WanVAE | None = None
        self.avae: Cosmos3AVAESoundDecoder | None = None
        self.scheduler: Cosmos3T2VWNScheduler | None = None

    def setup(self, args: Cosmos3WNArgs) -> None:  # type: ignore[override]
        """Build the transformer + VAE (+ AVAE), load weights, warm the scheduler."""
        if args.checkpoint_dir is None:
            raise ValueError(
                "Cosmos3WNArgs.checkpoint_dir is required (no random-weight debug "
                "path for a diffusion checkpoint this size)."
            )
        ckpt = Path(args.checkpoint_dir)
        eng = get_engine_config()
        device = eng.device.target
        dtype = eng.device.params_dtype

        config = (
            args.config
            if args.config is not None
            else load_config(ckpt / "transformer", Cosmos3Config)
        )

        self.transformer = Cosmos3Transformer(
            config, params_dtype=dtype, device=device
        ).eval()
        load_pretrained(
            self.transformer,
            ckpt / "transformer",
            remap=cosmos3_weight_remap,
            strict=args.weight_strict,
        )

        vae_config = load_config(ckpt / "vae", Cosmos3WanVAEConfig)
        self.vae = Cosmos3WanVAE(vae_config)
        load_pretrained(
            self.vae,
            ckpt / "vae",
            remap=cosmos3_vae_weight_remap,
            strict=args.weight_strict,
        )
        self.vae = self.vae.to(device=device, dtype=dtype).eval()

        if _should_load_sound(args, config):
            avae_config = load_config(ckpt / "sound_tokenizer", Cosmos3AVAESoundConfig)
            self.avae = Cosmos3AVAESoundDecoder(avae_config)
            load_pretrained(
                self.avae,
                ckpt / "sound_tokenizer",
                remap=cosmos3_avae_weight_remap,
                strict=args.weight_strict,
            )
            self.avae = self.avae.to(device=device, dtype=dtype).eval()

        use_karras = resolve_use_karras_sigmas(args.use_karras_sigmas, ckpt)
        self.scheduler = Cosmos3T2VWNScheduler(
            self.transformer,
            vae=self.vae,
            avae=self.avae,
            device=device,
            flow_shift=args.flow_shift,
            use_karras_sigmas=use_karras,
            torch_compile=args.torch_compile,
            compile_kwargs=args.compile_kwargs,
        )
        self.scheduler.setup()
        this_rank_log(
            logger,
            logging.INFO,
            "Cosmos3 tensor-parallel generation plugin ready (tp=%d, sound=%s, "
            "flow_shift=%s, use_karras_sigmas=%s).",
            self.scheduler.tp_size,
            self.avae is not None,
            args.flow_shift,
            use_karras,
        )

    def step(
        self, request: Cosmos3T2VRequest
    ) -> torch.Tensor | dict[str, torch.Tensor | int]:  # type: ignore[override]
        """Run one generation request and decode it to media.

        Runs on every TP rank (the collectives must fire everywhere); the result is
        identical across ranks. The caller / launcher persists it only from rank 0.
        """
        if self.scheduler is None:
            raise RuntimeError(
                "Cosmos3WNEntry.step called before setup; the scheduler is None."
            )
        out = self.scheduler.step(request)
        if isinstance(out, dict):
            return {
                "video": self.scheduler.decode(out["video"]),
                "sound": self.scheduler.decode_sound(out["sound"]),
                "sample_rate": self.scheduler.sound_sample_rate,
            }
        return self.scheduler.decode(out)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.transformer = None
        self.vae = None
        self.avae = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:  # type: ignore[override]
        """Expose the denoiser for engine-driven tensor dumping (empty pre-setup)."""
        if self.transformer is None:
            return {}
        return {"transformer": self.transformer}


__all__ = ["Cosmos3WNArgs", "Cosmos3WNEntry"]
