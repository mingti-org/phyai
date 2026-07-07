"""Cosmos3 generation plugin entry — the engine's cosmos3 hook."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.layers.quant.active import load_quant_plan, use_quant_plan
from phyai.models.cosmos3.avae_sound import (
    Cosmos3AVAESoundDecoder,
    cosmos3_avae_weight_remap,
)
from phyai.models.cosmos3.configuration_cosmos3 import (
    Cosmos3AVAESoundConfig,
    Cosmos3Config,
    Cosmos3WanVAEConfig,
)
from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Transformer,
    cosmos3_weight_remap,
)
from phyai.models.cosmos3.sampler_unipc import resolve_use_karras_sigmas
from phyai.models.cosmos3.scheduler_ws1_cosmos3 import (
    Cosmos3T2VRequest,
    Cosmos3T2VScheduler,
)
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE, cosmos3_vae_weight_remap
from phyai.utils import load_config, this_rank_log
from phyai.weights import load_pretrained


logger = logging.getLogger(__name__)


@dataclass
class Cosmos3Args(EntryArgs):
    """Args bundle for the cosmos3 generation plugin."""

    checkpoint_dir: str | Path | None = None
    config: Cosmos3Config | None = None
    # cosmos-framework native generation default (sample_args ``shift=10.0``).
    flow_shift: float = 10.0
    # UniPC sigma schedule. Default False = linear-flow + flow_shift, matching the
    # cosmos-framework native generation sampler. True = Karras (diffusers); ``None``
    # reads use_karras_sigmas from the checkpoint's scheduler_config.json.
    use_karras_sigmas: bool | None = False
    load_sound: bool | None = None
    weight_strict: bool = False
    torch_compile: bool = False
    compile_kwargs: dict | None = None


def _should_load_sound(args: Cosmos3Args, config: Cosmos3Config) -> bool:
    """Whether to build the AVAE: explicit ``load_sound`` wins, else ``sound_gen``."""
    if args.load_sound is not None:
        return bool(args.load_sound)
    return bool(config.sound_gen)


@Engine.register
class Cosmos3Entry(Entry):
    """Cosmos3 generation inference plugin entry (T2V / I2V / T2AV / I2AV)."""

    name: ClassVar[str] = "cosmos3"
    args_cls: ClassVar[type[EntryArgs]] = Cosmos3Args

    def __init__(self) -> None:
        # Default-init the slots so :meth:`step` / :meth:`close` /
        # :meth:`dump_targets` can check for "setup not yet run".
        self.transformer: Cosmos3Transformer | None = None
        self.vae: Cosmos3WanVAE | None = None
        self.avae: Cosmos3AVAESoundDecoder | None = None
        self.scheduler: Cosmos3T2VScheduler | None = None

    def setup(self, args: Cosmos3Args) -> None:  # type: ignore[override]
        """Build the transformer + VAE (+ AVAE), load weights, warm the scheduler."""
        if args.checkpoint_dir is None:
            raise ValueError(
                "Cosmos3Args.checkpoint_dir is required (no random-weight debug "
                "path for a diffusion checkpoint this size)."
            )
        ckpt = Path(args.checkpoint_dir)
        eng = get_engine_config()
        device = eng.device.target
        dtype = eng.device.params_dtype

        # Transformer config: explicit override > checkpoint folder.
        config = (
            args.config
            if args.config is not None
            else load_config(ckpt / "transformer", Cosmos3Config)
        )

        # The engine already ran P.init (mesh) + L.init (linear dispatcher) before
        # setup(), so the model constructors below find the dispatcher ready — no
        # register_mesh shim like the standalone example scripts need.
        with use_quant_plan(load_quant_plan(ckpt / "transformer")):
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
        self.scheduler = Cosmos3T2VScheduler(
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
            "Cosmos3 generation plugin ready (sound=%s, flow_shift=%s, "
            "use_karras_sigmas=%s).",
            self.avae is not None,
            args.flow_shift,
            use_karras,
        )

    def step(
        self, request: Cosmos3T2VRequest
    ) -> torch.Tensor | dict[str, torch.Tensor | int]:  # type: ignore[override]
        """Run one generation request and decode it to media.

        Video-only requests return pixels ``[B, 3, T, H, W]`` in ``[0, 1]``.
        Audio requests (``request.sound_frames`` set) return
        ``{"video": pixels, "sound": waveform [B, ch, samples] in [-1, 1],
        "sample_rate": int}``.
        """
        if self.scheduler is None:
            raise RuntimeError(
                "Cosmos3Entry.step called before setup; the scheduler is None."
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
        """Expose the denoiser for engine-driven tensor dumping.

        Returns ``{"transformer": self.transformer}`` so dumped operator keys
        align with the ``transformer/`` parameter names. Returns ``{}`` before
        :meth:`setup` so a dump-enabled engine that queries early records nothing
        instead of crashing.
        """
        if self.transformer is None:
            return {}
        return {"transformer": self.transformer}


__all__ = ["Cosmos3Args", "Cosmos3Entry"]
