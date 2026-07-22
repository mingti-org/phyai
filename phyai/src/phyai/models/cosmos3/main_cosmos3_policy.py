"""Cosmos3 action/policy plugin entry — the engine's cosmos3 policy hook."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.layers.quant.active import load_quant_plan, use_quant_plan
from phyai.models.cosmos3.configuration_cosmos3 import (
    Cosmos3Config,
    Cosmos3WanVAEConfig,
)
from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Transformer,
    cosmos3_weight_remap,
)
from phyai.models.cosmos3.sampler_unipc import resolve_use_karras_sigmas
from phyai.models.cosmos3.scheduler_ws1_cosmos3_policy import (
    Cosmos3ActionRequest,
    Cosmos3PolicyScheduler,
)
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE, cosmos3_vae_weight_remap
from phyai.utils import load_config, this_rank_log
from phyai.weights import load_pretrained


logger = logging.getLogger(__name__)


@dataclass
class Cosmos3PolicyArgs(EntryArgs):
    """Args bundle for the cosmos3 action/policy plugin."""

    checkpoint_dir: str | Path | None = None
    config: Cosmos3Config | None = None
    # note(chenghua): Released DROID policy serving uses linear-flow UniPC shift 5.
    flow_shift: float = 5.0
    # note(chenghua): Native policy inference uses linear-flow UniPC, including
    # for Nano checkpoints whose exported Diffusers config requests Karras sigmas.
    use_karras_sigmas: bool | None = False
    policy_modeling_mode: str | None = None
    decode_video: bool = False
    weight_strict: bool = False


@Engine.register
class Cosmos3PolicyEntry(Entry):
    """Cosmos3 action inference plugin (policy / forward_dynamics / inverse_dynamics)."""

    name: ClassVar[str] = "cosmos3_policy"
    args_cls: ClassVar[type[EntryArgs]] = Cosmos3PolicyArgs

    def __init__(self) -> None:
        self.transformer: Cosmos3Transformer | None = None
        self.vae: Cosmos3WanVAE | None = None
        self.scheduler: Cosmos3PolicyScheduler | None = None
        self.decode_video = False

    def setup(self, args: Cosmos3PolicyArgs) -> None:  # type: ignore[override]
        """Build the transformer (+ optional VAE) and warm the policy scheduler."""
        if args.checkpoint_dir is None:
            raise ValueError(
                "Cosmos3PolicyArgs.checkpoint_dir is required (no random-weight debug "
                "path for a diffusion checkpoint this size)."
            )
        ckpt = Path(args.checkpoint_dir)
        eng = get_engine_config()
        device = eng.device.target
        dtype = eng.device.params_dtype
        self.decode_video = bool(args.decode_video)

        config = (
            args.config
            if args.config is not None
            else load_config(ckpt / "transformer", Cosmos3Config)
        )
        if args.policy_modeling_mode is not None:
            config = replace(config, policy_modeling_mode=args.policy_modeling_mode)
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

        if self.decode_video:
            vae_config = load_config(ckpt / "vae", Cosmos3WanVAEConfig)
            self.vae = Cosmos3WanVAE(vae_config)
            load_pretrained(
                self.vae,
                ckpt / "vae",
                remap=cosmos3_vae_weight_remap,
                strict=args.weight_strict,
            )
            self.vae = self.vae.to(device=device, dtype=dtype).eval()

        use_karras = resolve_use_karras_sigmas(args.use_karras_sigmas, ckpt)
        self.scheduler = Cosmos3PolicyScheduler(
            self.transformer,
            vae=self.vae,
            device=device,
            flow_shift=args.flow_shift,
            use_karras_sigmas=use_karras,
            use_cuda_graph=eng.runtime.use_cuda_graph,
        )
        self.scheduler.setup()
        this_rank_log(
            logger,
            logging.INFO,
            "Cosmos3 policy plugin ready (decode_video=%s, flow_shift=%s, "
            "use_karras_sigmas=%s, modeling_mode=%s).",
            self.decode_video,
            args.flow_shift,
            use_karras,
            config.policy_modeling_mode,
        )

    def step(
        self, request: Cosmos3ActionRequest
    ) -> torch.Tensor | dict[str, torch.Tensor]:  # type: ignore[override]
        """Run one action request.

        Returns the action ``[1, chunk, raw_action_dim]`` by default, or the full
        ``{"action", "video", "pixels"}`` dict when the plugin was built with
        ``decode_video=True``.
        """
        if self.scheduler is None:
            raise RuntimeError(
                "Cosmos3PolicyEntry.step called before setup; the scheduler is None."
            )
        out = self.scheduler.step(request, decode_video=self.decode_video)
        if self.decode_video:
            return out
        return out["action"]

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.transformer = None
        self.vae = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:  # type: ignore[override]
        """Expose the denoiser for engine-driven tensor dumping (empty pre-setup)."""
        if self.transformer is None:
            return {}
        return {"transformer": self.transformer}


__all__ = ["Cosmos3PolicyArgs", "Cosmos3PolicyEntry"]
