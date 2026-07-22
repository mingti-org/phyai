"""Cosmos3 world-size-N (tensor-parallel) action/policy plugin entry.

Sibling of :mod:`phyai.models.cosmos3.main_cosmos3_policy` for the multi-GPU
tensor-parallel ("wn") path. Identical build except it constructs the
tensor-parallel
:class:`~phyai.models.cosmos3.scheduler_wn_cosmos3_policy.Cosmos3PolicyWNScheduler`.
Launch one process per rank under ``torchrun`` with ``ParallelConfig(world_size=N,
tp_size=N)`` (see ``examples/cosmos3/run_cosmos3_policy_wn.py``). Every rank runs
the identical denoise loop; only rank 0 should persist the returned action / video.
"""

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
from phyai.models.cosmos3.scheduler_ws1_cosmos3_policy import Cosmos3ActionRequest
from phyai.models.cosmos3.scheduler_wn_cosmos3_policy import Cosmos3PolicyWNScheduler
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE, cosmos3_vae_weight_remap
from phyai.utils import load_config, this_rank_log
from phyai.weights import load_pretrained


logger = logging.getLogger(__name__)


@dataclass
class Cosmos3PolicyWNArgs(EntryArgs):
    """Args bundle for the cosmos3 tensor-parallel action/policy plugin."""

    checkpoint_dir: str | Path | None = None
    config: Cosmos3Config | None = None
    flow_shift: float = 5.0
    use_karras_sigmas: bool | None = False
    policy_modeling_mode: str | None = None
    decode_video: bool = False
    weight_strict: bool = False


@Engine.register
class Cosmos3PolicyWNEntry(Entry):
    """Cosmos3 tensor-parallel action plugin (policy / forward / inverse dynamics)."""

    name: ClassVar[str] = "cosmos3_policy_wn"
    args_cls: ClassVar[type[EntryArgs]] = Cosmos3PolicyWNArgs

    def __init__(self) -> None:
        self.transformer: Cosmos3Transformer | None = None
        self.vae: Cosmos3WanVAE | None = None
        self.scheduler: Cosmos3PolicyWNScheduler | None = None
        self.decode_video = False

    def setup(self, args: Cosmos3PolicyWNArgs) -> None:  # type: ignore[override]
        """Build the transformer (+ optional VAE) and warm the policy scheduler."""
        if args.checkpoint_dir is None:
            raise ValueError(
                "Cosmos3PolicyWNArgs.checkpoint_dir is required (no random-weight "
                "debug path for a diffusion checkpoint this size)."
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
        self.scheduler = Cosmos3PolicyWNScheduler(
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
            "Cosmos3 tensor-parallel policy plugin ready (tp=%d, decode_video=%s, "
            "flow_shift=%s, use_karras_sigmas=%s, modeling_mode=%s).",
            self.scheduler.tp_size,
            self.decode_video,
            args.flow_shift,
            use_karras,
            config.policy_modeling_mode,
        )

    def step(
        self, request: Cosmos3ActionRequest
    ) -> torch.Tensor | dict[str, torch.Tensor]:  # type: ignore[override]
        """Run one action request (on every rank; result identical across ranks)."""
        if self.scheduler is None:
            raise RuntimeError(
                "Cosmos3PolicyWNEntry.step called before setup; the scheduler is None."
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


__all__ = ["Cosmos3PolicyWNArgs", "Cosmos3PolicyWNEntry"]
