"""DreamZero checkpoint assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn

import phyai.parallel as P
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroConfig
from phyai.models.dreamzero.image_encoder_wan import (
    DreamZeroWanImageEncoder,
    dreamzero_image_encoder_weight_remap,
)
from phyai.models.dreamzero.model_runner_image_encoder_dreamzero import (
    DreamZeroImageEncoderRunner,
)
from phyai.models.dreamzero.model_runner_text_encoder_dreamzero import (
    DreamZeroTextEncoderRunner,
)
from phyai.models.dreamzero.model_runner_vae_dreamzero import DreamZeroVAERunner
from phyai.models.dreamzero.modeling_dreamzero import (
    DreamZeroDiT,
    dreamzero_dit_weight_remap,
)
from phyai.models.dreamzero.pipeline_dreamzero import DreamZeroMinimalPipeline
from phyai.models.dreamzero.scheduler_ws1_dreamzero import DreamZeroWS1Scheduler
from phyai.models.dreamzero.text_encoder_wan import (
    DreamZeroWanTextEncoder,
    dreamzero_text_encoder_weight_remap,
)
from phyai.models.dreamzero.vae_wan import (
    DreamZeroWanVAE,
    dreamzero_vae_weight_remap,
)
from phyai.utils import get_logger, load_config
from phyai.weights import LoadReport, load_pretrained


logger = get_logger(__name__)

DreamZeroEncoderStrategy = Literal["replicated", "rank0_broadcast"]


@dataclass
class DreamZeroPipelineBundle:
    """Objects created for the minimal DreamZero inference path."""

    config: DreamZeroConfig
    dit: nn.Module
    text_encoder: nn.Module
    image_encoder: nn.Module
    vae: nn.Module
    text_runner: DreamZeroTextEncoderRunner | None
    image_runner: DreamZeroImageEncoderRunner | None
    vae_runner: DreamZeroVAERunner | None
    scheduler: DreamZeroWS1Scheduler
    pipeline: DreamZeroMinimalPipeline
    load_reports: dict[str, LoadReport]


@dataclass(frozen=True)
class DreamZeroBuildOptions:
    """Options for building DreamZero from a checkpoint directory."""

    checkpoint_dir: str | Path
    config: DreamZeroConfig | None = None
    dtype: torch.dtype = torch.bfloat16
    device: torch.device | str = "cuda"
    seed: int | None = None
    weight_strict: bool = True
    progress: bool | None = None
    attn_backend: str | None = None
    norm_backend: str | None = None
    official_attention: bool = False
    compile_scheduler: bool = False
    use_cfg_runner: bool = True
    encoder_strategy: DreamZeroEncoderStrategy = "replicated"
    image_use_cuda_graph: bool = False
    image_graph_batch_size: int = 1
    setup_scheduler: bool = True
    setup_encoders: bool = True
    sequential_cpu_offload: bool = False


def _load_component(
    model: nn.Module,
    checkpoint_dir: Path,
    *,
    remap,
    strict: bool,
    progress: bool | None,
) -> LoadReport:
    return load_pretrained(
        model,
        checkpoint_dir,
        remap=remap,
        strict=strict,
        progress=progress,
    )


def build_dreamzero_minimal_pipeline(
    options: DreamZeroBuildOptions,
) -> DreamZeroPipelineBundle:
    """Build real DreamZero components and connect the minimal pipeline.

    This function assumes process-level distributed state is already set up
    when tensor parallelism is desired. In normal serving that means the
    caller uses :class:`phyai.engine.Engine`; in standalone scripts it means
    initializing ``phyai.parallel`` before calling this builder.
    """

    if options.encoder_strategy not in ("replicated", "rank0_broadcast"):
        raise ValueError(
            "encoder_strategy must be 'replicated' or 'rank0_broadcast'; "
            f"got {options.encoder_strategy!r}."
        )

    checkpoint_dir = Path(options.checkpoint_dir)
    config = options.config or load_config(checkpoint_dir, DreamZeroConfig)
    device = torch.device(options.device)
    dtype = options.dtype
    model_device = torch.device("cpu") if options.sequential_cpu_offload else device
    encoder_rank = (
        not dist.is_available()
        or not dist.is_initialized()
        or options.encoder_strategy == "replicated"
        or P.default_mesh().group_rank("world") == 0
    )

    dit = DreamZeroDiT(
        config,
        params_dtype=dtype,
        device=model_device,
        attn_backend="official" if options.official_attention else options.attn_backend,
        norm_backend=options.norm_backend,
    ).eval()
    if encoder_rank:
        text_encoder = DreamZeroWanTextEncoder(
            config.text_encoder,
            params_dtype=dtype,
        ).eval()
        image_encoder = DreamZeroWanImageEncoder(
            config.image_encoder,
            # Match the official CLIP load path before the cast below. Direct
            # BF16 construction changes full-graph Inductor numerics on Thor.
            params_dtype=torch.float32,
        ).eval()
        vae = DreamZeroWanVAE(config.vae).eval()
    else:
        text_encoder = nn.Identity()
        image_encoder = nn.Identity()
        vae = nn.Identity()

    load_reports = {
        "dit": _load_component(
            dit,
            checkpoint_dir,
            remap=dreamzero_dit_weight_remap,
            strict=options.weight_strict,
            progress=options.progress,
        ),
    }
    if encoder_rank:
        load_reports.update(
            text_encoder=_load_component(
                text_encoder,
                checkpoint_dir,
                remap=dreamzero_text_encoder_weight_remap,
                strict=options.weight_strict,
                progress=options.progress,
            ),
            image_encoder=_load_component(
                image_encoder,
                checkpoint_dir,
                remap=dreamzero_image_encoder_weight_remap,
                strict=options.weight_strict,
                progress=options.progress,
            ),
            vae=_load_component(
                vae,
                checkpoint_dir,
                remap=dreamzero_vae_weight_remap,
                strict=options.weight_strict,
                progress=options.progress,
            ),
        )

    if options.sequential_cpu_offload:
        text_encoder = text_encoder.to(device="cpu", dtype=dtype).eval()
        image_encoder = image_encoder.to(device="cpu", dtype=dtype).eval()
        vae = vae.to(device="cpu", dtype=dtype).eval()
    else:
        text_encoder = text_encoder.to(device=device, dtype=dtype).eval()
        image_encoder = image_encoder.to(device=device, dtype=dtype).eval()
        vae = vae.to(device=device, dtype=dtype).eval()

    text_runner = (
        DreamZeroTextEncoderRunner(text_encoder, device=device, dtype=dtype)
        if encoder_rank
        else None
    )
    image_runner = (
        DreamZeroImageEncoderRunner(
            image_encoder,
            device=device,
            dtype=dtype,
            use_cuda_graph=options.image_use_cuda_graph,
            graph_batch_size=options.image_graph_batch_size,
        )
        if encoder_rank
        else None
    )
    vae_runner = (
        DreamZeroVAERunner(vae, device=device, dtype=dtype) if encoder_rank else None
    )

    if options.setup_encoders and encoder_rank:
        assert text_runner is not None
        assert image_runner is not None
        assert vae_runner is not None
        text_runner.setup()
        image_runner.setup()
        vae_runner.setup()

    scheduler = DreamZeroWS1Scheduler(
        dit,
        device=device,
        use_cfg_runner=options.use_cfg_runner,
        compile_updates=options.compile_scheduler,
    )
    if options.setup_scheduler:
        scheduler.setup()

    pipeline = DreamZeroMinimalPipeline(
        config=config,
        text_encoder=text_runner,
        image_encoder=image_runner,
        vae=vae_runner,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        seed=options.seed,
        sequential_cpu_offload=options.sequential_cpu_offload,
        encoder_rank0_broadcast=options.encoder_strategy == "rank0_broadcast",
    )

    logger.info_rank0(
        "DreamZero minimal pipeline ready from %s (dtype=%s, encoder_strategy=%s).",
        checkpoint_dir,
        dtype,
        options.encoder_strategy,
    )
    return DreamZeroPipelineBundle(
        config=config,
        dit=dit,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        vae=vae,
        text_runner=text_runner,
        image_runner=image_runner,
        vae_runner=vae_runner,
        scheduler=scheduler,
        pipeline=pipeline,
        load_reports=load_reports,
    )


__all__ = [
    "DreamZeroBuildOptions",
    "DreamZeroEncoderStrategy",
    "DreamZeroPipelineBundle",
    "build_dreamzero_minimal_pipeline",
]
