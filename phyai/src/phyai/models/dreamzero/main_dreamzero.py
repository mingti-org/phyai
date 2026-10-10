"""DreamZero engine plugin entry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import ParallelConfig, get_engine_config
from phyai.models.dreamzero.builder_dreamzero import (
    DreamZeroBuildOptions,
    DreamZeroEncoderStrategy,
    DreamZeroPipelineBundle,
    build_dreamzero_minimal_pipeline,
)
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroConfig
from phyai.models.dreamzero.pipeline_dreamzero import DreamZeroPipelineOutput


@dataclass
class DreamZeroArgs(EntryArgs):
    """Args bundle for the DreamZero minimal inference plugin."""

    checkpoint_dir: str | Path | None = None
    config: DreamZeroConfig | None = None
    seed: int | None = None
    weight_strict: bool = True
    progress: bool | None = None
    encoder_strategy: DreamZeroEncoderStrategy = "replicated"
    use_cfg_runner: bool = True
    image_use_cuda_graph: bool = False
    image_graph_batch_size: int = 1
    sequential_cpu_offload: bool = False
    num_inference_steps: int | None = None
    dynamic_dit: bool = False
    dynamic_dit_scheduler_steps: int = 16
    official_attention: bool = False
    compile_scheduler: bool = False


@Engine.register
class DreamZeroEntry(Entry):
    """DreamZero minimal pipeline entry.

    Requests must already be preprocessed by ``phyai-utils-tools`` or an
    equivalent caller-owned processor. The main ``phyai`` package deliberately
    does not own tokenizer or action denormalization logic.
    """

    name: ClassVar[str] = "dreamzero"
    args_cls: ClassVar[type[EntryArgs]] = DreamZeroArgs

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
        if parallel.outer.cfg_size not in (1, 2):
            raise ValueError("DreamZero supports CFG size 1 or 2.")
        if (
            resolved.dense.dp_size != 1
            or resolved.dense.sequence_parallel
            or resolved.attention.dp_size != 1
            or resolved.attention.cp_size != 1
            or resolved.attention.decode_cp_size != 1
            or resolved.attention.tp_size != resolved.dense.tp_size
        ):
            raise ValueError(
                "DreamZero requires matching dense/attention TP, DP=CP=1, and no sequence parallelism."
            )

    def __init__(self) -> None:
        self.bundle: DreamZeroPipelineBundle | None = None
        self.num_inference_steps: int | None = None
        self.dynamic_dit = False
        self.dynamic_dit_scheduler_steps = 16

    def setup(self, args: DreamZeroArgs) -> None:  # type: ignore[override]
        if args.checkpoint_dir is None:
            raise ValueError("DreamZeroArgs.checkpoint_dir is required.")
        self.num_inference_steps = args.num_inference_steps
        self.dynamic_dit = args.dynamic_dit
        self.dynamic_dit_scheduler_steps = args.dynamic_dit_scheduler_steps
        eng = get_engine_config()
        self.bundle = build_dreamzero_minimal_pipeline(
            DreamZeroBuildOptions(
                checkpoint_dir=args.checkpoint_dir,
                config=args.config,
                dtype=eng.device.params_dtype,
                device=eng.device.target,
                seed=args.seed,
                weight_strict=args.weight_strict,
                progress=args.progress,
                use_cfg_runner=args.use_cfg_runner,
                encoder_strategy=args.encoder_strategy,
                image_use_cuda_graph=args.image_use_cuda_graph,
                image_graph_batch_size=args.image_graph_batch_size,
                sequential_cpu_offload=args.sequential_cpu_offload,
                official_attention=args.official_attention,
                compile_scheduler=args.compile_scheduler,
            )
        )

    def step(self, request: Any) -> DreamZeroPipelineOutput:  # type: ignore[override]
        if self.bundle is None:
            raise RuntimeError("DreamZeroEntry.step called before setup.")
        with torch.inference_mode():
            return self.bundle.pipeline(
                request,
                num_inference_steps=self.num_inference_steps,
                dynamic_dit=self.dynamic_dit,
                dynamic_dit_scheduler_steps=self.dynamic_dit_scheduler_steps,
            )

    def close(self) -> None:
        self.bundle = None
        self.num_inference_steps = None
        self.dynamic_dit = False
        self.dynamic_dit_scheduler_steps = 16

    def dump_targets(self) -> dict[str, torch.nn.Module]:  # type: ignore[override]
        if self.bundle is None:
            return {}
        return {"dit": self.bundle.dit}


__all__ = ["DreamZeroArgs", "DreamZeroEntry"]
