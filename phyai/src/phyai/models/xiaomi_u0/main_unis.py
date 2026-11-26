"""Registered single-rank, tokenized U0 generation entry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.xiaomi_u0.configuration_unis import UNISConfig
from phyai.models.xiaomi_u0.modeling_unis_ar import UNISForCausalLM
from phyai.models.xiaomi_u0.requests_unis import (
    UNISGenerationRequest,
    UNISGenerationResult,
)
from phyai.models.xiaomi_u0.scheduler_unis import UNISGenerationScheduler
from phyai.weights.loader import load_pretrained


@dataclass
class UNISArgs(EntryArgs):
    """Local checkpoint, tokenizer mapping and an explicit cache budget.

    Before engine setup, set capacity to the larger conditional/unconditional
    input length plus the request's maximum number of generated tokens.
    """

    checkpoint_dir: str = ""
    tokenizer_mapping: dict[str, list[int]] = field(default_factory=dict)
    capacity: int = field(kw_only=True)
    paged_backend: str = "flashinfer"


@Engine.register
class UNISEntry(Entry):
    """Token generation plugin; image decode remains a separate operation."""

    name: ClassVar[str] = "xiaomi_u0"
    args_cls: ClassVar[type[EntryArgs]] = UNISArgs

    def __init__(self) -> None:
        """Create an entry with no model or scheduler allocated."""
        self.model: UNISForCausalLM | None = None
        self.scheduler: UNISGenerationScheduler | None = None

    def setup(self, args: UNISArgs) -> None:
        """Load local BF16 weights and initialize generation on the configured CUDA device.

        Reject CUDA Graph execution and checkpoint precision casts. Release
        partially initialized resources if model loading or scheduler setup fails.
        """
        if self.model is not None:
            raise RuntimeError("The U0 entry is already set up.")
        if not args.checkpoint_dir:
            raise ValueError("A local U0 checkpoint directory is required.")
        engine = get_engine_config()
        if torch.device(engine.device.target).type != "cuda":
            raise ValueError("The validated U0 entry requires CUDA.")
        if engine.device.params_dtype != torch.bfloat16:
            raise ValueError("The validated U0 entry requires BF16 model parameters.")
        if engine.runtime.use_cuda_graph:
            raise ValueError("U0 generation currently requires use_cuda_graph=False.")
        checkpoint = Path(args.checkpoint_dir)
        try:
            self.model = UNISForCausalLM(
                UNISConfig.from_json(checkpoint / "config.json"),
                dtype=engine.device.params_dtype,
                device=engine.device.target,
                attention_backend="flashinfer",
                lm_head_backend="torch",
            ).eval()
            report = load_pretrained(self.model, checkpoint, strict=True)
            if report.casts:
                raise ValueError("The U0 checkpoint must load without precision casts.")
            self.scheduler = UNISGenerationScheduler(
                self.model,
                capacity=args.capacity,
                tokenizer_mapping=args.tokenizer_mapping,
                paged_backend=args.paged_backend,
            )
            self.scheduler.setup()
        except Exception:
            self.close()
            raise

    def step(self, request: UNISGenerationRequest) -> UNISGenerationResult:
        """Generate tokens through the scheduler after successful setup."""
        if self.scheduler is None:
            raise RuntimeError("Call U0 entry setup before generation.")
        return self.scheduler.step(request)

    def close(self) -> None:
        """Release scheduler resources and clear the loaded model reference."""
        if self.scheduler is not None:
            self.scheduler.close()
        self.scheduler = None
        self.model = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        """Return the loaded model for engine inspection, or an empty mapping."""
        return {} if self.model is None else {"model": self.model}


__all__ = ["UNISArgs", "UNISEntry"]
