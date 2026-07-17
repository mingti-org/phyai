"""MiniCPM-V 4.6 GR00T engine plugin."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.minicpm_gr00t.configuration_minicpm_gr00t import (
    MiniCPMGR00TConfig,
)
from phyai.models.minicpm_gr00t.model_runner_minicpm_gr00t import (
    MiniCPMGR00TModelRunner,
)
from phyai.models.minicpm_gr00t.modeling_minicpm_gr00t import (
    MiniCPMGR00TModel,
    minicpm_gr00t_weight_remap,
)
from phyai.models.minicpm_gr00t.scheduler_ws1_minicpm_gr00t import (
    MiniCPMGR00TRequest,
    MiniCPMGR00TWS1Scheduler,
)
from phyai.weights import load_pretrained


def _compose_remap(
    user_remap: Callable[[str], str | None] | dict[str, str] | None,
) -> Callable[[str], str | None]:
    """Apply the required model remap before an optional caller remap."""

    if user_remap is None:
        return minicpm_gr00t_weight_remap
    if callable(user_remap):

        def _chained(key: str) -> str | None:
            key = minicpm_gr00t_weight_remap(key)
            return None if key is None else user_remap(key)

        return _chained
    if isinstance(user_remap, dict):
        rules = tuple(user_remap.items())

        def _chained_dict(key: str) -> str | None:
            key = minicpm_gr00t_weight_remap(key)
            if key is None:
                return None
            for source, destination in rules:
                key = key.replace(source, destination)
            return key

        return _chained_dict
    raise TypeError(
        f"weight_remap must be callable, dict, or None; got "
        f"{type(user_remap).__name__}."
    )


@dataclass
class MiniCPMGR00TArgs(EntryArgs):
    """Arguments for the single-card MiniCPM-GR00T inference plugin."""

    checkpoint: str | Path | None = None
    config: MiniCPMGR00TConfig | None = None
    weight_remap: Callable[[str], str | None] | dict[str, str] | None = None
    weight_strict: bool = True
    gdn_backend: str = "fla"


@Engine.register
class MiniCPMGR00TEntry(Entry):
    """Build, load, and execute MiniCPM-GR00T inference."""

    name: ClassVar[str] = "minicpm_gr00t"
    args_cls: ClassVar[type[EntryArgs]] = MiniCPMGR00TArgs

    def __init__(self) -> None:
        self.model: MiniCPMGR00TModel | None = None
        self.scheduler: MiniCPMGR00TWS1Scheduler | None = None

    def setup(self, args: MiniCPMGR00TArgs) -> None:  # type: ignore[override]
        engine = get_engine_config()
        config = args.config or MiniCPMGR00TConfig()
        self.model = MiniCPMGR00TModel(
            config,
            vlm_params_dtype=torch.bfloat16,
            action_params_dtype=torch.float32,
            gdn_backend=args.gdn_backend,
            device=engine.device.target,
        )
        if args.checkpoint is not None:
            checkpoint = Path(args.checkpoint)
            load_pretrained(
                self.model,
                checkpoint,
                remap=_compose_remap(args.weight_remap),
                strict=args.weight_strict,
            )
        runner = MiniCPMGR00TModelRunner(
            self.model,
            device=engine.device.target,
            use_cuda_graph=engine.runtime.use_cuda_graph,
        )
        self.scheduler = MiniCPMGR00TWS1Scheduler(
            runner,
            device=engine.device.target,
        )
        self.scheduler.setup()

    def step(
        self,
        request: MiniCPMGR00TRequest,
    ) -> torch.Tensor:  # type: ignore[override]
        if self.scheduler is None:
            raise RuntimeError(
                "MiniCPMGR00TEntry.step called before setup; scheduler is None."
            )
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        self.model = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        if self.model is None:
            return {}
        return {"model": self.model}


__all__ = ["MiniCPMGR00TArgs", "MiniCPMGR00TEntry"]
