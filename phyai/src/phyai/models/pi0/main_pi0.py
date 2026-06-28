"""pi0 plugin entry: the engine's pi0 hook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.pi0.configuration_pi0 import PI0Config
from phyai.models.pi0.modeling_pi0 import PI0Model
from phyai.models.pi0.scheduler_ws1_pi0 import PI0Request, PI0WS1Scheduler
from phyai.utils import load_config
from phyai.weights import load_pretrained

_PI0_DROP_KEYS: frozenset[str] = frozenset(
    {
        "paligemma_with_expert.gemma_expert.lm_head.weight",
    }
)


def _pi0_default_remap(k: str) -> str | None:
    """Apply pi0's built-in checkpoint compatibility rules."""

    if k.startswith("model."):
        k = k.removeprefix("model.")
    if k in _PI0_DROP_KEYS:
        return None
    if k.startswith("time_mlp_in."):
        return k.replace("time_mlp_in.", "action_time_mlp_in.", 1)
    if k.startswith("time_mlp_out."):
        return k.replace("time_mlp_out.", "action_time_mlp_out.", 1)
    return k


def _compose_remap(
    user_remap: Callable[[str], str | None] | dict[str, str] | None,
) -> Callable[[str], str | None]:
    """Combine pi0-specific dropped keys with the caller's remap."""

    if user_remap is None:
        return _pi0_default_remap
    if callable(user_remap):

        def _chained(k: str) -> str | None:
            k = _pi0_default_remap(k)
            if k is None:
                return None
            return user_remap(k)

        return _chained
    if isinstance(user_remap, dict):
        rules = list(user_remap.items())

        def _chained_dict(k: str) -> str | None:
            k = _pi0_default_remap(k)
            if k is None:
                return None
            for src, dst in rules:
                if src in k:
                    k = k.replace(src, dst)
            return k

        return _chained_dict
    raise TypeError(
        f"weight_remap must be callable, dict, or None; got {type(user_remap).__name__}"
    )


@dataclass
class PI0Args(EntryArgs):
    """Args bundle for the pi0 plugin."""

    checkpoint_dir: str | Path | None = None
    config: PI0Config | None = None
    max_batch_size: int = 1
    weight_remap: Callable[[str], str | None] | dict[str, str] | None = None
    weight_strict: bool = True
    vision_params_dtype: torch.dtype | None = torch.float32


@Engine.register
class PI0Entry(Entry):
    """pi0 inference plugin entry."""

    name: ClassVar[str] = "pi0"
    args_cls: ClassVar[type[EntryArgs]] = PI0Args

    def __init__(self) -> None:
        self.model: PI0Model | None = None
        self.scheduler: PI0WS1Scheduler | None = None

    def setup(self, args: PI0Args) -> None:  # type: ignore[override]
        """Build model, load weights, construct + warm the scheduler."""

        eng = get_engine_config()
        if args.config is not None:
            config = args.config
        elif args.checkpoint_dir is not None:
            config = load_config(args.checkpoint_dir, PI0Config)
        else:
            config = PI0Config()

        self.model = PI0Model(
            config,
            vision_params_dtype=args.vision_params_dtype,
            device=eng.device.target,
        )
        if args.checkpoint_dir is not None:
            load_pretrained(
                self.model,
                args.checkpoint_dir,
                remap=_compose_remap(args.weight_remap),
                strict=args.weight_strict,
            )

        self.scheduler = PI0WS1Scheduler(
            self.model,
            max_batch_size=args.max_batch_size,
            device=eng.device.target,
            use_cuda_graph=eng.runtime.use_cuda_graph,
        )
        self.scheduler.setup()

    def step(self, request: PI0Request) -> torch.Tensor:  # type: ignore[override]
        """Run one pi0 inference; return action chunk ``(B, chunk, action_dim)``."""

        if self.scheduler is None:
            raise RuntimeError("PI0Entry.step called before setup; scheduler is None.")
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        self.model = None


__all__ = ["PI0Args", "PI0Entry"]
