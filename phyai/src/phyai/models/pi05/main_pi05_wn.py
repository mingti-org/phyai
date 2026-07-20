"""pi0.5 data-parallel ("wn") plugin entry — the engine's pi0.5 DP hook.

Sibling of :mod:`phyai.models.pi05.main_pi05`. Identical model build + weight
load (a full replica per rank), except it sizes the single-card
:class:`PI05WS1Scheduler` for the per-rank shard and wraps it in a
:class:`~phyai.models.pi05.scheduler_wn_pi05.PI05WNScheduler`.

The engine bootstrap (``init_dist`` -> ``P.init`` 6-axis mesh -> ``L.init``) runs
before :meth:`setup`, so the ``dp`` axis is live here and ``per_rank_B`` can be
derived from it. Drive DP via ``ParallelConfig(world_size=N, dp_size=N,
tp_size=1)`` and launch one process per rank under ``torchrun`` (see
``examples/pi05/run_pi05_wn.py``). ``max_batch_size`` on the args is the TOTAL
batch across all ranks; each rank processes ``ceil(max_batch_size / dp_size)``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.layers.quant.active import load_quant_plan, use_quant_plan
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Entry, _compose_remap
from phyai.models.pi05.modeling_pi05 import PI05Model
from phyai.models.pi05.scheduler_wn_pi05 import PI05WNScheduler, _dp_rank_size
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request, PI05WS1Scheduler
from phyai.utils import load_config, this_rank_log
from phyai.weights import load_pretrained


logger = logging.getLogger(__name__)


@dataclass
class PI05WNArgs(EntryArgs):
    """Args bundle for the pi0.5 data-parallel plugin.

    Same shape as :class:`~phyai.models.pi05.main_pi05.PI05Args`, except
    ``max_batch_size`` is the **total** batch across all ``dp`` ranks (e.g. 32
    for the 8-GPU demo); the per-rank ws1 is sized ``ceil(max_batch_size /
    dp_size)``.
    """

    checkpoint_dir: str | Path | None = None
    config: PI05Config | None = None
    max_batch_size: int = 1
    weight_remap: Callable[[str], str | None] | dict[str, str] | None = None
    weight_strict: bool = True
    vision_params_dtype: torch.dtype | None = None
    inputs_image_shape: list[list[int]] | None = None


@Engine.register
class PI05WNEntry(Entry):
    """pi0.5 data-parallel inference plugin entry."""

    name: ClassVar[str] = "pi05_wn"
    args_cls: ClassVar[type[EntryArgs]] = PI05WNArgs

    def __init__(self) -> None:
        self.model: PI05Model | None = None
        self.scheduler: PI05WNScheduler | None = None

    def setup(self, args: PI05WNArgs) -> None:  # type: ignore[override]
        eng = get_engine_config()

        if args.config is not None:
            config = args.config
        elif args.checkpoint_dir is not None:
            config = load_config(args.checkpoint_dir, PI05Config)
        else:
            config = PI05Config()

        # Reuse pi05's recommended-engine overlay + num-image validation.
        eng = PI05Entry._apply_recommended_engine(eng, config)

        with use_quant_plan(load_quant_plan(args.checkpoint_dir)):
            self.model = PI05Model(
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
        num_images = PI05Entry._resolve_num_images(args.inputs_image_shape, config)

        _, dp_size = _dp_rank_size()
        per_rank_B = math.ceil(args.max_batch_size / dp_size)
        local = PI05WS1Scheduler(
            self.model,
            max_batch_size=per_rank_B,
            num_images=num_images,
            device=eng.device.target,
            use_cuda_graph=eng.runtime.use_cuda_graph,
        )
        self.scheduler = PI05WNScheduler(local, device=eng.device.target)
        self.scheduler.setup()
        this_rank_log(
            logger,
            logging.INFO,
            "pi0.5 DP plugin ready (dp=%d, total_batch=%d, per_rank_B=%d).",
            dp_size,
            args.max_batch_size,
            per_rank_B,
        )

    def step(self, request: PI05Request) -> torch.Tensor:  # type: ignore[override]
        if self.scheduler is None:
            raise RuntimeError(
                "PI05WNEntry.step called before setup; the scheduler is None."
            )
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        self.model = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:  # type: ignore[override]
        if self.model is None:
            return {}
        return {"model": self.model}


__all__ = ["PI05WNArgs", "PI05WNEntry"]
