from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from phyai.utils.logging import this_rank_log

if TYPE_CHECKING:
    from phyai.layers.quant.plan import QuantPlan

logger = logging.getLogger(__name__)

_active_plan: ContextVar["QuantPlan | None"] = ContextVar(
    "phyai_quant_plan", default=None
)


def get_active_plan() -> "QuantPlan | None":
    """The QuantPlan in effect for the model currently being built, if any."""
    return _active_plan.get()


@contextmanager
def use_quant_plan(plan: "QuantPlan | None") -> Iterator[None]:
    """Make ``plan`` the active plan for the duration of the ``with`` block."""
    token = _active_plan.set(plan)
    try:
        yield
    finally:
        _active_plan.reset(token)


def load_quant_plan(
    checkpoint_dir: str | Path | None,
    *,
    revision: str | None = None,
) -> "QuantPlan | None":
    """Build the QuantPlan implied by a checkpoint's quant config.

    Reads ``config.json``'s ``quantization_config`` / ``compression_config``
    and, for ModelOpt checkpoints, a standalone ``hf_quant_config.json``.
    Returns ``None`` for an unquantized checkpoint or when ``checkpoint_dir``
    is ``None`` — callers then keep the default bf16 behavior.
    """
    if checkpoint_dir is None:
        return None

    import json

    from phyai.layers.quant.importers import ConfigSources, build_quant_plan
    from phyai.utils.checkpoint import resolve_checkpoint

    # Resolve (download if a HF repo id) the checkpoint into a local folder.
    folder = Path(resolve_checkpoint(checkpoint_dir, revision=revision))

    # Try:            config.json  ->  quantization_config (else compression_config)
    # Quant Framework: HF-inline (fp8 / compressed-tensors, aka llm-compressor)
    hf_quant_config = None
    config_path = folder / "config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        hf_quant_config = raw.get("quantization_config") or raw.get(
            "compression_config"
        )

    # Try:            hf_quant_config.json  ->  standalone file
    # Quant Framework: ModelOpt (NVIDIA TensorRT Model Optimizer)
    standalone = None
    standalone_path = folder / "hf_quant_config.json"
    if standalone_path.is_file():
        with standalone_path.open("r", encoding="utf-8") as f:
            standalone = json.load(f)

    if not hf_quant_config and not standalone:
        return None

    loaded_from = []
    if hf_quant_config:
        loaded_from.append(str(config_path))
    if standalone:
        loaded_from.append(str(standalone_path))
    this_rank_log(
        logger, logging.INFO, "Loaded quant config from %s", ", ".join(loaded_from)
    )

    return build_quant_plan(
        ConfigSources(hf_quant_config=hf_quant_config, standalone=standalone)
    )


__all__ = ["get_active_plan", "use_quant_plan", "load_quant_plan"]
