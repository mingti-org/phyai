"""GR00T-N1.7 engine plugin entry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.gr00t_n17.configuration_gr00t_n17 import GR00TN17Config
from phyai.models.gr00t_n17.modeling_gr00t_n17 import GR00TN17Model
from phyai.models.gr00t_n17.scheduler_ws1_gr00t_n17 import (
    GR00TN17Request,
    GR00TN17WS1Scheduler,
)
from phyai.utils import load_config
from phyai.weights import load_pretrained


_GR00TN17_PENDING_DROP_PREFIXES: tuple[str, ...] = ()


def gr00t_n17_weight_remap(name: str) -> str | None:
    """Map a GR00T-N1.7 checkpoint key to its phyai parameter key.

    The native GR00T-N1.7 checkpoint keys already match the phyai parameters
    (each parameter carries its own ``hf_keys``), so this is identity except for
    keys under :data:`_GR00TN17_PENDING_DROP_PREFIXES`, which return ``None``
    (dropped). Follows the ``<model>_weight_remap`` convention.
    """
    if name.startswith(_GR00TN17_PENDING_DROP_PREFIXES):
        return None
    return name


def _compose_remap(
    user_remap: Callable[[str], str | None] | dict[str, str] | None,
) -> Callable[[str], str | None]:
    """Compose :func:`gr00t_n17_weight_remap` with an optional user remap."""
    if user_remap is None:
        return gr00t_n17_weight_remap
    if callable(user_remap):

        def _chained(k: str) -> str | None:
            mapped = gr00t_n17_weight_remap(k)
            if mapped is None:
                return None
            return user_remap(mapped)

        return _chained
    if isinstance(user_remap, dict):
        rules = list(user_remap.items())

        def _chained_dict(k: str) -> str | None:
            mapped = gr00t_n17_weight_remap(k)
            if mapped is None:
                return None
            for src, dst in rules:
                if src in mapped:
                    mapped = mapped.replace(src, dst)
            return mapped

        return _chained_dict
    raise TypeError(
        f"weight_remap must be callable, dict, or None; got {type(user_remap).__name__}"
    )


@dataclass
class GR00TN17Args(EntryArgs):
    """Args bundle for the GR00T-N1.7 plugin.

    ``checkpoint_dir`` is optional for import/shape-only development.
    When provided, ``config.json`` is loaded from that directory and
    safetensors are loaded into the native model.

    The engine consumes already-prepared model-input tensors (see
    :class:`GR00TN17Request`); image transform, Qwen3-VL patchify,
    tokenization, and state/action normalization are the caller's job via
    ``phyai_utils_tools.models.gr00t.GR00TProcessor``. Hence there is no
    ``embodiment_tag`` / processor knob here — those live with the processor.
    """

    checkpoint_dir: str | Path | None = None
    config: GR00TN17Config | None = None
    max_batch_size: int = 1
    weight_remap: Callable[[str], str | None] | dict[str, str] | None = None
    weight_strict: bool = True
    backbone_model_name_or_path: str | Path | None = None
    backbone_transformers_loading_kwargs: dict[str, Any] | None = None


@Engine.register
class GR00TN17Entry(Entry):
    """GR00T-N1.7 inference plugin entry."""

    name: ClassVar[str] = "gr00t_n17"
    args_cls: ClassVar[type[EntryArgs]] = GR00TN17Args

    def __init__(self) -> None:
        self.model: GR00TN17Model | None = None
        self.scheduler: GR00TN17WS1Scheduler | None = None

    def setup(self, args: EntryArgs) -> None:
        if not isinstance(args, GR00TN17Args):
            raise TypeError(
                "GR00TN17Entry.setup expected GR00TN17Args, got "
                f"{type(args).__name__}."
            )
        eng = get_engine_config()
        if args.config is not None:
            config = args.config
        elif args.checkpoint_dir is not None:
            config = load_config(args.checkpoint_dir, GR00TN17Config)
        else:
            config = GR00TN17Config()
        if args.backbone_model_name_or_path is not None:
            config = replace(
                config,
                backbone=replace(
                    config.backbone,
                    model_name=str(args.backbone_model_name_or_path),
                    qwen3vl=None,
                ),
            )

        self.model = GR00TN17Model(
            config,
            params_dtype=eng.device.params_dtype,
            device=eng.device.target,
            backbone_transformers_loading_kwargs=args.backbone_transformers_loading_kwargs,
        )

        if args.checkpoint_dir is not None:
            self.model.backbone._load_qwen3vl_model()
            load_pretrained(
                self.model,
                args.checkpoint_dir,
                remap=_compose_remap(args.weight_remap),
                strict=args.weight_strict,
            )

        self.scheduler = GR00TN17WS1Scheduler(
            self.model,
            max_batch_size=args.max_batch_size,
            device=eng.device.target,
            use_cuda_graph=eng.runtime.use_cuda_graph,
        )
        self.scheduler.setup()

    def step(self, request: GR00TN17Request) -> torch.Tensor:
        if self.scheduler is None:
            raise RuntimeError("GR00TN17Entry.step called before setup.")
        return self.scheduler.step(request)

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        self.model = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:
        """Expose the GR00T-N1.7 model for engine-driven tensor dumping.

        Returns ``{"model": self.model}`` so dumped operator keys read
        ``model.backbone.qwen3vl_model.model.language_model.layers.0...`` and
        ``model.action_head...`` (aligned with the checkpoint parameter names).
        Returns ``{}`` before :meth:`setup` has built the model, so a
        dump-enabled engine that queries early records nothing instead of crashing.
        """
        if self.model is None:
            return {}
        return {"model": self.model}


__all__ = ["GR00TN17Args", "GR00TN17Entry"]
