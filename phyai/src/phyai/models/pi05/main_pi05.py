"""pi0.5 plugin entry — the engine's pi0.5 hook.

Two pieces, both consumed by :class:`~phyai.engine.Engine`:

* :class:`PI05Args` — typed arg bundle. Carries an HF-style
  ``checkpoint_dir`` (the folder with ``config.json`` +
  ``model.safetensors`` / ``model.safetensors.index.json``), an
  optional :class:`PI05Config` override, and the model-specific
  scheduler knob ``max_batch_size``.
* :class:`PI05Entry` — :class:`~phyai.engine.Entry` subclass that
  parses the checkpoint folder's ``config.json`` via
  :func:`phyai.utils.load_config`, builds a :class:`PI05Model`,
  runs :func:`load_pretrained`, constructs and warms a
  :class:`PI05Scheduler`, then forwards :meth:`step` to it.

Importing this module registers ``PI05Entry`` with the engine via
``@Engine.register`` at class-definition time. The engine's own
``engine.py`` imports this module at the bottom of its file, so the
plugin is available the moment the engine module is loaded.

``device`` / ``params_dtype`` / ``*_backend`` / ``use_cuda_graph`` are
*not* fields on :class:`PI05Args`; they live on
:class:`~phyai.engine.EngineArgs` and are propagated to
:class:`PI05Model` and the scheduler via the
:class:`~phyai.engine_config.EngineConfig` singleton (which the
engine seeds in its ``__init__``). Adding them here would just create
a second source of truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.layers.quant.active import load_quant_plan, use_quant_plan
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.modeling_pi05 import PI05Model
from phyai.models.pi05.scheduler_pi05 import (
    PI05Request,
    PI05RolloutOutput,
    PI05RolloutRequest,
    PI05Scheduler,
)
from phyai.utils import get_logger, load_config
from phyai.weights import LoadReport, load_pretrained
from phyai.weights.loader import WeightLoadSession


logger = get_logger(__name__)


# Keys present in the upstream pi0.5 base safetensors that the inference
# model never consumes. The expert was trained with a lm_head sibling to
# the language model's, but at inference the expert produces flow-matching
# vectors (not tokens), so its lm_head weight has no parameter to land in. The
# source state dict may expose its tied PaliGemma embedding a second
# time through model.language_model.embed_tokens; the canonical lm_head alias
# already loads that same target.
# Dropping it silently keeps `weight_strict=True` honest for everything else.
_PI05_CANONICAL_DROP_KEYS: frozenset[str] = frozenset(
    {
        "paligemma_with_expert.gemma_expert.lm_head.weight",
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
    }
)


def _compose_remap(
    user_remap: Callable[[str], str | None] | dict[str, str] | None,
) -> Callable[[str], str | None]:
    """Normalize source keys, then filter canonical pi0.5 keys.

    ``user_remap`` maps checkpoint source keys into the canonical PhyAI/HF
    namespace used by :data:`_PI05_CANONICAL_DROP_KEYS`.
    """
    if user_remap is None:
        return lambda k: None if k in _PI05_CANONICAL_DROP_KEYS else k
    if callable(user_remap):

        def _chained(k: str) -> str | None:
            remapped = user_remap(k)
            return None if remapped in _PI05_CANONICAL_DROP_KEYS else remapped

        return _chained
    if isinstance(user_remap, dict):
        rules = list(user_remap.items())

        def _chained_dict(k: str) -> str | None:
            for src, dst in rules:
                if src in k:
                    k = k.replace(src, dst)
            return None if k in _PI05_CANONICAL_DROP_KEYS else k

        return _chained_dict
    raise TypeError(
        f"weight_remap must be callable, dict, or None; got {type(user_remap).__name__}"
    )


@dataclass
class PI05Args(EntryArgs):
    """Args bundle for the pi0.5 plugin.

    HuggingFace-style: ``checkpoint_dir`` is one folder containing both
    ``config.json`` and the safetensors shard(s) (single
    ``model.safetensors`` *or* ``model.safetensors.index.json`` plus
    its shards). It is empty by default for unit-test / debug paths —
    :meth:`PI05Entry.setup` then constructs the model with the default
    :class:`PI05Config` and skips :func:`load_pretrained`.

    ``config`` is an optional override; when ``None`` (the default) the
    config is read from ``checkpoint_dir/config.json`` if a directory
    is supplied, otherwise it falls back to ``PI05Config()`` (the
    public ``pi05_base`` geometry).

    ``weight_remap`` and ``weight_strict`` pass straight through to
    :func:`load_pretrained` for checkpoints whose key names diverge
    from upstream pi0.5 (HF rewrites, mid-training renames, etc.).

    ``vision_params_dtype`` selects the vision tower's compute precision
    independently of the engine dtype: pass ``torch.float32`` to run
    SigLIP + projector + their norms in fp32 (the openpi / lerobot parity
    path) while the language + expert stacks stay at the engine dtype
    (bf16). ``None`` (default) keeps the vision encoder and projector at the
    engine dtype; the patch stem and position addition remain fp32.

    ``inputs_image_shape`` declares the cameras the model consumes, one
    ``[H, W, C]`` per image (e.g. ``[[224, 224, 3], [224, 224, 3]]`` for two
    cameras). ``len(...)`` sets the camera count; each ``[H, W]`` is the
    native input size, resize-with-padded to the tower's ``image_size`` grid
    at request time. ``None`` (default) keeps the pi05_base contract of three
    cameras already at ``image_size``. ``C`` must equal
    ``config.vision.num_channels``.

    ``capture_rollout=True`` captures the expert's trajectory-producing
    graph instead of the final-action-only inference graph. Set it for an RL
    rollout engine; :class:`PI05RolloutRequest` remains functional without it
    but runs the expert loop eagerly.

    """

    checkpoint_dir: str | Path | None = None
    config: PI05Config | None = None
    max_batch_size: int = 1
    weight_remap: Callable[[str], str | None] | dict[str, str] | None = None
    weight_strict: bool = True
    vision_params_dtype: torch.dtype | None = None
    inputs_image_shape: list[list[int]] | None = None
    capture_rollout: bool = False


@Engine.register
class PI05Entry(Entry):
    """pi0.5 inference plugin entry."""

    name: ClassVar[str] = "pi05"
    args_cls: ClassVar[type[EntryArgs]] = PI05Args

    def __init__(self) -> None:
        # Default-init the slots so :meth:`step` / :meth:`close` can
        # check for "setup not yet run" without an attr-exists guard.
        self.model: PI05Model | None = None
        self.scheduler: PI05Scheduler | None = None
        self.weight_remap: Callable[[str], str | None] | dict[str, str] | None = None
        self.weight_update: WeightLoadSession | None = None

    def setup(self, args: PI05Args) -> None:  # type: ignore[override]
        """Build model, load weights, construct + warm the scheduler."""
        eng = get_engine_config()
        self.weight_remap = args.weight_remap

        # Resolve config: explicit override > checkpoint folder > defaults.
        if args.config is not None:
            config = args.config
        elif args.checkpoint_dir is not None:
            config = load_config(args.checkpoint_dir, PI05Config)
        else:
            config = PI05Config()

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

        num_images = self._resolve_num_images(args.inputs_image_shape, config)

        self.scheduler = PI05Scheduler(
            self.model,
            max_batch_size=args.max_batch_size,
            num_images=num_images,
            device=eng.device.target,
            use_cuda_graph=eng.runtime.use_cuda_graph,
            capture_rollout=args.capture_rollout,
        )
        self.scheduler.setup()

    @staticmethod
    def _resolve_num_images(
        inputs_image_shape: list[list[int]] | None, config: PI05Config
    ) -> int:
        """Validate ``inputs_image_shape`` and return the camera count.

        ``None`` defaults to 3 (the pi05_base contract). Otherwise each entry
        is a native ``[H, W, C]``; only ``C`` is constrained here (it must
        equal ``config.vision.num_channels`` — SigLIP's conv has a fixed
        channel count), while ``H`` / ``W`` are free because the scheduler
        resize-with-pads each camera to the tower's ``image_size`` grid. The
        count is ``len(inputs_image_shape)``.
        """
        if inputs_image_shape is None:
            return 3
        if len(inputs_image_shape) == 0:
            raise ValueError("inputs_image_shape must list at least one image.")
        num_channels = config.vision.num_channels
        for i, shape in enumerate(inputs_image_shape):
            if len(shape) != 3:
                raise ValueError(f"inputs_image_shape[{i}]={shape} must be [H, W, C].")
            h, w, c = shape
            if h <= 0 or w <= 0:
                raise ValueError(
                    f"inputs_image_shape[{i}] H/W must be positive, got [{h}, {w}]."
                )
            if c != num_channels:
                raise ValueError(
                    f"inputs_image_shape[{i}] channels {c} != "
                    f"config.vision.num_channels={num_channels}."
                )
        return len(inputs_image_shape)

    def step(self, request: PI05Request) -> torch.Tensor:  # type: ignore[override]
        """Run one pi0.5 inference; return the action chunk ``(B, chunk, action_dim)``."""
        if isinstance(request, PI05RolloutRequest):
            raise TypeError(
                "PI05RolloutRequest must be passed to rollout_step(), not step()."
            )
        if self.scheduler is None:
            raise RuntimeError(
                "PI05Entry.step called before setup; the scheduler is None."
            )
        return self.scheduler.step(request)

    def rollout_step(self, request: PI05RolloutRequest) -> PI05RolloutOutput:
        """Run a pi0.5 RL rollout and return model-space trajectory data."""
        if self.scheduler is None:
            raise RuntimeError("PI05Entry.rollout_step called before setup.")
        return self.scheduler.rollout_step(request)

    def begin_weight_update(self) -> None:
        """Start one incremental HF-named weight update."""
        if self.model is None:
            raise RuntimeError("PI05Entry weight update requires a loaded model.")
        if self.weight_update is not None:
            raise RuntimeError("A PI05Entry weight update is already active.")
        self.weight_update = WeightLoadSession(
            self.model,
            remap=_compose_remap(self.weight_remap),
            source_label="hot update",
        )

    def update_weights(
        self,
        weights: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        """Apply one streamed named-tensor batch to the active session."""
        if self.weight_update is None:
            raise RuntimeError("begin_weight_update() must be called first.")
        self.weight_update.load(weights)

    def finish_weight_update(self) -> LoadReport:
        """Validate the update and refresh scheduler constants in place."""
        if self.weight_update is None:
            raise RuntimeError("No PI05Entry weight update is active.")
        weight_update = self.weight_update
        try:
            if self.scheduler is None:
                raise RuntimeError("PI05Entry weight update requires a scheduler.")
            if not weight_update.report.loaded:
                raise RuntimeError(
                    "PI05Entry hot update did not match any PhyAI model weights."
                )
            report = weight_update.finish(strict=False, require_all=False)
            if report.unexpected:
                logger.warning_rank0(
                    "pi0.5 hot update skipped %d source-only or unknown tensors: %s",
                    len(report.unexpected),
                    report.unexpected[:5],
                )
            self.scheduler.refresh_weight_dependent_state()
            return report
        finally:
            self.weight_update = None

    def abort_weight_update(self) -> None:
        """Discard bookkeeping for a failed streamed update."""
        self.weight_update = None

    def close(self) -> None:
        self.weight_update = None
        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None
        self.model = None

    def dump_targets(self) -> dict[str, torch.nn.Module]:  # type: ignore[override]
        """Expose the pi0.5 model for engine-driven tensor dumping.

        Returns ``{"model": self.model}`` so dumped operator keys read
        ``model.paligemma_lm.layers.0.self_attn.o_proj`` etc. (aligned with
        the ``model.safetensors`` parameter names). Returns ``{}`` before
        :meth:`setup` has built the model, so a dump-enabled engine that
        somehow queries early just records nothing instead of crashing.
        """
        if self.model is None:
            return {}
        return {"model": self.model}


__all__ = ["PI05Args", "PI05Entry"]
