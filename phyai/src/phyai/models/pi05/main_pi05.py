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
  :class:`PI05WS1Scheduler`, then forwards :meth:`step` to it.

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

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config, set_engine_config
from phyai.layers.quant.active import load_quant_plan, use_quant_plan
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.modeling_pi05 import PI05Model
from phyai.models.pi05.scheduler_ws1_pi05 import (
    PI05Request,
    PI05WS1Scheduler,
)
from phyai.utils import load_config, this_rank_log
from phyai.weights import load_pretrained


logger = logging.getLogger(__name__)


# Keys present in the upstream pi0.5 base safetensors that the inference
# model never consumes. The expert was trained with a lm_head sibling to
# the language model's, but at inference the expert produces flow-matching
# vectors (not tokens), so its lm_head weight has no parameter to land in.
# Dropping it silently keeps `weight_strict=True` honest for everything else.
_PI05_DROP_KEYS: frozenset[str] = frozenset(
    {"paligemma_with_expert.gemma_expert.lm_head.weight"}
)


def _compose_remap(
    user_remap: Callable[[str], str | None] | dict[str, str] | None,
) -> Callable[[str], str | None]:
    """Combine the pi0.5-specific drop set with the caller's remap.

    The drop set runs first; any key it returns ``None`` for is removed
    before the user remap sees it.
    """
    if user_remap is None:
        return lambda k: None if k in _PI05_DROP_KEYS else k
    if callable(user_remap):

        def _chained(k: str) -> str | None:
            if k in _PI05_DROP_KEYS:
                return None
            return user_remap(k)

        return _chained
    if isinstance(user_remap, dict):
        rules = list(user_remap.items())

        def _chained_dict(k: str) -> str | None:
            if k in _PI05_DROP_KEYS:
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
    (bf16). ``None`` (default) keeps the tower at the engine dtype — the
    byte-identical single-dtype path.

    ``inputs_image_shape`` declares the cameras the model consumes, one
    ``[H, W, C]`` per image (e.g. ``[[224, 224, 3], [224, 224, 3]]`` for two
    cameras). ``len(...)`` sets the camera count; each ``[H, W]`` is the
    native input size, resize-with-padded to the tower's ``image_size`` grid
    at request time. ``None`` (default) keeps the pi05_base contract of three
    cameras already at ``image_size``. ``C`` must equal
    ``config.vision.num_channels``.
    """

    checkpoint_dir: str | Path | None = None
    config: PI05Config | None = None
    max_batch_size: int = 1
    weight_remap: Callable[[str], str | None] | dict[str, str] | None = None
    weight_strict: bool = True
    vision_params_dtype: torch.dtype | None = None
    inputs_image_shape: list[list[int]] | None = None


@Engine.register
class PI05Entry(Entry):
    """pi0.5 inference plugin entry."""

    name: ClassVar[str] = "pi05"
    args_cls: ClassVar[type[EntryArgs]] = PI05Args

    def __init__(self) -> None:
        # Default-init the slots so :meth:`step` / :meth:`close` can
        # check for "setup not yet run" without an attr-exists guard.
        self.model: PI05Model | None = None
        self.scheduler: PI05WS1Scheduler | None = None

    def setup(self, args: PI05Args) -> None:  # type: ignore[override]
        """Build model, load weights, construct + warm the scheduler."""
        eng = get_engine_config()

        # Resolve config: explicit override > checkpoint folder > defaults.
        if args.config is not None:
            config = args.config
        elif args.checkpoint_dir is not None:
            config = load_config(args.checkpoint_dir, PI05Config)
        else:
            config = PI05Config()

        # Apply pi0.5's recommended engine runtime knobs onto the singleton
        # *before* any model / attention wrapper is built (nothing has read
        # the workspace size or prefill backend yet — the flashinfer scratch
        # is allocated lazily at the first wrapper construction in
        # scheduler.setup). User / env choices always win: we only fill a
        # value the user left at its "unset" default.
        eng = self._apply_recommended_engine(eng, config)

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

        self.scheduler = PI05WS1Scheduler(
            self.model,
            max_batch_size=args.max_batch_size,
            num_images=num_images,
            device=eng.device.target,
            use_cuda_graph=eng.runtime.use_cuda_graph,
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

    @staticmethod
    def _apply_recommended_engine(eng, config: PI05Config):
        """Overlay ``config.recommended_engine`` onto the EngineConfig singleton.

        Returns the (possibly updated) :class:`EngineConfig`. Only fills
        knobs the user left unset so an explicit ``EngineConfig`` field or
        ``PHYAI_*`` env override always wins:

        * ``flashinfer_prefill_backend``: applied only if the runtime value
          is still ``None`` (the "defer to auto" sentinel).
        * ``flashinfer_workspace_bytes``: applied as a *floor* (``max``)
          and only when the effective prefill backend is ``"fa2"`` — FA2's
          split scratch for pi0.5's expert attention needs more than the
          128 MiB engine default.

        Shipping the recommendation on the model (and injecting it here at
        load time) keeps the shared attention backends model-agnostic
        instead of hardcoding pi0.5's kernel choice into them.
        """
        rec = config.recommended_engine
        runtime_kw: dict[str, object] = {}

        if (
            eng.runtime.flashinfer_prefill_backend is None
            and rec.flashinfer_prefill_backend is not None
        ):
            runtime_kw["flashinfer_prefill_backend"] = rec.flashinfer_prefill_backend

        effective_backend = runtime_kw.get(
            "flashinfer_prefill_backend", eng.runtime.flashinfer_prefill_backend
        )
        if (
            effective_backend == "fa2"
            and eng.runtime.flashinfer_workspace_bytes < rec.flashinfer_workspace_bytes
        ):
            runtime_kw["flashinfer_workspace_bytes"] = rec.flashinfer_workspace_bytes

        if not runtime_kw:
            return eng

        eng = replace(eng, runtime=replace(eng.runtime, **runtime_kw))
        set_engine_config(eng)
        ws_mib = eng.runtime.flashinfer_workspace_bytes // (1024 * 1024)
        this_rank_log(
            logger,
            logging.INFO,
            "pi0.5: applied model-recommended engine runtime %s "
            "(prefill_backend=%s, workspace=%d MiB). Rationale: the action "
            "expert's short-query/long-KV joint attention (head_dim 256) runs "
            "~2.5x faster on flashinfer's FA2 kernel than the auto-selected "
            "FA3. This is shipped as a per-model recommendation and injected "
            "into the EngineConfig at load time — preferred over hardcoding a "
            "model-specific kernel choice into the shared attention backends. "
            "Override via EngineConfig(runtime=RuntimeConfig("
            "flashinfer_prefill_backend=...)) or PHYAI_FLASHINFER_PREFILL_BACKEND.",
            runtime_kw,
            eng.runtime.flashinfer_prefill_backend,
            ws_mib,
        )
        return eng

    def step(self, request: PI05Request) -> torch.Tensor:  # type: ignore[override]
        """Run one pi0.5 inference; return the action chunk ``(B, chunk, action_dim)``."""
        if self.scheduler is None:
            raise RuntimeError(
                "PI05Entry.step called before setup; the scheduler is None."
            )
        return self.scheduler.step(request)

    def close(self) -> None:
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
