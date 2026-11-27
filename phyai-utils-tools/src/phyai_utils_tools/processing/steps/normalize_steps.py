"""Stats-based normalize / unnormalize steps — lerobot-exact schema.

Faithful port of lerobot's ``NormalizerProcessorStep`` /
``UnnormalizerProcessorStep`` (``.tmp/lerobot/.../processor/normalize_processor.py``)
so the serialized config round-trips with HuggingFace ``policy_*processor.json``:

* ``features`` — ``{feature_name: {"type": <FeatureType>, "shape": [...]}}``.
* ``norm_map`` — ``{<FeatureType>: <NormalizationMode>}`` (the *bucket* → mode).
* ``stats``   — ``{feature_name: {stat: tensor|list}}``, persisted to a sidecar
  ``.safetensors`` (flat keys ``"{feature_name}.{stat}"``) via
  :meth:`state_dict` / :meth:`load_state_dict`, NOT in the json config.

``__call__`` iterates ``features``; for each it looks up the mode by the
feature's ``type`` in ``norm_map`` and transforms the matching transition field
(:data:`STATE` for ``STATE``, :data:`ACTION` for ``ACTION``,
:data:`PIXEL_VALUES` for ``VISUAL``) — but only when that field is present and
stats exist for the feature. Empty ``features`` (the pi05_base default, which
ships no stats) ⇒ a pure no-op, so default numerics are bit-identical.

Supported modes (lerobot-exact): ``MEAN_STD``, ``MIN_MAX``, ``QUANTILES``
(q01/q99), ``QUANTILE10`` (q10/q90), ``IDENTITY``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    ACTION,
    PIXEL_VALUES,
    STATE,
    Transition,
)
from phyai_utils_tools.processing.types import (
    FeatureType,
    NormalizationMode,
    PolicyFeature,
)

# Which transition field each feature *bucket* normalizes. Other buckets
# (ENV/REWARD/LANGUAGE) have no inference-path field and are skipped.
_TYPE_TO_FIELD: dict[FeatureType, str] = {
    FeatureType.STATE: STATE,
    FeatureType.ACTION: ACTION,
    FeatureType.VISUAL: PIXEL_VALUES,
}


def _normalize_field(
    x: torch.Tensor,
    mode: NormalizationMode,
    s: dict[str, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Forward normalize one tensor by ``mode`` using stats ``s``."""
    if mode == NormalizationMode.MEAN_STD:
        return (x - s["mean"]) / (s["std"] + eps)
    if mode == NormalizationMode.MIN_MAX:
        return (x - s["min"]) / (s["max"] - s["min"] + eps) * 2.0 - 1.0
    if mode == NormalizationMode.QUANTILES:
        return (x - s["q01"]) / (s["q99"] - s["q01"] + eps) * 2.0 - 1.0
    if mode == NormalizationMode.QUANTILE10:
        return (x - s["q10"]) / (s["q90"] - s["q10"] + eps) * 2.0 - 1.0
    return x


def _unnormalize_field(
    x: torch.Tensor,
    mode: NormalizationMode,
    s: dict[str, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Inverse of :func:`_normalize_field`."""
    if mode == NormalizationMode.MEAN_STD:
        return x * (s["std"] + eps) + s["mean"]
    if mode == NormalizationMode.MIN_MAX:
        return (x + 1.0) / 2.0 * (s["max"] - s["min"] + eps) + s["min"]
    if mode == NormalizationMode.QUANTILES:
        return (x + 1.0) / 2.0 * (s["q99"] - s["q01"] + eps) + s["q01"]
    if mode == NormalizationMode.QUANTILE10:
        return (x + 1.0) / 2.0 * (s["q90"] - s["q10"] + eps) + s["q10"]
    return x


@dataclass
class _NormalizeBase(ProcessorStep):
    """Shared config / stats handling for the (un)normalize steps.

    ``features`` and ``norm_map`` accept either the typed objects or their
    JSON-string forms (so a config loaded from json constructs directly).
    ``stats`` is ``{feature_name: {stat: tensor|list}}``; absent stats (or
    ``IDENTITY`` / a feature type with no transition field) ⇒ no-op.
    """

    features: dict[str, Any] = field(default_factory=dict)
    norm_map: dict[Any, Any] = field(default_factory=dict)
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | str = torch.float32
    eps: float = 1e-8

    _features: dict[str, PolicyFeature] = field(
        default_factory=dict, init=False, repr=False
    )
    _norm_map: dict[FeatureType, NormalizationMode] = field(
        default_factory=dict, init=False, repr=False
    )
    _tensor_stats: dict[str, dict[str, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )
    _stats_explicit: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.dtype, str):
            name = self.dtype.removeprefix("torch.")
            if name not in ("float16", "bfloat16", "float32", "float64"):
                raise ValueError(f"Unsupported normalization dtype: {self.dtype!r}")
            self.dtype = getattr(torch, name)
        if self.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError("Normalization dtype must be a floating torch dtype")
        # features: {name: PolicyFeature | {"type","shape"}} -> {name: PolicyFeature}
        self._features = {}
        for name, feat in (self.features or {}).items():
            if isinstance(feat, PolicyFeature):
                self._features[name] = feat
            else:
                self._features[name] = PolicyFeature(
                    type=FeatureType(feat["type"]),
                    shape=tuple(feat.get("shape", ())),
                )
        # norm_map: {FeatureType|str: NormalizationMode|str} -> enums
        self._norm_map = {
            (k if isinstance(k, FeatureType) else FeatureType(k)): (
                v if isinstance(v, NormalizationMode) else NormalizationMode(v)
            )
            for k, v in (self.norm_map or {}).items()
        }
        self._stats_explicit = bool(self.stats)
        self._tensor_stats = self._to_tensor_stats(self.stats)

    def _to_tensor_stats(
        self, stats: dict[str, dict[str, Any]] | None
    ) -> dict[str, dict[str, torch.Tensor]]:
        out: dict[str, dict[str, torch.Tensor]] = {}
        for name, sub in (stats or {}).items():
            out[name] = {
                stat: torch.as_tensor(value, dtype=self.dtype, device=self.device)
                for stat, value in sub.items()
            }
        return out

    def get_config(self) -> dict[str, Any]:
        """lerobot schema: ``{eps, features:{name:{type,shape}}, norm_map}``.

        Stats are intentionally absent — they live in the sidecar state dict.
        """
        config = {
            "eps": self.eps,
            "features": {
                name: {"type": feat.type.value, "shape": list(feat.shape)}
                for name, feat in self._features.items()
            },
            "norm_map": {k.value: v.value for k, v in self._norm_map.items()},
        }

        # Omit the default to keep existing LeRobot configs unchanged. A
        # float64 sidecar alone is insufficient: loading casts to this dtype.
        if self.dtype != torch.float32:
            config["dtype"] = str(self.dtype).removeprefix("torch.")
        return config

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Flatten stats to ``{"{feature}.{stat}": tensor}`` (CPU) for the sidecar."""
        flat: dict[str, torch.Tensor] = {}
        for name, sub in self._tensor_stats.items():
            for stat, tensor in sub.items():
                flat[f"{name}.{stat}"] = tensor.detach().cpu()
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load flat ``"{feature}.{stat}"`` stats — unless stats were explicit.

        Mirrors lerobot: stats passed at construction win over a loaded sidecar.
        """
        if self._stats_explicit:
            return
        nested: dict[str, dict[str, torch.Tensor]] = {}
        for flat_key, tensor in state.items():
            name, stat = flat_key.rsplit(".", 1)
            nested.setdefault(name, {})[stat] = tensor.to(
                dtype=self.dtype, device=self.device
            )
        self._tensor_stats = nested

    def _apply(self, transition: Transition, inverse: bool) -> Transition:
        out = transition.copy()
        fn = _unnormalize_field if inverse else _normalize_field
        for name, feat in self._features.items():
            mode = self._norm_map.get(feat.type, NormalizationMode.IDENTITY)
            if mode == NormalizationMode.IDENTITY:
                continue
            field_name = _TYPE_TO_FIELD.get(feat.type)
            if field_name is None or field_name not in out:
                continue
            stats = self._tensor_stats.get(name)
            if not stats:
                continue
            out[field_name] = fn(out[field_name], mode, stats, self.eps)
        return out


@ProcessorStepRegistry.register("normalizer_processor")
@dataclass
class NormalizerStep(_NormalizeBase):
    """Forward-normalize the configured transition fields."""

    def __call__(self, transition: Transition) -> Transition:
        return self._apply(transition, inverse=False)


@ProcessorStepRegistry.register("unnormalizer_processor")
@dataclass
class UnnormalizerStep(_NormalizeBase):
    """Inverse-normalize the configured transition fields."""

    def __call__(self, transition: Transition) -> Transition:
        return self._apply(transition, inverse=True)


__all__ = [
    "FeatureType",
    "NormalizationMode",
    "NormalizerStep",
    "PolicyFeature",
    "UnnormalizerStep",
]
