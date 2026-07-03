"""GR00T-N1.7 processor — checkpoint-agnostic state/action/vision preprocessing.

Relocated out of the phyai engine (``phyai.models.gr00t_n17``) so the engine owns
only modeling/runner/scheduler and consumes already-prepared tensors. The package
is a workspace leaf: it depends only on numpy / torch / PIL and an injected (or
``phyai_utils_tools``-loaded) tokenizer — **no ``phyai`` import**.

GR00T ships its own structured checkpoint contract:

* ``processor_config.json`` — per-embodiment modality keys, horizons, action
  interpretation, and processor kwargs.
* ``statistics.json`` — min/max, percentile, mean, std for state/action norm.
* ``embodiment_id.json`` — checkpoint embodiment tag -> action-head slot.

Image/token processing follows the official deterministic eval transform, then a
native Qwen3-VL preprocessor (patchify + chat-template image-token expansion);
the pure transforms / normalization math live in :mod:`.ops_gr00t`.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from phyai_utils_tools.models.gr00t.ops_gr00t import (
    DEFAULT_EMBODIMENT_ID_MAPPING,
    apply_sin_cos_encoding,
    enum_value,
    eval_transform_image,
    load_json,
    load_preprocessor_config,
    normalize_values_meanstd,
    normalize_values_minmax,
    qwen3vl_process_image,
    relative_eef_to_absolute,
    relative_non_eef_to_absolute,
    resolve_embodiment_tag,
    tuple2,
    unnormalize_values_meanstd,
    unnormalize_values_minmax,
)
from phyai_utils_tools.tokenizer import get_tokenizer


@dataclass(frozen=True)
class GR00TActionConfig:
    """Per-action-group interpretation metadata."""

    rep: str
    type: str
    format: str
    state_key: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GR00TActionConfig":
        return cls(
            rep=enum_value(data["rep"]),
            type=enum_value(data["type"]),
            format=enum_value(data["format"]),
            state_key=data.get("state_key"),
        )


@dataclass(frozen=True)
class GR00TModalityConfig:
    """Checkpoint modality config for video/state/action/language."""

    delta_indices: list[int]
    modality_keys: list[str]
    action_configs: list[GR00TActionConfig] | None = None
    sin_cos_embedding_keys: list[str] | None = None
    mean_std_embedding_keys: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GR00TModalityConfig":
        raw_action_configs = data.get("action_configs")
        action_configs = None
        if raw_action_configs is not None:
            action_configs = [
                GR00TActionConfig.from_dict(v) for v in raw_action_configs
            ]
        return cls(
            delta_indices=[int(v) for v in data["delta_indices"]],
            modality_keys=[str(v) for v in data["modality_keys"]],
            action_configs=action_configs,
            sin_cos_embedding_keys=data.get("sin_cos_embedding_keys"),
            mean_std_embedding_keys=data.get("mean_std_embedding_keys"),
        )

    def __post_init__(self) -> None:
        if not self.delta_indices:
            raise ValueError("delta_indices must be non-empty.")
        if not self.modality_keys:
            raise ValueError("modality_keys must be non-empty.")
        if self.action_configs is not None and len(self.action_configs) != len(
            self.modality_keys
        ):
            raise ValueError(
                "action_configs length must match modality_keys length: "
                f"{len(self.action_configs)} != {len(self.modality_keys)}."
            )


@dataclass(frozen=True)
class GR00TObservation:
    """Batched raw observation in the public GR00T policy shape.

    * ``video[view]``: ``np.uint8`` array shaped ``(B, T, H, W, 3)``.
    * ``state[name]``: ``np.float32`` array shaped ``(B, T, D)``.
    * ``language[name]``: nested ``list[list[str]]`` shaped ``(B, T)``.
    """

    video: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    language: dict[str, list[list[str]]]


@dataclass(frozen=True)
class GR00TProcessedInputs:
    """Tensor inputs consumed by the native GR00T-N1.7 scheduler."""

    tensors: dict[str, torch.Tensor]
    raw_state: dict[str, np.ndarray] | None = None


def parse_modality_configs(
    modality_configs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, GR00TModalityConfig]]:
    """Parse a raw modality-config mapping into :class:`GR00TModalityConfig`."""
    parsed: dict[str, dict[str, GR00TModalityConfig]] = {}
    for embodiment_tag, cfg in modality_configs.items():
        parsed[embodiment_tag] = {}
        for modality, value in cfg.items():
            parsed[embodiment_tag][modality] = (
                value
                if isinstance(value, GR00TModalityConfig)
                else GR00TModalityConfig.from_dict(value)
            )
    return parsed


def ensure_numpy_observation(value: Any) -> GR00TObservation:
    """Validate / coerce the top-level raw observation shape."""
    if isinstance(value, GR00TObservation):
        return value
    if not isinstance(value, dict):
        raise TypeError("GR00T observation must be a GR00TObservation or dict.")
    try:
        return GR00TObservation(
            video=value["video"],
            state=value["state"],
            language=value["language"],
        )
    except KeyError as e:
        raise ValueError("observation requires video, state, and language keys.") from e


def _build_norm_params(
    statistics: dict[str, Any],
    modality_configs: dict[str, dict[str, GR00TModalityConfig]],
    *,
    use_percentiles: bool,
    use_relative_action: bool,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Build per-embodiment min/max/mean/std (+ relative) normalization params."""
    norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for embodiment_tag, stats_for_tag in statistics.items():
        norm_params[embodiment_tag] = {}
        for modality in ("state", "action"):
            if modality not in stats_for_tag:
                continue
            norm_params[embodiment_tag][modality] = {}
            for key, stats in stats_for_tag[modality].items():
                min_vals = np.asarray(
                    stats["q01"] if use_percentiles else stats["min"],
                    dtype=np.float32,
                )
                max_vals = np.asarray(
                    stats["q99"] if use_percentiles else stats["max"],
                    dtype=np.float32,
                )
                mean_vals = np.asarray(stats["mean"], dtype=np.float32)
                std_vals = np.asarray(stats["std"], dtype=np.float32)
                norm_params[embodiment_tag][modality][key] = {
                    "min": min_vals,
                    "max": max_vals,
                    "mean": mean_vals,
                    "std": std_vals,
                    "dim": np.asarray(min_vals.shape[0], dtype=np.int64),
                }
        if not use_relative_action:
            continue
        tag_cfg = modality_configs.get(embodiment_tag)
        if tag_cfg is None or "action" not in tag_cfg:
            continue
        action_cfg = tag_cfg["action"]
        if action_cfg.action_configs is None:
            continue
        for key, config in zip(action_cfg.modality_keys, action_cfg.action_configs):
            if config.rep != "relative":
                continue
            if "relative_action" not in stats_for_tag:
                raise ValueError(
                    f"Relative action statistics required for {embodiment_tag!r}."
                )
            action_dim = norm_params[embodiment_tag]["action"][key]["dim"]
            relative = stats_for_tag["relative_action"][key]
            norm_params[embodiment_tag]["action"][key] = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in relative.items()
                if k in ("min", "max", "mean", "std", "q01", "q99")
            }
            norm_params[embodiment_tag]["action"][key]["dim"] = action_dim
    return norm_params


def _has_local_tokenizer_files(path: Path) -> bool:
    """Whether ``path`` can satisfy ``AutoTokenizer.from_pretrained(path)``."""
    if not path.is_dir() or not (path / "tokenizer_config.json").exists():
        return False
    return (path / "tokenizer.json").exists() or (
        (path / "vocab.json").exists() and (path / "merges.txt").exists()
    )


class GR00TProcessor:
    """Checkpoint-agnostic GR00T-N1.7 state/action/vision processor.

    Mirrors Isaac-GR00T's ``Gr00tN1d7Processor`` contract but with a native
    (no ``transformers`` processor) Qwen3-VL image/token path. Construct
    programmatically, or via :meth:`from_pretrained` against a GR00T checkpoint
    directory. The primary API is :meth:`process_observation` (raw obs -> tensors)
    and :meth:`decode_action` (normalized action -> physical action).
    """

    def __init__(
        self,
        *,
        embodiment_tag: str,
        modality_configs: dict[str, dict[str, GR00TModalityConfig]] | None = None,
        statistics: dict[str, Any] | None = None,
        embodiment_id_mapping: dict[str, int] | None = None,
        max_state_dim: int = 132,
        max_action_dim: int = 132,
        max_action_horizon: int = 40,
        use_percentiles: bool = True,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
        use_mean_std: bool = False,
        exclude_state: bool = False,
        formalize_language: bool = True,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        model_type: str = "qwen",
        image_crop_size: list[int] | tuple[int, int] | None = (230, 230),
        image_target_size: list[int] | tuple[int, int] | None = (256, 256),
        shortest_image_edge: int | None = 256,
        crop_fraction: float | None = 0.95,
        use_albumentations: bool = True,
        transformers_loading_kwargs: dict[str, Any] | None = None,
        tokenizer: Any | None = None,
        vlm_processor: Any | None = None,
    ) -> None:
        self.embodiment_tag = resolve_embodiment_tag(embodiment_tag)
        self.modality_configs = modality_configs or {}
        self.statistics = deepcopy(statistics or {})
        self.embodiment_id_mapping = dict(DEFAULT_EMBODIMENT_ID_MAPPING)
        if embodiment_id_mapping is not None:
            self.embodiment_id_mapping.update(
                {str(k): int(v) for k, v in embodiment_id_mapping.items()}
            )
        self.max_state_dim = int(max_state_dim)
        self.max_action_dim = int(max_action_dim)
        self.max_action_horizon = int(max_action_horizon)
        self.use_percentiles = bool(use_percentiles)
        self.use_mean_std = bool(use_mean_std)
        self.clip_outliers = bool(clip_outliers)
        self.apply_sincos_state_encoding = bool(apply_sincos_state_encoding)
        self.use_relative_action = bool(use_relative_action)
        self.exclude_state = bool(exclude_state)
        self.formalize_language = bool(formalize_language)
        self.model_name = model_name
        self.model_type = model_type
        self.image_crop_size = tuple2(image_crop_size)
        self.image_target_size = tuple2(image_target_size)
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction
        self.use_albumentations = bool(use_albumentations)
        self.transformers_loading_kwargs = dict(transformers_loading_kwargs or {})
        self._tokenizer = tokenizer
        self.vlm_processor = vlm_processor
        if self.vlm_processor is not None and hasattr(self.vlm_processor, "tokenizer"):
            self.vlm_processor.tokenizer.padding_side = "left"
        self._image_params_cache: dict[str, Any] | None = None
        self.norm_params = _build_norm_params(
            self.statistics,
            self.modality_configs,
            use_percentiles=self.use_percentiles,
            use_relative_action=self.use_relative_action,
        )
        self._validate_embodiment()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        embodiment_tag: str,
        **overrides: Any,
    ) -> "GR00TProcessor":
        root = Path(pretrained_model_name_or_path)
        processor_root = (
            root / "processor"
            if (root / "processor").is_dir()
            and not (root / "processor_config.json").exists()
            else root
        )
        processor_config = load_json(processor_root / "processor_config.json")
        statistics = load_json(processor_root / "statistics.json")
        embodiment_id_path = processor_root / "embodiment_id.json"
        embodiment_id_mapping = (
            load_json(embodiment_id_path) if embodiment_id_path.exists() else None
        )

        kwargs = dict(processor_config["processor_kwargs"])
        local_model_dir = (
            processor_root if _has_local_tokenizer_files(processor_root) else root
        )
        if not _has_local_tokenizer_files(local_model_dir):
            local_model_dir = None
        kwargs.setdefault(
            "model_name",
            str(local_model_dir)
            if local_model_dir is not None
            else "nvidia/Cosmos-Reason2-2B",
        )
        kwargs.setdefault("model_type", "qwen")
        kwargs.setdefault("clip_outliers", True)
        kwargs["statistics"] = statistics
        kwargs["embodiment_id_mapping"] = embodiment_id_mapping
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        modality_configs = parse_modality_configs(kwargs.pop("modality_configs"))
        keep = {
            "max_state_dim",
            "max_action_dim",
            "max_action_horizon",
            "use_percentiles",
            "clip_outliers",
            "apply_sincos_state_encoding",
            "use_relative_action",
            "use_mean_std",
            "exclude_state",
            "formalize_language",
            "model_name",
            "model_type",
            "image_crop_size",
            "image_target_size",
            "shortest_image_edge",
            "crop_fraction",
            "use_albumentations",
            "transformers_loading_kwargs",
            "tokenizer",
            "vlm_processor",
        }
        init_kwargs = {k: kwargs[k] for k in keep if k in kwargs}
        return cls(
            embodiment_tag=embodiment_tag,
            modality_configs=modality_configs,
            statistics=statistics,
            embodiment_id_mapping=embodiment_id_mapping,
            **init_kwargs,
        )

    def process_observation(
        self, observation: GR00TObservation
    ) -> GR00TProcessedInputs:
        cfg = self.modality_config
        self._validate_observation(observation)
        state_keys = cfg["state"].modality_keys
        state_data = {key: observation.state[key] for key in state_keys}
        if self.exclude_state:
            normalized_states = torch.cat(
                [
                    torch.from_numpy(np.zeros_like(state_data[key]))
                    for key in state_keys
                ],
                dim=-1,
            )
        else:
            norm_state = self.apply_state(state_data)
            normalized_states = torch.cat(
                [torch.from_numpy(norm_state[key]) for key in state_keys], dim=-1
            )
        if normalized_states.shape[-1] > self.max_state_dim:
            raise ValueError(
                f"State dimension {normalized_states.shape[-1]} exceeds "
                f"max_state_dim {self.max_state_dim}."
            )
        padding_shape = (
            *normalized_states.shape[:-1],
            self.max_state_dim - normalized_states.shape[-1],
        )
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(padding_shape, dtype=normalized_states.dtype),
            ],
            dim=-1,
        )
        batch = normalized_states.shape[0]
        action_horizon = len(cfg["action"].delta_indices)
        if action_horizon > self.max_action_horizon:
            raise ValueError(
                f"Action horizon {action_horizon} exceeds "
                f"max_action_horizon {self.max_action_horizon}."
            )
        action_mask = torch.zeros((batch, self.max_action_horizon), dtype=torch.float32)
        action_mask[:, :action_horizon] = 1.0
        embodiment_id = torch.full(
            (batch,),
            self.embodiment_id,
            dtype=torch.int32,
        )
        vlm_tensors = self._process_vlm(observation)
        tensors = {
            "state": normalized_states.to(torch.get_default_dtype()),
            "embodiment_id": embodiment_id,
            "action_mask": action_mask,
        }
        tensors.update(vlm_tensors)
        return GR00TProcessedInputs(
            tensors=tensors,
            raw_state=deepcopy(state_data),
        )

    def decode_action(
        self,
        normalized_action: torch.Tensor | np.ndarray,
        *,
        raw_state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        if isinstance(normalized_action, torch.Tensor):
            action = normalized_action.detach().float().cpu().numpy()
        else:
            action = np.asarray(normalized_action)
        cfg = self.modality_config["action"]
        action_horizon = len(cfg.delta_indices)
        out_dict: dict[str, np.ndarray] = {}
        start_idx = 0
        for key in cfg.modality_keys:
            dim = int(self.norm_params[self.embodiment_tag]["action"][key]["dim"])
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + dim]
            start_idx += dim
        if action.shape[-1] < start_idx:
            raise ValueError(
                f"normalized_action last dimension {action.shape[-1]} is smaller "
                f"than required action dim {start_idx}."
            )
        state = None
        if raw_state is not None:
            state = {k.replace("state.", ""): v for k, v in raw_state.items()}
        decoded = self.unapply_action(out_dict, raw_state=state)
        return {f"action.{key}": value for key, value in decoded.items()}

    # -------------------------------------------- #

    def preprocess(
        self, observation: GR00TObservation | dict[str, Any]
    ) -> GR00TProcessedInputs:
        """Alias for :meth:`process_observation` (accepts a raw dict too)."""
        return self.process_observation(ensure_numpy_observation(observation))

    def postprocess(
        self,
        normalized_action: torch.Tensor | np.ndarray,
        *,
        raw_state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Alias for :meth:`decode_action`."""
        return self.decode_action(normalized_action, raw_state=raw_state)

    @property
    def modality_config(self) -> dict[str, GR00TModalityConfig]:
        return self.modality_configs[self.embodiment_tag]

    @property
    def embodiment_id(self) -> int:
        if self.embodiment_tag not in self.embodiment_id_mapping:
            raise ValueError(
                f"Embodiment tag {self.embodiment_tag!r} has no embodiment id. "
                f"Available tags include: {sorted(self.embodiment_id_mapping)[:8]}"
            )
        return self.embodiment_id_mapping[self.embodiment_tag]

    def apply_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalized_values: dict[str, np.ndarray] = {}
        state_cfg = self.modality_config["state"]
        sin_cos_keys = (
            set(state_cfg.sin_cos_embedding_keys or [])
            if self.apply_sincos_state_encoding
            else set()
        )
        mean_std_keys = set(state_cfg.mean_std_embedding_keys or [])
        for key in state_cfg.modality_keys:
            if key not in state:
                raise KeyError(
                    f"State key {key!r} missing for embodiment {self.embodiment_tag!r}."
                )
            values = state[key]
            if key in sin_cos_keys:
                normalized = apply_sin_cos_encoding(values)
            elif key in mean_std_keys:
                normalized = normalize_values_meanstd(
                    values, self.norm_params[self.embodiment_tag]["state"][key]
                )
            else:
                normalized = normalize_values_minmax(
                    values, self.norm_params[self.embodiment_tag]["state"][key]
                )
                if self.clip_outliers:
                    normalized = np.clip(normalized, -1.0, 1.0)
            normalized_values[key] = normalized.astype(np.float32, copy=False)
        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        *,
        raw_state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        action_cfg = self.modality_config["action"]
        mean_std_keys = set(action_cfg.mean_std_embedding_keys or [])
        unnormalized: dict[str, np.ndarray] = {}
        for key in action_cfg.modality_keys:
            if key not in action:
                raise KeyError(
                    f"Action key {key!r} missing for embodiment {self.embodiment_tag!r}."
                )
            params = self.norm_params[self.embodiment_tag]["action"][key]
            if key in mean_std_keys:
                value = unnormalize_values_meanstd(action[key], params)
            else:
                value = unnormalize_values_minmax(action[key], params)
            unnormalized[key] = value.astype(np.float32, copy=False)

        if action_cfg.action_configs is None:
            return unnormalized
        for key, config in zip(action_cfg.modality_keys, action_cfg.action_configs):
            if config.rep != "relative" or not self.use_relative_action:
                continue
            if raw_state is None:
                raise ValueError(
                    f"raw_state is required for relative action key {key!r}."
                )
            state_key = config.state_key or key
            if state_key not in raw_state:
                raise KeyError(
                    f"Reference state key {state_key!r} missing for relative action "
                    f"key {key!r}."
                )
            if config.type == "non_eef":
                unnormalized[key] = relative_non_eef_to_absolute(
                    unnormalized[key], raw_state[state_key]
                )
            elif config.type == "eef":
                unnormalized[key] = relative_eef_to_absolute(
                    unnormalized[key],
                    raw_state[state_key],
                    action_format=config.format,
                )
            else:
                raise ValueError(
                    f"Unsupported relative action type {config.type!r} for key {key!r}."
                )
        return unnormalized

    def _validate_embodiment(self) -> None:
        if self.modality_configs and self.embodiment_tag not in self.modality_configs:
            raise ValueError(
                f"Embodiment tag {self.embodiment_tag!r} is not present in "
                "processor modality_configs."
            )
        if self.statistics and self.embodiment_tag not in self.statistics:
            raise ValueError(
                f"Embodiment tag {self.embodiment_tag!r} is not present in statistics."
            )

    def _validate_observation(self, observation: GR00TObservation) -> None:
        cfg = self.modality_config
        for key in cfg["state"].modality_keys:
            if key not in observation.state:
                raise ValueError(f"observation.state requires key {key!r}.")
            arr = observation.state[key]
            if not isinstance(arr, np.ndarray) or arr.dtype != np.float32:
                raise TypeError(f"state {key!r} must be np.float32 ndarray.")
            if arr.ndim != 3:
                raise ValueError(f"state {key!r} must have shape (B,T,D).")
            if arr.shape[1] != len(cfg["state"].delta_indices):
                raise ValueError(
                    f"state {key!r} horizon mismatch: got {arr.shape[1]}, "
                    f"expected {len(cfg['state'].delta_indices)}."
                )
        for key in cfg["language"].modality_keys:
            if key not in observation.language:
                raise ValueError(f"observation.language requires key {key!r}.")
        for key in cfg["video"].modality_keys:
            if key not in observation.video:
                raise ValueError(f"observation.video requires key {key!r}.")
            arr = observation.video[key]
            if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8:
                raise TypeError(f"video {key!r} must be np.uint8 ndarray.")
            if arr.ndim != 5 or arr.shape[-1] != 3:
                raise ValueError(f"video {key!r} must have shape (B,T,H,W,3).")
            if arr.shape[1] != len(cfg["video"].delta_indices):
                raise ValueError(
                    f"video {key!r} horizon mismatch: got {arr.shape[1]}, "
                    f"expected {len(cfg['video'].delta_indices)}."
                )

    def _language(self, observation: GR00TObservation) -> list[str]:
        language_key = self.modality_config["language"].modality_keys[0]
        language = []
        for item in observation.language[language_key]:
            text = item[0]
            if self.formalize_language:
                text = re.sub(r"[^\w\s]", "", text.lower())
            language.append(text)
        return language

    def _process_vlm(self, observation: GR00TObservation) -> dict[str, torch.Tensor]:
        cfg = self.modality_config
        image_keys = cfg["video"].modality_keys
        images = [torch.from_numpy(observation.video[key]) for key in image_keys]
        stacked = torch.stack(images, dim=2)
        if stacked.ndim != 6:
            raise ValueError("stacked video must have shape (B,T,V,H,W,C).")
        batch, time, views, height, width, channels = stacked.shape
        images_flat = stacked.reshape(batch, time * views, height, width, channels)
        transformed_images = []
        for b in range(batch):
            transformed_frames = []
            for frame in images_flat[b]:
                transformed_frames.append(
                    eval_transform_image(
                        frame.numpy(),
                        shortest_image_edge=self.shortest_image_edge,
                        image_target_size=self.image_target_size,
                        image_crop_size=self.image_crop_size,
                        crop_fraction=self.crop_fraction,
                    )
                )
            transformed_images.append(np.stack(transformed_frames, axis=0))

        languages = self._language(observation)
        if self.vlm_processor is not None:
            return self._process_vlm_injected(transformed_images, languages)
        return self._process_vlm_native(transformed_images, languages)

    def _process_vlm_injected(
        self, transformed_images: list[np.ndarray], languages: list[str]
    ) -> dict[str, torch.Tensor]:
        """Path used by tests that inject a processor-like ``vlm_processor``."""
        processor = self.vlm_processor
        texts: list[str] = []
        all_images: list[Image.Image] = []
        for images_for_item, language in zip(transformed_images, languages):
            pil_images = [
                Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images_for_item
            ]
            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": img} for img in pil_images],
                        {"type": "text", "text": language},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            all_images.extend(pil_images)
        tokenized = processor(
            text=texts,
            images=all_images,
            return_tensors="pt",
            padding=True,
        )
        return {k: v for k, v in tokenized.items() if isinstance(v, torch.Tensor)}

    def _process_vlm_native(
        self, transformed_images: list[np.ndarray], languages: list[str]
    ) -> dict[str, torch.Tensor]:
        """Native Qwen3-VL preprocessing — no ``transformers`` processor.

        Replicates ``Qwen3VLProcessor.__call__``: the Qwen2-VL image processor
        (``smart_resize`` -> rescale/normalize -> patchify -> ``pixel_values`` +
        ``image_grid_thw``), the tokenizer's own chat template, and the per-image
        ``<|image_pad|>`` expansion by ``grid_thw.prod() // merge_size**2``.
        """
        tokenizer = self._native_tokenizer()
        params = self._image_params()
        merge_length = int(params["merge_size"]) ** 2
        image_token = "<|image_pad|>"

        pixel_chunks: list[torch.Tensor] = []
        grid_list: list[tuple[int, int, int]] = []
        texts: list[str] = []
        for images_for_item, language in zip(transformed_images, languages):
            for frame in images_for_item:
                pixel_values, grid_thw = qwen3vl_process_image(frame, params)
                pixel_chunks.append(pixel_values)
                grid_list.append(grid_thw)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image"} for _ in images_for_item],
                        {"type": "text", "text": language},
                    ],
                }
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            )

        # Expand each placeholder image token to one token per merged patch,
        # consuming grids in image order (mirrors Qwen3VLProcessor.__call__).
        grid_index = 0
        expanded: list[str] = []
        for text in texts:
            while image_token in text:
                num = int(np.prod(grid_list[grid_index])) // merge_length
                text = text.replace(image_token, "<|placeholder|>" * num, 1)
                grid_index += 1
            expanded.append(text.replace("<|placeholder|>", image_token))

        tokenized = tokenizer(expanded, return_tensors="pt", padding=True)
        outputs: dict[str, torch.Tensor] = {
            k: v for k, v in tokenized.items() if isinstance(v, torch.Tensor)
        }
        outputs["pixel_values"] = torch.cat(pixel_chunks, dim=0)
        outputs["image_grid_thw"] = torch.tensor(grid_list, dtype=torch.long)
        # mm_token_type_ids: 1 at image tokens, 0 elsewhere.
        input_ids = outputs.get("input_ids")
        if input_ids is not None:
            image_id = tokenizer.convert_tokens_to_ids(image_token)
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[input_ids == image_id] = 1
            outputs["mm_token_type_ids"] = mm_token_type_ids
        return outputs

    def _native_tokenizer(self) -> Any:
        if self._tokenizer is None:
            tokenizer = get_tokenizer(
                self.model_name, **self.transformers_loading_kwargs
            )
            tokenizer.padding_side = "left"
            self._tokenizer = tokenizer
        return self._tokenizer

    def _image_params(self) -> dict[str, Any]:
        if self._image_params_cache is not None:
            return self._image_params_cache
        config = load_preprocessor_config(
            self.model_name,
            local_files_only=bool(
                self.transformers_loading_kwargs.get("local_files_only", False)
            ),
        )
        size = config.get("size", {}) if config else {}
        params = {
            "patch_size": int(config.get("patch_size", 16)) if config else 16,
            "temporal_patch_size": int(config.get("temporal_patch_size", 2))
            if config
            else 2,
            "merge_size": int(config.get("merge_size", 2)) if config else 2,
            "image_mean": list(config.get("image_mean", [0.5, 0.5, 0.5]))
            if config
            else [0.5, 0.5, 0.5],
            "image_std": list(config.get("image_std", [0.5, 0.5, 0.5]))
            if config
            else [0.5, 0.5, 0.5],
            "min_pixels": int(size.get("shortest_edge", 4 * 28 * 28)),
            "max_pixels": int(size.get("longest_edge", 16777216)),
        }
        self._image_params_cache = params
        return params


__all__ = [
    "GR00TActionConfig",
    "GR00TModalityConfig",
    "GR00TObservation",
    "GR00TProcessedInputs",
    "GR00TProcessor",
    "ensure_numpy_observation",
    "parse_modality_configs",
]
