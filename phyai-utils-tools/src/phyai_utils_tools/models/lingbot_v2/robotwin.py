"""RoboTwin boundary adapter for LingBot-VLA 2.0 inference.

The model processor intentionally consumes a robot-independent canonical
state/action layout.  RoboTwin exposes the two grippers interleaved with the
two six-joint arms, so the WebSocket deployment needs a small, explicit
boundary adapter.  The mappings in this module mirror
``configs/robot_configs/robotwin.yaml`` from the official LingBot repository.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROBOTWIN_STATE_KEY = "observation.state"
ROBOTWIN_ACTION_KEY = "action"
ROBOTWIN_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)

_STATE_ARM_FEATURE = "observation.state.arm.position"
_STATE_EFFECTOR_FEATURE = "observation.state.effector.position"
_ACTION_ARM_FEATURE = "action.arm.position"
_ACTION_EFFECTOR_FEATURE = "action.effector.position"


def _to_numpy(value: Any, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if dtype is not None and hasattr(value, "float"):
            value = value.float()
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _require_last_dim(value: np.ndarray, expected: int, *, name: str) -> None:
    if value.ndim == 0 or value.shape[-1] != expected:
        raise ValueError(
            f"{name} must have last dimension {expected}, got {value.shape}."
        )


def raw_state_to_canonical(state: Any) -> np.ndarray:
    """Convert RoboTwin ``[arm_l6, grip_l, arm_r6, grip_r]`` to 12+2."""

    raw = _to_numpy(state, dtype=np.float32)
    _require_last_dim(raw, 14, name=ROBOTWIN_STATE_KEY)
    return np.concatenate(
        (raw[..., :6], raw[..., 7:13], raw[..., 6:7], raw[..., 13:14]),
        axis=-1,
    )


def canonical_action_to_raw(action: Any) -> np.ndarray:
    """Convert canonical ``[arm_l6, arm_r6, grip_l, grip_r]`` to RoboTwin."""

    canonical = _to_numpy(action, dtype=np.float32)
    _require_last_dim(canonical, 14, name="canonical action")
    return np.concatenate(
        (
            canonical[..., :6],
            canonical[..., 12:13],
            canonical[..., 6:12],
            canonical[..., 13:14],
        ),
        axis=-1,
    )


def raw_action_to_canonical(action: Any) -> np.ndarray:
    """Inverse of :func:`canonical_action_to_raw`, useful for parity checks."""

    return raw_state_to_canonical(action)


def load_robotwin_stats(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_feature_stats(
    stats: Mapping[str, Any],
    feature_names: Sequence[str],
) -> dict[str, list[Any]]:
    feature_stats = [stats[name] for name in feature_names]
    combined = {}
    for stat_name in feature_stats[0]:
        arrays = [
            _to_numpy(values[stat_name], dtype=np.float32) for values in feature_stats
        ]
        if any(array.ndim == 0 for array in arrays):
            continue
        combined[stat_name] = np.concatenate(arrays, axis=-1).tolist()
    return combined


def canonical_robotwin_stats(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert official split stats into the processor's 14-D stats schema."""

    nested = payload.get("norm_stats", payload)
    return {
        "norm_stats": {
            ROBOTWIN_STATE_KEY: _canonical_feature_stats(
                nested,
                (_STATE_ARM_FEATURE, _STATE_EFFECTOR_FEATURE),
            ),
            ROBOTWIN_ACTION_KEY: _canonical_feature_stats(
                nested,
                (_ACTION_ARM_FEATURE, _ACTION_EFFECTOR_FEATURE),
            ),
        }
    }


def _single_prompt(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    prompt = value[0]
    return prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt


@dataclass(frozen=True)
class RoboTwinLingBotV2Adapter:
    """Map one RoboTwin WebSocket observation to/from canonical model fields."""

    use_length: int = 50

    def prepare_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        images = [_to_numpy(observation[key]) for key in ROBOTWIN_CAMERA_KEYS]
        state = raw_state_to_canonical(observation[ROBOTWIN_STATE_KEY])
        if state.ndim == 1:
            state = state[None, :]
        return {
            "images": images,
            "state": state,
            "task": [_single_prompt(observation["prompt"])],
        }

    def format_action_chunk(self, canonical_action: Any) -> dict[str, np.ndarray]:
        action = _to_numpy(canonical_action, dtype=np.float32)
        if action.ndim == 3:
            action = action[0]
        _require_last_dim(action, 14, name="canonical action")
        return {ROBOTWIN_ACTION_KEY: canonical_action_to_raw(action[: self.use_length])}


__all__ = [
    "ROBOTWIN_ACTION_KEY",
    "ROBOTWIN_CAMERA_KEYS",
    "ROBOTWIN_STATE_KEY",
    "RoboTwinLingBotV2Adapter",
    "canonical_action_to_raw",
    "canonical_robotwin_stats",
    "load_robotwin_stats",
    "raw_action_to_canonical",
    "raw_state_to_canonical",
]
