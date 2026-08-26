"""Check RoboTwin mapping and normalization against the official formulas."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from phyai_utils_tools.models.lingbot_v2 import (
    canonical_action_to_raw,
    canonical_robotwin_stats,
    load_robotwin_stats,
    raw_state_to_canonical,
)

EPS = 1e-6


def normalize(value: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    return (value - low) / (high - low + EPS) * 2.0 - 1.0


def unnormalize(value: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    return (value + 1.0) / 2.0 * (high - low + EPS) + low


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2e-6)
    args = parser.parse_args()

    payload = load_robotwin_stats(args.stats_json)
    split = payload["norm_stats"]
    combined = canonical_robotwin_stats(payload)["norm_stats"]

    raw_state = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
    canonical_state = raw_state_to_canonical(raw_state)
    official_state = np.concatenate(
        (
            normalize(
                canonical_state[:12],
                split["observation.state.arm.position"],
            ),
            normalize(
                canonical_state[12:],
                split["observation.state.effector.position"],
            ),
        )
    )
    phyai_state = normalize(canonical_state, combined["observation.state"])

    normalized_action = np.linspace(-1.25, 1.25, 50 * 14, dtype=np.float32).reshape(
        50, 14
    )
    official_canonical_action = np.concatenate(
        (
            unnormalize(
                normalized_action[:, :12],
                split["action.arm.position"],
            ),
            unnormalize(
                normalized_action[:, 12:],
                split["action.effector.position"],
            ),
        ),
        axis=-1,
    )
    official_raw_action = canonical_action_to_raw(official_canonical_action)
    phyai_raw_action = canonical_action_to_raw(
        unnormalize(normalized_action, combined["action"])
    )

    state_error = max_abs(official_state, phyai_state)
    action_error = max_abs(official_raw_action, phyai_raw_action)
    print("RoboTwin adapter parity")
    print(f"stats              : {args.stats_json}")
    print(f"normalization      : bounds_99_woclip, eps={EPS}")
    print(f"state max abs      : {state_error:.9e}")
    print(f"action max abs     : {action_error:.9e}")
    print("camera order       : cam_high, cam_left_wrist, cam_right_wrist")
    print("action chunk       : 50 x 14")
    if state_error > args.atol or action_error > args.atol:
        raise RuntimeError(
            f"adapter parity failed with atol={args.atol}: "
            f"state={state_error}, action={action_error}"
        )
    print("Adapter parity passed")


if __name__ == "__main__":
    main()
