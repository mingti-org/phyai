"""Offline tests for the LingBot V2 RoboTwin boundary adapter."""

from __future__ import annotations

import numpy as np

from phyai_utils_tools.models.lingbot_v2 import (
    ROBOTWIN_CAMERA_KEYS,
    RoboTwinLingBotV2Adapter,
    canonical_action_to_raw,
    canonical_robotwin_stats,
    raw_action_to_canonical,
    raw_state_to_canonical,
)


def make_stats() -> dict:
    def feature(start: float, width: int) -> dict[str, list[float]]:
        values = np.arange(start, start + width, dtype=np.float32)
        return {
            "mean": values.tolist(),
            "std": (values + 1).tolist(),
            "q01": (values - 2).tolist(),
            "q99": (values + 2).tolist(),
            "q02": (values - 1).tolist(),
            "q98": (values + 1).tolist(),
        }

    return {
        "norm_stats": {
            "observation.state.arm.position": feature(0, 12),
            "observation.state.effector.position": feature(20, 2),
            "action.arm.position": feature(40, 12),
            "action.effector.position": feature(60, 2),
        }
    }


def test_state_and_action_joint_order_roundtrip():
    raw = np.arange(14, dtype=np.float32)
    expected = np.array(
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13],
        dtype=np.float32,
    )

    canonical = raw_state_to_canonical(raw)

    np.testing.assert_array_equal(canonical, expected)
    np.testing.assert_array_equal(canonical_action_to_raw(canonical), raw)
    np.testing.assert_array_equal(raw_action_to_canonical(raw), expected)


def test_official_split_stats_are_concatenated_in_model_order():
    stats = canonical_robotwin_stats(make_stats())["norm_stats"]

    assert len(stats["observation.state"]["q01"]) == 14
    assert len(stats["action"]["q99"]) == 14
    assert stats["observation.state"]["mean"] == [
        *np.arange(0, 12, dtype=np.float32).tolist(),
        20.0,
        21.0,
    ]
    assert stats["action"]["mean"] == [
        *np.arange(40, 52, dtype=np.float32).tolist(),
        60.0,
        61.0,
    ]


def test_observation_camera_prompt_and_state_mapping():
    observation = {
        key: np.full((8, 9, 3), index, dtype=np.uint8)
        for index, key in enumerate(ROBOTWIN_CAMERA_KEYS)
    }
    observation["observation.state"] = np.arange(14, dtype=np.float32)
    observation["prompt"] = ["pick up the block"]

    prepared = RoboTwinLingBotV2Adapter().prepare_observation(observation)

    assert [int(image[0, 0, 0]) for image in prepared["images"]] == [0, 1, 2]
    assert prepared["task"] == ["pick up the block"]
    assert prepared["state"].shape == (1, 14)
    assert prepared["state"][0, 12:].tolist() == [6.0, 13.0]


def test_action_chunk_is_sliced_and_restored_to_robotwin_order():
    canonical = np.broadcast_to(
        np.arange(14, dtype=np.float32),
        (1, 50, 14),
    ).copy()

    result = RoboTwinLingBotV2Adapter(use_length=7).format_action_chunk(canonical)

    assert result["action"].shape == (7, 14)
    assert result["action"][0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        12.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        13.0,
    ]
