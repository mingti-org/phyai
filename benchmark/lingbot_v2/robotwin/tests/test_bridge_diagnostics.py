from __future__ import annotations

import numpy as np

from benchmark.lingbot_v2.robotwin.legacy_policy_bridge import (
    LingBotV2LegacyBridge,
)


def observation() -> dict:
    return {
        "instruction": "lift the pot",
        "vision": {
            "cam_head": {"color": np.zeros((4, 4, 3), dtype=np.uint8)},
            "cam_left_wrist": {"color": np.zeros((4, 4, 3), dtype=np.uint8)},
            "cam_right_wrist": {"color": np.zeros((4, 4, 3), dtype=np.uint8)},
        },
        "state": {
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_joint_state": np.zeros(1, dtype=np.float32),
            "right_arm_joint_state": np.zeros(6, dtype=np.float32),
            "right_ee_joint_state": np.zeros(1, dtype=np.float32),
        },
    }


class Client:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def infer(self, request: dict) -> dict:
        self.requests.append(request)
        return {"action": np.zeros((50, 14), dtype=np.float32)}

    def reset(self) -> None:
        pass


def test_paired_noise_is_case_stable_and_request_indexed() -> None:
    clients = [Client(), Client()]
    bridges = [
        LingBotV2LegacyBridge(client, paired_noise=True, noise_seed=7)
        for client in clients
    ]
    case = {"task_name": "hanging_mug", "seed": 100002}

    for bridge in bridges:
        bridge.prepare_case(case)
        bridge.update_obs(observation())
        bridge.get_action()

    first = clients[0].requests[0]["_lingbot_diagnostic_noise"]
    second = clients[1].requests[0]["_lingbot_diagnostic_noise"]
    assert first.shape == (50, 55)
    np.testing.assert_array_equal(first, second)

    bridges[0].update_obs(observation())
    bridges[0].get_action()
    assert not np.array_equal(
        first,
        clients[0].requests[1]["_lingbot_diagnostic_noise"],
    )
