"""Compare one fixed request through direct and external bridge paths."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.lingbot_v2.robotwin.legacy_policy_bridge import (
    DIAGNOSTIC_NOISE_KEY,
    DirectWebSocketPolicyClient,
)
from benchmark.lingbot_v2.robotwin.replay_first_request import (
    action_array,
    array_metrics,
    array_sha256,
    load_captured_request,
)


def xpolicylab_observation(request: dict[str, Any]) -> dict[str, Any]:
    state = np.asarray(request["observation.state"], dtype=np.float32)
    prompt = str(request["task"])
    return {
        "data_format_version": "v1.0",
        "instruction": prompt,
        "instructions": [prompt],
        "env_idx": 0,
        "vision": {
            "cam_head": {"color": request["observation.images.cam_high"]},
            "cam_left_wrist": {"color": request["observation.images.cam_left_wrist"]},
            "cam_right_wrist": {"color": request["observation.images.cam_right_wrist"]},
        },
        "state": {
            "left_arm_joint_state": state[:6],
            "left_ee_joint_state": state[6:7],
            "right_arm_joint_state": state[7:13],
            "right_ee_joint_state": state[13:14],
        },
        "additional_info": {"frequency": 30},
    }


def paired_noise(*, noise_seed: int, case_seed: int, task_name: str) -> np.ndarray:
    task_hash = zlib.crc32(task_name.encode("utf-8"))
    generator = np.random.default_rng(
        np.random.SeedSequence((noise_seed, case_seed, 0, task_hash))
    )
    return generator.standard_normal((50, 55), dtype=np.float32)


def direct_actions(
    request: dict[str, Any], noise: np.ndarray, host: str, port: int, repeat: int
) -> list[np.ndarray]:
    client = DirectWebSocketPolicyClient(host=host, port=port)
    actions = []
    try:
        for _ in range(repeat):
            client.reset()
            payload = dict(request)
            payload[DIAGNOSTIC_NOISE_KEY] = np.array(noise, copy=True)
            actions.append(action_array(client.infer(payload)))
    finally:
        client.close()
    return actions


def bridge_actions(
    observation: dict[str, Any], url: str, task_name: str, case_seed: int, repeat: int
) -> list[np.ndarray]:
    from client_server.ws import WsModelClient

    client = WsModelClient(
        url=url,
        evaluation_id="direct-bridge-equivalence",
        trial_id="fixed-request",
        action_case_id="fixed-request",
    )
    actions = []
    try:
        for _ in range(repeat):
            client.call(
                func_name="prepare_case",
                obs={
                    "task_name": task_name,
                    "seed": case_seed,
                    "instruction": observation["instruction"],
                    "action_type": "joint",
                },
            )
            client.call(func_name="reset")
            client.call(func_name="update_obs", obs=observation)
            actions.append(
                np.asarray(client.call(func_name="get_action"), dtype=np.float32)
            )
    finally:
        client.close()
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--direct-host", default="127.0.0.1")
    parser.add_argument("--direct-port", type=int, default=9330)
    parser.add_argument("--bridge-url", default="ws://127.0.0.1:18087")
    parser.add_argument("--task-name", default="hanging_mug")
    parser.add_argument("--case-seed", type=int, default=100000)
    parser.add_argument("--noise-seed", type=int, default=20260825)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = load_captured_request(args.request)
    observation = xpolicylab_observation(request)
    noise = paired_noise(
        noise_seed=args.noise_seed,
        case_seed=args.case_seed,
        task_name=args.task_name,
    )
    direct = direct_actions(
        request, noise, args.direct_host, args.direct_port, args.repeat
    )
    bridge = bridge_actions(
        observation, args.bridge_url, args.task_name, args.case_seed, args.repeat
    )

    report = {
        "request": str(args.request),
        "noise_sha256": array_sha256(noise),
        "direct_action_sha256": [array_sha256(value) for value in direct],
        "bridge_action_sha256": [array_sha256(value) for value in bridge],
        "direct_repeat": array_metrics(direct[0], direct[1]),
        "bridge_repeat": array_metrics(bridge[0], bridge[1]),
        "direct_vs_bridge": array_metrics(direct[0], bridge[0]),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.save(args.output / "direct-action.npy", direct[0])
    np.save(args.output / "bridge-action.npy", bridge[0])
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
