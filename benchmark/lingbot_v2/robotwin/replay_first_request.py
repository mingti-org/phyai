"""Replay one captured RoboTwin request against Official and PHYAI."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.lingbot_v2.robotwin.legacy_policy_bridge import (
    CAPTURE_FORMAT_VERSION,
)
from benchmark.lingbot_v2.robotwin.msgpack_numpy import Packer, unpackb

DIAGNOSTIC_NOISE_KEY = "_lingbot_diagnostic_noise"


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_captured_request(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        version = int(payload["format_version"])
        if version != CAPTURE_FORMAT_VERSION:
            raise ValueError(
                f"capture format {version} is unsupported; "
                f"expected {CAPTURE_FORMAT_VERSION}."
            )
        prompt = str(payload["prompt"].item())
        result = {
            "observation.images.cam_high": np.array(payload["cam_high"], copy=True),
            "observation.images.cam_left_wrist": np.array(
                payload["cam_left_wrist"], copy=True
            ),
            "observation.images.cam_right_wrist": np.array(
                payload["cam_right_wrist"], copy=True
            ),
            "observation.state": np.array(payload["state"], copy=True),
            "task": prompt,
            "prompt": [prompt],
        }
    for key in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        image = result[key]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"captured {key} must be HWC RGB, got {image.shape}.")
    if result["observation.state"].shape != (14,):
        raise ValueError(
            "captured observation.state must have shape (14,), got "
            f"{result['observation.state'].shape}."
        )
    return result


def action_array(response: Mapping[str, Any]) -> np.ndarray:
    if "action" not in response:
        raise KeyError("policy response is missing 'action'.")
    action = np.asarray(response["action"], dtype=np.float32)
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 2 or action.shape[-1] != 14:
        raise ValueError(f"policy action must be (T,14), got {action.shape}.")
    return action


def array_metrics(reference: Any, candidate: Any) -> dict[str, Any]:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"metric shapes differ: {left.shape} != {right.shape}.")
    difference = right - left
    left_norm = float(np.linalg.norm(left.reshape(-1)))
    right_norm = float(np.linalg.norm(right.reshape(-1)))
    denominator = left_norm * right_norm
    cosine = (
        float(np.dot(left.reshape(-1), right.reshape(-1)) / denominator)
        if denominator > 0.0
        else float(left_norm == right_norm)
    )
    return {
        "exact": bool(np.array_equal(left, right)),
        "cosine": cosine,
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(
            np.linalg.norm(difference.reshape(-1)) / max(left_norm, 1e-30)
        ),
    }


def receive_mapping(connection: Any) -> Mapping[str, Any]:
    message = connection.recv()
    if isinstance(message, str):
        raise RuntimeError(f"policy server returned text:\n{message}")
    value = unpackb(message)
    if not isinstance(value, Mapping):
        raise TypeError(
            f"policy response must be a mapping, got {type(value).__name__}."
        )
    return value


def run_backend(
    *,
    name: str,
    url: str,
    observation: Mapping[str, Any],
    noise: np.ndarray,
    repeat: int,
) -> tuple[list[np.ndarray], Mapping[str, Any]]:
    try:
        from websockets.sync.client import connect
    except ImportError as error:
        raise RuntimeError(
            "websockets with the sync client API is required."
        ) from error

    actions: list[np.ndarray] = []
    with connect(
        url,
        compression=None,
        max_size=None,
        ping_interval=None,
        ping_timeout=None,
        proxy=None,
        open_timeout=30,
    ) as connection:
        metadata = receive_mapping(connection)
        packer = Packer()
        connection.send(packer.pack({"reset": True, "robo_name": "robotwin"}))
        receive_mapping(connection)
        for index in range(repeat):
            request = dict(observation)
            request[DIAGNOSTIC_NOISE_KEY] = np.array(noise, copy=True)
            connection.send(packer.pack(request))
            response = receive_mapping(connection)
            action = action_array(response)
            actions.append(action)
            print(
                f"{name} repeat {index + 1}/{repeat}: "
                f"shape={action.shape} sha256={array_sha256(action)}"
            )
    return actions, metadata


def input_hashes(observation: Mapping[str, Any], noise: np.ndarray) -> dict[str, str]:
    return {
        "cam_high": array_sha256(observation["observation.images.cam_high"]),
        "cam_left_wrist": array_sha256(
            observation["observation.images.cam_left_wrist"]
        ),
        "cam_right_wrist": array_sha256(
            observation["observation.images.cam_right_wrist"]
        ),
        "state": array_sha256(observation["observation.state"]),
        "noise": array_sha256(noise),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--official-url", default="ws://127.0.0.1:9330")
    parser.add_argument("--phyai-url", default="ws://127.0.0.1:9331")
    parser.add_argument("--noise-seed", type=int, default=20260819)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--model-action-dim", type=int, default=55)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.noise_seed < 0:
        raise ValueError("--noise-seed must be non-negative.")
    if args.chunk_size <= 0 or args.model_action_dim <= 0:
        raise ValueError("noise dimensions must be positive.")
    if args.repeat < 2:
        raise ValueError("--repeat must be at least 2 for self-repeat validation.")

    observation = load_captured_request(args.request)
    random_state = np.random.RandomState(args.noise_seed)
    noise = random_state.standard_normal(
        (args.chunk_size, args.model_action_dim)
    ).astype(np.float32)

    print("===== Fixed request =====")
    print(f"capture    : {args.request}")
    print(f"prompt     : {observation['task']}")
    print(f"noise seed : {args.noise_seed}")
    hashes = input_hashes(observation, noise)
    for key, value in hashes.items():
        print(f"{key:15}: {value}")

    print("===== Official =====")
    official_actions, official_metadata = run_backend(
        name="Official",
        url=args.official_url,
        observation=observation,
        noise=noise,
        repeat=args.repeat,
    )
    print("===== PHYAI =====")
    phyai_actions, phyai_metadata = run_backend(
        name="PHYAI",
        url=args.phyai_url,
        observation=observation,
        noise=noise,
        repeat=args.repeat,
    )

    official_repeat = array_metrics(official_actions[0], official_actions[1])
    phyai_repeat = array_metrics(phyai_actions[0], phyai_actions[1])
    cross_backend = array_metrics(official_actions[0], phyai_actions[0])
    report = {
        "request": str(args.request),
        "noise_seed": args.noise_seed,
        "input_sha256": hashes,
        "official_metadata": dict(official_metadata),
        "phyai_metadata": dict(phyai_metadata),
        "official_repeat": official_repeat,
        "phyai_repeat": phyai_repeat,
        "official_vs_phyai": cross_backend,
        "official_action_sha256": [array_sha256(x) for x in official_actions],
        "phyai_action_sha256": [array_sha256(x) for x in phyai_actions],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "official-action.npy", official_actions[0])
    np.save(args.output / "phyai-action.npy", phyai_actions[0])
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("===== Result =====")
    print(f"Official repeat : {official_repeat}")
    print(f"PHYAI repeat    : {phyai_repeat}")
    print(f"Official/PHYAI  : {cross_backend}")
    print(f"Report          : {args.output / 'report.json'}")
    if not official_repeat["exact"] or not phyai_repeat["exact"]:
        raise RuntimeError("a backend was not exact across fixed-noise repeats.")


if __name__ == "__main__":
    main()
