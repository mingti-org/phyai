"""Bridge the current XPolicyLab protocol to a LingBot V2 legacy server.

The official LingBot V2 repository and the PHYAI compatibility server expose
the same direct websocket contract: send one flattened RoboTwin observation
and receive one action chunk.  Current RoboTwin releases use XPolicyLab's RPC
protocol instead.  This module adapts only that transport boundary; it does not
change preprocessing, normalization, model execution, or action values.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.lingbot_v2.robotwin.msgpack_numpy import Packer, unpackb

LOGGER = logging.getLogger("phyai.lingbot_v2.robotwin.bridge")

CAPTURE_FORMAT_VERSION = 1
DIAGNOSTIC_NOISE_KEY = "_lingbot_diagnostic_noise"
MODEL_ACTION_DIM = 55
MODEL_CHUNK_SIZE = 50

CAMERA_ALIASES = {
    "observation.images.cam_high": ("cam_head", "cam_high"),
    "observation.images.cam_left_wrist": ("cam_left_wrist",),
    "observation.images.cam_right_wrist": ("cam_right_wrist",),
}


def camera_color(vision: Mapping[str, Any], aliases: Sequence[str]) -> np.ndarray:
    """Find an RGB HWC image under the supplied camera aliases."""

    for name in aliases:
        if name not in vision:
            continue
        camera = vision[name]
        if isinstance(camera, Mapping):
            for key in ("color", "rgb", "image"):
                if key in camera:
                    camera = camera[key]
                    break
            else:
                raise KeyError(f"vision camera {name!r} has no color image.")
        image = np.asarray(camera)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"vision camera {name!r} must have HWC RGB shape, got {image.shape}."
            )
        return image
    raise KeyError(f"missing vision camera; tried {tuple(aliases)}.")


def state_field(state: Mapping[str, Any], key: str, *, expected_dim: int) -> np.ndarray:
    """Read and validate one flat floating-point state field."""

    value = np.asarray(state[key], dtype=np.float32).reshape(-1)
    if value.shape != (expected_dim,):
        raise ValueError(
            f"state field {key!r} must have shape ({expected_dim},), got {value.shape}."
        )
    return value


def observation_prompt(observation: Mapping[str, Any]) -> str:
    """Extract the non-empty instruction used by the legacy policy protocol."""

    instruction = observation.get("instruction")
    if not instruction:
        instructions = observation.get("instructions")
        if isinstance(instructions, Sequence) and not isinstance(
            instructions, (str, bytes, bytearray)
        ):
            instruction = instructions[0] if instructions else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("XPolicyLab observation must contain a non-empty instruction.")
    return instruction


def xpolicylab_to_legacy_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one current RoboTwin observation to the official V2 boundary."""

    vision = observation["vision"]
    state = observation["state"]
    result: dict[str, Any] = {
        output_key: camera_color(vision, aliases)
        for output_key, aliases in CAMERA_ALIASES.items()
    }
    result["observation.state"] = np.concatenate(
        (
            state_field(state, "left_arm_joint_state", expected_dim=6),
            state_field(state, "left_ee_joint_state", expected_dim=1),
            state_field(state, "right_arm_joint_state", expected_dim=6),
            state_field(state, "right_ee_joint_state", expected_dim=1),
        )
    )
    prompt = observation_prompt(observation)
    # Official FeatureTransform consumes a scalar task, while the PHYAI
    # compatibility boundary consumes a one-item prompt batch.
    result["task"] = prompt
    result["prompt"] = [prompt]
    return result


def save_legacy_observation(
    path: Path,
    observation: Mapping[str, Any],
) -> None:
    """Save one direct-server request without pickle or lossy conversions."""

    prompt = observation.get("task")
    if not isinstance(prompt, str):
        prompts = observation.get("prompt")
        if isinstance(prompts, Sequence) and not isinstance(
            prompts, (str, bytes, bytearray)
        ):
            prompt = prompts[0] if len(prompts) == 1 else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("legacy observation must contain one non-empty prompt.")
    payload = {
        "format_version": np.asarray(CAPTURE_FORMAT_VERSION, dtype=np.int64),
        "cam_high": np.asarray(observation["observation.images.cam_high"]),
        "cam_left_wrist": np.asarray(observation["observation.images.cam_left_wrist"]),
        "cam_right_wrist": np.asarray(
            observation["observation.images.cam_right_wrist"]
        ),
        "state": np.asarray(observation["observation.state"], dtype=np.float32),
        "prompt": np.asarray(prompt),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


class DirectWebSocketPolicyClient:
    """Synchronous client for the official LingBot direct websocket server."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        startup_timeout: float = 1200.0,
        retry_interval: float = 2.0,
    ) -> None:
        from websockets.sync.client import connect

        uri = f"ws://{host}:{port}"
        deadline = time.monotonic() + startup_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                self.connection = connect(
                    uri,
                    compression=None,
                    max_size=None,
                    ping_interval=None,
                    ping_timeout=None,
                    proxy=None,
                )
                metadata = self.connection.recv()
                if isinstance(metadata, str):
                    raise TypeError(
                        f"legacy policy server returned text metadata: {metadata}"
                    )
                self.metadata = unpackb(metadata)
                self.packer = Packer()
                LOGGER.info("connected to legacy policy server at %s", uri)
                return
            except (OSError, RuntimeError) as error:
                last_error = error
                time.sleep(retry_interval)
        raise TimeoutError(
            f"legacy policy server did not become ready at {uri} within "
            f"{startup_timeout:g}s: {last_error}"
        )

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send one MessagePack observation and decode the policy response."""

        self.connection.send(self.packer.pack(dict(observation)))
        response = self.connection.recv()
        if isinstance(response, str):
            raise TypeError(f"legacy policy server returned text:\n{response}")
        result = unpackb(response)
        return result

    def reset(self) -> None:
        """Reset the remote legacy policy session."""

        self.infer({"reset": True, "robo_name": "robotwin"})

    def close(self) -> None:
        """Close the direct policy WebSocket connection."""

        self.connection.close()


class LingBotV2LegacyBridge:
    """Expose an official-style LingBot V2 server as an XPolicyLab model."""

    def __init__(
        self,
        client: Any,
        *,
        capture_first_request: Path | None = None,
        paired_noise: bool = False,
        noise_seed: int = 0,
    ) -> None:
        self.client = client
        self.capture_first_request = capture_first_request
        self.paired_noise = paired_noise
        self.noise_seed = noise_seed
        self.capture_written = False
        self.observation: Mapping[str, Any] | None = None
        self.case_task = ""
        self.case_seed = 0
        self.request_index = 0

    def prepare_case(self, case_meta: Mapping[str, Any] | None = None) -> None:
        """Clear per-case state and record task metadata for diagnostics."""

        self.observation = None
        self.request_index = 0
        if case_meta:
            self.case_task = str(case_meta.get("task_name", ""))
            self.case_seed = int(case_meta.get("seed", 0))
            LOGGER.info(
                "prepare case task=%s seed=%s instruction=%r",
                case_meta.get("task_name"),
                case_meta.get("seed"),
                case_meta.get("instruction"),
            )

    def reset(self) -> None:
        """Reset both bridge bookkeeping and the remote policy session."""

        self.observation = None
        self.request_index = 0
        self.client.reset()

    def update_obs(self, observation: Mapping[str, Any]) -> None:
        """Store the latest XPolicyLab observation for ``get_action``."""

        self.observation = observation

    def get_action(self) -> np.ndarray:
        """Convert the observation, request a chunk, and return raw actions."""

        if self.observation is None:
            raise RuntimeError("update_obs must be called before get_action.")
        legacy_observation = xpolicylab_to_legacy_observation(self.observation)
        if self.paired_noise:
            task_hash = zlib.crc32(self.case_task.encode("utf-8"))
            generator = np.random.default_rng(
                np.random.SeedSequence(
                    (self.noise_seed, self.case_seed, self.request_index, task_hash)
                )
            )
            legacy_observation[DIAGNOSTIC_NOISE_KEY] = generator.standard_normal(
                (MODEL_CHUNK_SIZE, MODEL_ACTION_DIM), dtype=np.float32
            )
            self.request_index += 1
        if self.capture_first_request is not None and not self.capture_written:
            save_legacy_observation(self.capture_first_request, legacy_observation)
            self.capture_written = True
            LOGGER.info(
                "captured first direct-server request at %s",
                self.capture_first_request,
            )
        response = self.client.infer(legacy_observation)
        action = np.asarray(response["action"], dtype=np.float32)
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 2 or action.shape[-1] != 14:
            raise ValueError(
                "legacy policy action must have shape (T, 14) or (1, T, 14), "
                f"got {action.shape}."
            )
        return action

    def trial_end(self, result: Mapping[str, Any] | None = None) -> None:
        """Discard per-trial state after the evaluator records its result."""

        if result:
            LOGGER.info(
                "trial end task=%s seed=%s success=%s",
                result.get("task_name"),
                result.get("seed"),
                result.get("success"),
            )
        self.observation = None
        self.request_index = 0

    def on_trial_end(self, result: Mapping[str, Any] | None = None) -> None:
        """Compatibility callback forwarding to :meth:`trial_end`."""

        self.trial_end(result)

    def close(self) -> None:
        """Close the bridge's underlying direct policy client."""

        self.client.close()


def add_xpolicylab_paths(robotwin_root: Path) -> None:
    """Add RoboTwin and its XPolicyLab checkout to the import search path."""

    xpolicylab_root = robotwin_root / "XPolicyLab"
    for path in (robotwin_root, xpolicylab_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def parse_args() -> argparse.Namespace:
    """Parse command-line settings for the transport-only bridge."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=9330)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--capture-first-request", type=Path)
    parser.add_argument(
        "--execution-mode",
        choices=("chunk",),
        default="chunk",
        help="action chunks are executed exactly as returned by the Official CLI",
    )
    parser.add_argument(
        "--paired-noise",
        action="store_true",
        help="inject identical per-case diffusion noise for A/B diagnostics",
    )
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Start an XPolicyLab-compatible bridge for one direct policy server."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    robotwin_root = args.robotwin_root.resolve()
    add_xpolicylab_paths(robotwin_root)

    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    client = DirectWebSocketPolicyClient(
        host=args.backend_host,
        port=args.backend_port,
        startup_timeout=args.startup_timeout,
    )
    model = LingBotV2LegacyBridge(
        client,
        capture_first_request=args.capture_first_request,
        paired_noise=args.paired_noise,
        noise_seed=args.noise_seed,
    )
    server = PolicyServer(
        model=model,
        config=PolicyServerConfig(host=args.host, port=args.port),
    )
    try:
        asyncio.run(server.serve_forever())
    finally:
        model.close()


if __name__ == "__main__":
    main()
