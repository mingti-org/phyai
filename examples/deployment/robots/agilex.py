"""AgileX CompositeRobot and a bounded RPC bridge to the system ROS interpreter.

RobotDeployment remains the only action scheduler. This backend snapshots the
worker's sensor cache or submits one complete dual-arm position command.
"""

from __future__ import annotations

import math
import os
import shlex
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from phyai_robot import CompositeRobot, FeatureSpec, ObservationNotReady, Sample

from .agilex_protocol import (
    SIDES,
    VIEWS,
    decode_array,
    receive_packet,
    send_packet,
)

ACTION_SCHEMA = {
    **{
        f"joint_position_{side}": FeatureSpec((6,), "float64", unit="rad")
        for side in SIDES
    },
    **{f"gripper_{side}": FeatureSpec((1,), "float64", unit="m") for side in SIDES},
}


@dataclass
class AgilexRobotConfig:
    ros_python: str = "/usr/bin/python3"
    ros_setup: str = "/opt/ros/humble/setup.bash"
    node_name: str = "phyai_agilex"
    domain_id: int = 0
    localhost_only: bool = False
    camera_height: int = 480
    camera_width: int = 640
    # Standard sensor_msgs are sufficient; custom driver overlays are not imported.
    image_topics: dict[str, str] = field(
        default_factory=lambda: {
            "front": "/camera_f/color/image_raw",
            "left": "/camera_l/color/image_raw",
            "right": "/camera_r/color/image_raw",
        }
    )
    state_topics: dict[str, str] = field(
        default_factory=lambda: {
            "left": "/puppet/joint_left",
            "right": "/puppet/joint_right",
        }
    )
    command_topics: dict[str, str] = field(
        default_factory=lambda: {
            "left": "/joint_left_states",
            "right": "/joint_right_states",
        }
    )
    startup_timeout_s: float = 15.0
    observation_timeout_s: float = 2.0
    io_timeout_s: float = 0.25

    def validate(self):
        if self.camera_height < 1 or self.camera_width < 1:
            raise ValueError("Camera dimensions must be positive")
        if not 0 <= self.domain_id <= 232:
            raise ValueError("ROS domain_id must be in [0, 232]")
        if not self.node_name.replace("_", "").isalnum() or self.node_name[0].isdigit():
            raise ValueError("Invalid ROS node_name")
        for attr, keys in (
            ("image_topics", VIEWS),
            ("state_topics", SIDES),
            ("command_topics", SIDES),
        ):
            topics = getattr(self, attr)
            if set(topics) != set(keys) or any(
                not value.startswith("/") for value in topics.values()
            ):
                raise ValueError(
                    f"robot.{attr} must map {keys} to absolute ROS topic names"
                )
            if len(set(topics.values())) != len(topics):
                raise ValueError(f"robot.{attr} cannot contain duplicate topics")
        if set(self.state_topics.values()) & set(self.command_topics.values()):
            raise ValueError("State topics and command topics must differ")
        for attr in ("startup_timeout_s", "observation_timeout_s", "io_timeout_s"):
            if not math.isfinite(getattr(self, attr)) or getattr(self, attr) <= 0:
                raise ValueError(f"robot.{attr} must be finite and positive")
        if not Path(self.ros_setup).expanduser().is_file():
            raise ValueError("robot.ros_setup does not exist")
        if not os.access(Path(self.ros_python).expanduser(), os.X_OK):
            raise ValueError("robot.ros_python must be an executable interpreter")


def observation_schema(config):
    return {
        **{
            f"{view}_camera": FeatureSpec(
                (config.camera_height, config.camera_width, 3), "uint8"
            )
            for view in VIEWS
        },
        **ACTION_SCHEMA,
    }


def state_from_observation(observation):
    return np.concatenate(
        [
            observation.samples[key].value
            for side in SIDES
            for key in (f"joint_position_{side}", f"gripper_{side}")
        ]
    ).astype(np.float32)


def action_from_target(target):
    target = np.asarray(target, dtype=np.float64).reshape(14)
    return {
        key: target[part].copy()
        for index, side in enumerate(SIDES)
        for key, part in (
            (f"joint_position_{side}", slice(index * 7, index * 7 + 6)),
            (f"gripper_{side}", slice(index * 7 + 6, index * 7 + 7)),
        )
    }


def target_from_action(action):
    return np.concatenate(
        [
            action[key]
            for side in SIDES
            for key in (f"joint_position_{side}", f"gripper_{side}")
        ]
    )


class AgilexBackend:
    def __init__(self, config: AgilexRobotConfig):
        self.config = config
        self.observation_keys = frozenset(observation_schema(config))
        self.action_keys = frozenset(ACTION_SCHEMA)
        self.process = None
        self.channel = None
        self.last_status = {}
        self.closed = False
        self.stopped = False

    def connect(self):
        if self.closed:
            raise RuntimeError("Backend is closed")
        if self.process is not None:
            return
        self.config.validate()
        parent, child = socket.socketpair()
        self.channel = parent
        worker = Path(__file__).with_name("agilex_worker.py")
        command = (
            "source "
            + shlex.quote(str(Path(self.config.ros_setup).expanduser()))
            + " >&2; exec "
            + shlex.join(
                [
                    str(Path(self.config.ros_python).expanduser()),
                    "-u",
                    str(worker),
                    "--fd",
                    str(child.fileno()),
                ]
            )
        )
        env = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(name, None)
        env.update(
            ROS_DOMAIN_ID=str(self.config.domain_id),
            ROS_LOCALHOST_ONLY="1" if self.config.localhost_only else "0",
            PYTHONNOUSERSITE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        try:
            self.process = subprocess.Popen(
                ["bash", "-e", "-c", command],
                pass_fds=(child.fileno(),),
                env=env,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            child.close()
            self._rpc(
                "init",
                timeout=self.config.startup_timeout_s,
                config=asdict(self.config),
            )
        except BaseException:
            child.close()
            self._abort()
            raise

    def _rpc(self, command, *, timeout=None, **kwargs):
        if self.channel is None:
            raise RuntimeError("ROS worker is not connected")
        timeout = self.config.io_timeout_s if timeout is None else timeout
        try:
            self.channel.settimeout(timeout)
            send_packet(self.channel, {"command": command, **kwargs})
            result = receive_packet(self.channel, timeout)
        except (OSError, EOFError, ValueError):
            # Do not reuse a partially received stream or an uncertain write.
            self._abort()
            raise
        if not result.get("ok"):
            if result.get("kind") == "not_ready":
                raise ObservationNotReady(result["error"])
            raise RuntimeError(result.get("error", "ROS worker request failed"))
        return result

    def read(self):
        result = self._rpc("observe", timeout=self.config.observation_timeout_s)
        self.last_status = {
            key: value for key, value in result.items() if key != "samples"
        }
        samples = {}
        for key, value in result["samples"].items():
            array = decode_array(value["array"])
            stamp = value["received_at_ns"]
            if key in VIEWS:
                samples[f"{key}_camera"] = Sample(array, stamp)
            else:
                side = key.removeprefix("state_")
                values = array.astype(np.float64).reshape(7)
                samples[f"joint_position_{side}"] = Sample(values[:6].copy(), stamp)
                samples[f"gripper_{side}"] = Sample(values[6:].copy(), stamp)
        return samples

    def write(self, action):
        if self.stopped:
            raise RuntimeError("Robot is stopped")
        self._rpc("write", target=target_from_action(action).tolist())

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.channel is not None:
            self._rpc("stop", timeout=self.config.observation_timeout_s)

    def _abort(self):
        if self.channel is not None:
            self.channel.close()
            self.channel = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
            self.process = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.stop()
            if self.channel is not None:
                self._rpc("close", timeout=self.config.observation_timeout_s)
                if self.process is not None:
                    self.process.wait(timeout=3)
        finally:
            self._abort()


def make_agilex_robot(config):
    backend = AgilexBackend(config)
    robot = CompositeRobot(
        observation_schema=observation_schema(config),
        action_schema=ACTION_SCHEMA,
        backends=[backend],
    )
    return robot


def wait_for_observation(robot, *, timeout_s, max_age_s):
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            observation = robot.get_observation()
            now = time.monotonic_ns()
            stale = [
                key
                for key, sample in observation.samples.items()
                if not 0 <= (now - sample.received_at_ns) / 1e9 <= max_age_s
            ]
            if stale:
                raise ObservationNotReady(f"Stale sensors: {stale}")
            return observation
        except ObservationNotReady:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)
