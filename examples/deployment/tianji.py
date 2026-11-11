"""Tianji (Marvin Pro) physical robot client entry point.

Architecture & Execution Flow:
    1. Reads TianjiClientConfig from dataclass defaults and bundled YAML (configs/tianji.yaml).
    2. Constructs `TianjiConfig` (specifying ROS 2 domain 24, topics, QoS, mode 3 impedance,
       gripper calibration, and H.264 mosaic decoding).
    3. Runs the standardized `phyai_robot.remote.run_remote` lifecycle:
       - Queries model spec from serve_pi05.py over ZMQ.
       - Assembles `CompositeRobot` and `VectorAdapter`.
       - Connects ROS 2 subscriptions and publishers.
       - Executes one warmup forward pass and discards output.
       - Invokes `TianjiController.activate()` to command controller services
         (set_ready -> set_mode(3) -> set_input(3) takeover).
       - Runs the 200 Hz control loop executing StopAndRefill trajectory chunks.
       - On termination (Ctrl-C or error), safely calls `set_input(0)` to disengage.

Usage:
    # Print merged configuration:
    python -m examples.deployment.tianji --print-config

    # Launch live execution:
    python -m examples.deployment.tianji schedule.execution_horizon=20
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from .client import RemoteConfig, ControlConfig, ScheduleConfig, main
from .robots.tianji import TianjiConfig, build

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "tianji.yaml"


@dataclass
class TianjiClientConfig:
    """Complete client configuration schema for Tianji (Marvin Pro) robot deployments.

    Attributes:
        max_steps: Maximum number of physical commands to dispatch before stopping (null = run indefinitely).
        robot: Native ROS 2 topics, controller parameters, gripper calibration, and camera settings.
        remote: ZMQ RPC connection parameters to the policy server.
        control: 200 Hz actuator send loop timing and sensor freshness age budgets.
        schedule: StopAndRefill trajectory chunk execution and action consumption pacing.
    """

    max_steps: int | None = (
        None  # Actual Robot.send_action calls; null runs until Ctrl-C.
    )
    robot: TianjiConfig = field(default_factory=TianjiConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


if __name__ == "__main__":
    main(TianjiClientConfig, build, default_config=DEFAULT_CONFIG)
