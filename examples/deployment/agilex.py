"""AgileX dual-Piper robot client entry point.

Architecture & Execution Flow:
    1. Reads AgilexClientConfig from dataclass defaults and bundled YAML (configs/agilex.yaml).
    2. Constructs `AgilexConfig` (specifying ROS 2 domain 0, Orbbec camera topics, Piper CAN
       JointState topics, and 14-DoF action schema).
    3. Runs the standardized `phyai_robot.remote.run_remote` lifecycle:
       - Queries model spec from serve_pi05.py over ZMQ.
       - Assembles `CompositeRobot` (3 camera backends + 1 joint backend) and `VectorAdapter`.
       - Connects ROS 2 subscriptions and publishers.
       - Executes one warmup forward pass and discards output.
       - Runs the 200 Hz control loop executing StopAndRefill trajectory chunks.
       - On termination (Ctrl-C or error), cleanly stops topic publishing without homing.

Usage:
    # Print merged configuration:
    python -m examples.deployment.agilex --print-config

    # Launch live execution:
    python -m examples.deployment.agilex schedule.execution_horizon=20
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import field, dataclass

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "examples.deployment"

from .client import RemoteConfig, ControlConfig, ScheduleConfig, main
from .robots.agilex import AgilexConfig, build

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "agilex.yaml"


@dataclass
class AgilexClientConfig:
    """Complete client configuration schema for AgileX dual-Piper robot deployments.

    Attributes:
        max_steps: Maximum number of physical commands to dispatch before stopping (null = run indefinitely).
        robot: Native ROS 2 topics, Piper arm JointState bindings, and Orbbec camera parameters.
        remote: ZMQ RPC connection parameters to the policy server.
        control: 200 Hz actuator send loop timing and sensor freshness age budgets.
        schedule: StopAndRefill trajectory chunk execution and action consumption pacing.
    """

    max_steps: int | None = (
        None  # Actual Robot.send_action calls; null runs until Ctrl-C.
    )
    robot: AgilexConfig = field(default_factory=AgilexConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)


if __name__ == "__main__":
    main(AgilexClientConfig, build, default_config=DEFAULT_CONFIG)
