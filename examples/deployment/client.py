"""Shared robot-client configuration schemas and entry point coordinator.

Architecture & Scope:
    This module provides the generic client infrastructure shared by all physical robot
    entry points (e.g., Tianji, AgileX):
    1. Configuration Data Classes:
       - `RemoteConfig`: Connection parameters and timeout budgets for remote model RPC.
       - `ControlConfig`: Real-time control loop frequency (control_hz) and sensor freshness
         age thresholds (max_sample_age_s).
       - `ScheduleConfig`: Trajectory chunk pacing parameters (action_hz, execution_horizon).
    2. Lifecycle Execution (`main`):
       - Parses layered configuration (dataclass defaults < bundled YAML < custom YAML < CLI).
       - Instantiates `RemotePolicy` client and `Scheduler`.
       - Invokes `phyai_robot.remote.run_remote` to execute the verified 11-step lifecycle.
       - Catches KeyboardInterrupt (Ctrl-C) to exit cleanly without unhandled stack traces.
"""

from __future__ import annotations

import sys
from typing import Any, TypeVar
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable, Sequence

from phyai_robot import Scheduler, StopAndRefill, RealTimeChunking, DeploymentOptions
from phyai_robot.remote import PolicySpec, Embodiment, RemotePolicy, run_remote

from .configuration import parse_config


@dataclass
class RemoteConfig:
    """Configuration parameters for connecting to the remote PhyAI policy server over ZMQ.

    Attributes:
        endpoint: ZMQ socket endpoint URL (default: loopback 'tcp://127.0.0.1:5555').
        timeout_s: Synchronous timeout budget (in seconds) for standard RPC calls
            (info, begin, predict, end).
        warmup_timeout_s: Extended timeout budget (in seconds) for the initial warmup forward pass
            (allows first-run JIT kernel compilation and CUDA graph capture).
    """

    endpoint: str = "tcp://127.0.0.1:5555"
    timeout_s: float = 2.0
    warmup_timeout_s: float = 120.0

    def create(self) -> RemotePolicy:
        """Instantiate a RemotePolicy client using these configured parameters."""
        return RemotePolicy(
            self.endpoint,
            timeout_s=self.timeout_s,
            warmup_timeout_s=self.warmup_timeout_s,
        )


@dataclass
class ControlConfig:
    """Actuation frequency and sensor data freshness parameters for RobotDeployment.

    Attributes:
        control_hz: Physical actuator command dispatch frequency in Hz (e.g. 200.0 Hz = 5 ms tick).
            Sets the linear interpolation density between discrete model trajectory targets.
        max_sample_age_s: Maximum permitted reception age (in seconds) for any sensor sample.
            If any sensor frame in an observation is older than this threshold, inference pauses
            until fresh data arrives.
        observation_timeout_s: Maximum duration (in seconds) to wait for all sensors to become
            available and fresh before halting control and shutting down.
        shutdown_timeout_s: Maximum duration (in seconds) to wait for a background inference thread
            to finish after hardware stop has been issued.
        max_control_lateness_s: Maximum tolerated delay past a scheduled send deadline (e.g. 10 ms).
            Ticks delayed within this budget skip dropped frames; delays beyond this limit safely halt.
    """

    control_hz: float = 200.0
    max_sample_age_s: float = 0.5
    observation_timeout_s: float = 10.0
    shutdown_timeout_s: float = 2.0
    max_control_lateness_s: float = 0.01

    def to_options(self) -> DeploymentOptions:
        """Convert to phyai_robot.DeploymentOptions dataclass."""
        return DeploymentOptions(
            control_hz=self.control_hz,
            max_sample_age_s=self.max_sample_age_s,
            observation_timeout_s=self.observation_timeout_s,
            shutdown_timeout_s=self.shutdown_timeout_s,
            max_control_lateness_s=self.max_control_lateness_s,
        )


@dataclass
class ScheduleConfig:
    """Action chunk trajectory pacing and execution horizon configuration.

    Attributes:
        strategy: Scheduling algorithm identifier (default: "stop_and_refill").
        action_hz: Rate (in Hz) at which predicted trajectory targets are consumed (e.g. 25.0 Hz).
        execution_horizon: Number of leading targets executed per chunk before requesting a refill.
            In RTC, the interval between inference anchors is execution_horizon / action_hz.
        prefix_steps: RTC committed targets, including the snapshot-time target. Inference
            must finish in less than (prefix_steps - 1) / action_hz seconds.
    """

    strategy: str = "stop_and_refill"
    action_hz: float = 25.0
    execution_horizon: int | None = 20
    prefix_steps: int = 8

    def create(self) -> Scheduler:
        """Instantiate the configured trajectory Scheduler."""
        if self.strategy == "rtc":
            return RealTimeChunking(
                action_hz=self.action_hz,
                execution_horizon=self.execution_horizon,
                prefix_steps=self.prefix_steps,
            )
        if self.strategy != "stop_and_refill":
            raise ValueError(f"unknown schedule.strategy: {self.strategy!r}")
        return StopAndRefill(
            action_hz=self.action_hz, execution_horizon=self.execution_horizon
        )


ConfigT = TypeVar("ConfigT")


def validate_client(config: Any) -> None:
    """Perform cross-field semantic validation across remote, control, and robot settings.

    Args:
        config: Populated deployment configuration instance.

    Raises:
        ValueError: If settings violate physical or temporal constraints.
    """
    if config.max_steps is not None and config.max_steps < 1:
        raise ValueError("max_steps must be positive or null")
    if not config.remote.endpoint.strip():
        raise ValueError("remote.endpoint must not be empty")
    config.remote.create()
    config.control.to_options()
    config.schedule.create()
    config.robot.validate()


def main(
    schema: type[ConfigT],
    build: Callable[[Any, PolicySpec], Embodiment],
    *,
    default_config: Path,
    argv: Sequence[str] | None = None,
) -> None:
    """Universal CLI entry point for physical robot deployment clients.

    Parses command-line arguments, merges YAML defaults and overrides, executes
    the remote deployment session, and handles Ctrl-C termination gracefully.

    Args:
        schema: Dataclass schema type defining client configuration structure.
        build: Factory callable accepting (robot_config, policy_spec) and returning an Embodiment.
        default_config: Path to bundled default YAML file for this robot.
        argv: Optional command-line argument list (default: sys.argv[1:]).
    """
    config: Any = parse_config(
        schema, default_config=default_config, argv=argv, validate=validate_client
    )
    try:
        run_remote(
            config.remote.create(),
            build=lambda spec: build(config.robot, spec),
            scheduler=config.schedule.create(),
            options=config.control.to_options(),
            max_steps=config.max_steps,
            report=lambda message: print(message, flush=True),
        )
    except KeyboardInterrupt:
        print("Deployment interrupted; robot cleanup completed.", file=sys.stderr)
        raise SystemExit(130) from None
