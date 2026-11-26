"""The standard robot-side deployment lifecycle against a resident model server.

Lifecycle Protocol & Invariants:
    Every robot client follows an identical deterministic 11-step execution sequence:
    1. Query Model Spec: `spec = policy.info()` retrieves server schemas and metadata
       (e.g. required camera image size, input state dimensionality).
    2. Zero-I/O Embodiment Construction: `embodiment = build(spec)` constructs the
       CompositeRobot and VectorAdapter without opening any ROS nodes or physical sockets.
    3. Handshake Schema Verification: Verifies that `adapter.request_schema` matches the
       server's declared `spec.inputs` contract before reserving the server or connecting.
    4. Register Reverse Cleanup: Registers `policy.close` and `robot.close` in an ExitStack.
       Guarantee: Hardware is stopped and closed BEFORE the model run reservation is freed.
    5. Reserve Run: `policy.begin_run()` reserves exclusive server access and resets RNG noise.
    6. Connect Hardware: `robot.connect()` establishes native ROS 2 or transport sessions.
    7. Initial Snapshot: `wait_for_observation()` polls until all sensors report fresh samples.
    8. Warmup Execution: `policy.warmup(request)` executes the first forward pass, and the
       returned action chunk is validated in full and discarded. (Surfaces compilation or
       dimension errors before physical motion is enabled).
    9. Re-verify Freshness: Because warmup may take several seconds, sensor freshness is
       re-audited before actuation.
    10. Actuator Enablement: `embodiment.activate()` enables controller user command routing
        (e.g., Tianji calling `set_ready` -> `set_mode` -> `set_input(3)`).
    11. Control Loop Execution: `deployment.run(max_steps=max_steps)` starts the real-time loop.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple
from contextlib import ExitStack
from collections.abc import Callable, Mapping

import numpy as np

from ..robot import Robot, wait_for_observation
from ..policy import PolicyAdapter
from ._wire import PolicySpec, schema_to_json
from .policy import RemotePolicy
from ..scheduling import Scheduler, InferenceContext, RealTimeChunking
from ..deployment import RobotDeployment, DeploymentOptions


class Embodiment(NamedTuple):
    """Container holding an assembled robot, its policy adapter, and an activation hook.

    Attributes:
        robot: Connected or unconnected Robot instance.
        adapter: PolicyAdapter converting physical Observations to model request dictionaries.
            Its `request_schema` must match the model server's `spec.inputs`.
        activate: Optional zero-argument callable executed after successful warmup and freshness
            verification, immediately before the control loop starts. Used to command controller
            services (e.g. enabling user joint multiplexing). Stopping the robot must undo this.
    """

    robot: Robot
    adapter: PolicyAdapter[Any, Any]
    activate: Callable[[], None] | None = None


def run_remote(
    policy: RemotePolicy,
    *,
    build: Callable[[PolicySpec], Embodiment],
    scheduler: Scheduler,
    options: DeploymentOptions,
    max_steps: int | None = None,
    report: Callable[[str], None] = lambda message: None,
) -> None:
    """Execute a complete robot deployment session against a remote policy server.

    Args:
        policy: Connected RemotePolicy instance.
        build: Factory callable accepting PolicySpec and returning an Embodiment.
            Must not perform physical I/O.
        scheduler: Action chunk scheduling strategy (e.g. StopAndRefill).
        options: Deployment options specifying control frequency and freshness bounds.
        max_steps: Optional maximum number of commands to dispatch before stopping.
        report: Callback for logging lifecycle milestones (e.g. print).

    Raises:
        ValueError: If robot adapter schemas do not match server input requirements.
        TimeoutError: If sensor snapshots or RPC calls exceed timeout thresholds.
    """
    with ExitStack() as resources:
        # ExitStack unwinds in reverse registration order:
        # 1. robot.close() halts physical actuators and closes hardware sessions.
        # 2. policy.close() ends the remote server run reservation.
        resources.callback(policy.close)

        # Step 1: Query model server specifications (no run reservation).
        spec = policy.info()
        if isinstance(scheduler, RealTimeChunking):
            rtc = spec.metadata.get("rtc")
            scheduler.configure_model(
                horizon=spec.metadata.get("horizon"),
                max_prefix_len=rtc.get("max_prefix_len")
                if isinstance(rtc, Mapping)
                else None,
            )
            prefix = spec.inputs.get("action_prefix")
            length = spec.inputs.get("prefix_length")
            actions = spec.outputs.get("actions")
            if (
                prefix is None
                or length is None
                or actions is None
                or len(actions.shape) != 2
                or actions.shape[0] != spec.metadata["horizon"]
                or prefix.shape != (rtc["max_prefix_len"], actions.shape[1])
                or np.dtype(prefix.dtype) != np.dtype("float32")
                or length.shape != (1,)
                or np.dtype(length.dtype) != np.dtype("int64")
            ):
                raise ValueError("RTC model prefix schemas do not match its capability")

        # Step 2: Build robot and adapter based on model image_size and horizons.
        embodiment = build(spec)
        robot, adapter = embodiment.robot, embodiment.adapter

        # Step 3: Verify physical contract parity between adapter output and model inputs.
        expected = schema_to_json(spec.inputs)
        actual = schema_to_json(getattr(adapter, "request_schema", {}))
        if actual != expected:
            raise ValueError(
                f"robot adapter produces {actual}, but the model expects {expected}"
            )

        # Register hardware cleanup before opening hardware connections.
        resources.callback(robot.close)

        deployment = RobotDeployment(
            robot=robot,
            policy=policy,
            adapter=adapter,
            scheduler=scheduler,
            options=options,
        )

        # Step 5 & 6: Reserve model run and connect robot transports.
        policy.begin_run()
        robot.connect()

        def fresh_observation():
            return wait_for_observation(
                robot,
                timeout_s=options.observation_timeout_s,
                max_age_s=options.max_sample_age_s,
            )

        # Step 7: Wait for all sensors to deliver complete, fresh data frames.
        observation = fresh_observation()

        # Step 8: Execute first warmup inference forward pass and discard output.
        started = time.monotonic()
        result = policy.warmup(adapter.to_request(observation, InferenceContext()))
        warmup = deployment.prepare_chunk(adapter.to_actions(result, observation))
        report(
            f"Warmup took {time.monotonic() - started:.3f}s; "
            f"discarded {len(warmup.actions)} targets"
        )

        # Step 9: Re-verify sensor freshness after warmup delay.
        fresh_observation()

        # Step 10: Enable controller actuator command routing.
        if embodiment.activate is not None:
            embodiment.activate()

        report("Robot commands enabled; Ctrl-C stops")

        # Step 11: Execute the real-time control loop.
        deployment.run(max_steps=max_steps)
