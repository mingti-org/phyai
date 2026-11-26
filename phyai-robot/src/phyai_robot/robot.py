"""Whole-robot interface, composite session composition, and snapshot helpers.

This module provides:
1. `Robot`: Structural Protocol defining the abstract contract for robot hardware.
   All implementations provide schema inspection, connection lifecycle, observation
   acquisition, command submission, and safe stopping.
2. `CompositeRobot`: Implementation that composes multiple heterogeneous transport
   backends (e.g. ROS 2 joint publishers, ROS 2 video decoders, CAN grippers) into a
   single unified robot interface. Field routing is determined strictly by declared
   key ownership.
3. `take_snapshot` & `wait_for_observation`: Bounded polling utilities for acquiring
   complete, schema-validated, fresh sensor snapshots without blocking the control loop.
"""

from __future__ import annotations

import time
import threading
from types import MappingProxyType
from typing import Protocol
from contextlib import AbstractContextManager, nullcontext
from collections.abc import Mapping, Callable, Sequence

from .types import (
    Action,
    Sample,
    Observation,
    ActionSchema,
    ObservationSchema,
    ObservationNotReady,
)
from ._compat import BaseExceptionGroup
from .validation import (
    freeze_values,
    freeze_samples,
    check_freshness,
    validate_schema,
    validate_values,
    validate_observation,
)
from .backends.base import Backend


class Robot(Protocol):
    """Protocol defining the complete single-tick robot interface.

    Concurrency Contract:
        Methods on Robot are NOT required to be internally thread-safe.
        Callers (e.g., RobotDeployment) must serialize access using an external lock
        (such as `_io_lock`). Concurrent calls to get_observation, send_action, stop,
        or close without serialization are not permitted.
    """

    @property
    def observation_schema(self) -> ObservationSchema:
        """Return the immutable sensor schema declared before connecting."""
        ...

    @property
    def action_schema(self) -> ActionSchema:
        """Return the immutable single-step command schema accepted by the robot."""
        ...

    def connect(self) -> None:
        """Establish physical connections or start underlying communication sessions.

        Successful return indicates that transports are initialized. It does not
        guarantee that every sensor has delivered its initial data frame.
        """
        ...

    def get_observation(self) -> Observation:
        """Acquire a detached, complete sensor snapshot across all declared fields.

        Returns:
            Observation containing immutable Samples for every key in observation_schema.

        Raises:
            ObservationNotReady: If one or more sensors have not produced their first sample.
            RuntimeError: If called prior to connect() or after close().
        """
        ...

    def send_action(self, action: Action) -> None:
        """Submit one complete single-tick command mapping to physical actuators.

        Args:
            action: Complete mapping of action field names to single-tick command arrays.

        Raises:
            ValueError: If action keys do not match action_schema or values contain NaN/inf.
            RuntimeError: If called before connect(), after close(), or after stop().
        """
        ...

    def stop(self) -> None:
        """Command the robot to enter its designated hardware-safe state.

        Idempotent: Repeated invocations must be safe and harmless. Once stopped,
        subsequent calls to send_action are rejected until re-instantiated.
        """
        ...

    def close(self) -> None:
        """Terminate active sessions, stop devices, and release all system resources.

        Safe to call on partially initialized or already closed instances.
        Instances are strictly single-use after close() or unrecoverable error.
        """
        ...


class CompositeRobot:
    """Assembles a unified Robot by routing field subsets to reusable transport backends.

    Architectural Invariants:
    1. Zero I/O at Construction: Initializing CompositeRobot only validates schema
       consistency and ownership partitions without opening sockets, topics, or devices.
    2. Exact Covering: Each key declared in observation_schema and action_schema must
       be owned by exactly one Backend. Duplicate or missing keys are rejected at construction.
    3. Non-Transactional Submission: Full command validation occurs before the first
       backend write begins. However, writes across separate physical buses (e.g. ROS + CAN)
       are not atomic. If an intermediate backend write fails, all opened backends are
       stopped to transition the robot to a safe state.
    4. Comprehensive Exception Aggregation: Cleanup failures across multiple backends
       are accumulated using BaseExceptionGroup rather than silently dropping subsequent errors.
    """

    def __init__(
        self,
        *,
        observation_schema: ObservationSchema,  # Complete sensor layout mapping.
        action_schema: ActionSchema,  # Complete single-step command layout mapping.
        backends: Sequence[Backend],  # Ordered sequence of transport session backends.
        action_guard: Callable[[Action], None] | None = None,
        # Optional user hook to verify physical domain safety limits before writing to hardware.
    ) -> None:
        validate_schema(observation_schema)
        validate_schema(action_schema)
        self._observation_schema: ObservationSchema = MappingProxyType(
            dict(observation_schema)
        )
        self._action_schema: ActionSchema = MappingProxyType(dict(action_schema))
        self._backends: tuple[Backend, ...] = tuple(backends)
        self._action_guard = action_guard

        # Ensure no backend instance is supplied more than once.
        if len({id(backend) for backend in self._backends}) != len(self._backends):
            raise ValueError("a backend instance must not be supplied twice")

        # Verify that the provided backends form a partition over the schemas (exact covering).
        for attr, schema in (
            ("observation_keys", observation_schema),
            ("action_keys", action_schema),
        ):
            seen: set[str] = set()
            for backend in self._backends:
                keys: frozenset[str] = getattr(backend, attr)
                if seen & keys:
                    raise ValueError(f"duplicate {attr}: {seen & keys}")
                seen.update(keys)
            if seen != set(schema):
                raise ValueError(f"backend {attr} do not exactly cover the schema")

        self._opened: list[
            Backend
        ] = []  # Tracks backends currently connecting or connected.
        self._connected = (
            False  # True only after all backend sessions connect successfully.
        )
        self._stopped = False  # True once stop() has been executed.
        self._closed = False  # True once close() has terminated sessions.

    @property
    def observation_schema(self) -> ObservationSchema:
        """Return the immutable observation schema for the composed robot."""
        return self._observation_schema

    @property
    def action_schema(self) -> ActionSchema:
        """Return the immutable action command schema for the composed robot."""
        return self._action_schema

    @property
    def backends(self) -> tuple[Backend, ...]:
        """Access underlying backends in registration order for auxiliary device services."""
        return self._backends

    def connect(self) -> None:
        """Sequentially connect all composed backends; clean up on partial failure."""
        if self._closed or self._stopped:
            raise RuntimeError("robot is closed or stopped; construct a new instance")
        if self._connected:
            return
        try:
            for backend in self._backends:
                self._opened.append(backend)
                backend.connect()
            self._connected = True
        except BaseException as error:
            # Atomic rollback: If any backend fails to connect, close already opened ones.
            try:
                self.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - aggregate both errors
                raise BaseExceptionGroup(
                    "connection and cleanup failed", [error, cleanup_error]
                )
            raise

    def _require_connected(self) -> None:
        """Raise RuntimeError if robot is not currently connected or has closed."""
        if not self._connected or self._closed:
            raise RuntimeError("robot is not connected")

    def get_observation(self) -> Observation:
        """Collect and assemble sensor samples from all backends into an Observation.

        Queries cached samples from each backend in sequence. Verifies that all declared
        fields are present and non-empty. Array buffers are frozen to prevent external
        mutation while preserving their local receive timestamps.

        Returns:
            Observation with frozen samples.

        Raises:
            ObservationNotReady: If one or more declared sensor fields lack a valid sample.
            ValueError: If a backend returns an undeclared field.
        """
        self._require_connected()
        samples: dict[str, Sample] = {}
        for backend in self._backends:
            received: Mapping[str, Sample] = backend.read()
            if not set(received) <= backend.observation_keys:
                raise ValueError("backend returned undeclared observation fields")
            samples.update(received)

        # Confirm all required sensor fields have reported at least one sample.
        if set(samples) != set(self.observation_schema):
            raise ObservationNotReady(
                f"missing sensor samples: {set(self.observation_schema) - set(samples)}"
            )

        observation: Observation = Observation(freeze_samples(samples))
        validate_observation(observation, self.observation_schema)
        return observation

    def send_action(self, action: Action) -> None:
        """Validate and dispatch a complete single-tick command across backends.

        Validates all fields in action against action_schema, freezes array buffers,
        executes the optional action_guard safety check, and writes the respective
        action subsets to each backend in sequence.

        Args:
            action: Complete single-tick command mapping.

        Raises:
            RuntimeError: If robot is not connected or has been stopped.
            BaseExceptionGroup: If a write fails and subsequent stop also encounters an error.
        """
        self._require_connected()
        if self._stopped:
            raise RuntimeError("robot has been stopped")
        validate_values(action, self.action_schema)
        command: Action = freeze_values(action)

        # Optional user hook (e.g. check joint limits or gripper ranges) before dispatching to I/O.
        if self._action_guard is not None:
            self._action_guard(command)

        try:
            for backend in self._backends:
                if backend.action_keys:
                    backend.write({key: command[key] for key in backend.action_keys})
        except BaseException as error:
            # On write failure, immediately trigger emergency stop to safeguard physical hardware.
            try:
                self.stop()
            except BaseException as stop_error:  # noqa: BLE001
                raise BaseExceptionGroup("command and stop failed", [error, stop_error])
            raise

    def stop(self) -> None:
        """Request all opened backends to transition into their hardware-safe state."""
        if self._stopped:
            return
        self._stopped = True
        errors: list[BaseException] = []
        # Stop in reverse initialization order so upstream dependencies halt last.
        for backend in reversed(self._opened):
            try:
                backend.stop()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("one or more backends could not stop", errors)

    def close(self) -> None:
        """Safely halt the robot and release all physical and network resources."""
        if self._closed:
            return
        errors: list[BaseException] = []
        try:
            self.stop()
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
        for backend in reversed(self._opened):
            try:
                backend.close()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
        self._closed = True
        self._connected = False
        if errors:
            raise BaseExceptionGroup("robot cleanup failed", errors)


def take_snapshot(robot: Robot, *, max_age_s: float) -> Observation:
    """Acquire a single detached, schema-validated, fresh Observation snapshot.

    Enforces uniform detachment and freshness validation across both CompositeRobot
    and custom Robot implementations.

    Args:
        robot: Robot instance from which to retrieve observation.
        max_age_s: Maximum permitted sample age in seconds.

    Returns:
        Observation whose samples are guaranteed fresh (age <= max_age_s).

    Raises:
        ObservationNotReady: If sensors are missing or exceed max_age_s.
    """
    raw = robot.get_observation()
    observation = Observation(freeze_samples(raw.samples))
    validate_observation(observation, robot.observation_schema)
    check_freshness(observation, max_age_s)
    return observation


def wait_for_observation(
    robot: Robot,
    *,
    timeout_s: float,
    max_age_s: float,
    lock: AbstractContextManager[object] | None = None,
    stop_event: threading.Event | None = None,
    on_snapshot: Callable[[], None] | None = None,
) -> Observation | None:
    """Poll take_snapshot until all sensors report fresh samples, or timeout occurs.

    Designed for use by deployment loops and warmup routines:
    - Serializes each read attempt under `lock` (if provided) to avoid racing with control sends.
    - Gracefully exits returning None if `stop_event` is signaled during polling.
    - Retries only recoverable ObservationNotReady conditions (missing/stale frames).
    - Unrecoverable transport or schema errors raise immediately without retrying.

    Args:
        robot: Robot instance to query.
        timeout_s: Total wait budget in seconds before timing out.
        max_age_s: Maximum permitted sensor sample age in seconds.
        lock: Optional context manager (e.g. deployment._io_lock) acquired during read.
        stop_event: Optional threading.Event signaling early cancellation.
        on_snapshot: Optional bounded callback under the read lock after a fresh
            snapshot, used to capture its action timeline before another send.

    Returns:
        Fresh Observation snapshot, or None if stop_event was signaled.

    Raises:
        TimeoutError: If timeout_s expires before a complete, fresh observation is available.
    """
    deadline = time.monotonic() + timeout_s
    lock = nullcontext() if lock is None else lock
    while True:
        with lock:
            if stop_event is not None and stop_event.is_set():
                return None
            try:
                observation = take_snapshot(robot, max_age_s=max_age_s)
            except ObservationNotReady as error:
                reason = error
            else:
                if on_snapshot is not None:
                    on_snapshot()
                return observation
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for a complete, fresh observation: {reason}"
            ) from reason
        if stop_event is None:
            time.sleep(0.01)
        elif stop_event.wait(0.01):
            return None
