"""Robot deployment coordinator: background inference worker and real-time control loop.

Architecture:
    RobotDeployment coordinates two concurrent execution contexts:
    1. Background Producer Thread (_produce):
       - Monitors scheduler state (via `scheduler.wants_inference()`).
       - Acquires fresh observation snapshots (under `_io_lock`).
       - Translates snapshots via `adapter.to_request(obs, context)`.
       - Calls synchronous model prediction via `policy.predict(request)`.
         (Crucial: Prediction runs WITHOUT holding `_io_lock`, ensuring stalled GPU
         inference never blocks hardware emergency shutdown).
       - Validates predictions and submits them to `scheduler.add_chunk()`.
    2. Real-Time Actuator Control Thread (_execute):
       - Driven by a monotonic host clock at `options.control_hz` (e.g. 200 Hz).
       - Queries `scheduler.action_at(tick_ns)` to retrieve interpolated targets.
       - Enforces strict lateness bounds (`max_control_lateness_s`): skips dropped ticks
         rather than emitting dangerous high-frequency catch-up command bursts.
       - Dispatches commands to hardware via `robot.send_action(action)` (under `_io_lock`).

Synchronization & Shutdown:
    - An internal `threading.Condition` coordinates producer-consumer signaling.
    - `_io_lock` serializes all hardware operations on `Robot` (get_observation, send_action, stop, close).
    - Shutdown Sequence: When stopping, `robot.stop()` and `robot.close()` execute immediately
      under `_io_lock`, preempting actuator motion even if a background model forward pass
      is still running or blocked.
"""

from __future__ import annotations

import math
import time
import threading
from typing import Any, Generic, TypeVar
from _thread import LockType
from dataclasses import dataclass

from .robot import Robot, wait_for_observation
from .types import ActionChunk
from .policy import Policy, PolicyAdapter
from ._compat import BaseExceptionGroup
from .scheduling import Scheduler, InferenceContext
from .validation import freeze_values, validate_values


@dataclass(frozen=True)
class DeploymentOptions:
    """Configuration options governing control loop timing, timeouts, and freshness.

    Attributes:
        control_hz: Physical actuator command dispatch frequency in Hz (e.g., 200.0 Hz = 5 ms).
            Defines the interpolation density between model trajectory targets.
        max_sample_age_s: Maximum permitted local receive age in seconds for any sensor sample.
            If any sensor frame in an observation is older than this limit, inference pauses
            and waits for a fresh sample.
        observation_timeout_s: Maximum duration in seconds to wait for a complete, fresh observation
            before raising TimeoutError and shutting down hardware.
        shutdown_timeout_s: Maximum duration in seconds to wait for a background inference thread
            to terminate after hardware has been stopped and closed.
        max_control_lateness_s: Maximum tolerated delay past a scheduled control tick deadline.
            If host execution jitter or I/O contention exceeds this threshold (e.g. 10 ms),
            the control loop halts safely rather than dispatching delayed commands.
    """

    control_hz: float = (
        200.0  # Actuator send frequency in Hz during active trajectory execution.
    )
    max_sample_age_s: float = (
        0.5  # Maximum sensor sample age at snapshot time in seconds.
    )
    observation_timeout_s: float = (
        5.0  # Timeout for acquiring a complete fresh observation snapshot.
    )
    shutdown_timeout_s: float = (
        2.0  # Grace period for background worker thread exit on shutdown.
    )
    max_control_lateness_s: float = (
        0.01  # Maximum allowable delay past scheduled send deadline.
    )

    def __post_init__(self) -> None:
        """Validate that all options are positive, finite numbers."""
        for name in (
            "control_hz",
            "max_sample_age_s",
            "observation_timeout_s",
            "shutdown_timeout_s",
            "max_control_lateness_s",
        ):
            value: Any = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if 1e9 / self.control_hz < 1:
            raise ValueError("control period must be at least one nanosecond")


RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class RobotDeployment(Generic[RequestT, ResultT]):
    """Coordinates physical robot I/O, trajectory scheduling, and asynchronous policy inference.

    Single-Use Lifecycle:
        Instances of RobotDeployment are strictly single-use. Once `run()` completes or raises,
        hardware is closed and a new deployment instance must be constructed for subsequent runs.
    """

    def __init__(
        self,
        *,
        robot: Robot,
        policy: Policy[RequestT, ResultT],
        adapter: PolicyAdapter[RequestT, ResultT],
        scheduler: Scheduler,
        options: DeploymentOptions,
    ) -> None:
        """Initialize the deployment coordinator.

        Args:
            robot: Robot hardware or composite abstraction.
            policy: Model policy predictor (local engine or RemotePolicy over ZMQ).
            adapter: Bridge converting between physical observations and model requests/actions.
            scheduler: Trajectory scheduling strategy (e.g. StopAndRefill).
            options: Deployment timing parameters and freshness thresholds.
        """
        scheduler.bind(robot.action_schema)
        self.robot = robot
        self.policy = policy
        self.adapter = adapter
        self.scheduler = scheduler
        self.options = options

        self._condition = (
            threading.Condition()
        )  # Coordinates signaling between worker and control loop.
        self._io_lock: LockType = (
            threading.Lock()
        )  # Serializes all hardware operations on self.robot.
        self._stop_event = threading.Event()  # Signals cancellation across all threads.
        self._worker_error: BaseException | None = None
        self._started = False

    def stop(self) -> None:
        """Request immediate shutdown. Guarantees no new Robot I/O begins after returning.

        Sets the cancellation event, wakes threads blocked on condition variables,
        and waits for any already-executing Robot I/O call to complete.
        """
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        # Barrier: Wait for any currently executing get_observation or send_action to finish.
        with self._io_lock:
            pass

    def prepare_chunk(self, chunk: ActionChunk) -> ActionChunk:
        """Validate and freeze an entire predicted trajectory chunk.

        Enforces that every action in the chunk conforms to the robot's ActionSchema and
        contains only finite values. Freezes array buffers and checks scheduler capacity.

        Args:
            chunk: ActionChunk produced by adapter.to_actions().

        Returns:
            Validated ActionChunk containing immutable (frozen) command arrays.

        Raises:
            ValueError: If chunk is empty or any action violates schema / scheduler constraints.
        """
        if not isinstance(chunk, ActionChunk) or not chunk.actions:
            raise ValueError("adapter must return a nonempty ActionChunk")
        for action in chunk.actions:
            validate_values(action, self.robot.action_schema)
        prepared = ActionChunk(tuple(freeze_values(a) for a in chunk.actions))
        with self._condition:
            self.scheduler.check(prepared)
        return prepared

    def _produce(self) -> None:
        """Background worker loop: acquire observations, query model, and submit chunks."""
        try:
            while True:
                # Wait until the scheduler indicates it is ready for a new prediction.
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_event.is_set()
                        or self.scheduler.wants_inference()
                    )
                    if self._stop_event.is_set():
                        return

                context: InferenceContext | None = None

                def capture_context() -> None:
                    nonlocal context
                    # Keep snapshot and timeline capture ordered against sends.
                    with self._condition:
                        if not self._stop_event.is_set():
                            context = self.scheduler.begin_inference(
                                time.monotonic_ns()
                            )

                # Acquire a fresh sensor observation snapshot under the I/O lock.
                observation = wait_for_observation(
                    self.robot,
                    timeout_s=self.options.observation_timeout_s,
                    max_age_s=self.options.max_sample_age_s,
                    lock=self._io_lock,
                    stop_event=self._stop_event,
                    on_snapshot=capture_context,
                )
                if observation is None or context is None:
                    return

                # Prepare model request via adapter.
                request = self.adapter.to_request(observation, context)
                if self._stop_event.is_set():
                    return

                # Execute model inference forward pass WITHOUT holding _io_lock.
                # This ensures long or stalled inference never blocks hardware stop/close.
                result = self.policy.predict(request)
                if self._stop_event.is_set():
                    return

                # Convert result to physical actions and validate complete chunk.
                chunk = self.prepare_chunk(self.adapter.to_actions(result, observation))

                # Submit the new chunk to the scheduler under the condition lock.
                with self._condition:
                    if self._stop_event.is_set():
                        return
                    self.scheduler.add_chunk(chunk, context, time.monotonic_ns())
                    self._condition.notify_all()
        except BaseException as error:  # noqa: BLE001 - forward background exceptions to main thread
            with self._condition:
                self._worker_error = error
                self._condition.notify_all()
            self.stop()

    def _raise_worker_error(self) -> None:
        """Re-raise background worker exceptions in the main thread."""
        with self._condition:
            if self._worker_error is not None:
                raise self._worker_error

    def _execute(self, remaining_steps: int | None) -> int:
        """Execute control ticks at control_hz until the active chunk completes or lateness occurs.

        Enforces temporal determinism:
        - Ticks are scheduled strictly on a monotonic clock: deadline = start + tick * period.
        - If host jitter causes a delay within max_control_lateness_s, missed intermediate ticks
          are skipped and the latest due target is emitted, avoiding high-frequency command bursts.
        - If lateness exceeds max_control_lateness_s, a detailed TimeoutError is raised.

        Args:
            remaining_steps: Maximum sends permitted before returning (None = no limit).

        Returns:
            Total count of physical commands successfully dispatched during this execution block.
        """
        period_ns = 1e9 / self.options.control_hz
        max_lateness_ns = self.options.max_control_lateness_s * 1e9
        started_at_ns = time.monotonic_ns()
        tick = 0
        sent = 0
        previous_send_ns = 0

        while True:
            scheduled_tick = tick
            deadline_ns = started_at_ns + round(tick * period_ns)

            # Sleep until scheduled tick deadline, aborting early if stop was signaled.
            if self._stop_event.wait(max(0, (deadline_ns - time.monotonic_ns()) / 1e9)):
                return sent
            woke_at_ns = time.monotonic_ns()

            with self._io_lock:
                if self._stop_event.is_set():
                    return sent
                locked_at_ns = time.monotonic_ns()

                # Determine the latest due tick based on current monotonic clock time.
                tick = max(tick, math.floor((locked_at_ns - started_at_ns) / period_ns))

                with self._condition:
                    action = self.scheduler.action_at(
                        started_at_ns + round(tick * period_ns)
                    )
                    # Notify producer worker thread as soon as the scheduler needs inference.
                    if action is None or self.scheduler.wants_inference():
                        self._condition.notify_all()
                    if action is None:
                        # Chunk has finished executing; return to idle to wait for refill.
                        return sent

                now_ns = time.monotonic_ns()
                lateness_ns = now_ns - deadline_ns

                # Enforce control lateness limit to safeguard hardware against timing stalls.
                if lateness_ns > max_lateness_ns:
                    raise TimeoutError(
                        "control deadline exceeded: "
                        f"lateness_ms={lateness_ns / 1e6:.6f}, "
                        f"limit_ms={max_lateness_ns / 1e6:.6f}, "
                        f"scheduled_tick={scheduled_tick}, "
                        f"control_hz={self.options.control_hz:g}, "
                        f"wake_lateness_ms={(woke_at_ns - deadline_ns) / 1e6:.6f}, "
                        f"io_wait_ms={(locked_at_ns - woke_at_ns) / 1e6:.6f}, "
                        f"prepare_ms={(now_ns - locked_at_ns) / 1e6:.6f}, "
                        f"previous_send_ms={previous_send_ns / 1e6:.6f}"
                    )

                # Transmit the physical command to robot actuators.
                self.robot.send_action(action)
                previous_send_ns = time.monotonic_ns() - now_ns

            sent += 1
            if remaining_steps is not None and sent >= remaining_steps:
                return sent

            # Advance to the next scheduled tick on the original chunk clock.
            tick = max(tick + 1, math.floor((now_ns - started_at_ns) / period_ns) + 1)

    def run(self, *, max_steps: int | None = None) -> None:
        """Connect the robot, start background worker, and run the real-time control loop.

        Guarantees that hardware is safely stopped and closed upon exit or exception.

        Args:
            max_steps: Optional maximum number of commands to dispatch before stopping.
                None runs indefinitely until interrupted with Ctrl-C or stopped externally.

        Raises:
            RuntimeError: If called more than once on the same deployment instance.
            BaseExceptionGroup: If deployment runtime errors or cleanup failures occur.
        """
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        if self._started:
            raise RuntimeError("deployment instances are single-use")
        self._started = True

        worker: threading.Thread | None = None
        errors: list[BaseException] = []

        try:
            with self._io_lock:
                if not self._stop_event.is_set():
                    self.robot.connect()

            if not self._stop_event.is_set():
                worker = threading.Thread(
                    target=self._produce, name="phyai-policy", daemon=True
                )
                worker.start()

            sent = 0
            while not self._stop_event.is_set():
                # Wait until the scheduler becomes active with executable actions.
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_event.is_set() or self.scheduler.is_active()
                    )
                self._raise_worker_error()
                if self._stop_event.is_set():
                    break

                # Execute actions from the active chunk.
                sent += self._execute(None if max_steps is None else max_steps - sent)
                if max_steps is not None and sent >= max_steps:
                    break

            self._raise_worker_error()
        except BaseException as error:  # noqa: BLE001 - capture all errors to ensure cleanup
            errors.append(error)
        finally:
            self.stop()
            # Hardware Safety Guarantee: Stop and close physical actuators first
            # before waiting on a potentially blocked background inference call.
            with self._io_lock:
                for cleanup in (self.robot.stop, self.robot.close):
                    try:
                        cleanup()
                    except BaseException as error:  # noqa: BLE001
                        errors.append(error)

            # Join background worker thread up to shutdown_timeout_s budget.
            if worker is not None:
                worker.join(self.options.shutdown_timeout_s)
                if worker.is_alive():
                    errors.append(
                        TimeoutError(
                            "policy is still running after robot shutdown; "
                            "do not destroy its resources until predict() returns"
                        )
                    )

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("deployment and/or cleanup failed", errors)
