"""Empty-queue inference and independently paced, single-step robot control.

The queue contains model-rate targets, NOT pre-expanded control ticks. A policy
worker reads one observation only when the previous chunk has fully drained.
The control loop interpolates that chunk at its own frequency, then pauses all
command publication while the worker obtains and processes the next snapshot.
This is stop-and-refill chunk execution, not RTC or a hard real-time scheduler.
"""

from __future__ import annotations

import math
import time
import threading
from typing import Any
from _thread import LockType
from collections import deque
from dataclasses import dataclass

import numpy as np

from .robot import Robot
from .types import Action, ActionChunk, Observation, ActionSchema, ObservationNotReady
from .policy import Policy, PolicyAdapter
from .validation import (
    freeze_values,
    freeze_samples,
    validate_values,
    validate_observation,
)


@dataclass(frozen=True)
class DeploymentOptions:
    control_hz: float  # Robot.send_action frequency during execution, not inference.
    max_sample_age_s: float  # Maximum local receive age when taking a snapshot.
    startup_timeout_s: float = 5.0  # Budget to obtain the first executable chunk.
    observation_timeout_s: float = 5.0  # Per-refill wait for complete, fresh sensors.
    max_queued_actions: int = 50  # Capacity in SOURCE/model targets, not control ticks.
    max_prediction_age_s: float = 0.5  # Snapshot-to-send limit, including inference.
    shutdown_timeout_s: float = 2.0  # Maximum wait for an in-flight prediction.
    interpolate_keys: frozenset[str] = frozenset()  # Other fields use sample-and-hold.
    execution_horizon: int | None = None  # Prefix of each prediction; None keeps all.
    action_hz: float | None = None  # None uses the adapter's ActionChunk.step_period_s.
    max_control_lateness_s: float = 0.01  # Allowed delay past a scheduled send.

    def __post_init__(self) -> None:
        for name in (
            "control_hz",
            "max_sample_age_s",
            "startup_timeout_s",
            "observation_timeout_s",
            "max_prediction_age_s",
            "shutdown_timeout_s",
            "max_control_lateness_s",
            "action_hz",
        ):
            value: Any = getattr(self, name)
            if name == "action_hz" and value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if 1e9 / self.control_hz < 1:
            raise ValueError("control period must be at least one nanosecond")
        if type(self.max_queued_actions) is not int or self.max_queued_actions < 1:
            raise ValueError("max_queued_actions must be a positive integer")
        if self.execution_horizon is not None and (
            type(self.execution_horizon) is not int or self.execution_horizon < 1
        ):
            raise ValueError("execution_horizon must be a positive integer or None")
        if (
            self.execution_horizon is not None
            and self.execution_horizon > self.max_queued_actions
        ):
            raise ValueError(
                "queue capacity must cover execution_horizon source actions"
            )
        object.__setattr__(self, "interpolate_keys", frozenset(self.interpolate_keys))


def _prepare_chunk(
    chunk: ActionChunk, schema: ActionSchema, options: DeploymentOptions
) -> ActionChunk:
    """Validate the whole prediction, then detach the retained model-rate prefix.

    A malformed discarded tail still indicates a broken prediction. Capacity
    overflow fails explicitly instead of silently shortening execution_horizon.
    No array of 200 Hz commands is allocated here; interpolation is the control
    loop's job. The configured action rate overrides the adapter's nominal rate.
    """
    if not isinstance(chunk, ActionChunk) or not chunk.actions:
        raise ValueError("adapter must return a nonempty ActionChunk")
    if not math.isfinite(chunk.step_period_s) or chunk.step_period_s <= 0:
        raise ValueError("chunk period must be finite and positive")
    for action in chunk.actions:
        validate_values(action, schema)
    count = len(chunk.actions)
    if options.execution_horizon is not None:
        count = min(count, options.execution_horizon)
    if count > options.max_queued_actions:
        raise ValueError(
            "predicted prefix exceeds queue capacity; set execution_horizon"
        )
    period = chunk.step_period_s if options.action_hz is None else 1 / options.action_hz
    ticks = count * period * options.control_hz
    if not math.isfinite(ticks) or ticks <= 0:
        raise ValueError("source/control period ratio cannot be represented")
    return ActionChunk(tuple(freeze_values(a) for a in chunk.actions[:count]), period)


def _interpolate(
    left: Action, right: Action, fraction: float, keys: frozenset[str]
) -> Action:
    """Linear interpolation for continuous absolute targets; hold all other fields.

    Convex interpolation does not overshoot either endpoint, unlike an arbitrary
    cubic spline. This is joint/normalized-gripper interpolation, not an SE(3)
    pose interpolator or a velocity/acceleration limiter. At the chunk boundary
    right == left: the final retained target is held for one source interval.
    """
    if fraction == 0 or left is right:
        return left
    values: dict[str, Any] = dict(left)
    for key in keys:
        # Work in float64 to avoid overflow in the intermediate subtraction for
        # lower-precision schemas, then restore the schema's original dtype.
        values[key] = (
            (1 - fraction) * left[key].astype(np.float64)
            + fraction * right[key].astype(np.float64)
        ).astype(left[key].dtype)
    return freeze_values(values)


class RobotDeployment[RequestT, ResultT]:
    """Run one observation/inference worker and one independent control loop.

    A Condition protects queue contents, chunk timing and worker failures. The
    last source target stays in the queue through its final interval and send,
    so an apparently empty queue can never trigger observation mid-command.
    Robot is not required to be thread-safe: an I/O lock serializes worker reads,
    control writes, shutdown and external stop requests. predict() never holds
    that lock, so stopping hardware does not wait for a stalled model.
    """

    def __init__(
        self,
        *,
        robot: Robot,
        policy: Policy[RequestT, ResultT],
        adapter: PolicyAdapter[RequestT, ResultT],
        options: DeploymentOptions,
    ) -> None:
        if not options.interpolate_keys <= set(robot.action_schema):
            raise ValueError("interpolate_keys contains unknown command fields")
        for key in options.interpolate_keys:
            if np.dtype(robot.action_schema[key].dtype).kind != "f":
                raise ValueError(f"{key}: interpolation requires a floating dtype")
        self.robot = robot
        self.policy = policy
        self.adapter = adapter
        self.options = options
        self._condition = threading.Condition()
        self._io_lock: LockType = threading.Lock()
        self._stop_event = threading.Event()
        self._queue: deque[Action] = deque()
        self._source_period_s = 0.0  # Metadata changes atomically with queue refill.
        self._observed_at_ns = 0
        self._worker_error: BaseException | None = None
        self._started = False

    def stop(self) -> None:
        """Request shutdown; no new Robot I/O starts after this method returns.

        The run thread performs Robot.stop/close. Calls already inside bounded
        Robot I/O finish first. In-flight inference is discarded on return.
        """
        # Publish cancellation before waiting for the I/O barrier, so a worker
        # finishing read() cannot start another send while stop() waits its turn.
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        with self._io_lock:
            pass  # Wait for any already-started Robot call to finish.

    def _observe(self) -> tuple[Observation, int] | None:
        with self._io_lock:
            if self._stop_event.is_set():
                return None
            raw = self.robot.get_observation()
        # Custom Robots receive the same detached snapshot/validation contract as
        # CompositeRobot. These copies happen only between execution chunks.
        observation = Observation(freeze_samples(raw.samples))
        validate_observation(observation, self.robot.observation_schema)
        now_ns = time.monotonic_ns()
        for key, sample in observation.samples.items():
            if (
                not 0
                <= now_ns - sample.received_at_ns
                <= self.options.max_sample_age_s * 1e9
            ):
                raise TimeoutError(f"stale observation: {key}")
        return observation, now_ns

    def _produce(self) -> None:
        try:
            while not self._stop_event.is_set():
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_event.is_set() or not self._queue
                    )
                if self._stop_event.is_set():
                    return
                # Empty means idle, including time spent waiting for new sensor
                # frames. A populated cache is not necessarily a fresh snapshot.
                # Retry without sending anything; never relax the age threshold.
                observation_deadline = (
                    time.monotonic() + self.options.observation_timeout_s
                )
                while not self._stop_event.is_set():
                    try:
                        snapshot = self._observe()
                        break
                    except (ObservationNotReady, TimeoutError) as error:
                        if time.monotonic() >= observation_deadline:
                            raise TimeoutError(
                                f"timed out waiting for fresh observation: {error}"
                            ) from error
                        self._stop_event.wait(0.01)
                else:
                    return
                if snapshot is None:
                    return
                observation, observed_at_ns = snapshot
                request = self.adapter.to_request(observation)
                if self._stop_event.is_set():
                    return
                result = self.policy.predict(request)
                if self._stop_event.is_set():
                    return
                chunk = _prepare_chunk(
                    self.adapter.to_actions(result, observation),
                    self.robot.action_schema,
                    self.options,
                )
                with self._condition:
                    if self._stop_event.is_set():
                        return
                    # Only this worker refills, and only a completely drained
                    # chunk triggers it. No prefetch, merging, or low watermark.
                    self._queue.extend(chunk.actions)
                    self._source_period_s = chunk.step_period_s
                    self._observed_at_ns = observed_at_ns
                    self._condition.notify_all()
        except BaseException as error:  # noqa: BLE001 - forward failures across threads
            with self._condition:
                self._worker_error = error
                self._condition.notify_all()
            self.stop()

    def _raise_worker_error(self) -> None:
        with self._condition:
            if self._worker_error is not None:
                raise self._worker_error

    def _execute_chunk(self, remaining_steps: int | None) -> int:
        """Consume source targets lazily using a fresh clock for this chunk.

        Target i is at i * source_period. The chunk covers [0, N * period),
        including a final held interval. Without missed ticks, 25/200 Hz gives
        eight sends per source interval, or 160 sends over 0.8 seconds for 20
        targets. Tolerated lateness skips old ticks without stretching source
        timing. The inference gap is deliberately outside this control clock.
        """
        with self._condition:
            count = len(self._queue)
            source_period = self._source_period_s
            observed_at_ns = self._observed_at_ns
        ratio = source_period * self.options.control_hz
        total_ticks = math.ceil(count * ratio)
        started_at_ns = time.monotonic_ns()
        period_ns = 1e9 / self.options.control_hz
        max_lateness_ns = self.options.max_control_lateness_s * 1e9
        source_index = 0
        sent = 0
        tick = 0
        previous_send_ns = 0
        while tick < total_ticks:
            scheduled_tick = tick
            deadline_ns = started_at_ns + round(tick * period_ns)
            if self._stop_event.wait(max(0, (deadline_ns - time.monotonic_ns()) / 1e9)):
                return sent
            woke_at_ns = time.monotonic_ns()
            with self._io_lock:
                if self._stop_event.is_set():
                    return sent
                locked_at_ns = time.monotonic_ns()
                # Keep the original deadline for the lateness check. Moving it
                # forward first would hide stalls larger than the configured limit.
                # Within that limit, select the latest due tick instead of replaying
                # stale interpolated commands in a catch-up burst.
                tick = min(
                    total_ticks - 1,
                    max(tick, math.floor((locked_at_ns - started_at_ns) / period_ns)),
                )
                position = tick / ratio
                index = min(math.floor(position), count - 1)
                with self._condition:
                    while source_index < index:
                        self._queue.popleft()
                        source_index += 1
                    left = self._queue[0]
                    right = self._queue[1] if len(self._queue) > 1 else left
                action = _interpolate(
                    left, right, position - index, self.options.interpolate_keys
                )
                now_ns = time.monotonic_ns()
                lateness_ns = now_ns - deadline_ns
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
                if now_ns - observed_at_ns > self.options.max_prediction_age_s * 1e9:
                    raise TimeoutError("queued prediction expired before execution")
                self.robot.send_action(action)
                previous_send_ns = time.monotonic_ns() - now_ns
            sent += 1
            if remaining_steps is not None and sent >= remaining_steps:
                return sent
            # Preparation can also cross a tick boundary. Do not replay that tick
            # immediately after this send; keep the next deadline on the chunk clock.
            tick = max(tick + 1, math.floor((now_ns - started_at_ns) / period_ns) + 1)
        # Keep the last target queued until its interval has elapsed. This also
        # prevents a refill/read racing with the final in-flight send. We send
        # nothing during this wait or during the subsequent observation/inference.
        end_ns = started_at_ns + round(count * source_period * 1e9)
        if not self._stop_event.wait(max(0, (end_ns - time.monotonic_ns()) / 1e9)):
            with self._condition:
                self._queue.clear()
                self._condition.notify_all()
        return sent

    def run(self, *, max_steps: int | None = None) -> None:
        """Connect, execute, and always stop/close; max_steps counts actual sends.

        An empty queue is normal: sleep until inference enqueues a whole prefix.
        No duplicate hold commands are sent across that gap. During an active
        chunk, lateness above max_control_lateness_s stops execution; tolerable
        missed ticks are skipped without catch-up bursts. The first-chunk timeout
        does not limit later idle gaps.
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
            startup_deadline = time.monotonic() + self.options.startup_timeout_s
            if not self._stop_event.is_set():
                worker = threading.Thread(
                    target=self._produce, name="phyai-policy", daemon=True
                )
                worker.start()
            sent = 0
            while not self._stop_event.is_set():
                with self._condition:
                    timeout = (
                        max(0, startup_deadline - time.monotonic())
                        if sent == 0
                        else None
                    )
                    ready = self._condition.wait_for(
                        lambda: self._stop_event.is_set() or bool(self._queue), timeout
                    )
                self._raise_worker_error()
                if self._stop_event.is_set():
                    break
                if not ready:
                    raise TimeoutError("timed out waiting for first observation/action")
                sent += self._execute_chunk(
                    None if max_steps is None else max_steps - sent
                )
                if max_steps is not None and sent >= max_steps:
                    break
            self._raise_worker_error()
        except BaseException as error:  # noqa: BLE001 - preserve errors through cleanup
            errors.append(error)
        finally:
            self.stop()
            # Stop hardware before waiting for potentially blocked prediction.
            # The I/O lock also excludes the producer's bounded observation read.
            with self._io_lock:
                for cleanup in (self.robot.stop, self.robot.close):
                    try:
                        cleanup()
                    except BaseException as error:  # noqa: BLE001 - preserve cleanup failures
                        errors.append(error)
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
