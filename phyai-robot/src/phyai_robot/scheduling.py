"""Action-chunk scheduling: coordinating model predictions with actuator control ticks.

Architecture:
    In phyai-robot, hardware control execution is decoupled into two independent domains:
    1. Execution Loop (`RobotDeployment`): Manages threads, physical I/O locks, hardware
       timing (at `control_hz`), and graceful shutdown sequences.
    2. Trajectory Scheduling (`Scheduler`): Dictates the action timeline — determining when
       to request new inference snapshots, accepting incoming predictions, and computing
       the exact interpolated command to dispatch at any control tick timestamp.

    All methods on `Scheduler` are invoked exclusively while `RobotDeployment` holds its
    internal synchronization Condition lock. Consequently:
    - Schedulers do not need internal locks or thread-synchronization primitives.
    - Schedulers must perform bounded synchronous CPU work without I/O.

Available Schedulers:
    - `StopAndRefill` (Default): Traditional chunk-by-chunk execution. The robot executes
      an action prefix, halts motion when the chunk drains, acquires a fresh sensor snapshot,
      queries the model policy, and resumes execution.
    - `Real-Time Chunking (RTC)`: Schedulers implementing continuous motion query inference
      while the current trajectory prefix is still executing. The `InferenceContext.pending`
      tuple provides the model adapter with the exact future actions scheduled to run
      during the model forward pass, enabling smooth trajectory splicing.
"""

from __future__ import annotations

import math
from typing import Any, Protocol
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from .types import Action, ActionChunk, ActionSchema
from .validation import freeze_values


@dataclass(frozen=True)
class InferenceContext:
    """Snapshot context provided to PolicyAdapter when initiating inference.

    Attributes:
        pending: Ordered tuple of single-tick physical actions that are currently scheduled
            to execute on hardware while the background model prediction is running.
            - In `StopAndRefill`, motion is paused during inference, so `pending` is empty.
            - In Real-Time Chunking (RTC), `pending` samples the committed trajectory
              from the observation timestamp at action_hz, including its current target.
    """

    pending: tuple[Action, ...] = ()


class Scheduler(Protocol):
    """Protocol governing the trajectory timeline between predictions and actuator ticks.

    All methods are invoked by RobotDeployment while holding its internal condition lock.
    """

    def bind(self, action_schema: ActionSchema) -> None:
        """Validate and bind strategy settings against the robot's physical action schema.

        Called during RobotDeployment initialization. Schedulers inspect action_schema to
        identify continuous floating-point fields eligible for linear interpolation.
        """
        ...

    def check(self, chunk: ActionChunk) -> None:
        """Verify that an incoming predicted chunk satisfies strategy prerequisites.

        Raises:
            ValueError: If the chunk contains fewer targets than execution_horizon.
        """
        ...

    def wants_inference(self) -> bool:
        """Query whether the background worker thread should trigger a new prediction now."""
        ...

    def begin_inference(self, now_ns: int) -> InferenceContext:
        """Record that an observation snapshot was taken and return the active context.

        Args:
            now_ns: Monotonic timestamp in nanoseconds when the observation was captured.

        Returns:
            InferenceContext describing pending actions executing during this inference.
        """
        ...

    def add_chunk(
        self, chunk: ActionChunk, context: InferenceContext, now_ns: int
    ) -> None:
        """Enqueue or splice a newly received action prediction into the timeline.

        Args:
            chunk: Validated ActionChunk returned by PolicyAdapter.to_actions().
            context: The InferenceContext originally returned by begin_inference().
            now_ns: Monotonic timestamp in nanoseconds when the prediction arrived.
        """
        ...

    def is_active(self) -> bool:
        """Return True if the scheduler has actions available to dispatch via action_at()."""
        ...

    def action_at(self, tick_ns: int) -> Action | None:
        """Compute the interpolated actuator command scheduled for host timestamp tick_ns.

        Args:
            tick_ns: Monotonic timestamp in nanoseconds of the current control tick.

        Returns:
            Action mapping with interpolated targets, or None to signal the end of the active
            chunk (transitioning the deployment control loop to idle).
        """
        ...


def interpolate(
    left: Action, right: Action, fraction: float, keys: frozenset[str]
) -> Action:
    """Perform linear convex interpolation for continuous targets; hold discrete fields.

    Mathematical guarantee:
        Convex interpolation: target = (1 - fraction) * left + fraction * right.
        Never overshoots either endpoint, unlike cubic splines or polynomial fits.
        Continuous fields listed in `keys` are linearly blended in float64 arithmetic to
        prevent intermediate precision overflow, then cast back to the field's native dtype.
        Fields not in `keys` perform zero-order sample-and-hold (retaining values from `left`).

    Args:
        left: Starting target Action at fraction = 0.0.
        right: Ending target Action at fraction = 1.0.
        fraction: Normalized temporal progress in [0.0, 1.0].
        keys: Set of action field names configured for continuous interpolation.

    Returns:
        Complete Action mapping containing interpolated array values (read-only buffers).
    """
    if fraction == 0 or left is right:
        return left
    values: dict[str, Any] = dict(left)
    for key in keys:
        # Compute in float64 to prevent numeric overflow on float16/float32 schemas.
        value = (
            (1 - fraction) * left[key].astype(np.float64)
            + fraction * right[key].astype(np.float64)
        ).astype(left[key].dtype)
        value.flags.writeable = False
        values[key] = value
    return values


class StopAndRefill:
    """Chunk-by-chunk scheduling: execute an action prefix, halt, refill, and repeat.

    Execution Invariants:
    1. Action Consumption: Target i is scheduled at host time `start_ns + i / action_hz`.
       The chunk duration spans `[0, execution_horizon / action_hz)`.
    2. Tail Holding: The final retained target is held constant across its final interval;
       interpolation never drifts toward unpredicted or discarded horizons.
    3. Motion Separation: Inference occurs strictly when the previous chunk has drained.
       Actuation is suspended during observation acquisition and model forward pass,
       eliminating sensor-actuator race conditions without complex queuing.
    """

    def __init__(
        self,
        *,
        action_hz: float,  # Consumption rate of model targets in Hz (e.g. 25.0 Hz).
        execution_horizon: int | None = None,  # Number of targets to execute per chunk.
        interpolate_keys: Iterable[str]
        | None = None,  # Continuous fields (None = all floats).
    ) -> None:
        """Initialize StopAndRefill scheduler.

        Args:
            action_hz: Target consumption frequency in Hz (must be positive and finite).
            execution_horizon: Integer count of leading targets to execute from each chunk.
                If None, the entire predicted chunk horizon is executed.
            interpolate_keys: Optional collection of field names to interpolate. If None,
                all floating-point fields in the action schema are automatically selected.
        """
        if (
            isinstance(action_hz, bool)
            or not math.isfinite(action_hz)
            or action_hz <= 0
        ):
            raise ValueError("action_hz must be finite and positive")
        if execution_horizon is not None and (
            type(execution_horizon) is not int or execution_horizon < 1
        ):
            raise ValueError("execution_horizon must be a positive integer or None")

        self.action_hz = float(action_hz)
        self.execution_horizon = execution_horizon
        self._requested_keys = (
            None if interpolate_keys is None else frozenset(interpolate_keys)
        )
        self._keys: frozenset[str] = frozenset()
        self._chunk: tuple[Action, ...] | None = None
        self._start_ns: int | None = None

    def bind(self, action_schema: ActionSchema) -> None:
        """Inspect action_schema to identify floating-point fields for interpolation."""
        floating = {
            key
            for key, spec in action_schema.items()
            if np.dtype(spec.dtype).kind == "f"
        }
        if self._requested_keys is None:
            self._keys = frozenset(floating)
            return
        if not self._requested_keys <= set(action_schema):
            raise ValueError("interpolate_keys contains unknown command fields")
        if not self._requested_keys <= floating:
            raise ValueError("interpolation requires floating-point command fields")
        self._keys = self._requested_keys

    def check(self, chunk: ActionChunk) -> None:
        """Verify that the predicted chunk contains at least execution_horizon targets."""
        if self.execution_horizon is not None and (
            len(chunk.actions) < self.execution_horizon
        ):
            raise ValueError(
                f"chunk has {len(chunk.actions)} targets; "
                f"execution_horizon is {self.execution_horizon}"
            )

    def wants_inference(self) -> bool:
        """Trigger inference only when the previous chunk has completely drained."""
        return self._chunk is None

    def begin_inference(self, now_ns: int) -> InferenceContext:
        """Return empty context because no actions execute during StopAndRefill inference."""
        return InferenceContext()

    def add_chunk(
        self, chunk: ActionChunk, context: InferenceContext, now_ns: int
    ) -> None:
        """Store the retained prefix of the predicted chunk; clock starts on first tick."""
        self.check(chunk)
        self._chunk = chunk.actions[: self.execution_horizon]
        self._start_ns = None  # Reset chunk clock; set upon first call to action_at().

    def is_active(self) -> bool:
        """Return True while a chunk is loaded and executing."""
        return self._chunk is not None

    def action_at(self, tick_ns: int) -> Action | None:
        """Compute the linearly interpolated action target for monotonic timestamp tick_ns."""
        if self._chunk is None:
            return None
        if self._start_ns is None:
            self._start_ns = tick_ns

        # Compute fractional position in target steps: position = dt * action_hz.
        position = (tick_ns - self._start_ns) * self.action_hz / 1e9
        if position >= len(self._chunk):
            # Chunk duration has elapsed. Transition to idle to trigger next observation.
            self._chunk = self._start_ns = None
            return None

        index = math.floor(position)
        left = self._chunk[index]
        right = self._chunk[min(index + 1, len(self._chunk) - 1)]
        return interpolate(left, right, position - index, self._keys)


class RealTimeChunking(StopAndRefill):
    """Overlap inference with a fixed, committed prefix of the current trajectory.

    ``execution_horizon`` is the number of action periods between inference anchors.
    The initial anchor starts on the first control tick. Later anchors are observation
    timestamps, never prediction arrival times. For P prefix targets, inference must
    finish strictly before anchor + (P - 1) / action_hz. The old interpolation remains
    intact until that endpoint; only the predicted suffix is then used.

    This bounded implementation keeps one current and one future chunk. It requires
    2 <= P <= execution_horizon and execution_horizon + P <= the model horizon.
    """

    def __init__(
        self,
        *,
        action_hz: float,
        prefix_steps: int,
        execution_horizon: int,
        interpolate_keys: Iterable[str] | None = None,
    ) -> None:
        super().__init__(
            action_hz=action_hz,
            execution_horizon=execution_horizon,
            interpolate_keys=interpolate_keys,
        )
        if type(prefix_steps) is not int or prefix_steps < 2:
            raise ValueError("RTC prefix_steps must be an integer of at least 2")
        if type(execution_horizon) is not int or execution_horizon < prefix_steps:
            raise ValueError("RTC requires prefix_steps <= execution_horizon")
        if action_hz > 1e9:
            raise ValueError("RTC action period must be at least one nanosecond")
        self.prefix_steps = prefix_steps
        self._period_ns = 1e9 / self.action_hz
        self._horizon: int | None = None
        self._inflight: InferenceContext | None = None
        self._anchor_ns: int | None = None
        self._request_at_ns: float | None = None
        self._handoff_ns: float | None = None
        self._replacement: tuple[Action, ...] | None = None
        self._last_tick_ns: int | None = None

    def configure_model(self, *, horizon: int, max_prefix_len: int) -> None:
        """Validate advertised training support before reserving a run or opening hardware."""
        if type(horizon) is not int or horizon < 1:
            raise ValueError("RTC requires a positive integer model horizon")
        if (
            type(max_prefix_len) is not int
            or not self.prefix_steps <= max_prefix_len < horizon
        ):
            raise ValueError(
                "RTC requires trained max_prefix_len >= prefix_steps and < horizon"
            )
        if self.execution_horizon + self.prefix_steps > horizon:
            raise ValueError("RTC requires execution_horizon + prefix_steps <= horizon")
        if self._inflight is not None or self._chunk is not None:
            raise RuntimeError("cannot configure RTC after inference has started")
        self._horizon = horizon

    def check(self, chunk: ActionChunk) -> None:
        """Require the complete model horizon, including its known prefix."""
        if self._horizon is None:
            raise ValueError("RTC model capability has not been negotiated")
        if len(chunk.actions) != self._horizon:
            raise ValueError(
                f"RTC chunk has {len(chunk.actions)} targets; "
                f"model horizon is {self._horizon}"
            )

    def wants_inference(self) -> bool:
        if self._inflight is not None or self._replacement is not None:
            return False
        return self._chunk is None or (
            self._request_at_ns is not None
            and self._last_tick_ns is not None
            and self._last_tick_ns >= self._request_at_ns
        )

    def begin_inference(self, now_ns: int) -> InferenceContext:
        """Freeze the trajectory at the snapshot timestamp under the robot I/O lock."""
        if self._horizon is None:
            raise ValueError("RTC model capability has not been negotiated")
        if not self.wants_inference():
            raise RuntimeError("RTC inference is already pending or not due")
        if self._last_tick_ns is not None and now_ns < self._last_tick_ns:
            raise ValueError("RTC observation timestamp precedes the control timeline")

        pending: tuple[Action, ...] = ()
        if self._chunk is not None:
            handoff_ns = now_ns + (self.prefix_steps - 1) * self._period_ns
            if handoff_ns >= self._start_ns + len(self._chunk) * self._period_ns:
                raise TimeoutError(
                    "RTC has insufficient remaining overlap for inference"
                )
            pending = tuple(
                freeze_values(self._sample(now_ns + step * self._period_ns))
                for step in range(self.prefix_steps)
            )
            self._handoff_ns = handoff_ns
        context = InferenceContext(pending=pending)
        self._inflight = context
        self._anchor_ns = now_ns
        return context

    def add_chunk(
        self, chunk: ActionChunk, context: InferenceContext, now_ns: int
    ) -> None:
        if context is not self._inflight or context is None:
            raise ValueError("RTC received a stale or foreign inference context")
        self.check(chunk)
        if now_ns < self._anchor_ns or (
            self._last_tick_ns is not None and now_ns < self._last_tick_ns
        ):
            raise ValueError("RTC prediction timestamp precedes the active timeline")
        if not context.pending:
            self._chunk = chunk.actions
        else:
            if now_ns >= self._handoff_ns:
                raise TimeoutError(
                    "RTC prediction missed the committed-prefix deadline"
                )
            # Keep the actual endpoint, avoiding round-trip drift in the model prefix.
            self._replacement = (
                context.pending[-1],
                *chunk.actions[self.prefix_steps :],
            )
        self._inflight = None

    def _sample(self, tick_ns: float) -> Action:
        assert self._chunk is not None and self._start_ns is not None
        position = (tick_ns - self._start_ns) / self._period_ns
        if position < 0:
            raise ValueError("RTC timestamp precedes the active trajectory")
        if position >= len(self._chunk):
            raise TimeoutError("RTC trajectory underrun")
        index = math.floor(position)
        left = self._chunk[index]
        right = self._chunk[min(index + 1, len(self._chunk) - 1)]
        return interpolate(left, right, position - index, self._keys)

    def action_at(self, tick_ns: int) -> Action | None:
        if self._chunk is None:
            return None
        if self._last_tick_ns is not None and tick_ns < self._last_tick_ns:
            raise ValueError("RTC control timestamps must be monotonic")
        self._last_tick_ns = tick_ns
        if self._start_ns is None:
            self._start_ns = tick_ns
            self._request_at_ns = tick_ns + self.execution_horizon * self._period_ns
        if self._handoff_ns is not None and tick_ns >= self._handoff_ns:
            if self._replacement is None:
                raise TimeoutError("RTC inference missed the committed-prefix deadline")
            self._chunk = self._replacement
            self._start_ns = self._handoff_ns
            self._request_at_ns = (
                self._anchor_ns + self.execution_horizon * self._period_ns
            )
            self._replacement = None
            self._handoff_ns = None
        return self._sample(tick_ns)
