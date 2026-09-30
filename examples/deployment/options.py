"""YAML-friendly settings for the shared RobotDeployment control loop.

This schema is independent of robot hardware and policy layout. The entry point
chooses which continuous action fields to interpolate and checks model-specific
horizon constraints. DeploymentOptions remains the authority for timing and
queue-capacity validation; no second scheduler is implemented here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterable

from phyai_robot import DeploymentOptions


@dataclass
class DeploymentConfig:
    """Source/control frequencies in Hz, timeout and age limits in seconds.

    Queue sizes and execution_horizon count source targets, not interpolated
    control ticks. The plain YAML fields intentionally omit interpolate_keys:
    that set is chosen from the actual robot action schema during composition.
    """

    # Commands are sent at this nominal rate only while a chunk is executing.
    # An empty queue means no sends, including during observation and inference.
    control_hz: float = 200.0
    # Override ActionChunk.step_period_s with 1/action_hz. Changing this value
    # changes trajectory timing; it does not request another model prediction.
    action_hz: float = 25.0
    # Keep this many targets from each prediction and discard the remaining tail.
    execution_horizon: int = 20
    # Capacity is measured before interpolation, so it must cover the prefix.
    max_queued_actions: int = 20
    # Every field must have been received locally within this age at snapshot time.
    max_sample_age_s: float = 0.5
    # First executable chunk after run() starts; model loading/warmup is separate.
    startup_timeout_s: float = 10.0
    # Each queue refill gets its own budget to obtain a complete, fresh snapshot.
    observation_timeout_s: float = 10.0
    # Checked at each send, measured from the snapshot used for the prediction.
    # This budget includes preprocessing, inference, and time executing the prefix.
    max_prediction_age_s: float = 2.0
    # Hardware is stopped/closed before waiting this long for an active predictor.
    shutdown_timeout_s: float = 2.0
    # Tolerated lateness relative to the scheduled host send, not motor latency.
    # Missed ticks within the limit are skipped, never replayed in a catch-up burst.
    max_control_lateness_s: float = 0.01

    def to_options(self, *, interpolate_keys: Iterable[str] = ()) -> DeploymentOptions:
        """Validate loop settings and bind interpolation to the caller's fields.

        interpolate_keys names continuous floating-point action fields. The core
        loop linearly interpolates those fields and sample-and-holds all others.
        RobotDeployment checks the names/dtypes against the actual robot schema;
        this configuration layer does not infer whether a field is continuous.
        Constructing DeploymentOptions validates positive finite rates/timeouts
        and queue capacity without creating threads or connecting a robot.
        """
        return DeploymentOptions(
            **asdict(self), interpolate_keys=frozenset(interpolate_keys)
        )
