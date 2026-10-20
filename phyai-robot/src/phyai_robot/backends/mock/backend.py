"""In-memory device feedback for integration tests, unit tests, and CPU-only examples."""

from __future__ import annotations

import time
from typing import Any
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from ...types import Action, Sample
from ...validation import freeze_values


class MockBackend:
    """Simulated transport session reflecting commanded actions directly into sensor feedback.

    Design & Testing Purpose:
        MockBackend allows verifying composite robot aggregation, policy adapters, and control loop
        scheduling without requiring access to real robot actuators, CAN buses, or ROS nodes.
        - Upon each `write(action)`, commanded fields are mapped directly to feedback observation fields.
        - `read()` returns fresh simulated measurements timestamped with the current monotonic clock.
        - Tracks submission counts (`write_count`) and state transitions (`stop_count`).
    """

    def __init__(
        self,
        *,
        initial_values: Mapping[
            str, NDArray[Any]
        ],  # Initial array data for all observation fields.
        action_to_observation: Mapping[
            str, str
        ],  # Mapping: command_key -> feedback_field_name.
    ) -> None:
        """Initialize MockBackend with initial sensor states and reflection mapping.

        Args:
            initial_values: Dictionary of initial NumPy arrays for all observation fields.
            action_to_observation: Mapping from action command keys to observation fields that
                reflect their values upon write().
        """
        self._values: dict[str, NDArray[Any]] = dict(freeze_values(initial_values))
        self._mapping: dict[str, str] = dict(action_to_observation)

        if not set(self._mapping.values()) <= set(self._values):
            raise ValueError("all command feedback targets must exist")
        if len(set(self._mapping.values())) != len(self._mapping):
            raise ValueError(
                "multiple commands cannot overwrite the same feedback field"
            )

        self._connected: bool = False
        self._closed: bool = False
        self._stopped: bool = False
        self.write_count: int = 0  # Number of write() calls processed.
        self.stop_count: int = 0  # Number of effective stop() state transitions.
        self.last_action: Action | None = None  # Last submitted command mapping.

    @property
    def observation_keys(self) -> frozenset[str]:
        """Return the set of observation field names provided by this mock backend."""
        return frozenset(self._values)

    @property
    def action_keys(self) -> frozenset[str]:
        """Return the set of action command field names accepted by this mock backend."""
        return frozenset(self._mapping)

    def connect(self) -> None:
        """Mark mock session connected."""
        if self._closed or self._stopped:
            raise RuntimeError("mock session is closed or stopped")
        self._connected = True

    def read(self) -> Mapping[str, Sample]:
        """Generate and return simulated sensor measurements with current monotonic timestamp."""
        if not self._connected:
            raise RuntimeError("mock session is not connected")
        now_ns: int = time.monotonic_ns()
        return {
            key: Sample(value, now_ns)
            for key, value in freeze_values(self._values).items()
        }

    def write(self, action: Action) -> None:
        """Copy command values into mapped observation feedback fields."""
        if not self._connected or self._stopped:
            raise RuntimeError("mock session is not accepting commands")
        if set(action) != self.action_keys:
            raise ValueError("mock command fields do not match")

        for key, target in self._mapping.items():
            if (
                action[key].shape != self._values[target].shape
                or action[key].dtype != self._values[target].dtype
            ):
                raise ValueError(
                    f"{key}: command must match its feedback shape and dtype"
                )

        copied: Action = freeze_values(action)
        self._values.update(
            {self._mapping[key]: value for key, value in copied.items()}
        )
        self.last_action = copied
        self.write_count += 1

    def stop(self) -> None:
        """Mark mock session stopped and increment stop_count."""
        if not self._stopped:
            self.stop_count += 1
            self._stopped = True

    def close(self) -> None:
        """Halt and close mock session."""
        if not self._closed:
            self.stop()
            self._connected = False
            self._closed = True
