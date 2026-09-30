"""In-memory device feedback for integration tests and CPU-only examples."""

from __future__ import annotations

import time
from typing import Any
from collections.abc import Mapping

from numpy.typing import NDArray

from ...types import Action, Sample
from ...validation import freeze_values


class MockBackend:
    """Immediately copy commands into mapped observations, without dynamics.

    Each read represents a newly generated simulated measurement. Real cached
    transports must instead retain the timestamp of the last received message.
    """

    def __init__(
        self,
        *,
        initial_values: Mapping[
            str, NDArray[Any]
        ],  # Independent initial sensor arrays.
        action_to_observation: Mapping[str, str],  # Command key -> feedback field.
    ) -> None:
        self._values: dict[str, NDArray[Any]] = dict(freeze_values(initial_values))
        self._mapping: dict[str, str] = dict(action_to_observation)
        if not set(self._mapping.values()) <= set(self._values):
            raise ValueError("all command feedback targets must exist")
        if len(set(self._mapping.values())) != len(self._mapping):
            raise ValueError(
                "multiple commands cannot overwrite the same feedback field"
            )
        self._connected: bool = False  # Mock has the same explicit connection boundary.
        self._closed: bool = False  # Closing makes this session single-use.
        self._stopped: bool = False  # Stopping prevents subsequent commands.
        self.write_count: int = (
            0  # Bounded diagnostic state; no unbounded action history.
        )
        self.stop_count: int = (
            0  # Counts effective stop transitions, not repeated calls.
        )
        self.last_action: Action | None = None  # Detached last command for inspection.

    @property
    def observation_keys(self) -> frozenset[str]:
        return frozenset(self._values)

    @property
    def action_keys(self) -> frozenset[str]:
        return frozenset(self._mapping)

    def connect(self) -> None:
        if self._closed or self._stopped:
            raise RuntimeError("mock session is closed or stopped")
        self._connected = True

    def read(self) -> Mapping[str, Sample]:
        if not self._connected:
            raise RuntimeError("mock session is not connected")
        now_ns: int = time.monotonic_ns()
        return {
            key: Sample(value, now_ns)
            for key, value in freeze_values(self._values).items()
        }

    def write(self, action: Action) -> None:
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
        if not self._stopped:
            self.stop_count += 1
            self._stopped = True

    def close(self) -> None:
        if not self._closed:
            self.stop()
            self._connected = False
            self._closed = True
