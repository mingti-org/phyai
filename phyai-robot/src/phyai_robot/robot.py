"""Whole-robot interface and a small implementation assembled from sessions."""

from __future__ import annotations

from types import MappingProxyType
from typing import Protocol
from collections.abc import Mapping, Callable, Sequence

from .types import (
    Action,
    Sample,
    Observation,
    ActionSchema,
    ObservationSchema,
    ObservationNotReady,
)
from .validation import (
    freeze_values,
    freeze_samples,
    validate_schema,
    validate_values,
    validate_observation,
)
from .backends.base import Backend


class Robot(Protocol):
    """Expose named observations and complete single-tick commands.

    Robot I/O must be serialized by the caller (RobotDeployment uses an I/O
    lock). Neither this interface nor CompositeRobot promises thread-safe
    concurrent read/write/stop.
    """

    @property
    def observation_schema(self) -> ObservationSchema:
        """Return the immutable input layout before connecting."""
        ...

    @property
    def action_schema(self) -> ActionSchema:
        """Return the immutable single-command layout before connecting."""
        ...

    def connect(self) -> None:
        """Connect without implying that every sensor has delivered a first frame."""
        ...

    def get_observation(self) -> Observation:
        """Return a detached complete snapshot, or raise ObservationNotReady."""
        ...

    def send_action(self, action: Action) -> None:
        """Submit one complete command; return does not mean physical completion."""
        ...

    def stop(self) -> None:
        """Enter the configured safe state; allow repeated calls."""
        ...

    def close(self) -> None:
        """Stop and release resources; allow partial connection and repeated calls."""
        ...


class CompositeRobot:
    """Route fixed field subsets to reusable transport sessions.

    Construction performs no I/O. Commands are validated in full before the
    first write, but writes across separate transports are not transactional.
    Any device-side calibration and stop behavior belongs to backend codecs.
    """

    def __init__(
        self,
        *,
        observation_schema: ObservationSchema,  # Complete, fixed sensor layout.
        action_schema: ActionSchema,  # Complete, fixed single-step command layout.
        backends: Sequence[Backend],  # Sessions in normal write order.
        action_guard: Callable[[Action], None] | None = None,
        # Pure, nonmutating check for static physical limits before the first write.
    ) -> None:
        validate_schema(observation_schema)
        validate_schema(action_schema)
        self._observation_schema: ObservationSchema = MappingProxyType(
            dict(observation_schema)
        )
        self._action_schema: ActionSchema = MappingProxyType(dict(action_schema))
        self._backends: tuple[Backend, ...] = tuple(backends)
        self._action_guard: Callable[[Action], None] | None = action_guard
        if len({id(backend) for backend in self._backends}) != len(self._backends):
            raise ValueError("a backend instance must not be supplied twice")
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
        self._opened: list[Backend] = []  # Includes the session currently connecting.
        self._connected: bool = (
            False  # True only after all sessions opened successfully.
        )
        self._stopped: bool = (
            False  # A stopped robot cannot implicitly resume on write.
        )
        self._closed: bool = False  # Instances are single-use after close or failure.

    @property
    def observation_schema(self) -> ObservationSchema:
        return self._observation_schema

    @property
    def action_schema(self) -> ActionSchema:
        return self._action_schema

    def connect(self) -> None:
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
            try:
                self.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - re-raise both failures
                raise BaseExceptionGroup(
                    "connection and cleanup failed", [error, cleanup_error]
                )
            raise

    def _require_connected(self) -> None:
        if not self._connected or self._closed:
            raise RuntimeError("robot is not connected")

    def get_observation(self) -> Observation:
        self._require_connected()
        samples: dict[str, Sample] = {}
        for backend in self._backends:
            received: Mapping[str, Sample] = backend.read()
            if not set(received) <= backend.observation_keys:
                raise ValueError("backend returned undeclared observation fields")
            samples.update(received)
        if set(samples) != set(self.observation_schema):
            raise ObservationNotReady(
                f"missing sensor samples: {set(self.observation_schema) - set(samples)}"
            )
        observation: Observation = Observation(freeze_samples(samples))
        validate_observation(observation, self.observation_schema)
        return observation

    def send_action(self, action: Action) -> None:
        self._require_connected()
        if self._stopped:
            raise RuntimeError("robot has been stopped")
        validate_values(action, self.action_schema)
        command: Action = freeze_values(action)
        if self._action_guard is not None:
            self._action_guard(command)
        try:
            for backend in self._backends:
                if backend.action_keys:
                    backend.write({key: command[key] for key in backend.action_keys})
        except BaseException as error:
            try:
                self.stop()
            except BaseException as stop_error:  # noqa: BLE001 - re-raise both failures
                raise BaseExceptionGroup("command and stop failed", [error, stop_error])
            raise

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        errors: list[BaseException] = []
        for backend in reversed(self._opened):
            try:
                backend.stop()
            except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("one or more backends could not stop", errors)

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        try:
            self.stop()
        except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
            errors.append(error)
        for backend in reversed(self._opened):
            try:
                backend.close()
            except BaseException as error:  # noqa: BLE001 - preserve failures across cleanup/thread boundaries
                errors.append(error)
        self._closed = True
        self._connected = False
        if errors:
            raise BaseExceptionGroup("robot cleanup failed", errors)
