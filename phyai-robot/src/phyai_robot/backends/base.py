"""The reusable session boundary behind a composed Robot.

This module defines the abstract `Backend` protocol representing an independent
communication session (such as a ROS 2 node, a ZMQ subscriber/requester, a serial bus,
or an in-memory test double).
"""

from typing import Protocol
from collections.abc import Mapping

from ..types import Action, Sample


class Backend(Protocol):
    """Owns one transport session and a disjoint, fixed subset of the robot's fields.

    Concurrency Contract:
        Public methods (connect, read, write, stop, close) are called exclusively by
        the Robot caller under serialization (e.g. RobotDeployment's _io_lock).
        Internal background workers (such as ROS spin or ZMQ poll threads) may update
        internal sample caches concurrently, but must expose thread-safe immutable
        snapshots and propagate background exceptions upon subsequent public calls.
    """

    @property
    def observation_keys(self) -> frozenset[str]:
        """Return the immutable set of sensor field names owned by this session."""
        ...

    @property
    def action_keys(self) -> frozenset[str]:
        """Return the immutable set of actuator command field names owned by this session."""
        ...

    def connect(self) -> None:
        """Initialize transport resources, sockets, nodes, and worker threads.

        If initialization fails partway through, the backend must clean up any partially
        allocated resources before raising.
        """
        ...

    def read(self) -> Mapping[str, Sample]:
        """Return the latest cached sensor samples owned by this session.

        Non-Blocking Contract:
            read() must return immediately from cache and must NEVER block waiting for
            the next incoming network packet or sensor frame. During system startup,
            returning a subset of declared keys is permitted before initial frames arrive.
        """
        ...

    def write(self, action: Action) -> None:
        """Submit the owned subset of actuator commands within a bounded I/O timeout.

        Args:
            action: Command mapping containing exactly the keys declared in action_keys.

        Raises:
            TimeoutError: If the transport fails to transmit or acknowledge within io_timeout_s.
            RuntimeError: If called when stopped or disconnected.
        """
        ...

    def stop(self) -> None:
        """Command the session's owned actuators to enter their safe state.

        Idempotent: Repeated invocations must be safe and non-blocking. Bounded by stop_timeout_s.
        """
        ...

    def close(self) -> None:
        """Halt worker threads and release all native communication handles."""
        ...
