"""The reusable session boundary behind a composed Robot."""

from typing import Protocol
from collections.abc import Mapping

from ..types import Action, Sample


class Backend(Protocol):
    """Own one transport session and a fixed subset of the robot's fields.

    Public methods are serialized by the Robot's caller. Receive workers may
    update private caches concurrently, but must return stable snapshots and
    propagate worker errors. A cache read must not wait for the next sensor frame.
    """

    @property
    def observation_keys(self) -> frozenset[str]:
        """Return the fixed fields this session can observe, before connecting."""
        ...

    @property
    def action_keys(self) -> frozenset[str]:
        """Return the fixed command fields owned by this session."""
        ...

    def connect(self) -> None:
        """Open resources; clean up partial initialization if opening fails."""
        ...

    def read(self) -> Mapping[str, Sample]:
        """Read cached fields; missing fields are permitted during startup only."""
        ...

    def write(self, action: Action) -> None:
        """Submit the exact owned command subset with a bounded wait."""
        ...

    def stop(self) -> None:
        """Request the configured safe state; repeated calls must be harmless."""
        ...

    def close(self) -> None:
        """Release resources, including after a partial connect failure."""
        ...
