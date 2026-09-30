"""Small, transport-independent data contracts shared by robots and policies."""

from __future__ import annotations

from typing import Any, TypeAlias
from dataclasses import dataclass
from collections.abc import Mapping

from numpy.typing import NDArray


@dataclass(frozen=True)
class FeatureSpec:
    """Describe one value, without a batch dimension or a trajectory dimension.

    Units, component order, and reference frames are documentation contracts:
    device codecs must perform the actual conversions. Field names are opaque
    identifiers, not paths through a robot's physical structure.
    """

    shape: tuple[int, ...]  # One sample's shape; () is a scalar and (7,) a vector.
    dtype: str  # A NumPy real numeric or boolean dtype, such as "float32".
    unit: str | None = None  # Physical unit, such as rad, m, or N; None for RGB.
    names: tuple[str, ...] | None = None  # Component names for a 1-D vector.
    frame: str | None = None  # Reference frame for spatial values, when relevant.


# Sensor names and layouts are independent of the physical robot structure.
ObservationSchema: TypeAlias = Mapping[str, FeatureSpec]
# Command fields describe exactly one control tick.
ActionSchema: TypeAlias = Mapping[str, FeatureSpec]
# Dtypes can differ between fields; FeatureSpec validates each array at runtime.
Action: TypeAlias = Mapping[str, NDArray[Any]]


@dataclass(frozen=True)
class Sample:
    """A value and the local monotonic time at which its message was received."""

    value: NDArray[Any]  # Value in the robot schema's units, order, and dtype.
    received_at_ns: int  # time.monotonic_ns(), not a remote sensor clock.


@dataclass(frozen=True)
class Observation:
    """A complete snapshot; individual fields need not have been sampled together.

    Dataclass freezing alone does not freeze arrays. Robot implementations must
    detach snapshots from mutable receive buffers before returning them.
    """

    samples: Mapping[str, Sample]  # All declared observation fields are required.


@dataclass(frozen=True)
class ActionChunk:
    """Predicted targets; Deployment resamples them into individual commands."""

    actions: tuple[Action, ...]  # Nonempty, ordered, complete single-step targets.
    step_period_s: float  # Time between source targets, in seconds.


class ObservationNotReady(RuntimeError):
    """At least one required sensor has not produced its first sample yet."""
