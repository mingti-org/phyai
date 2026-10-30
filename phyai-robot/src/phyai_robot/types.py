"""Small, transport-independent data contracts shared by robots and policies.

This module defines the foundational data structures and type definitions for
phyai-robot. All contracts are designed around four principles:
1. Data layout independence: Sensor observations, motor commands, and model
   tensors are identified by opaque string names (e.g. 'joint_position_left',
   'head_camera') rather than hardcoded hierarchical part trees.
2. Strict type & shape specifications: Each feature describes its shape and
   dtype without implicit broadcasting, runtime casting, or hidden unit conversion.
3. Decoupled time domains: Hardware sampling time, local arrival timestamp, and
   actuator execution schedule are explicitly separated.
4. Thread-safe snapshot semantics: Data classes detach mutable buffers so
   observations passed across concurrency boundaries remain immutable.
"""

from __future__ import annotations

from typing import Any, TypeAlias
from dataclasses import dataclass
from collections.abc import Mapping

from numpy.typing import NDArray


@dataclass(frozen=True)
class FeatureSpec:
    """Specification of a single named feature without batch or trajectory dimensions.

    FeatureSpec defines the structural and numerical contract of a single tensor or
    array value. It serves as the single source of truth for schema validation before
    any hardware connection is opened or any model weight is evaluated.

    Attributes:
        shape: Dimensions of a single sample. An empty tuple () denotes a scalar,
            (7,) denotes a 7-element 1D vector (e.g. 7-DoF joint angles), and
            (224, 224, 3) denotes an HWC image tensor.
        dtype: Canonical NumPy data type string (e.g. 'float32', 'float64', 'uint8',
            'int32', 'bool'). Must belong to real numeric, integer, or boolean kinds.
        unit: Physical engineering unit of the data (e.g. 'rad', 'm', 'N', 'normalized').
            This acts as an explicit documentation contract; transport codecs and
            adapters must perform any required physical conversions. None for images.
        names: Optional component names for a 1D vector (e.g. joint names ('j1', ..., 'j7')
            or wrench components ('fx', 'fy', 'fz', 'tx', 'ty', 'tz')). When provided,
            len(names) must strictly equal shape[0] and elements must be unique.
        frame: Optional reference coordinate frame identifier for spatial vectors
            (e.g. 'base_link', 'tool0', 'world').
    """

    shape: tuple[
        int, ...
    ]  # Shape of one individual sample: () for scalar, (N,) for vector.
    dtype: str  # NumPy dtype name, such as "float32", "float64", "uint8".
    unit: str | None = None  # Physical unit documentation, e.g. 'rad', 'm', 'N'.
    names: tuple[str, ...] | None = None  # Component names for 1-D vector features.
    frame: str | None = None  # Reference coordinate frame for spatial features.


# Mapping from opaque field names to their FeatureSpec.
# Used universally across observation layouts, action layouts, and model RPC schemas.
Schema: TypeAlias = Mapping[str, FeatureSpec]

# Schema describing all declared sensor observation fields produced by a robot.
ObservationSchema: TypeAlias = Schema

# Schema describing all declared single-step actuator command fields accepted by a robot.
ActionSchema: TypeAlias = Schema

# A complete single-tick command mapping field names to their corresponding NumPy arrays.
# Each array must strictly conform to the shape and dtype in ActionSchema.
Action: TypeAlias = Mapping[str, NDArray[Any]]


@dataclass(frozen=True)
class Sample:
    """A sensor measurement paired with its host reception timestamp.

    Attributes:
        value: Array containing the sensor measurement in the robot schema's units,
            ordering, and dtype. Memory must be detached from native receive buffers.
        received_at_ns: Local monotonic clock timestamp in nanoseconds
            (time.monotonic_ns()) captured immediately when the message arrived
            at the host process. This timestamp is used for freshness evaluation
            and is independent of any remote sensor clock or message header stamp.
    """

    value: NDArray[Any]  # Array data in declared layout, units, and dtype.
    received_at_ns: int  # Host time.monotonic_ns() timestamp upon packet arrival.


@dataclass(frozen=True)
class Observation:
    """A complete, consistent snapshot of robot sensor states across all declared fields.

    An Observation aggregates the latest available Sample for every key declared
    in the robot's ObservationSchema. While individual sensor fields may have been
    sampled at slightly different arrival times (e.g., asynchronous cameras and
    high-rate joint encoders), an Observation guarantees completeness: no declared
    field may be missing.

    Note on array immutability:
        Dataclass freezing alone makes the mapping reference immutable, but does not
        prevent in-place modification of contained NumPy array buffers. Robot
        implementations and validation routines must freeze array buffers (flags.writeable=False)
        or detach copies before returning them to downstream consumers.

    Attributes:
        samples: Immutable mapping from observation field names to timestamped Samples.
    """

    samples: Mapping[str, Sample]  # Mapping of all required sensor fields to Samples.


@dataclass(frozen=True)
class ActionChunk:
    """A sequence of ordered single-step action targets predicted by a policy.

    ActionChunk represents the discrete trajectory output of a robot policy for a
    single inference step. Each element in 'actions' is a complete single-step Action
    mapping containing every command field declared in the robot's ActionSchema.

    The rate at which individual actions are consumed and interpolated is governed
    by the deployment Scheduler (e.g., StopAndRefill.action_hz), decoupling the
    model's predicted trajectory horizon from the physical send rate.

    Attributes:
        actions: Non-empty ordered tuple of complete single-step Action mappings.
    """

    actions: tuple[Action, ...]  # Non-empty sequence of complete single-tick targets.


class ObservationNotReady(RuntimeError):
    """Exception raised when an observation cannot be fulfilled.

    Raised by robot backends or snapshot helpers when one or more required sensor
    fields have not yet produced their first sample, or when one or more samples
    exceed the maximum permitted age threshold (max_sample_age_s).
    """
