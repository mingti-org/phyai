"""Schema validation, array freezing, and observation freshness checks.

This module enforces contract boundaries between physical hardware, transport
sessions, policy adapters, and control execution loops:
- Schema validation: Structural verification of FeatureSpec definitions prior to
  opening physical connections or allocating device resources.
- Value validation: Strict runtime verification of array shapes, dtypes, and
  numerical finiteness (no NaNs or infinities in floating-point commands).
- Single-copy freezing: Arrays are frozen once (flags.writeable=False) upon receipt
  or preparation. Already frozen arrays are reused without redundant allocation.
- Freshness auditing: Verification that local host receive ages do not exceed
  stipulated latency budgets, preventing control execution on stale sensor data.
"""

from __future__ import annotations

import time
from types import MappingProxyType
from typing import Any
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .types import Action, Sample, Observation, FeatureSpec, ObservationNotReady


def validate_schema(schema: Mapping[str, FeatureSpec]) -> None:
    """Validate a physical data schema before opening any hardware connection.

    Ensures all keys are non-empty strings, specs are valid FeatureSpec instances,
    shapes consist of positive integers, dtypes belong to supported real/integer/boolean
    NumPy categories, and component names (if present) match vector dimensionality.

    Args:
        schema: Mapping of field names to their FeatureSpec descriptions.

    Raises:
        ValueError: If a field name is empty, shape contains invalid dimensions,
            dtype is unsupported, or component names are malformed/duplicated.
        TypeError: If schema values are not FeatureSpec instances.
    """
    for key, spec in schema.items():
        if not isinstance(key, str) or not key:
            raise ValueError("schema keys must be nonempty strings")
        if not isinstance(spec, FeatureSpec):
            raise TypeError(f"{key}: expected FeatureSpec")
        if not isinstance(spec.shape, tuple) or any(
            type(size) is not int or size <= 0 for size in spec.shape
        ):
            raise ValueError(f"{key}: shape must be a tuple of positive integers")
        if np.dtype(spec.dtype).kind not in "biuf":
            raise ValueError(
                f"{key}: only boolean, integer, and real dtypes are supported"
            )
        if spec.names is not None and (
            not isinstance(spec.names, tuple)
            or len(spec.shape) != 1
            or len(spec.names) != spec.shape[0]
            or any(not isinstance(name, str) or not name for name in spec.names)
            or len(set(spec.names)) != len(spec.names)
        ):
            raise ValueError(f"{key}: names must identify each vector component once")
        for label in (spec.unit, spec.frame):
            if label is not None and (not isinstance(label, str) or not label):
                raise ValueError(f"{key}: unit/frame must be nonempty strings or None")


def validate_values(values: Action, schema: Mapping[str, FeatureSpec]) -> None:
    """Validate a complete mapping of values against their target schema.

    Enforces that:
    1. The set of provided keys matches the schema exactly (no missing or extra fields).
    2. Each value is an instance of numpy.ndarray.
    3. The array shape and dtype match the schema's FeatureSpec specification.
    4. Floating-point arrays contain only finite values (rejection of NaN or +/-inf).

    Args:
        values: Field name to NumPy array mapping (e.g. Action or Observation values).
        schema: Expected schema against which values are validated.

    Raises:
        ValueError: If keys do not match schema, shapes mismatch, or values contain NaN/inf.
        TypeError: If an entry is not an ndarray or dtype mismatches.
    """
    if set(values) != set(schema):
        raise ValueError(
            f"field mismatch: missing={set(schema) - set(values)}, "
            f"extra={set(values) - set(schema)}"
        )
    for key, spec in schema.items():
        value: NDArray[Any] = values[key]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{key}: expected numpy.ndarray")
        if value.shape != spec.shape:
            raise ValueError(f"{key}: expected shape {spec.shape}, got {value.shape}")
        if value.dtype != np.dtype(spec.dtype):
            raise TypeError(f"{key}: expected dtype {spec.dtype}, got {value.dtype}")
        # Numerical integrity: Reject NaN and infinity for floating-point tensors.
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"{key}: contains NaN or infinity")


def _is_frozen(value: NDArray[Any]) -> bool:
    """Check whether an array buffer is immutable and free from external mutation.

    An array is considered frozen if its own writeable flag is False and its base
    memory buffer cannot be mutated through another writeable view.

    Args:
        value: Candidate NumPy ndarray to inspect.

    Returns:
        True if the array owns read-only memory or views an immutable base (e.g. bytes);
        False if any writeable buffer underlies the array.
    """
    array: Any = value
    while isinstance(array, np.ndarray):
        if array.flags.writeable:
            return False
        array = array.base
    return array is None or isinstance(array, bytes)


def freeze(value: NDArray[Any]) -> NDArray[Any]:
    """Ensure an array is read-only, reusing already frozen instances without copying.

    If the array is already frozen, it is returned directly. Otherwise, a contiguous
    copy is allocated, its writeable flag is cleared, and the immutable copy is returned.
    This guarantees zero redundant memory allocations when snapshots traverse
    multiple internal layers (e.g. Backend -> CompositeRobot -> Deployment).

    Args:
        value: NumPy array to freeze.

    Returns:
        An immutable (read-only) NumPy array sharing or containing the same data.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError("snapshots require numpy.ndarray values")
    if _is_frozen(value):
        return value
    copied = np.array(value, copy=True)
    copied.flags.writeable = False
    return copied


def freeze_values(values: Action) -> Action:
    """Freeze all arrays in a mapping and return an immutable MappingProxy.

    Args:
        values: Mapping of field names to NumPy arrays.

    Returns:
        MappingProxyType wrapping a dictionary of frozen read-only arrays.
    """
    return MappingProxyType({key: freeze(value) for key, value in values.items()})


def freeze_samples(samples: Mapping[str, Sample]) -> Mapping[str, Sample]:
    """Detach sensor buffers and return an immutable mapping of timestamped Samples.

    Reuses already frozen arrays within Samples without allocating new memory,
    while preserving the original host monotonic receive timestamps.

    Args:
        samples: Mapping of field names to Sample instances.

    Returns:
        MappingProxyType containing detached, immutable Samples.
    """
    return MappingProxyType(
        {
            key: sample
            if _is_frozen(sample.value)
            else Sample(freeze(sample.value), sample.received_at_ns)
            for key, sample in samples.items()
        }
    )


def validate_observation(
    observation: Observation, schema: Mapping[str, FeatureSpec]
) -> None:
    """Verify structural validity and host clock domain of an Observation.

    Checks that all schema fields are present and valid, and verifies that sample
    reception timestamps belong to the local monotonic clock domain in the past.

    Args:
        observation: Observation instance to validate.
        schema: Expected observation schema.

    Raises:
        ValueError: If schema values mismatch or receive timestamps are negative/in future.
    """
    validate_values({k: s.value for k, s in observation.samples.items()}, schema)
    now_ns: int = time.monotonic_ns()
    for key, sample in observation.samples.items():
        if (
            type(sample.received_at_ns) is not int
            or not 0 <= sample.received_at_ns <= now_ns
        ):
            raise ValueError(f"{key}: invalid local monotonic receive timestamp")


def check_freshness(
    observation: Observation, max_age_s: float, now_ns: int | None = None
) -> None:
    """Enforce sample freshness bounds across all fields in an observation snapshot.

    Compares the host reception time of each field against the current local monotonic
    clock. If any sensor sample has an age exceeding max_age_s (or arrives with a future
    timestamp), an ObservationNotReady exception is raised listing all stale fields.

    Args:
        observation: Complete observation snapshot to check.
        max_age_s: Maximum permitted sample latency in seconds.
        now_ns: Optional reference monotonic timestamp in nanoseconds (default: current time).

    Raises:
        ObservationNotReady: If one or more sensor fields are older than max_age_s or in future.
    """
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    stale = [
        key
        for key, sample in observation.samples.items()
        if not 0 <= now_ns - sample.received_at_ns <= max_age_s * 1e9
    ]
    if stale:
        raise ObservationNotReady(f"stale sensor samples: {stale}")
