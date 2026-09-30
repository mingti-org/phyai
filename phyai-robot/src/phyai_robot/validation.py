"""Schema checks and detached snapshots, without implicit unit or dtype changes."""

from __future__ import annotations

import time
from types import MappingProxyType
from typing import Any
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .types import Action, Sample, FeatureSpec, Observation


def validate_schema(schema: Mapping[str, FeatureSpec]) -> None:
    """Reject ambiguous field layouts before opening any hardware connection."""
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
    """Validate the entire mapping before any command can reach a device."""
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
        if not np.isfinite(value).all():
            raise ValueError(f"{key}: contains NaN or infinity")


def freeze_values(values: Action) -> Action:
    """Copy arrays and prevent ordinary in-place mutation by downstream readers.

    This is an ownership convention, not a security boundary. Callers must not
    deliberately re-enable array writes or mutate snapshots after submission.
    """
    copied: dict[str, NDArray[Any]] = {}
    for key, value in values.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{key}: snapshots require numpy.ndarray values")
        copied[key] = np.array(value, copy=True)
        copied[key].flags.writeable = False
    return MappingProxyType(copied)


def freeze_samples(samples: Mapping[str, Sample]) -> Mapping[str, Sample]:
    """Detach sensor buffers without refreshing cached receive timestamps."""
    values: Action = freeze_values(
        {key: sample.value for key, sample in samples.items()}
    )
    return MappingProxyType(
        {
            key: Sample(values[key], sample.received_at_ns)
            for key, sample in samples.items()
        }
    )


def validate_observation(
    observation: Observation, schema: Mapping[str, FeatureSpec]
) -> None:
    """Check values and clock domains; freshness limits belong to Deployment."""
    validate_values({k: s.value for k, s in observation.samples.items()}, schema)
    now_ns: int = time.monotonic_ns()
    for key, sample in observation.samples.items():
        if (
            type(sample.received_at_ns) is not int
            or not 0 <= sample.received_at_ns <= now_ns
        ):
            raise ValueError(f"{key}: invalid local monotonic receive timestamp")
