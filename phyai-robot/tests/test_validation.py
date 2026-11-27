"""Contract failures must be caught before model input or hardware submission."""

import time
from typing import Any
from collections.abc import Mapping

import numpy as np
import pytest
from phyai_robot import Sample, FeatureSpec, Observation, ObservationNotReady
from numpy.typing import NDArray
from phyai_robot.validation import (
    freeze,
    freeze_samples,
    check_freshness,
    validate_schema,
    validate_values,
    validate_observation,
)


@pytest.mark.parametrize(
    "spec",
    [
        FeatureSpec((-1,), "float32"),
        FeatureSpec((0,), "float32"),
        FeatureSpec((True,), "float32"),
        FeatureSpec((1,), "object"),
        FeatureSpec((2,), "float32", names=("a",)),
        FeatureSpec((2,), "float32", names=("a", "a")),
        FeatureSpec((), "float32", names=()),
    ],
)
def test_invalid_schema(spec: FeatureSpec) -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_schema({"value": spec})


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"other": np.zeros(2, dtype=np.float32)},
        {"q": np.zeros(3, dtype=np.float32)},
        {"q": np.zeros(2, dtype=np.float64)},
        {"q": np.array([np.nan, 0], dtype=np.float32)},
        {"q": [0, 1]},
    ],
)
def test_invalid_values(values: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_values(values, {"q": FeatureSpec((2,), "float32")})


def test_snapshot_detaches_array_and_preserves_receive_time(
    vector: NDArray[np.float32],
) -> None:
    stamp: int = time.monotonic_ns()
    snapshot: Mapping[str, Sample] = freeze_samples({"q": Sample(vector, stamp)})
    vector[:] = 8
    np.testing.assert_array_equal(snapshot["q"].value, [0, 0])
    assert snapshot["q"].received_at_ns == stamp
    with pytest.raises(ValueError):
        snapshot["q"].value[:] = 2
    with pytest.raises(TypeError):
        snapshot["other"] = snapshot["q"]  # type: ignore[index]  # Exercise immutable mapping rejection.


def test_timestamp_must_be_local_and_in_the_past(vector: NDArray[np.float32]) -> None:
    observation: Observation = Observation(
        {"q": Sample(vector, time.monotonic_ns() + 10**12)}
    )
    with pytest.raises(ValueError, match="timestamp"):
        validate_observation(observation, {"q": FeatureSpec((2,), "float32")})


def test_snapshot_does_not_silently_convert_lists_to_arrays() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        freeze_samples(
            {"q": Sample([1.0, 2.0], time.monotonic_ns())}  # type: ignore[arg-type]  # Deliberately pass a list.
        )


def test_freeze_copies_only_arrays_that_can_still_change() -> None:
    owned = np.arange(4.0)
    frozen = freeze(owned)
    assert frozen is not owned and not frozen.flags.writeable
    assert freeze(frozen) is frozen
    view = frozen[1:]
    assert freeze(view) is view  # A read-only view of frozen memory stays shared.
    writable_base = np.arange(4.0)
    readonly_view = writable_base[:]
    readonly_view.flags.writeable = False
    assert freeze(readonly_view) is not readonly_view  # Its base can still change.
    immutable = np.frombuffer(b"\x01\x02", dtype=np.uint8)
    assert freeze(immutable) is immutable


def test_check_freshness_reports_every_stale_field() -> None:
    now = time.monotonic_ns()
    observation = Observation(
        {
            "fresh": Sample(np.zeros(1), now),
            "old": Sample(np.zeros(1), now - 2 * 10**9),
            "future": Sample(np.zeros(1), now + 10**9),
        }
    )
    with pytest.raises(ObservationNotReady, match="'old', 'future'"):
        check_freshness(observation, 1.0, now)
    check_freshness(Observation({"fresh": observation.samples["fresh"]}), 1.0, now)
