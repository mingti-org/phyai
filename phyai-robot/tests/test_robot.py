"""Exercise composite ownership, mixed sessions, and partial-failure cleanup."""

import numpy as np
import pytest
from phyai_robot import (
    Action,
    FeatureSpec,
    Observation,
    CompositeRobot,
    ObservationNotReady,
)
from numpy.typing import NDArray
from phyai_robot.backends.mock import MockBackend


def make_pair() -> tuple[CompositeRobot, MockBackend, MockBackend]:
    left: MockBackend = MockBackend(
        initial_values={"left": np.zeros(1, dtype=np.float32)},
        action_to_observation={"left_target": "left"},
    )
    right: MockBackend = MockBackend(
        initial_values={"right": np.zeros(1, dtype=np.float32)},
        action_to_observation={"right_target": "right"},
    )
    robot: CompositeRobot = CompositeRobot(
        observation_schema={
            "left": FeatureSpec((1,), "float32"),
            "right": FeatureSpec((1,), "float32"),
        },
        action_schema={
            "left_target": FeatureSpec((1,), "float32"),
            "right_target": FeatureSpec((1,), "float32"),
        },
        backends=[left, right],
    )
    return robot, left, right


def test_mixed_sessions_route_only_owned_fields_and_close() -> None:
    robot, left, right = make_pair()
    robot.connect()
    robot.connect()
    command: Action = {
        "left_target": np.array([1], dtype=np.float32),
        "right_target": np.array([2], dtype=np.float32),
    }
    robot.send_action(command)
    command["left_target"][:] = 100
    observed: Observation = robot.get_observation()
    assert observed.samples["left"].value[0] == 1
    assert observed.samples["right"].value[0] == 2
    assert left.last_action is not None and right.last_action is not None
    assert set(left.last_action) == {"left_target"}
    assert set(right.last_action) == {"right_target"}
    robot.close()
    robot.close()
    assert left.stop_count == right.stop_count == 1
    with pytest.raises(RuntimeError):
        robot.connect()


def test_bad_second_field_does_not_send_first_field() -> None:
    robot, left, right = make_pair()
    robot.connect()
    with pytest.raises(TypeError):
        robot.send_action(
            {
                "left_target": np.ones(1, dtype=np.float32),
                "right_target": np.ones(1, dtype=np.float64),
            }
        )
    assert left.write_count == right.write_count == 0
    robot.close()


def test_second_write_failure_stops_both_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, left, right = make_pair()
    robot.connect()

    def fail(action: Action) -> None:
        raise OSError("CAN-like write failure")

    monkeypatch.setattr(right, "write", fail)
    with pytest.raises(OSError, match="write failure"):
        robot.send_action(
            {
                "left_target": np.ones(1, dtype=np.float32),
                "right_target": np.ones(1, dtype=np.float32),
            }
        )
    assert left.write_count == 1
    assert left.stop_count == right.stop_count == 1
    with pytest.raises(RuntimeError, match="stopped"):
        robot.send_action({})
    robot.close()


def test_connect_failure_closes_partial_and_prior_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, left, right = make_pair()

    def fail() -> None:
        raise OSError("connect failed")

    monkeypatch.setattr(right, "connect", fail)
    with pytest.raises(OSError, match="connect failed"):
        robot.connect()
    assert left.stop_count == right.stop_count == 1
    with pytest.raises(RuntimeError):
        left.read()


def test_stop_attempts_every_session_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, left, right = make_pair()
    robot.connect()

    def fail() -> None:
        raise OSError("stop failed")

    monkeypatch.setattr(right, "stop", fail)
    with pytest.raises(ExceptionGroup):
        robot.stop()
    assert left.stop_count == 1
    # Restore the device hook so test cleanup can release its state.
    monkeypatch.undo()
    robot.close()


def test_schema_ownership_rejected_at_construction(vector: NDArray[np.float32]) -> None:
    backend: MockBackend = MockBackend(
        initial_values={"q": vector}, action_to_observation={"target": "q"}
    )
    with pytest.raises(ValueError, match="twice"):
        CompositeRobot(
            observation_schema={}, action_schema={}, backends=[backend, backend]
        )
    with pytest.raises(ValueError, match="cover"):
        CompositeRobot(observation_schema={}, action_schema={}, backends=[backend])


def test_missing_first_sample_is_not_zero_filled(
    robot_and_backend: tuple[CompositeRobot, MockBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot, backend = robot_and_backend
    robot.connect()
    monkeypatch.setattr(backend, "read", dict)
    with pytest.raises(ObservationNotReady):
        robot.get_observation()
    robot.close()


def test_guard_runs_before_any_write(vector: NDArray[np.float32]) -> None:
    backend: MockBackend = MockBackend(
        initial_values={"q": vector}, action_to_observation={"target": "q"}
    )

    def guard(action: Action) -> None:
        if np.any(action["target"] > 1):
            raise ValueError("outside physical limits")

    robot: CompositeRobot = CompositeRobot(
        observation_schema={"q": FeatureSpec((2,), "float32")},
        action_schema={"target": FeatureSpec((2,), "float32")},
        backends=[backend],
        action_guard=guard,
    )
    robot.connect()
    with pytest.raises(ValueError, match="limits"):
        robot.send_action({"target": np.full(2, 2, dtype=np.float32)})
    assert backend.write_count == 0
    robot.close()
