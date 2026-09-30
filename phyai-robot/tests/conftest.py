"""CPU-only fixtures; invoke pytest with --confcutdir=phyai-robot."""

import numpy as np
import pytest
from phyai_robot import FeatureSpec, CompositeRobot
from numpy.typing import NDArray
from phyai_robot.backends.mock import MockBackend


@pytest.fixture
def vector() -> NDArray[np.float32]:
    return np.array([0.0, 0.0], dtype=np.float32)


@pytest.fixture
def robot_and_backend(
    vector: NDArray[np.float32],
) -> tuple[CompositeRobot, MockBackend]:
    backend: MockBackend = MockBackend(
        initial_values={"q": vector, "camera": np.zeros((2, 3, 3), dtype=np.uint8)},
        action_to_observation={"target": "q"},
    )
    robot: CompositeRobot = CompositeRobot(
        observation_schema={
            "q": FeatureSpec((2,), "float32"),
            "camera": FeatureSpec((2, 3, 3), "uint8"),
        },
        action_schema={"target": FeatureSpec((2,), "float32")},
        backends=[backend],
    )
    return robot, backend
