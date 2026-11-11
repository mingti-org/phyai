"""Verify the mock's own behavior separately from whole-robot routing."""

from collections.abc import Mapping

import numpy as np
import pytest
from phyai_robot import Sample
from numpy.typing import NDArray
from phyai_robot.backends.mock import MockBackend


def test_feedback_preserves_independent_camera_and_detaches_buffers(
    vector: NDArray[np.float32],
) -> None:
    camera: NDArray[np.uint8] = np.zeros((2, 2, 3), dtype=np.uint8)
    backend: MockBackend = MockBackend(
        initial_values={"q": vector, "camera": camera},
        action_to_observation={"target": "q"},
    )
    backend.connect()
    target: NDArray[np.float32] = np.ones_like(vector)
    backend.write({"target": target})
    target[:] = 9
    observed: Mapping[str, Sample] = backend.read()
    np.testing.assert_array_equal(observed["q"].value, [1, 1])
    np.testing.assert_array_equal(observed["camera"].value, camera)
    backend.stop()
    with pytest.raises(RuntimeError, match="accepting"):
        backend.write({"target": vector})
    backend.close()
    assert backend.stop_count == 1
