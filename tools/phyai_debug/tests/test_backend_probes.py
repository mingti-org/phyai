"""Reuse the upstream fake ROS transport, with no physical ROS context."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.phyai_debug.capture import Session
from tools.phyai_debug.compat import Probes
from phyai_robot.backends.ros2 import Ros2Backend, RosObservation, RosCommand


@pytest.fixture
def fake_ros(monkeypatch):
    file = Path(__file__).resolve().parents[3] / "phyai-robot/tests/conftest.py"
    spec = importlib.util.spec_from_file_location("debug_fake_ros", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FakeRos(monkeypatch)


class Sink:
    def __init__(self):
        self.rows = []

    def emit(self, kind, **fields):
        self.rows.append({"kind": kind, **fields})
        return len(self.rows)


def test_probes_preserve_callback_stamp_values_and_publish_once(fake_ros):
    sink = Sink()
    session = Session(sink, 0.5)

    def encode(value):
        msg = fake_ros.JointState()
        msg.position = value.tolist()
        return msg

    backend = Ros2Backend(
        observations={
            "q": RosObservation(
                "/feedback", fake_ros.JointState, lambda m: np.array(m.position)
            )
        },
        actions={"q": RosCommand("/command", fake_ros.JointState, encode, stamp=True)},
        on_stop=lambda b: None,
    )
    original = backend._cache
    with Probes(session) as probes:
        probes.backend(backend)
        backend.connect()
        try:
            message = fake_ros.JointState()
            message.header.stamp = SimpleNamespace(sec=12, nanosec=34)
            message.position = [1.0, 2.0]
            fake_ros.feed("/feedback", message)
            sample = backend.read()["q"]
            np.testing.assert_array_equal(sample.value, [1, 2])
            meta = session.sample_metadata("q", sample)
            assert meta["source_stamp_ns"] == 12_000_000_034
            assert (
                sample.received_at_ns
                <= meta["callback_probe_ns"]
                <= meta["cache_ready_at_ns"]
            )
            session.command = {"command_id": 9, "request_id": 3}
            backend.write({"q": np.array([0.3, 0.4])})
            assert len(fake_ros.publications) == 1
            assert fake_ros.publications[0][1].position == [0.3, 0.4]
            row = next(row for row in sink.rows if row["kind"] == "ros_publish")
            assert row["command_id"] == 9 and row["status"] == "success"
        finally:
            backend.close()
    assert backend._cache == original


def test_decode_error_is_preserved_and_recorded(fake_ros):
    sink = Sink()
    session = Session(sink, 0.5)

    def broken(message):
        raise ValueError("bad camera frame")

    backend = Ros2Backend(
        observations={
            "camera": RosObservation(
                "/camera", fake_ros.JointState, broken, recover_on_next_message=True
            )
        },
        actions={},
    )
    with Probes(session) as probes:
        probes.backend(backend)
        backend.connect()
        try:
            fake_ros.feed("/camera", fake_ros.JointState())
            with pytest.raises(Exception, match="bad camera frame"):
                backend.read()
            assert any(row.get("status") == "decode_error" for row in sink.rows)
        finally:
            backend.close()
