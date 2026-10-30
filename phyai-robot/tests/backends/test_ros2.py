"""Test the ROS backend contract with in-memory topics, never real ROS contexts."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from phyai_robot.backends.ros2 import (
    RosCommand,
    Ros2Backend,
    RosObservation,
    resolve_qos,
    image_backends,
)
from phyai_robot.backends.ros2.codecs import (
    decode_pose,
    decode_wrench,
    decode_rgb_image,
    decode_nv12_image,
    decode_joint_positions,
)


def test_grouped_topics_preserve_fields_stamp_order_and_ownership(fake_ros):
    fields = ("joints", "gripper")

    def encode(values):
        msg = fake_ros.JointState()
        msg.position = np.concatenate([values[k] for k in fields]).tolist()
        return msg

    backend = Ros2Backend(
        observations={
            fields: RosObservation(
                "/feedback",
                fake_ros.JointState,
                lambda m: {
                    "joints": np.array(m.position[:2]),
                    "gripper": np.array(m.position[2:]),
                },
            )
        },
        actions={
            fields: RosCommand("/first", fake_ros.JointState, encode, stamp=True),
            ("other",): RosCommand(
                "/second",
                fake_ros.JointState,
                lambda v: fake_ros.JointState(),
                stamp=True,
            ),
        },
        on_stop=lambda node: None,
        domain_id=37,
    )
    try:
        backend.connect()
        msg = fake_ros.JointState()
        msg.position = [1.0, 2.0, 3.0]
        fake_ros.feed("/feedback", msg)
        observed = backend.read()
        assert set(observed) == set(fields)
        assert observed["joints"].received_at_ns == observed["gripper"].received_at_ns
        np.testing.assert_array_equal(observed["joints"].value, [1.0, 2.0])
        msg.position[0] = 99.0
        np.testing.assert_array_equal(backend.read()["joints"].value, [1.0, 2.0])
        backend.write(
            {
                "other": np.array([9.0]),
                "gripper": np.array([0.2]),
                "joints": np.array([0.3, 0.4]),
            }
        )
        assert [topic for topic, _ in fake_ros.publications] == ["/first", "/second"]
        assert fake_ros.publications[0][1].position == [0.3, 0.4, 0.2]
        assert [msg.header.stamp for _, msg in fake_ros.publications] == [1, 2]
        assert fake_ros.contexts[0].domain_id == 37
        backend.stop()
        with pytest.raises(RuntimeError, match="stopped"):
            backend.write(
                {"other": np.zeros(1), "gripper": np.zeros(1), "joints": np.zeros(2)}
            )
        assert len(fake_ros.publications) == 2
    finally:
        backend.close()
    assert all(thread != threading.get_ident() for _, _, thread in fake_ros.events)
    assert len({thread for _, _, thread in fake_ros.events}) == 1


def test_single_field_legacy_order_and_no_stamp(fake_ros):
    backend = Ros2Backend(
        observations={},
        actions={
            key: RosCommand(
                "/" + key, fake_ros.JointState, lambda value: fake_ros.JointState()
            )
            for key in ("a", "b")
        },
        on_stop=lambda node: None,
    )
    try:
        backend.connect()
        backend.write({"b": np.zeros(1), "a": np.ones(1)})
        assert [topic for topic, _ in fake_ros.publications] == ["/b", "/a"]
        assert all(message.header.stamp is None for _, message in fake_ros.publications)
        assert fake_ros.contexts[0].domain_id is None
    finally:
        backend.close()


@pytest.mark.parametrize(
    "bindings",
    [{(): None}, {("a", "a"): None}, {("a", "b"): None, "b": None}, {("a", ""): None}],
)
def test_reject_ambiguous_field_bindings(bindings):
    with pytest.raises(ValueError):
        Ros2Backend(observations=bindings, actions={})


def test_recoverable_decode_error_clears_only_after_valid_message(fake_ros):
    def decode(message):
        if message.bad:
            raise ValueError("bad frame")
        return np.array([message.value])

    backend = Ros2Backend(
        observations={
            "camera": RosObservation(
                "/camera", object, decode, recover_on_next_message=True
            )
        },
        actions={},
    )
    try:
        backend.connect()
        fake_ros.feed("/camera", SimpleNamespace(bad=True))
        with pytest.raises(RuntimeError, match="bad frame"):
            backend.read()
        fake_ros.feed("/camera", SimpleNamespace(bad=False, value=7))
        np.testing.assert_array_equal(backend.read()["camera"].value, [7])
    finally:
        backend.close()


def test_bad_group_cannot_cache_partial_observation(fake_ros):
    backend = Ros2Backend(
        observations={
            ("a", "b"): RosObservation(
                "/s", object, lambda msg: msg, recover_on_next_message=True
            )
        },
        actions={},
    )
    try:
        backend.connect()
        fake_ros.feed("/s", {"a": np.array([1])})
        with pytest.raises(RuntimeError, match="fields do not match"):
            backend.read()
        fake_ros.feed("/s", {"a": np.array([2]), "b": np.array([3])})
        assert set(backend.read()) == {"a", "b"}
    finally:
        backend.close()


def test_invalid_command_type_publishes_nothing(fake_ros):
    backend = Ros2Backend(
        observations={},
        actions={
            ("a",): RosCommand(
                "/a", fake_ros.JointState, lambda values: fake_ros.JointState()
            ),
            ("b",): RosCommand("/b", fake_ros.JointState, lambda values: object()),
        },
        on_stop=lambda node: None,
    )
    backend.connect()
    with pytest.raises(TypeError, match="message type"):
        backend.write({"a": np.zeros(1), "b": np.zeros(1)})
    with pytest.raises(RuntimeError, match="transport worker failed"):
        backend.close()
    assert not fake_ros.publications


@pytest.mark.parametrize(
    "encoding,data,expected",
    [
        ("rgb8", [1, 2, 3, 4, 5, 6, 0, 0], [1, 2, 3, 4, 5, 6]),
        ("bgr8", [1, 2, 3, 4, 5, 6, 0, 0], [3, 2, 1, 6, 5, 4]),
        ("rgba8", [1, 2, 3, 99, 4, 5, 6, 99], [1, 2, 3, 4, 5, 6]),
        ("bgra8", [1, 2, 3, 99, 4, 5, 6, 99], [3, 2, 1, 6, 5, 4]),
        ("mono8", [1, 2, 0, 0], [1, 1, 1, 2, 2, 2]),
    ],
)
def test_rgb_image_row_padding_and_encoding(encoding, data, expected):
    result = decode_rgb_image(
        SimpleNamespace(
            encoding=encoding, data=bytes(data), height=1, width=2, step=len(data)
        )
    )
    np.testing.assert_array_equal(
        result.reshape(-1), np.array(expected, dtype=np.uint8)
    )
    assert (
        result.shape == (1, 2, 3)
        and result.dtype == np.uint8
        and result.flags.c_contiguous
    )


def test_joint_positions_follow_requested_names():
    result = decode_joint_positions(
        SimpleNamespace(name=["b", "a", "extra"], position=[2, 1, 3]), ("a", "b")
    )
    np.testing.assert_array_equal(result, [1.0, 2.0])
    assert result.dtype == np.float64
    with pytest.raises(ValueError):
        decode_joint_positions(
            SimpleNamespace(name=["a", "a"], position=[1, 2]), ("a",)
        )


def test_multiple_bindings_can_share_one_topic(fake_ros):
    backend = Ros2Backend(
        observations={
            key: RosObservation(
                "/shared", object, lambda message, key=key: message[key]
            )
            for key in ("a", "b")
        },
        actions={},
    )
    try:
        backend.connect()
        fake_ros.feed("/shared", {"a": np.array([1.0]), "b": np.array([2.0])})
        result = backend.read()
        np.testing.assert_array_equal(result["a"].value, [1.0])
        np.testing.assert_array_equal(result["b"].value, [2.0])
    finally:
        backend.close()


def test_default_decode_error_remains_fatal(fake_ros):
    def fail(message):
        raise ValueError("bad decode")

    backend = Ros2Backend(
        observations={"q": RosObservation("/q", object, fail)}, actions={}
    )
    backend.connect()
    fake_ros.feed("/q", object())
    with pytest.raises(RuntimeError, match="transport worker failed"):
        backend.close()
    assert not fake_ros.publications


def _request(data=None):
    return SimpleNamespace(data=data)


def test_service_calls_run_on_the_worker_and_check_success(fake_ros):
    fake_ros.services["/svc/ok"] = lambda request: SimpleNamespace(success=True)
    fake_ros.services["/svc/no"] = lambda request: SimpleNamespace(
        success=False, message="busy"
    )
    backend = Ros2Backend(observations={}, actions={})
    try:
        backend.connect()
        backend.call_service(object, "/svc/ok", _request(3), timeout_s=1)
        assert fake_ros.calls[0][:2] == ("/svc/ok", 3)
        assert fake_ros.calls[0][2] != threading.get_ident()
        with pytest.raises(RuntimeError, match="rejected the request: busy"):
            backend.call_service(object, "/svc/no", _request(), timeout_s=1)
        with pytest.raises(TimeoutError, match="not available"):
            backend.call_service(object, "/svc/missing", _request(), timeout_s=0.05)
        # A rejected or unavailable service does not fault the session.
        backend.call_service(object, "/svc/ok", _request(4), timeout_s=1)
    finally:
        backend.close()


def test_stop_hook_has_its_own_timeout_longer_than_io(fake_ros):
    """A controller service slower than io_timeout_s still completes on stop."""
    import time

    def slow_idle(request):
        time.sleep(0.3)
        return SimpleNamespace(success=True)

    fake_ros.services["/tj/control/set_input"] = slow_idle

    def on_stop(backend):
        backend.call_service(object, "/tj/control/set_input", _request(0), timeout_s=1)

    backend = Ros2Backend(
        observations={},
        actions={
            "a": RosCommand("/a", fake_ros.JointState, lambda v: fake_ros.JointState())
        },
        on_stop=on_stop,
        io_timeout_s=0.05,
        stop_timeout_s=1.5,
    )
    backend.connect()
    backend.stop()
    backend.close()
    assert [name for name, *_ in fake_ros.calls] == ["/tj/control/set_input"]


def test_stop_reports_timeout_when_stop_hook_exceeds_stop_timeout(fake_ros):
    import time

    def slow_idle(request):
        time.sleep(0.4)
        return SimpleNamespace(success=True)

    fake_ros.services["/idle"] = slow_idle
    backend = Ros2Backend(
        observations={},
        actions={
            "a": RosCommand("/a", fake_ros.JointState, lambda v: fake_ros.JointState())
        },
        on_stop=lambda b: b.call_service(object, "/idle", _request(), timeout_s=1),
        io_timeout_s=0.05,
        stop_timeout_s=0.1,
    )
    backend.connect()
    with pytest.raises(TimeoutError, match="stop timed out"):
        backend.stop()
    # The worker still finishes the in-flight request and is not asked twice.
    with pytest.raises(RuntimeError, match="transport worker failed"):
        backend.close()
    assert len(fake_ros.calls) == 1


def test_named_qos_profiles_are_resolved_on_the_worker(fake_ros):
    backend = Ros2Backend(
        observations={
            "a": RosObservation("/a", object, lambda m: m, "sensor_data"),
            "b": RosObservation("/b", object, lambda m: m, "reliable"),
        },
        actions={
            "c": RosCommand("/c", fake_ros.JointState, lambda v: v, "best_effort")
        },
        on_stop=lambda backend: None,
    )
    try:
        backend.connect()
        node = fake_ros.nodes[0]
        assert node.subscriptions["/a"][0][2] is fake_ros.sensor_qos
        assert node.subscriptions["/b"][0][2].reliability == "reliable"
        assert node.publishers["/c"].qos.reliability == "best_effort"
    finally:
        backend.close()
    with pytest.raises(ValueError, match="unknown QoS"):
        resolve_qos("fast")


def test_image_backends_decode_each_camera_on_its_own_worker(fake_ros):
    backends = image_backends(
        {"front": "/front", "side": "/side"},
        message_type=object,
        decode=lambda message: np.full((2, 2, 3), message, dtype=np.uint8),
        node_name="cams",
    )
    try:
        for backend in backends:
            backend.connect()
        fake_ros.feed("/front", 1)
        fake_ros.feed("/side", 2)
        assert [backend.observation_keys for backend in backends] == [
            {"front"},
            {"side"},
        ]
        assert [node.name for node in fake_ros.nodes] == ["cams_front", "cams_side"]
        assert backends[1].read()["side"].value[0, 0, 0] == 2
    finally:
        for backend in backends:
            backend.close()


def test_nv12_decoder_uses_opencv_and_checks_size():
    cv2 = pytest.importorskip("cv2")
    frame = np.random.default_rng(0).integers(0, 256, (6, 4), dtype=np.uint8)
    message = SimpleNamespace(height=4, width=4, data=frame.tobytes())
    np.testing.assert_array_equal(
        decode_nv12_image(message), cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV12)
    )
    with pytest.raises(ValueError, match="NV12 data"):
        decode_nv12_image(SimpleNamespace(height=4, width=4, data=b"\0" * 10))


def test_pose_and_wrench_decoders_flatten_in_documented_order():
    vector = SimpleNamespace(x=1.0, y=2.0, z=3.0)
    pose = SimpleNamespace(
        pose=SimpleNamespace(
            position=vector, orientation=SimpleNamespace(x=4.0, y=5.0, z=6.0, w=7.0)
        )
    )
    wrench = SimpleNamespace(wrench=SimpleNamespace(force=vector, torque=vector))
    np.testing.assert_array_equal(decode_pose(pose), [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(decode_wrench(wrench), [1, 2, 3, 1, 2, 3])


def test_decoder_returning_none_caches_nothing(fake_ros):
    backend = Ros2Backend(
        observations={"q": RosObservation("/q", object, lambda m: m)}, actions={}
    )
    try:
        backend.connect()
        fake_ros.feed("/q", None)
        assert "q" not in backend.read()
        fake_ros.feed("/q", np.array([1.0]))
        first = backend.read()["q"]
        fake_ros.feed("/q", None)
        assert backend.read()["q"] is first  # The old sample keeps its timestamp.
    finally:
        backend.close()
