"""Tianji layout, codecs, calibration and controller order, without ROS."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
from phyai_robot import Sample, Observation, InferenceContext
from phyai_robot.remote import PolicySpec

from examples.deployment.robots import tianji
from examples.deployment.robots.tianji_codecs import (
    encode_arm,
    encode_gripper,
    decode_arm_positions,
    decode_gripper_feedback,
)

SPEC = PolicySpec({}, {}, {"image_size": 8})


def observation(config):
    now = time.monotonic_ns()
    samples = {}
    for key, spec in tianji.observation_schema(config, 8).items():
        value = np.zeros(spec.shape, dtype=spec.dtype)
        samples[key] = Sample(value, now)
    samples["joint_position_left"] = Sample(np.arange(7, dtype=np.float64), now)
    samples["joint_position_right"] = Sample(np.arange(7, 14, dtype=np.float64), now)
    samples["gripper_feedback_left"] = Sample(
        np.array([0.4, 1, 1, 1, 1], dtype=np.float32), now
    )
    samples["gripper_feedback_right"] = Sample(
        np.array([-0.08, 1, 1, 1, 1], dtype=np.float32), now
    )
    return Observation(samples)


def test_state_layout_and_gripper_calibration_match_the_dataset():
    config = tianji.TianjiConfig()
    adapter = tianji.make_adapter(config, tianji.observation_schema(config, 8))
    request = adapter.to_request(observation(config), InferenceContext())
    # Feedback is float32; calibration runs in float64 like the dataset export.
    grip = [float(np.float32(value)) / 1.6 for value in (0.4, -0.08)]
    expected = np.concatenate(
        [np.arange(7), [grip[0]], np.arange(7, 14), [grip[1]]]
    ).astype(np.float32)
    np.testing.assert_array_equal(request["state"], expected)  # Not clipped.
    assert set(request) == {
        "state",
        "image.head_left",
        "image.left_wrist",
        "image.right_wrist",
    }


def test_actions_follow_dataset_order_and_clamp_grippers():
    config = tianji.TianjiConfig()
    adapter = tianji.make_adapter(config, tianji.observation_schema(config, 8))
    row = np.concatenate([np.arange(7), [1.4], np.arange(7, 14), [-0.2]])
    chunk = adapter.to_actions({"actions": row[None]}, observation(config))
    action = chunk.actions[0]
    np.testing.assert_array_equal(action["joint_position_left"], np.arange(7))
    np.testing.assert_array_equal(action["joint_position_right"], np.arange(7, 14))
    assert action["gripper_left"][0] == 1.0 and action["gripper_right"][0] == 0.0
    assert action["gripper_left"].dtype == np.float32
    tianji.check_gripper_range(action)
    with pytest.raises(ValueError, match="gripper_left"):
        tianji.check_gripper_range({**action, "gripper_left": np.array([1.5])})


def test_codecs_split_both_arms_from_one_message():
    message = SimpleNamespace(arm_positions=list(range(14)))
    fields = decode_arm_positions(message)
    np.testing.assert_array_equal(fields["joint_position_left"], np.arange(7))
    np.testing.assert_array_equal(fields["joint_position_right"], np.arange(7, 14))
    with pytest.raises(ValueError):
        decode_arm_positions(SimpleNamespace(arm_positions=[0.0] * 13))
    feedback = decode_gripper_feedback(SimpleNamespace(data=[1, 2, 3, 4, 5]))
    assert feedback.dtype == np.float32 and feedback.shape == (5,)
    arm = encode_arm(np.arange(7.0), SimpleNamespace)
    assert arm.positions == list(np.arange(7.0))
    assert encode_gripper(np.array([0.25], np.float32), SimpleNamespace).data == 0.25


def test_build_binds_topics_and_decodes_the_mosaic_on_one_worker(ros_messages):
    config = tianji.TianjiConfig(domain_id=24)
    config.wrench_topics = {"left": "/tj/info/wrench_left"}
    embodiment = tianji.build(config, SPEC)
    cameras, command = embodiment.robot.backends
    fields = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    assert cameras.observation_keys == set(fields) and not cameras.action_keys
    binding = cameras._observations[fields]
    assert binding.topic == "/quad_tile/compressed_undistorted"
    assert binding.message_type is ros_messages["CompressedImage"]
    assert command.action_keys == set(tianji.ACTION_SCHEMA)
    assert {"joint_position_left", "joint_position_right"} <= command.observation_keys
    assert "wrench_left" in command.observation_keys
    assert "eef_left" not in command.observation_keys
    assert command._stop_timeout_s == config.service_timeout_s + 0.5
    assert all(b._domain_id == 24 for b in embodiment.robot.backends)
    assert embodiment.robot.observation_schema["head_camera"].shape == (8, 8, 3)
    feedback = command._observations[("joint_position_left", "joint_position_right")]
    assert feedback.topic == "/tj/info/joint_feedback"
    for side, suffix in (("left", "L"), ("right", "R")):
        output = command._actions[f"gripper_{side}"]
        assert output.topic == f"/tj/control/user/gripperValue{suffix}"
        assert output.message_type is ros_messages["Float32"]


def test_mosaic_tiles_are_stretched_like_training_then_letterboxed(ros_messages):
    config = tianji.TianjiConfig(camera_size=[540, 960])
    embodiment = tianji.build(config, PolicySpec({}, {}, {"image_size": 224}))
    cameras = embodiment.robot.backends[0]
    mosaic = next(iter(cameras._observations.values())).decode
    frame = np.zeros((1488, 1920, 3), dtype=np.uint8)
    frame[:744, :960] = 50  # head_left tile
    frame[744:, :960] = 100  # left wrist tile
    frame[744:, 960:] = 150  # right wrist tile
    mosaic.decoder = lambda message: frame
    tiles = mosaic(object())
    assert {k: int(v.max()) for k, v in tiles.items()} == {
        "head_camera": 50,
        "left_wrist_camera": 100,
        "right_wrist_camera": 150,
    }
    rows = np.where(tiles["head_camera"].max(axis=(1, 2)) > 0)[0]
    # 960x540 letterboxed into 224x224 keeps 126 centered rows.
    assert tiles["head_camera"].shape == (224, 224, 3)
    assert (rows.min(), rows.max()) == (49, 174)


def test_raw_source_uses_one_undistorted_topic_per_camera(ros_messages):
    config = tianji.TianjiConfig(camera_source="raw")
    embodiment = tianji.build(config, SPEC)
    cameras = embodiment.robot.backends[:3]
    assert [b.observation_keys for b in cameras] == [
        {"head_camera"},
        {"left_wrist_camera"},
        {"right_wrist_camera"},
    ]
    topics = [next(iter(b._observations.values())).topic for b in cameras]
    assert all(topic.endswith("_undistorted") for topic in topics)


def test_camera_size_null_keeps_the_aspect_ratio():
    transform = tianji.camera_transform(tianji.TianjiConfig(camera_size=None), 224)
    rows = np.where(transform(np.full((744, 960, 3), 9, np.uint8)).max(axis=(1, 2)))[0]
    assert len(rows) == 173  # 224 * 744 / 960


def test_controller_enables_in_order_and_idles_only_after_enabling(ros_messages):
    config = tianji.TianjiConfig(velocity_ratio=30)
    controller = tianji.TianjiController(config)
    calls = []

    class Backend:
        def call_service(self, service_type, name, request, *, timeout_s):
            calls.append((name, getattr(request, "data", None), timeout_s))

    backend = Backend()
    controller.on_stop(backend)
    assert calls == []  # Never enabled: stopping changes nothing.
    controller.backend = backend
    controller.activate()
    controller.on_stop(backend)
    assert calls == [
        ("/tj/control/set_ready", None, 3.0),
        ("/tj/control/set_mode", 3, 3.0),
        ("/tj/control/set_vel_ratio", 30, 3.0),
        ("/tj/control/set_input", 3, 3.0),
        ("/tj/control/set_input", 0, 3.0),
    ]


def test_failed_enable_still_requests_idle_on_stop(ros_messages):
    controller = tianji.TianjiController(tianji.TianjiConfig())
    calls = []

    class Backend:
        def call_service(self, service_type, name, request, *, timeout_s):
            calls.append(name)
            if name.endswith("set_mode"):
                raise RuntimeError("rejected")

    controller.backend = Backend()
    with pytest.raises(RuntimeError):
        controller.activate()
    controller.on_stop(controller.backend)
    assert calls[-1] == "/tj/control/set_input"


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": 2},
        {"velocity_ratio": 0},
        {"domain_id": 300},
        {"gripper_limits_rad": [1.0, 1.0]},
        {"model_cameras": {"head_left": "missing"}},
        {"joint_command_topics": {"left": "/a"}},
        {"qos": {"camera": "sensor_data"}},
        {"camera_source": "webrtc"},
        {"camera_size": [540]},
        {"mosaic_tiles": {"head_camera": [2, 0]}},
        {"mosaic_tiles": {"head_camera": [0, 0], "left_wrist_camera": [0, 0]}},
        {"mosaic_max_rate_hz": 0},
        {"camera_source": "raw", "camera_topics": {"head_camera": "/a"}},
    ],
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        tianji.TianjiConfig(**overrides).validate()
