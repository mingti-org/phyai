"""AgileX layout and JointState codecs, without ROS."""

import time
from types import SimpleNamespace

import numpy as np
from phyai_robot import Sample, Observation, InferenceContext
from phyai_robot.remote import PolicySpec

from examples.deployment.robots import agilex
from examples.deployment.robots.agilex_codecs import JOINT_NAMES, decode_arm, encode_arm


def test_joint_state_round_trip_keeps_order_and_units():
    message = SimpleNamespace(
        name=list(JOINT_NAMES)[::-1], position=list(range(7))[::-1]
    )
    fields = decode_arm(message, "left")
    np.testing.assert_array_equal(fields["joint_position_left"], np.arange(6))
    np.testing.assert_array_equal(fields["gripper_left"], [6])
    encoded = encode_arm(fields, "left", SimpleNamespace)
    assert encoded.name == list(JOINT_NAMES)
    assert encoded.position == list(map(float, range(7)))


def test_state_and_actions_use_left_then_right_with_grippers():
    config = agilex.AgilexConfig()
    schema = agilex.observation_schema(config, 4)
    adapter = agilex.make_adapter(config, schema)
    now = time.monotonic_ns()
    samples = {
        key: Sample(np.full(spec.shape, i, dtype=spec.dtype), now)
        for i, (key, spec) in enumerate(schema.items())
    }
    request = adapter.to_request(Observation(samples), InferenceContext())
    expected = [
        samples[k].value
        for k in (
            "joint_position_left",
            "gripper_left",
            "joint_position_right",
            "gripper_right",
        )
    ]
    np.testing.assert_array_equal(request["state"], np.concatenate(expected))
    assert set(request) == {"state", "image.front", "image.left", "image.right"}
    row = np.arange(14, dtype=np.float32)
    action = adapter.to_actions({"actions": row[None]}, Observation(samples)).actions[0]
    np.testing.assert_array_equal(action["joint_position_right"], np.arange(7, 13))
    np.testing.assert_array_equal(action["gripper_right"], [13])


def test_build_uses_configured_domain_and_one_worker_per_camera(ros_messages):
    config = agilex.AgilexConfig(domain_id=3)
    embodiment = agilex.build(config, PolicySpec({}, {}, {"image_size": 4}))
    backends = embodiment.robot.backends
    assert len(backends) == 4 and all(b._domain_id == 3 for b in backends)
    assert embodiment.activate is None
