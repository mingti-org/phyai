"""AgileX dual Piper robot: ROS bindings, camera decoders, and model adapter.

Architecture & Invariants:
    1. Sensor Ingestion:
       - Cameras: Three independent Orbbec RGB cameras (/camera_f|l|r/color/image_raw).
         Allocates one dedicated worker thread per camera (`image_backends`) to decode and
         letterbox frames to the model's required image size (224x224).
       - Joint & Gripper Feedback: Piper drivers publish `sensor_msgs.msg.JointState` for each
         arm (/puppet/joint_left|right) containing 7 values: 6 revolute arm joint angles (rad)
         followed by 1 gripper position (meters). A grouped tuple binding decodes both fields
         together, ensuring strictly synchronized arrival timestamps.
    2. Model Mapping (VectorAdapter):
       - Assembles the 14-DoF state vector: [L_j0..L_j5, L_grip, R_j0..R_j5, R_grip].
       - Arm angles in radians, gripper positions in meters.
    3. Hardware Commands:
       - Actuator commands publish to /joint_left_states and /joint_right_states at control_hz (200 Hz).
       - Each arm command is a stamped JointState containing 6 arm angles and 1 gripper position.
       - No motor-enable, homing, or service calls are performed; stopping simply ends topic publication.
"""

from __future__ import annotations

from functools import partial
from dataclasses import field, dataclass

from phyai_robot import (
    StateSlot,
    CommandSlot,
    FeatureSpec,
    VectorAdapter,
    CompositeRobot,
)
from phyai_robot.images import resize_with_pad
from phyai_robot.remote import PolicySpec, Embodiment
from phyai_robot.backends.ros2 import (
    RosCommand,
    Ros2Backend,
    RosObservation,
    image_backends,
)
from phyai_robot.backends.ros2.codecs import decode_rgb_image

from .agilex_codecs import decode_arm, encode_arm

SIDES = ("left", "right")


@dataclass
class AgilexConfig:
    """AgileX dual Piper robot ROS 2 connection and topic configuration.

    Attributes:
        domain_id: ROS 2 domain ID matching Piper and Orbbec driver launch (default: 0).
            Set to null to inherit ROS_DOMAIN_ID from the calling shell.
        node_name: Base name for ROS 2 nodes instantiated by this client.
        io_timeout_s: Maximum budget for publishing single-tick commands.
        camera_topics: Mapping from camera field names to native ROS 2 RGB image topics.
        model_cameras: Mapping from model camera keys to robot observation fields.
        state_topics: Per-side JointState feedback topics published by Piper drivers.
        command_topics: Per-side JointState command topics accepted by Piper drivers.
        qos: Mapping of topic categories to QoS profile aliases.
    """

    domain_id: int | None = None
    node_name: str = "phyai_agilex"
    io_timeout_s: float = 0.1
    camera_topics: dict[str, str] = field(
        default_factory=lambda: {
            "front_camera": "/camera_f/color/image_raw",
            "left_camera": "/camera_l/color/image_raw",
            "right_camera": "/camera_r/color/image_raw",
        }
    )
    model_cameras: dict[str, str] = field(
        default_factory=lambda: {
            "front": "front_camera",
            "left": "left_camera",
            "right": "right_camera",
        }
    )
    state_topics: dict[str, str] = field(
        default_factory=lambda: {
            "left": "/puppet/joint_left",
            "right": "/puppet/joint_right",
        }
    )
    command_topics: dict[str, str] = field(
        default_factory=lambda: {
            "left": "/joint_left_states",
            "right": "/joint_right_states",
        }
    )
    qos: dict[str, str] = field(
        default_factory=lambda: {
            "camera": "sensor_data",
            "state": "sensor_data",
            "command": "reliable",
        }
    )

    def validate(self) -> None:
        """Reject malformed or conflicting topic and domain configurations."""
        if self.domain_id is not None and not 0 <= self.domain_id <= 232:
            raise ValueError("robot.domain_id must be in [0, 232] or null")
        if not self.node_name.replace("_", "").isalnum():
            raise ValueError("robot.node_name must be a valid ROS node name")
        if not self.io_timeout_s > 0:
            raise ValueError("robot.io_timeout_s must be positive")
        if not set(self.model_cameras.values()) <= set(self.camera_topics):
            raise ValueError("robot.model_cameras must name camera_topics fields")
        for attr in ("state_topics", "command_topics"):
            if set(getattr(self, attr)) != set(SIDES):
                raise ValueError(f"robot.{attr} must map left and right")
        if set(self.state_topics.values()) & set(self.command_topics.values()):
            raise ValueError("state topics and command topics must differ")
        if {"camera", "state", "command"} - set(self.qos):
            raise ValueError("robot.qos needs camera, state and command")


ACTION_SCHEMA = {
    **{
        f"joint_position_{side}": FeatureSpec((6,), "float64", unit="rad")
        for side in SIDES
    },
    **{f"gripper_{side}": FeatureSpec((1,), "float64", unit="m") for side in SIDES},
}


def observation_schema(config: AgilexConfig, image_size: int) -> dict[str, FeatureSpec]:
    """Construct ObservationSchema based on active camera and Piper feedback topics."""
    return {
        **{
            key: FeatureSpec((image_size, image_size, 3), "uint8")
            for key in config.camera_topics
        },
        **ACTION_SCHEMA,
    }


def make_adapter(config: AgilexConfig, observations: dict[str, FeatureSpec]):
    """Construct a VectorAdapter binding AgileX 14-DoF joints and gripper travel."""
    fields = [
        key for side in SIDES for key in (f"joint_position_{side}", f"gripper_{side}")
    ]
    return VectorAdapter(
        observation_schema=observations,
        action_schema=ACTION_SCHEMA,
        cameras=config.model_cameras,
        state=[StateSlot(key) for key in fields],
        commands=[CommandSlot(key) for key in fields],
    )


def build(config: AgilexConfig, spec: PolicySpec) -> Embodiment:
    """Construct AgileX CompositeRobot, VectorAdapter, and embodiment container."""
    from sensor_msgs.msg import Image, JointState

    image_size = int(spec.metadata["image_size"])
    ros = dict(domain_id=config.domain_id, io_timeout_s=config.io_timeout_s)

    # Grouped tuple bindings for Piper arms: joint angles and gripper travel share one JointState packet.
    states, commands = {}, {}
    for side in SIDES:
        fields = (f"joint_position_{side}", f"gripper_{side}")
        states[fields] = RosObservation(
            config.state_topics[side],
            JointState,
            partial(decode_arm, side=side),
            config.qos["state"],
            recover_on_next_message=True,
        )
        commands[fields] = RosCommand(
            config.command_topics[side],
            JointState,
            partial(encode_arm, side=side, message_type=JointState),
            config.qos["command"],
            stamp=True,
        )

    joints = Ros2Backend(
        observations=states,
        actions=commands,
        # Stopping ends command publication without homing arms or disabling motors.
        on_stop=lambda backend: None,
        node_name=config.node_name,
        **ros,
    )
    cameras = image_backends(
        config.camera_topics,
        message_type=Image,
        decode=lambda message: resize_with_pad(decode_rgb_image(message), image_size),
        node_name=config.node_name,
        qos=config.qos["camera"],
        **ros,
    )
    schema = observation_schema(config, image_size)
    robot = CompositeRobot(
        observation_schema=schema,
        action_schema=ACTION_SCHEMA,
        backends=[*cameras, joints],
    )
    return Embodiment(robot, make_adapter(config, schema))
