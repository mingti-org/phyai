"""Tianji (Marvin Pro) robot: ROS bindings, model adapter and controller enable.

Architecture & Data Flow:
    1. Sensor Ingestion:
       - Cameras: The default `camera_source="mosaic"` subscribes to the compressed
         H.264 preview stream (/quad_tile/compressed_undistorted) from the multi-camera
         Argus container. A single worker thread decodes the access units via PyAV and
         splits the 2x2 grid into three logical camera views with synchronized timestamps.
       - Joint & Gripper Feedback: Joint feedback (/tj/info/joint_feedback) provides
         14-DoF arm positions, and gripper feedback topics (/info/gripper_feedback_L/R)
         provide motor position readings.
    2. Model Mapping (VectorAdapter):
       - Assembles the 16-DoF state vector: [L_j0..L_j6, L_grip, R_j0..R_j6, R_grip].
       - Normalizes gripper feedback from motor radians [0.0, 1.6] to [0.0, 1.0].
    3. Controller Activation & Safety (TianjiController):
       - Activation sequence: set_ready -> set_mode(3, impedance) -> optional set_vel_ratio
         -> set_input(3, user channel takeover).
       - Stop sequence: On halt or error, calls set_input(0) to safely disengage the user
         channel without disabling motors, homing the arms, or dropping held payloads.
"""

from __future__ import annotations

from typing import Any
from functools import partial
from dataclasses import field, dataclass
from collections.abc import Mapping

import numpy as np
from phyai_robot import (
    Action,
    StateSlot,
    CommandSlot,
    FeatureSpec,
    VectorAdapter,
    CompositeRobot,
)
from phyai_robot.images import resize, resize_with_pad
from phyai_robot.adapters import ACTION_PREFIX_KEY, PREFIX_LENGTH_KEY
from phyai_robot.remote import PolicySpec, Embodiment
from phyai_robot.backends.ros2 import (
    RosCommand,
    Ros2Backend,
    RosObservation,
    image_backends,
)
from phyai_robot.backends.ros2.video import H264Decoder, MosaicDecoder
from phyai_robot.backends.ros2.backend import SERVICE_POLL_MARGIN_S
from phyai_robot.backends.ros2.codecs import (
    POSE_NAMES,
    WRENCH_NAMES,
    decode_pose,
    decode_wrench,
    decode_nv12_image,
)

from .tianji_codecs import (
    ARM_DOF,
    GRIPPER_FEEDBACK_SIZE,
    encode_arm,
    encode_gripper,
    decode_arm_positions,
    decode_gripper_feedback,
)

SIDES = ("left", "right")
MODE_JOINT_POSITION = 1
MODE_JOINT_IMPEDANCE = 3
INPUT_IDLE = 0
INPUT_USER = 3
CAMERA_SOURCES = ("mosaic", "raw")
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, ARM_DOF + 1))
GRIPPER_FEEDBACK_NAMES = (
    "position",
    "velocity",
    "torque",
    "mos_temperature",
    "motor_temperature",
)


def _sides(**topics: str) -> Any:
    return field(default_factory=lambda: dict(topics))


@dataclass
class TianjiConfig:
    """Tianji ROS connection, controller service, and calibration configuration.

    Attributes:
        domain_id: ROS 2 domain ID (must match Apex controller stack, default: 24).
        node_name: Base name for ROS 2 nodes instantiated by this client.
        mode: Controller operating mode (1: Joint Position, 3: Joint Impedance).
        velocity_ratio: Controller velocity scaling percentage [1, 100] (null leaves unchanged).
        service_timeout_s: Synchronous timeout budget for ROS 2 controller service calls.
        io_timeout_s: Single-step command publication timeout.
        gripper_limits_rad: Gripper motor stroke calibration [lower, upper] in radians.
        camera_source: "mosaic" for H.264 compressed stream (~30 fps), "raw" for NV12 topics.
        camera_size: Optional [height, width] stretch target before square letterboxing.
        camera_resize_hw: Alias for camera_size for backward compatibility.
        mosaic_topic: ROS 2 topic name for the compressed H.264 multi-camera mosaic.
        mosaic_grid: [rows, columns] grid shape of the mosaic image.
        mosaic_tiles: Mapping from camera field names to [row, column] cell indices in mosaic.
        mosaic_max_rate_hz: Maximum RGB conversion rate in Hz for mosaic decoder.
        camera_topics: Mapping of camera fields to raw NV12 topics (used when camera_source="raw").
        model_cameras: Mapping from model camera keys to robot observation fields.
        joint_feedback_topic: Dual arm Jointfeedback topic.
        gripper_feedback_topics: Per-side gripper motor Float32MultiArray topics.
        eef_topics: Optional end-effector PoseStamped topics.
        wrench_topics: Optional wrist WrenchStamped topics.
        joint_command_topics: Per-side JointcmdArm command topics.
        gripper_command_topics: Per-side Float32 gripper command topics.
        service_namespace: Controller service topic namespace (default: '/tj/control').
        qos: Mapping of topic categories to QoS profile aliases.
    """

    domain_id: int | None = None
    node_name: str = "phyai_tianji"
    mode: int = MODE_JOINT_IMPEDANCE
    velocity_ratio: int | None = None
    service_timeout_s: float = 3.0
    io_timeout_s: float = 0.1
    gripper_limits_rad: list[float] = field(default_factory=lambda: [0.0, 1.6])
    camera_source: str = "mosaic"
    camera_size: list[int] | None = None
    camera_resize_hw: list[int] | None = None
    mosaic_topic: str = "/quad_tile/compressed_undistorted"
    mosaic_grid: list[int] = field(default_factory=lambda: [2, 2])
    mosaic_tiles: dict[str, list[int]] = field(
        default_factory=lambda: {
            "head_camera": [0, 0],
            "left_wrist_camera": [1, 0],
            "right_wrist_camera": [1, 1],
        }
    )
    mosaic_max_rate_hz: float = 15.0
    camera_topics: dict[str, str] = field(
        default_factory=lambda: {
            "head_camera": "/camera/left_eye/image_nv12_undistorted",
            "left_wrist_camera": "/camera/left_wrist/image_nv12_undistorted",
            "right_wrist_camera": "/camera/right_wrist/image_nv12_undistorted",
        }
    )
    model_cameras: dict[str, str] = field(
        default_factory=lambda: {
            "head_left": "head_camera",
            "left_wrist": "left_wrist_camera",
            "right_wrist": "right_wrist_camera",
        }
    )
    joint_feedback_topic: str = "/tj/info/joint_feedback"
    gripper_feedback_topics: dict[str, str] = _sides(
        left="/info/gripper_feedback_L", right="/info/gripper_feedback_R"
    )
    eef_topics: dict[str, str] = field(default_factory=dict)
    wrench_topics: dict[str, str] = field(default_factory=dict)
    joint_command_topics: dict[str, str] = _sides(
        left="/tj/control/user/joint_cmd_A", right="/tj/control/user/joint_cmd_B"
    )
    # Send VLA targets through the source mux, which owns the final driver topics.
    gripper_command_topics: dict[str, str] = _sides(
        left="/tj/control/user/gripperValueL", right="/tj/control/user/gripperValueR"
    )
    service_namespace: str = "/tj/control"
    qos: dict[str, str] = field(
        default_factory=lambda: {
            "camera": "sensor_data",
            "joint_feedback": "sensor_data",
            "gripper_feedback": "reliable",
            "eef": "reliable",
            "wrench": "sensor_data",
            "joint_command": "sensor_data",
            "gripper_command": "reliable",
        }
    )

    def validate(self) -> None:
        """Reject incompatible settings before creating ROS resources."""
        if self.domain_id is not None and not 0 <= self.domain_id <= 232:
            raise ValueError("robot.domain_id must be in [0, 232] or null")
        if self.mode not in (MODE_JOINT_POSITION, MODE_JOINT_IMPEDANCE):
            raise ValueError("robot.mode must be 1 (joint position) or 3 (impedance)")
        if self.velocity_ratio is not None and not 1 <= self.velocity_ratio <= 100:
            raise ValueError("robot.velocity_ratio must be in [1, 100] or null")
        if not self.node_name.replace("_", "").isalnum():
            raise ValueError("robot.node_name must be a valid ROS node name")
        for name in ("service_timeout_s", "io_timeout_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"robot.{name} must be positive")
        low, high = self.gripper_limits_rad
        if not np.isfinite((low, high)).all() or low >= high:
            raise ValueError("robot.gripper_limits_rad must be finite and increasing")
        if self.camera_source not in CAMERA_SOURCES:
            raise ValueError(f"robot.camera_source must be one of {CAMERA_SOURCES}")

        # Support camera_resize_hw as an alias for camera_size.
        if self.camera_size is None and self.camera_resize_hw is not None:
            self.camera_size = self.camera_resize_hw

        if self.camera_size is not None and (
            len(self.camera_size) != 2 or min(self.camera_size) < 1
        ):
            raise ValueError("robot.camera_size must be [height, width] or null")
        if self.camera_resize_hw is not None and (
            len(self.camera_resize_hw) != 2 or min(self.camera_resize_hw) < 1
        ):
            raise ValueError("robot.camera_resize_hw must be [height, width] or null")

        if len(self.mosaic_grid) != 2 or min(self.mosaic_grid) < 1:
            raise ValueError("robot.mosaic_grid must be [rows, columns]")
        rows, columns = self.mosaic_grid
        tiles = [tuple(tile) for tile in self.mosaic_tiles.values()]
        if len(set(tiles)) != len(tiles) or any(
            len(tile) != 2 or not (0 <= tile[0] < rows and 0 <= tile[1] < columns)
            for tile in tiles
        ):
            raise ValueError("robot.mosaic_tiles must name distinct grid cells")
        if not self.mosaic_max_rate_hz > 0:
            raise ValueError("robot.mosaic_max_rate_hz must be positive")
        if not set(self.model_cameras.values()) <= set(camera_fields(self)):
            raise ValueError(
                f"robot.model_cameras must name {self.camera_source} camera fields"
            )
        for attr in (
            "gripper_feedback_topics",
            "joint_command_topics",
            "gripper_command_topics",
        ):
            if set(getattr(self, attr)) != set(SIDES):
                raise ValueError(f"robot.{attr} must map left and right")
        for attr in ("eef_topics", "wrench_topics"):
            if not set(getattr(self, attr)) <= set(SIDES):
                raise ValueError(f"robot.{attr} keys must be left/right")
        missing = {
            "camera",
            "joint_feedback",
            "gripper_feedback",
            "eef",
            "wrench",
            "joint_command",
            "gripper_command",
        } - set(self.qos)
        if missing:
            raise ValueError(f"robot.qos is missing {sorted(missing)}")


def camera_fields(config: TianjiConfig) -> list[str]:
    """Return observation fields provided by the selected camera source."""
    if config.camera_source == "mosaic":
        return list(config.mosaic_tiles)
    return list(config.camera_topics)


def camera_transform(config: TianjiConfig, image_size: int):
    """Factory creating an image transform pipeline matching training geometry."""
    size = config.camera_size

    def transform(image: np.ndarray) -> np.ndarray:
        if size is not None:
            image = resize(image, int(size[0]), int(size[1]))
        return resize_with_pad(image, image_size)

    return transform


ACTION_SCHEMA = {
    **{
        f"joint_position_{side}": FeatureSpec(
            (ARM_DOF,), "float64", unit="rad", names=JOINT_NAMES
        )
        for side in SIDES
    },
    **{
        f"gripper_{side}": FeatureSpec(
            (1,), "float32", unit="normalized", names=("target_position",)
        )
        for side in SIDES
    },
}


def observation_schema(config: TianjiConfig, image_size: int) -> dict[str, FeatureSpec]:
    """Construct ObservationSchema based on active camera and sensor topic configuration."""
    schema = {
        key: FeatureSpec((image_size, image_size, 3), "uint8")
        for key in camera_fields(config)
    }
    for side in SIDES:
        schema[f"joint_position_{side}"] = ACTION_SCHEMA[f"joint_position_{side}"]
        schema[f"gripper_feedback_{side}"] = FeatureSpec(
            (GRIPPER_FEEDBACK_SIZE,), "float32", names=GRIPPER_FEEDBACK_NAMES
        )
    for side in config.eef_topics:
        schema[f"eef_{side}"] = FeatureSpec((7,), "float64", names=POSE_NAMES)
    for side in config.wrench_topics:
        schema[f"wrench_{side}"] = FeatureSpec((6,), "float64", names=WRENCH_NAMES)
    return schema


def make_adapter(
    config: TianjiConfig,
    observations: dict[str, FeatureSpec],
    *,
    max_prefix_len: int = 0,
):
    """Construct a VectorAdapter binding Tianji physical joints and calibrated grippers."""
    limits = tuple(config.gripper_limits_rad)
    return VectorAdapter(
        observation_schema=observations,
        action_schema=ACTION_SCHEMA,
        cameras=config.model_cameras,
        state=[
            slot
            for side in SIDES
            for slot in (
                StateSlot(f"joint_position_{side}"),
                StateSlot(f"gripper_feedback_{side}", index=(0,), limits=limits),
            )
        ],
        commands=[
            slot
            for side in SIDES
            for slot in (
                CommandSlot(f"joint_position_{side}"),
                CommandSlot(f"gripper_{side}", clip=(0.0, 1.0)),
            )
        ],
        max_prefix_len=max_prefix_len,
    )


def rtc_max_prefix_len(spec: PolicySpec) -> int:
    """Validate RTC metadata against the wire schema before constructing ROS resources."""
    rtc = spec.metadata.get("rtc")
    prefix = spec.inputs.get(ACTION_PREFIX_KEY)
    length = spec.inputs.get(PREFIX_LENGTH_KEY)
    if rtc is None:
        if prefix is not None or length is not None:
            raise ValueError("RTC input fields require rtc metadata")
        return 0
    if not isinstance(rtc, Mapping):
        raise ValueError("rtc metadata must be a mapping")
    maximum = rtc.get("max_prefix_len")
    action_dim = sum(value.shape[0] for value in ACTION_SCHEMA.values())
    if type(maximum) is not int or maximum < 1:
        raise ValueError("rtc.max_prefix_len must be a positive integer")
    if (
        prefix is None
        or prefix.shape != (maximum, action_dim)
        or np.dtype(prefix.dtype) != np.dtype("float32")
        or length is None
        or length.shape != (1,)
        or np.dtype(length.dtype) != np.dtype("int64")
    ):
        raise ValueError("RTC metadata and Tianji input schema do not match")
    return maximum


def check_gripper_range(action: Action) -> None:
    """Action guard: Reject normalized gripper commands outside [0.0, 1.0]."""
    for side in SIDES:
        value = float(action[f"gripper_{side}"][0])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"gripper_{side} target must be in [0, 1], got {value}")


class TianjiController:
    """Manages Marvin Pro controller state transitions via ROS 2 services.

    Services run on the command backend's dedicated worker thread via backend.call_service().
    """

    def __init__(self, config: TianjiConfig) -> None:
        self._config = config
        self._enabled = False
        self.backend: Ros2Backend | None = None

    def _call(self, backend: Ros2Backend, name: str, value: int | None) -> None:
        from std_srvs.srv import Trigger
        from marvin_msgs.srv import Int

        if value is None:
            service_type, request = Trigger, Trigger.Request()
        else:
            service_type, request = Int, Int.Request(data=int(value))
        backend.call_service(
            service_type,
            f"{self._config.service_namespace}/{name}",
            request,
            timeout_s=self._config.service_timeout_s,
        )

    def activate(self) -> None:
        """Execute controller startup sequence: ready -> mode -> vel_ratio -> input=3."""
        assert self.backend is not None
        config = self._config
        self._enabled = True
        self._call(self.backend, "set_ready", None)
        self._call(self.backend, "set_mode", config.mode)
        if config.velocity_ratio is not None:
            self._call(self.backend, "set_vel_ratio", config.velocity_ratio)
        self._call(self.backend, "set_input", INPUT_USER)

    def on_stop(self, backend: Ros2Backend) -> None:
        """Safety callback: Switch input mux back to idle (0) upon stop or shutdown."""
        if self._enabled:
            self._call(backend, "set_input", INPUT_IDLE)


def build(config: TianjiConfig, spec: PolicySpec) -> Embodiment:
    """Construct Tianji CompositeRobot, VectorAdapter, and controller activation hook."""
    max_prefix_len = rtc_max_prefix_len(spec)
    from std_msgs.msg import Float32, Float32MultiArray
    from marvin_msgs.msg import JointcmdArm, Jointfeedback
    from sensor_msgs.msg import Image, CompressedImage
    from geometry_msgs.msg import PoseStamped, WrenchStamped

    image_size = int(spec.metadata["image_size"])
    qos = config.qos
    ros = dict(domain_id=config.domain_id, io_timeout_s=config.io_timeout_s)

    # Decode both left and right arm feedback from a single Jointfeedback topic.
    observations: dict[str | tuple[str, ...], RosObservation] = {
        ("joint_position_left", "joint_position_right"): RosObservation(
            config.joint_feedback_topic,
            Jointfeedback,
            decode_arm_positions,
            qos["joint_feedback"],
        ),
    }
    for side, topic in config.gripper_feedback_topics.items():
        observations[f"gripper_feedback_{side}"] = RosObservation(
            topic, Float32MultiArray, decode_gripper_feedback, qos["gripper_feedback"]
        )
    for side, topic in config.eef_topics.items():
        observations[f"eef_{side}"] = RosObservation(
            topic, PoseStamped, decode_pose, qos["eef"]
        )
    for side, topic in config.wrench_topics.items():
        observations[f"wrench_{side}"] = RosObservation(
            topic, WrenchStamped, decode_wrench, qos["wrench"]
        )

    actions: dict[str | tuple[str, ...], RosCommand] = {}
    for side in SIDES:
        actions[f"joint_position_{side}"] = RosCommand(
            config.joint_command_topics[side],
            JointcmdArm,
            partial(encode_arm, message_type=JointcmdArm),
            qos["joint_command"],
            stamp=True,
        )
        actions[f"gripper_{side}"] = RosCommand(
            config.gripper_command_topics[side],
            Float32,
            partial(encode_gripper, message_type=Float32),
            qos["gripper_command"],
        )

    controller = TianjiController(config)
    controller.backend = Ros2Backend(
        observations=observations,
        actions=actions,
        on_stop=controller.on_stop,
        node_name=config.node_name,
        stop_timeout_s=config.service_timeout_s + SERVICE_POLL_MARGIN_S,
        **ros,
    )

    transform = camera_transform(config, image_size)
    if config.camera_source == "mosaic":
        # Decodes the multi-camera H.264 stream on a dedicated worker with frame_num protection.
        mosaic = MosaicDecoder(
            tiles={key: tuple(tile) for key, tile in config.mosaic_tiles.items()},
            grid=tuple(config.mosaic_grid),
            transform=transform,
            decoder=H264Decoder(max_rate_hz=config.mosaic_max_rate_hz),
        )
        cameras = [
            Ros2Backend(
                observations={
                    mosaic.fields: RosObservation(
                        config.mosaic_topic, CompressedImage, mosaic, qos["camera"]
                    )
                },
                actions={},
                node_name=f"{config.node_name}_cameras",
                **ros,
            )
        ]
    else:
        # Fallback: Allocate one worker per uncompressed NV12 camera topic.
        cameras = image_backends(
            config.camera_topics,
            message_type=Image,
            decode=lambda message: transform(decode_nv12_image(message)),
            node_name=config.node_name,
            qos=qos["camera"],
            **ros,
        )

    schema = observation_schema(config, image_size)
    robot = CompositeRobot(
        observation_schema=schema,
        action_schema=ACTION_SCHEMA,
        backends=[*cameras, controller.backend],
        action_guard=check_gripper_range,
    )
    return Embodiment(
        robot,
        make_adapter(config, schema, max_prefix_len=max_prefix_len),
        controller.activate,
    )
