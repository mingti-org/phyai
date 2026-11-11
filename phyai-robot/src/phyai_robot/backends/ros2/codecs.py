"""Pure decoders for standard ROS messages; importing them never imports ROS.

Every decoder returns arrays that own their data rather than aliasing the
incoming message, and none of them changes physical units.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def decode_joint_positions(
    message: Any, joint_names: Sequence[str]
) -> NDArray[np.float64]:
    """Return detached float64 positions in the requested JointState name order.

    Extra message joints are ignored; duplicate names, missing required names,
    or inconsistent name and position counts are errors.
    """
    positions = np.asarray(message.position, dtype=np.float64)
    names = list(message.name)
    if len(names) != len(set(names)) or any(name not in names for name in joint_names):
        raise ValueError(
            f"Expected unique joint names containing {tuple(joint_names)}, got {names}"
        )
    if len(positions) != len(names):
        raise ValueError("JointState names and positions differ in length")
    return np.array(
        [positions[names.index(name)] for name in joint_names], dtype=np.float64
    )


def decode_rgb_image(message: Any) -> NDArray[np.uint8]:
    """Return contiguous HWC uint8 RGB, respecting ROS Image row padding.

    RGB/BGR/RGBA/BGRA/mono 8-bit encodings are accepted. Alpha is discarded and
    grayscale is repeated across three channels; no resize or normalization is
    applied. NV12 has its own decoder and is never guessed from buffer size.
    """
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    encoding = str(message.encoding).lower()
    if encoding not in channels:
        raise ValueError(f"Unsupported image encoding: {encoding}")
    count = channels[encoding]
    height, width, stride = int(message.height), int(message.width), int(message.step)
    if height <= 0 or width <= 0 or stride < width * count:
        raise ValueError("Invalid image dimensions or row stride")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != height * stride:
        raise ValueError("Image buffer length does not match height * step")
    pixels = raw.reshape(height, stride)[:, : width * count].reshape(
        height, width, count
    )
    if count == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    elif encoding.startswith("bgr"):
        pixels = pixels[:, :, [2, 1, 0]]
    else:
        pixels = pixels[:, :, :3]
    return np.ascontiguousarray(pixels).copy()


def decode_nv12_image(message: Any) -> NDArray[np.uint8]:
    """Convert a tightly packed NV12 Image to HWC uint8 RGB with OpenCV.

    NV12 is a full-resolution Y plane followed by an interleaved half-resolution
    UV plane. OpenCV is required: a silent fallback with different color
    coefficients would feed the model systematically different colors.
    """
    import cv2

    height, width = int(message.height), int(message.width)
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError("NV12 images need positive, even dimensions")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != height * width * 3 // 2:
        raise ValueError(
            f"NV12 data has {raw.size} bytes, expected {height * width * 3 // 2}"
        )
    frame = raw.reshape(height * 3 // 2, width)
    return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV12))


def decode_pose(message: Any) -> NDArray[np.float64]:
    """PoseStamped -> [x, y, z, qx, qy, qz, qw] in the message frame."""
    pose = message.pose
    return np.array(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )


def decode_wrench(message: Any) -> NDArray[np.float64]:
    """WrenchStamped -> [fx, fy, fz, tx, ty, tz], unfiltered."""
    wrench = message.wrench
    return np.array(
        [
            wrench.force.x,
            wrench.force.y,
            wrench.force.z,
            wrench.torque.x,
            wrench.torque.y,
            wrench.torque.z,
        ],
        dtype=np.float64,
    )


POSE_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")
WRENCH_NAMES = ("fx", "fy", "fz", "tx", "ty", "tz")
