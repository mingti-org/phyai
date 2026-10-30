"""AgileX (Piper) JointState conversions; importing them never imports ROS."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from phyai_robot.backends.ros2.codecs import decode_joint_positions

# Six arm joints followed by the gripper, as published by the Piper driver.
JOINT_NAMES = tuple(f"joint{i}" for i in range(7))


def decode_arm(message: Any, side: str) -> dict[str, NDArray[np.float64]]:
    """Split one seven-value feedback into six arm joints and one gripper."""
    values = decode_joint_positions(message, JOINT_NAMES)
    return {
        f"joint_position_{side}": values[:6].copy(),
        f"gripper_{side}": values[6:].copy(),
    }


def encode_arm(values: Any, side: str, message_type: type[Any]) -> Any:
    """Join radians and gripper meters without clipping, scaling or reordering.

    The backend stamps the header just before publishing.
    """
    message = message_type()
    message.name = list(JOINT_NAMES)
    message.position = np.concatenate(
        [values[f"joint_position_{side}"], values[f"gripper_{side}"]]
    ).tolist()
    return message
