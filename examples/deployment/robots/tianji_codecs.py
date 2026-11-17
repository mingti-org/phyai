"""Tianji (Marvin Pro) message conversions; importing them never imports ROS.

Message classes are injected so these conversions can be tested offline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

ARM_DOF = 7
GRIPPER_FEEDBACK_SIZE = 5


def decode_arm_positions(message: Any) -> dict[str, NDArray[np.float64]]:
    """Split Jointfeedback.arm_positions (L1..L7, R1..R7) into both arms.

    Both fields come from one message, so they share a receive timestamp.
    Velocities, efforts, body and head values are not part of the observation.
    """
    positions = np.asarray(message.arm_positions, dtype=np.float64)
    if positions.shape != (2 * ARM_DOF,):
        raise ValueError(f"unexpected arm_positions shape {positions.shape}")
    return {
        "joint_position_left": positions[:ARM_DOF].copy(),
        "joint_position_right": positions[ARM_DOF:].copy(),
    }


def decode_gripper_feedback(message: Any) -> NDArray[np.float32]:
    """Copy [position, velocity, torque, MOS temperature, motor temperature].

    Position stays in motor radians; calibration to [0, 1] is model input
    conversion and happens in the adapter.
    """
    values = np.array(message.data, dtype=np.float32)
    if values.shape != (GRIPPER_FEEDBACK_SIZE,):
        raise ValueError(f"unexpected gripper feedback shape {values.shape}")
    return values


def encode_arm(values: NDArray[Any], message_type: type[Any]) -> Any:
    """Seven absolute radian targets as JointcmdArm; the backend stamps it."""
    message = message_type()
    message.positions = np.asarray(values, dtype=np.float64).tolist()
    return message


def encode_gripper(values: NDArray[Any], message_type: type[Any]) -> Any:
    """One normalized motor-coordinate target as Float32.

    Physical opening direction depends on the installation and calibration;
    preserve the coordinate convention used by the training data.

    The gripper driver maps it to motor travel; the adapter's feedback
    calibration is not applied to commands.
    """
    message = message_type()
    message.data = float(values[0])
    return message
