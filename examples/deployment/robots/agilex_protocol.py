"""AgileX codecs shared with the Python 3.10 ROS worker.

Keep this module free of PhyAI, Torch, ROS and Python 3.12 syntax. The worker
uses an inherited Unix socket, not a network listener or a pickle endpoint.
"""

from __future__ import annotations

import base64
import json
import math
import struct
import time

import numpy as np

VIEWS = ("front", "left", "right")
SIDES = ("left", "right")
JOINT_NAMES = tuple(f"joint{i}" for i in range(7))
MAX_PACKET_BYTES = 32 * 1024 * 1024


def decode_joint_state(message):
    positions = np.asarray(message.position, dtype=np.float64)
    names = list(message.name)
    if len(names) != len(set(names)) or any(name not in names for name in JOINT_NAMES):
        raise ValueError(f"Expected unique joint0..joint6 names, got {names}")
    if len(positions) != len(names):
        raise ValueError("JointState names and positions differ in length")
    return np.array(
        [positions[names.index(name)] for name in JOINT_NAMES], dtype=np.float64
    )


def decode_image(message):
    """Respect ROS row stride; preserve native RGB pixels for the checkpoint processor."""
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


def encode_array(array):
    array = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_array(value):
    dtype = np.dtype(value["dtype"])
    shape = tuple(value["shape"])
    if dtype.kind not in "buif" or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid wire array dtype or shape")
    raw = base64.b64decode(value["data"], validate=True)
    if math.prod(shape) * dtype.itemsize != len(raw):
        raise ValueError("Invalid wire array length")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def send_packet(channel, message):
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError("Worker packet exceeds size limit")
    channel.sendall(struct.pack("!I", len(payload)) + payload)


def receive_packet(channel, timeout_s=None):
    # One deadline covers the entire packet, including large image snapshots.
    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    def exact(size):
        data = bytearray()
        while len(data) < size:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ROS worker response timed out")
                channel.settimeout(remaining)
            else:
                channel.settimeout(None)
            part = channel.recv(size - len(data))
            if not part:
                raise EOFError("ROS worker connection closed")
            data.extend(part)
        return data

    size = struct.unpack("!I", exact(4))[0]
    if size > MAX_PACKET_BYTES:
        raise ValueError("Worker packet exceeds size limit")
    return json.loads(exact(size))
