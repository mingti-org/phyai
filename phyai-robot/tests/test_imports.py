"""Core imports must work on a robot computer without a model/transport stack."""

import os
import sys
import subprocess
from pathlib import Path


def test_imports_do_not_load_optional_dependencies() -> None:
    # A fresh process avoids contamination by optional backend tests.
    script: str = """
import sys
import phyai_robot
from phyai_robot.backends.mock import MockBackend
from phyai_robot.backends.ros2 import Ros2Backend
from phyai_robot.backends.zmq import ZmqBackend
from phyai_robot.backends.ros2.codecs import decode_nv12_image
from phyai_robot.images import resize_with_pad
from phyai_robot.remote import RemotePolicy, PolicyServer, run_remote
loaded = {'torch', 'phyai', 'rclpy', 'zmq', 'can', 'PIL', 'cv2'} & set(sys.modules)
assert not loaded, loaded
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        # Root pytest can add this package through pythonpath without installing
        # it. Pass the source location explicitly to the fresh interpreter too.
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
