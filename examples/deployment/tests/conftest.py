"""CPU-only tests for the deployment examples; no ROS, model or GPU is used."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# The examples are a package of the checkout, not an installed distribution.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import omegaconf  # noqa: F401
except ImportError:  # The deployment dependency group is not installed.
    collect_ignore_glob = ["test_*.py"]


class Message(SimpleNamespace):
    """Stand-in for a ROS message or service request class."""

    Request = None

    def __init__(self, **values):
        super().__init__(**values)


@pytest.fixture
def ros_messages(monkeypatch):
    """Install fake message/service modules used by the robot factories."""
    names = {}

    def module(name, **classes):
        obj = types.ModuleType(name)
        obj.__dict__.update(classes)
        monkeypatch.setitem(sys.modules, name, obj)
        return obj

    def message(name):
        cls = type(name, (Message,), {})
        names[name] = cls
        return cls

    def service(name):
        cls = type(name, (), {"Request": type(f"{name}Request", (Message,), {})})
        names[name] = cls
        return cls

    for package, classes in {
        "std_msgs": ["Float32", "Float32MultiArray"],
        "sensor_msgs": ["Image", "CompressedImage", "JointState"],
        "geometry_msgs": ["PoseStamped", "WrenchStamped"],
        "marvin_msgs": ["JointcmdArm", "Jointfeedback"],
    }.items():
        module(package)
        module(f"{package}.msg", **{name: message(name) for name in classes})
    module("std_srvs")
    module("std_srvs.srv", Trigger=service("Trigger"))
    module("marvin_msgs.srv", Int=service("Int"))
    return names
