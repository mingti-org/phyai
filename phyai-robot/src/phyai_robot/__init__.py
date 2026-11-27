"""Optional robot support; importing this package loads no GPU or transport SDK."""

from .robot import Robot, CompositeRobot, take_snapshot, wait_for_observation
from .types import (
    Action,
    Sample,
    Schema,
    ActionChunk,
    FeatureSpec,
    Observation,
    ActionSchema,
    ObservationSchema,
    ObservationNotReady,
)
from .policy import Policy, EnginePolicy, PolicyAdapter
from .adapters import StateSlot, CommandSlot, VectorAdapter
from .scheduling import Scheduler, StopAndRefill, InferenceContext, RealTimeChunking
from .deployment import RobotDeployment, DeploymentOptions

__all__ = [
    "Action",
    "ActionChunk",
    "ActionSchema",
    "CommandSlot",
    "CompositeRobot",
    "DeploymentOptions",
    "EnginePolicy",
    "FeatureSpec",
    "InferenceContext",
    "Observation",
    "ObservationNotReady",
    "ObservationSchema",
    "Policy",
    "PolicyAdapter",
    "RealTimeChunking",
    "Robot",
    "RobotDeployment",
    "Sample",
    "Scheduler",
    "Schema",
    "StateSlot",
    "StopAndRefill",
    "VectorAdapter",
    "take_snapshot",
    "wait_for_observation",
]
