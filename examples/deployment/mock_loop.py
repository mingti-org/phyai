"""Run a small dual-arm deployment with in-memory feedback and a CPU policy.

Install the editable phyai-robot package, then execute this file. No PhyAI
Engine, GPU, ROS installation, or external transport dependency is required.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import numpy as np
from phyai_robot import (
    Action,
    Policy,
    ActionChunk,
    FeatureSpec,
    Observation,
    PolicyAdapter,
    CompositeRobot,
    RobotDeployment,
    DeploymentOptions,
)
from numpy.typing import NDArray
from phyai_robot.backends.mock import MockBackend

# Both mappings have "left.target_q" and "right.target_q" keys. Each value is
# a float32 array of shape (7,), in radians and in the robot schema joint order.
# A request contains current positions; a prediction contains absolute targets.
type JointPositions = Mapping[str, NDArray[np.float32]]


class SmallStepPolicy:
    """Generate absolute arm targets from the current joint feedback."""

    def predict(self, request: JointPositions) -> JointPositions:
        """Add 0.01 radians to each current joint position in the request."""
        # The request is an application-defined mapping, just as an Engine policy
        # would accept its model's Request class. The core loop does not inspect it.
        return {key: value + np.float32(0.01) for key, value in request.items()}


class DualArmAdapter:
    """Select independent observations and map them to absolute target fields."""

    def to_request(self, observation: Observation) -> JointPositions:
        """Extract joint arrays from the robot's named, timestamped samples.

        observation.samples also contains gripper widths and an RGB camera.
        Each Sample stores its array in value and its receive time separately.
        """
        # The camera remains a valid observation although this toy policy uses
        # only joint positions. There is no requirement to consume every sensor.
        return {
            "left.target_q": observation.samples["left.q"].value.copy(),
            "right.target_q": observation.samples["right.q"].value.copy(),
        }

    def to_actions(
        self, result: JointPositions, observation: Observation
    ) -> ActionChunk:
        """Complete each command with gripper targets, then hold it for 8 ticks.

        result contains absolute arm targets. This adapter does not need the
        paired observation again; a relative-action adapter would use it to
        add predicted deltas to the original measured positions.
        """
        action: Action = {
            **result,
            "left.target_width": np.array([0.02], dtype=np.float32),
            "right.target_width": np.array([0.02], dtype=np.float32),
        }
        return ActionChunk(actions=tuple(action for _ in range(8)), step_period_s=0.05)


def main() -> None:
    """Run a bounded CPU-only deployment using independent in-memory backends.

    No ROS node, checkpoint, or physical robot is used. MockBackend copies target
    values directly into mapped feedback fields, so this checks composition and
    shutdown behavior rather than robot dynamics or real-time performance.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of control commands before shutdown",
    )
    args: argparse.Namespace = parser.parse_args()
    arm_backend: MockBackend = MockBackend(
        initial_values={
            "left.q": np.zeros(7, dtype=np.float32),
            "right.q": np.zeros(7, dtype=np.float32),
            "camera.front.rgb": np.zeros((32, 32, 3), dtype=np.uint8),
        },
        action_to_observation={"left.target_q": "left.q", "right.target_q": "right.q"},
    )
    gripper_backend: MockBackend = MockBackend(
        initial_values={
            "left.width": np.zeros(1, dtype=np.float32),
            "right.width": np.zeros(1, dtype=np.float32),
        },
        action_to_observation={
            "left.target_width": "left.width",
            "right.target_width": "right.width",
        },
    )
    robot: CompositeRobot = CompositeRobot(
        observation_schema={
            "left.q": FeatureSpec((7,), "float32", unit="rad"),
            "right.q": FeatureSpec((7,), "float32", unit="rad"),
            "left.width": FeatureSpec((1,), "float32", unit="m"),
            "right.width": FeatureSpec((1,), "float32", unit="m"),
            "camera.front.rgb": FeatureSpec((32, 32, 3), "uint8"),
        },
        action_schema={
            "left.target_q": FeatureSpec((7,), "float32", unit="rad"),
            "right.target_q": FeatureSpec((7,), "float32", unit="rad"),
            "left.target_width": FeatureSpec((1,), "float32", unit="m"),
            "right.target_width": FeatureSpec((1,), "float32", unit="m"),
        },
        backends=(arm_backend, gripper_backend),
    )
    # These annotations tie adapter output -> policy input -> adapter input.
    policy: Policy[JointPositions, JointPositions] = SmallStepPolicy()
    adapter: PolicyAdapter[JointPositions, JointPositions] = DualArmAdapter()
    deployment: RobotDeployment[JointPositions, JointPositions] = RobotDeployment(
        robot=robot,
        policy=policy,
        adapter=adapter,
        options=DeploymentOptions(
            control_hz=20,
            max_sample_age_s=0.5,
            max_prediction_age_s=2,
            action_hz=20,  # Model targets have a configurable 50ms spacing.
            max_queued_actions=16,
        ),
    )
    # run connects the robot and owns both stop and close on completion/failure.
    # Here action_hz equals control_hz, so each of the eight targets yields one
    # send before the next observation/inference cycle refills the empty queue.
    deployment.run(max_steps=args.steps)
    print(f"Completed {arm_backend.write_count} steps; both backends stopped.")


if __name__ == "__main__":
    main()
