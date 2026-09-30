"""Tianji observation/action layout for a converted pi0.5 policy.

The robot does not know the task, tokenizer, dataset normalization, camera order,
or action delta convention. This adapter supplies those model-specific choices
while preserving EEF, wrench, and full gripper feedback in robot observations.
No training-framework runtime or command-line example is imported here.
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
from dataclasses import field, replace, dataclass
from collections.abc import Mapping

import numpy as np
from phyai_robot import Action, ActionChunk, Observation
from numpy.typing import NDArray

from ..policies import Pi05PolicyRequest
from ..robots.tianji import ARM_DOF, GRIPPER_FEEDBACK_SIZE, resize_rgb

# Dataset layout, shared by state and decoded actions:
# [left joints 0:7, left gripper 7, right joints 8:15, right gripper 15].
# Joint coordinates are radians; gripper coordinates are normalized positions.
ACTION_DIM = 16
ACTION_HORIZON = 50
ACTION_PERIOD_S = 0.04  # Source targets are sampled at 25 Hz.
# Default closed/open motor positions for the adapter's affine state mapping.
# These are calibration inputs, not a discovered device limit or a safety bound.
GRIPPER_LIMITS_RAD = (0.0, 1.6)


@dataclass
class TianjiPi05AdapterConfig:
    """Model metadata and the gripper calibration used by this adapter.

    These values describe the model/embodiment mapping, not ROS transport.
    The operator must supply calibration matching the hardware; the adapter
    applies it locally without querying any external status service.
    """

    # None uses unambiguous checkpoint discovery; relative names use checkpoint.
    metadata_file: str | None = None
    # Both hands use this [closed, open] pair. ROS feedback remains in radians;
    # only model state is converted to a fraction of this measured travel range.
    gripper_limits_rad: list[float] = field(
        default_factory=lambda: list(GRIPPER_LIMITS_RAD)
    )

    def validate(self, *, execution_horizon: int) -> None:
        """Check calibration and this model's retained-prefix limit before startup."""
        if not 1 <= execution_horizon <= ACTION_HORIZON:
            raise ValueError(
                f"execution_horizon must be between 1 and {ACTION_HORIZON}"
            )
        if self.metadata_file is not None and not self.metadata_file.strip():
            raise ValueError("adapter.metadata_file must be a non-empty path or null")
        limits = self.gripper_limits_rad
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(
                "adapter.gripper_limits_rad must contain two finite increasing values"
            )


def load_processor(
    checkpoint: Path, metadata_file: str | None = None
) -> tuple[Any, dict[str, Any]]:
    """Load a local tokenizer, normalization statistics, and export metadata.

    An explicit metadata path is relative to the checkpoint unless absolute.
    Otherwise prefer deployment_metadata.json, or require exactly one
    *_metadata.json file. Discovery preserves compatibility with existing
    converted checkpoints without coupling inference to their exporter names.
    Ambiguous exports fail rather than silently choosing a different contract.
    """
    import torch
    from phyai_utils_tools.models.pi05 import PI05Processor
    from phyai_utils_tools.processing.steps import NormalizerStep, UnnormalizerStep

    if metadata_file is not None:
        metadata_path = checkpoint / Path(metadata_file).expanduser()
    elif (checkpoint / "deployment_metadata.json").is_file():
        metadata_path = checkpoint / "deployment_metadata.json"
    else:
        candidates = sorted(checkpoint.glob("*_metadata.json"))
        if len(candidates) != 1:
            raise ValueError("Set adapter.metadata_file: expected one metadata JSON")
        metadata_path = candidates[0]
    metadata = json.loads(metadata_path.read_text())
    validate_metadata(metadata)
    # The export names its statistics "state" and "actions"; the shared
    # processor expects canonical observation.state and action feature names.
    # Pass the local checkpoint as tokenizer source to use its vocabulary.
    stats = json.loads((checkpoint / "norm_stats.json").read_text())["norm_stats"]
    processor = PI05Processor(
        tokenizer_name=str(checkpoint),
        dataset_stats={"observation.state": stats["state"], "action": stats["actions"]},
        action_dim=metadata["action_dim"],
        normalize_pixels=True,
        params_dtype=torch.float32,
    )
    # Dataset statistics are float64. Retain that arithmetic and the exported
    # epsilon; float32 normalization can change tokenized state and predictions.
    for pipeline in (processor.preprocessor, processor.postprocessor):
        pipeline.steps = [
            replace(step, eps=metadata["norm_eps"], dtype=torch.float64)
            if isinstance(step, (NormalizerStep, UnnormalizerStep))
            else step
            for step in pipeline.steps
        ]
    return processor, metadata


def prepare_observation(
    processor: Any, metadata: Mapping[str, Any], payload: Mapping[str, Any]
) -> Any:
    """Apply training-compatible image geometry before the shared processor.

    Image order comes from validated metadata. Pixel scaling, state tokenization,
    and text tokenization are delegated to PhyAI's PI05Processor. Resizing an
    already padded square image leaves it unchanged, so camera-side resizing
    and direct full-resolution observations follow the same numerical contract.
    """
    import torch

    images = []
    for camera in metadata["camera_names"]:
        image = np.asarray(payload[camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{camera} must be an RGB uint8 HWC image")
        # HWC uint8 [0, 255] -> batched CHW float32 [0, 1]. Keep the three
        # cameras as an ordered list: swapping hands is shape-correct but wrong
        # for the learned policy. Further pixel normalization is processor-owned.
        padded = resize_rgb(image, processor.image_size)
        images.append(torch.from_numpy(padded).permute(2, 0, 1)[None].float() / 255.0)
    state = np.asarray(payload["state"], dtype=np.float32)
    if state.shape != (metadata["action_dim"],):
        raise ValueError(
            f"Expected state shape {(metadata['action_dim'],)}, got {state.shape}"
        )
    # Batch dimension is always one. Copy state so preprocessing cannot modify
    # the paired observation that will later anchor the predicted joint deltas.
    # The processor normalizes/tokenizes state together with the instruction.
    return processor.preprocess(
        {
            "images": images,
            "task": str(payload["task"]),
            "state": torch.from_numpy(state.copy())[None],
        }
    )


def absolute_actions(
    processor: Any,
    metadata: Mapping[str, Any],
    normalized: Any,
    state: NDArray[np.float32],
) -> Any:
    """Unnormalize real action coordinates and anchor arm deltas to one snapshot.

    The model pads to its internal action dimension. Slice that padding before
    applying the 16-coordinate dataset statistics. The delta mask anchors only
    arm joints; both gripper targets stay absolute and are clipped by the adapter.
    """
    import torch

    actions = processor.postprocess(
        normalized[..., : metadata["action_dim"]].float().cpu()
    )
    # Broadcast one masked 16-value anchor across the whole horizon. Deltas
    # are relative to the observed pose, not incremental changes to be summed
    # over time. False mask entries (grippers) receive no state offset.
    anchor = torch.as_tensor(np.asarray(state), dtype=actions.dtype)
    mask = torch.tensor(metadata["delta_mask"], dtype=torch.bool)
    return actions + torch.where(mask, anchor, 0)[..., None, :]


def validate_metadata(metadata: Mapping[str, Any]) -> None:
    """Reject exports whose shapes are compatible but whose meaning differs.

    Camera order and delta_mask are as important as the array dimensions. An
    incompatible export must fail before control is enabled, rather than drive
    an arm using another arm's coordinates. action_stride is an export-contract
    check; runtime source-target spacing is controlled by DeploymentOptions.
    norm_eps is retained from the export for matching normalization arithmetic.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a JSON object")
    expected = {
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_stride": 2,
        "camera_names": ["head_left", "left_wrist", "right_wrist"],
        "delta_mask": [True] * 7 + [False] + [True] * 7 + [False],
    }
    eps = metadata.get("norm_eps")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (float, int))
        or not np.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("checkpoint norm_eps must be finite and positive")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Unsupported Tianji checkpoint {key}: {metadata.get(key)!r}"
            )


class TianjiPi05Adapter:
    """Implement PolicyAdapter without changing the CompositeRobot contract.

    State is left arm (7 radians), left gripper, right arm (7 radians), right
    gripper. Only position feedback enters state. All three RGB cameras enter
    the vision encoder. EEF and wrench stay in Observation for future models.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        task: str,
        metadata_file: str | None = None,
        gripper_limits_rad: tuple[float, float] = GRIPPER_LIMITS_RAD,
    ) -> None:
        """Load preprocessing assets and bind one task plus gripper calibration.

        checkpoint contains tokenizer, norm_stats.json, and validated metadata.
        This creates no engine or ROS connection. image_size is exposed so the
        robot factory can cache policy-sized RGB frames before snapshot copies.
        Task text stays in the adapter: changing it does not change Robot's API.
        """
        if not task.strip():
            raise ValueError("task must be non-empty")
        low, high = map(float, gripper_limits_rad)
        if not np.isfinite((low, high)).all() or high <= low:
            raise ValueError("gripper limits must be finite and increasing")
        self._processor, self.metadata = load_processor(checkpoint, metadata_file)
        self.image_size = self._processor.image_size
        self._task = task.strip()
        self._gripper_limits_rad = low, high

    def state_from_observation(self, observation: Observation) -> NDArray[np.float32]:
        """Convert motor-radian feedback to the dataset's normalized gripper state.

        Observation retains all five raw gripper feedback values. Calibration
        maps position to (position-low)/(high-low); do not clip measured state:
        slight negative readings occur at the physical stop and in the dataset.
        Commands, unlike observations, are clamped to the valid [0, 1] range.
        """
        # Only index 0 of each five-value gripper feedback vector is position.
        # Velocity, torque, and temperatures remain available in Observation,
        # but must not leak into this checkpoint's 16-dimensional state vector.
        values: list[NDArray[np.float32]] = []
        low, high = self._gripper_limits_rad
        for side in ("left", "right"):
            joints = np.asarray(
                observation.samples[f"joint_position_{side}"].value, dtype=np.float32
            )
            feedback = observation.samples[f"gripper_feedback_{side}"].value
            if joints.shape != (ARM_DOF,) or feedback.shape != (GRIPPER_FEEDBACK_SIZE,):
                raise ValueError(f"Invalid {side} arm/gripper state shape")
            grip = (float(feedback[0]) - low) / (high - low)
            values.extend((joints, np.asarray([grip], dtype=np.float32)))
        # Append arm then gripper for each side, not both arms then both hands:
        # the checkpoint's normalization statistics depend on this exact order.
        state = np.concatenate(values)
        if not np.isfinite(state).all():
            raise ValueError("Tianji policy state contains a non-finite value")
        return state

    def to_request(self, observation: Observation) -> Pi05PolicyRequest:
        """Use the reference RGB resize/padding, state normalization and tokenizer.

        The configured instruction is injected here, not stored in the Robot. The
        right eye is intentionally absent: head_left uses the left-eye camera.
        """
        # Convert application-level sensor names to checkpoint camera names.
        # EEF/wrench are deliberately not removed from the robot schema: another
        # adapter can consume them without changing transport or control code.
        payload = {
            "head_left": observation.samples["head_camera"].value,
            "left_wrist": observation.samples["left_wrist_camera"].value,
            "right_wrist": observation.samples["right_wrist_camera"].value,
            "state": self.state_from_observation(observation),
            "task": self._task,
        }
        return Pi05PolicyRequest(
            prepare_observation(self._processor, self.metadata, payload)
        )

    def to_actions(self, result: Any, observation: Observation) -> ActionChunk:
        """Decode all 50 predictions; Deployment selects the execution prefix.

        Arm deltas are anchored to the SAME observation used for the request,
        never to newer feedback acquired while inference was running. The
        checkpoint delta_mask leaves both gripper outputs absolute.
        """
        state = self.state_from_observation(observation)
        absolute = absolute_actions(self._processor, self.metadata, result, state)
        values = absolute.detach().cpu().numpy()
        expected = (1, ACTION_HORIZON, ACTION_DIM)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(
                f"Expected finite model actions with shape {expected}, "
                f"got {values.shape}"
            )
        # Validate all 50 targets before any prefix is queued. A malformed tail
        # is still a broken prediction even when execution_horizon is only 20.
        # Copy each arm slice to detach commands from the shared model tensor.
        actions: list[Action] = []
        for target in values[0]:
            actions.append(
                {
                    "joint_position_left": np.asarray(
                        target[:7], dtype=np.float64
                    ).copy(),
                    "joint_position_right": np.asarray(
                        target[8:15], dtype=np.float64
                    ).copy(),
                    "gripper_left": np.asarray(
                        [np.clip(target[7], 0.0, 1.0)], dtype=np.float32
                    ),
                    "gripper_right": np.asarray(
                        [np.clip(target[15], 0.0, 1.0)], dtype=np.float32
                    ),
                }
            )
        # These are source-rate targets, not 200 Hz sends. The deployment owns
        # prefix selection, its thread-safe queue, and interpolation. A configured
        # action_hz overrides this nominal checkpoint period of 40 ms.
        return ActionChunk(tuple(actions), ACTION_PERIOD_S)
