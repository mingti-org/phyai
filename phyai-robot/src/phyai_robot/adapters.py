"""Robot-level adapter bridging physical named robot schemas to vector model contracts.

Most Vision-Language-Action (VLA) models (such as Pi0, Pi0.5, OpenVLA, Octo) share a
canonical input/output tensor structure:
- Inputs:
    - Multiple ordered RGB camera images: passed as individual tensors under
      names 'image.<camera_name>' (e.g. 'image.head_left', 'image.left_wrist').
    - One concatenated proprioceptive state vector: passed under the key 'state'
      (shape: (state_dim,), dtype: float32).
- Outputs:
    - One 2D matrix of action trajectories: passed under the key 'actions'
      (shape: (horizon, action_dim), dtype: float32 or float64).

VectorAdapter serves as the physical robot's ambassador to this vector contract:
- It configures which physical sensor fields fill which slots of the state vector,
  along with optional linear calibration limits (e.g. mapping raw gripper encoder
  travel [0.0, 1.6] rad to normalized range [0.0, 1.0]).
- It decomposes output rows of the action matrix back into physical robot action
  fields (e.g. splitting columns into arm joint positions and gripper commands),
  applying optional safety validity clamping (clip).
- Decoupling guarantee: By encapsulating embodiment mapping in VectorAdapter, the
  model server remains completely agnostic of the physical robot (zero robot imports).
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .types import Action, Schema, ActionChunk, FeatureSpec, Observation
from .scheduling import InferenceContext
from .validation import validate_schema

# Reserved dictionary keys for model RPC array payloads.
IMAGE_PREFIX = "image."  # Prefix for camera image inputs (e.g. "image.front").
STATE_KEY = "state"  # Key for the concatenated 1D state vector.
ACTIONS_KEY = "actions"  # Key for the 2D predicted action trajectory matrix (horizon, action_dim).


@dataclass(frozen=True)
class StateSlot:
    """Configures extraction of an observation field into the model state vector.

    Attributes:
        field: Name of the physical observation field in ObservationSchema.
        index: Optional tuple of integer indices for selecting a subset of components
            from a 1D vector field (e.g. index=(0,) to select only the position element
            from a 5-element gripper feedback vector). None selects all components.
        limits: Optional linear normalization limits (low, high). When provided, raw values
            are linearly mapped: normalized = (raw - low) / (high - low).
            Notice: No clipping is performed here, preserving small out-of-range readings
            (such as mechanical hardstop offsets) present in physical training data.
    """

    field: str
    index: tuple[int, ...] | None = (
        None  # Component indices within the field (None = all).
    )
    limits: tuple[float, float] | None = (
        None  # (low, high) calibration limits for [0, 1] normalization.
    )


@dataclass(frozen=True)
class CommandSlot:
    """Configures unpacking of model action columns into a physical actuator command field.

    Attributes:
        field: Name of the physical actuator command field in ActionSchema.
        clip: Optional validity range (low, high) applied before casting to the target dtype.
            For instance, clip=(0.0, 1.0) clamps normalized gripper targets to [0, 1].
            This represents valid domain boundary enforcement, not collision prevention.
    """

    field: str
    clip: tuple[float, float] | None = (
        None  # (min, max) clipping limits applied to predictions.
    )


def _check_range(name: str, value: tuple[float, float] | None) -> None:
    """Validate that a range tuple consists of two strictly increasing finite floats."""
    if value is None:
        return
    if len(value) != 2 or not all(math.isfinite(v) for v in value):
        raise ValueError(f"{name} must contain two finite values")
    if value[0] >= value[1]:
        raise ValueError(f"{name} must be increasing")


class VectorAdapter:
    """Concrete PolicyAdapter mapping physical robot schemas to/from VLA vector contracts.

    Invariants:
    1. Zero Image Copies: Image arrays are passed directly from Observation into the request
       mapping without re-allocation. Camera fields must already match the model's required
       HWC RGB uint8 dimensions (handled upstream by camera decoders or letterbox steps).
    2. Precision Preservation: State extraction accumulates in float64 arithmetic to avoid
       numerical loss during linear affine scaling, then converts to float32 for the model.
    3. Trajectory Unpacking: Model predictions shaped (horizon, action_dim) are validated and
       sliced column-wise into individual single-tick Action dictionaries.
    """

    def __init__(
        self,
        *,
        observation_schema: Schema,
        action_schema: Schema,
        cameras: Mapping[
            str, str
        ],  # Mapping: model camera identifier -> observation field name.
        state: Sequence[StateSlot],  # Ordered sequence of StateSlot extractors.
        commands: Sequence[CommandSlot],  # Ordered sequence of CommandSlot unpackers.
    ) -> None:
        """Construct and validate a VectorAdapter against robot schemas.

        Args:
            observation_schema: Schema of the robot's physical observations.
            action_schema: Schema of the robot's physical commands.
            cameras: Mapping from model camera names (e.g. 'head_left') to physical
                observation fields (e.g. 'head_camera').
            state: Ordered sequence defining how physical fields assemble into the state vector.
            commands: Ordered sequence defining how action columns map into command fields.

        Raises:
            ValueError: If cameras, state slots, or command slots fail validation or do not
                exactly partition the target schemas.
        """
        validate_schema(observation_schema)
        validate_schema(action_schema)

        if not cameras:
            raise ValueError("at least one camera is required")
        for name, field in cameras.items():
            spec = observation_schema.get(field)
            if spec is None:
                raise ValueError(f"camera {name!r}: unknown observation {field!r}")
            if spec.dtype != "uint8" or len(spec.shape) != 3 or spec.shape[2] != 3:
                raise ValueError(f"camera {name!r}: {field!r} is not RGB uint8 HWC")
        self._cameras = dict(cameras)

        # Validate state slot definitions.
        self._state: list[tuple[str, NDArray[np.intp], tuple[float, float] | None]] = []
        for slot in state:
            spec = observation_schema.get(slot.field)
            if spec is None or len(spec.shape) != 1:
                raise ValueError(
                    f"state field {slot.field!r} must be a 1-D observation"
                )
            index = np.arange(spec.shape[0]) if slot.index is None else slot.index
            if not len(index) or any(not 0 <= i < spec.shape[0] for i in index):
                raise ValueError(f"state field {slot.field!r}: invalid index")
            _check_range(f"{slot.field} limits", slot.limits)
            self._state.append((slot.field, np.asarray(index, np.intp), slot.limits))
        if not self._state:
            raise ValueError("at least one state slot is required")

        # Validate command slot definitions and compute column slice partitions.
        self._commands: list[tuple[str, slice, FeatureSpec, Any]] = []
        start = 0
        for slot in commands:
            spec = action_schema.get(slot.field)
            if spec is None or len(spec.shape) != 1:
                raise ValueError(f"command field {slot.field!r} must be a 1-D action")
            _check_range(f"{slot.field} clip", slot.clip)
            width = spec.shape[0]
            self._commands.append(
                (slot.field, slice(start, start + width), spec, slot.clip)
            )
            start += width

        # Ensure command slots exactly cover the declared action schema without omissions or duplicates.
        if sorted(field for field, *_ in self._commands) != sorted(action_schema):
            raise ValueError("command slots must cover every action field once")

        self.action_dim = start
        self.state_dim = sum(len(index) for _, index, _ in self._state)

        # Declare the expected model request schema for handshake validation with the server.
        self.request_schema: Schema = MappingProxyType(
            {
                **{
                    IMAGE_PREFIX + name: observation_schema[field]
                    for name, field in cameras.items()
                },
                STATE_KEY: FeatureSpec((self.state_dim,), "float32"),
            }
        )

    def state_vector(self, observation: Observation) -> NDArray[np.float32]:
        """Extract and assemble the concatenated 1D float32 state vector from an observation.

        Args:
            observation: Robot observation snapshot.

        Returns:
            Concatenated 1D float32 array of shape (state_dim,).

        Raises:
            ValueError: If extracted state contains non-finite values (NaN / inf).
        """
        parts = []
        for field, index, limits in self._state:
            values = observation.samples[field].value[index].astype(np.float64)
            if limits is not None:
                low, high = limits
                values = (values - low) / (high - low)
            parts.append(values)
        state = np.concatenate(parts).astype(np.float32)
        if not np.isfinite(state).all():
            raise ValueError("model state contains a non-finite value")
        return state

    def to_request(
        self, observation: Observation, context: InferenceContext
    ) -> Mapping[str, NDArray[Any]]:
        """Construct the model input dictionary containing camera views and state vector.

        Images are referenced directly without copying.
        """
        return {
            **{
                IMAGE_PREFIX + name: observation.samples[field].value
                for name, field in self._cameras.items()
            },
            STATE_KEY: self.state_vector(observation),
        }

    def to_actions(
        self, result: Mapping[str, NDArray[Any]], observation: Observation
    ) -> ActionChunk:
        """Unpack a predicted action trajectory matrix into an ActionChunk.

        Args:
            result: Model output mapping containing ACTIONS_KEY with shape (horizon, action_dim).
            observation: The original Observation snapshot associated with this prediction.

        Returns:
            ActionChunk containing a sequence of complete physical Action mappings.

        Raises:
            ValueError: If action matrix shape or dimension does not match action_dim.
        """
        values = np.asarray(result[ACTIONS_KEY])
        if (
            values.ndim != 2
            or values.shape[1] != self.action_dim
            or values.dtype.kind != "f"
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"expected finite float actions with shape (H, {self.action_dim}), "
                f"got {values.dtype} {values.shape}"
            )

        actions: list[Action] = []
        for row in values:
            action = {}
            for field, part, spec, clip in self._commands:
                value = row[part]
                if clip is not None:
                    value = np.clip(value, *clip)
                action[field] = np.asarray(value, dtype=spec.dtype)
            actions.append(action)
        return ActionChunk(tuple(actions))
