"""Action postprocessing step — trim the action chunk to the real action dim.

pi0.5 (and openpi VLAs generally) pad the action vector to a fixed
``max_action_dim`` for the model, then slice back to the dataset's true action
dimensionality on the way out. :class:`SliceActionStep` is that slice — the
minimal postprocess every model needs after (optional) unnormalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import ACTION, STATE, Transition


@ProcessorStepRegistry.register("slice_action_step")
@dataclass
class SliceActionStep(ProcessorStep):
    """Trim ``ACTION`` to ``[..., :action_dim]``.

    ``action_dim`` is the dataset's real action width (``<= max_action_dim``).
    ``None`` leaves the action untouched (pass-through), useful when the caller
    already wants the full padded chunk.
    """

    action_dim: int | None = None

    def __call__(self, transition: Transition) -> Transition:
        if self.action_dim is None:
            return transition
        action = transition.get(ACTION)
        if action is None:
            raise ValueError("SliceActionStep requires an ACTION entry.")
        out = transition.copy()
        out[ACTION] = action[..., : self.action_dim]
        return out

    def get_config(self) -> dict[str, Any]:
        return {"action_dim": self.action_dim}


@ProcessorStepRegistry.register("delta_action_processor")
@dataclass
class DeltaActionStep(ProcessorStep):
    """Add the current raw state to selected leading action dimensions.

    Each timestep uses the same observation anchor. False mask entries retain
    absolute predictions, such as gripper commands. Padded dimensions are kept.
    """

    delta_mask: list[bool]

    def __call__(self, transition: Transition) -> Transition:
        action = transition[ACTION]
        width = len(self.delta_mask)
        if not width or any(type(value) is not bool for value in self.delta_mask):
            raise ValueError("delta_mask must be a nonempty list of booleans")
        if action.ndim not in (2, 3) or action.shape[-1] < width:
            raise ValueError("Action must have shape [time, dim] or [batch, time, dim]")
        state = torch.as_tensor(
            transition[STATE], device=action.device, dtype=action.dtype
        )
        if action.ndim == 3 and state.ndim == 1 and action.shape[0] == 1:
            state = state.unsqueeze(0)
        if state.shape != (*action.shape[:-2], width):
            raise ValueError(
                "Raw state shape must match the action batch and delta_mask width"
            )
        if not torch.isfinite(state).all() or not torch.isfinite(action).all():
            raise ValueError("Action and raw state must contain only finite values")
        result = action.clone()
        mask = torch.tensor(self.delta_mask, device=action.device, dtype=torch.bool)
        result[..., :width] += torch.where(mask, state, 0).unsqueeze(-2)
        return {**transition, ACTION: result}

    def get_config(self) -> dict[str, Any]:
        return {"delta_mask": list(self.delta_mask)}


__all__ = ["DeltaActionStep", "SliceActionStep"]
