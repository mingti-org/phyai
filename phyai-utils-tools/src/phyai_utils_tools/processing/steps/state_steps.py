"""Checkpoint-declared constant coordinates in the input state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from phyai_utils_tools.processing.pipeline import ProcessorStep, ProcessorStepRegistry
from phyai_utils_tools.processing.transition import STATE, Transition


@ProcessorStepRegistry.register("state_constants_processor")
@dataclass
class StateConstantsStep(ProcessorStep):
    """Copy the state and replace selected last-axis coordinates before normalization.

    Values use the input state's units. This step affects only its transition;
    postprocessing receives the original request state as the action anchor.
    """

    values: dict[str, float]
    _values: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for key, value in self.values.items():
            if isinstance(key, bool) or not str(key).isdecimal():
                raise ValueError("State constant indices must be nonnegative integers")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("State constants must be finite numbers")
            index = int(key)
            if index in self._values:
                raise ValueError("State constant indices must be unique")
            self._values[index] = float(value)

    def __call__(self, transition: Transition) -> Transition:
        state = transition[STATE]
        if (
            not isinstance(state, torch.Tensor)
            or state.ndim < 1
            or not state.is_floating_point()
        ):
            raise ValueError(
                "StateConstantsStep requires a state tensor with a final coordinate axis"
            )
        if self._values and max(self._values) >= state.shape[-1]:
            raise ValueError("State constant index exceeds the input state width")
        result = state.clone()
        for index, value in self._values.items():
            result[..., index] = value
        return {**transition, STATE: result}

    def get_config(self) -> dict[str, Any]:
        return {"values": {str(index): value for index, value in self._values.items()}}


__all__ = ["StateConstantsStep"]
