"""pi0-specific prompt and state preparation steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from phyai_utils_tools.processing.pipeline import ProcessorStep, ProcessorStepRegistry
from phyai_utils_tools.processing.transition import PROMPT, STATE, TASK, Transition


@ProcessorStepRegistry.register("pi0_prompt_prepare_step")
@dataclass
class Pi0PromptPrepareStep(ProcessorStep):
    """Ensure pi0 task prompts end with a newline."""

    def __call__(self, transition: Transition) -> Transition:
        tasks = transition.get(TASK)
        if tasks is None:
            raise ValueError("Pi0PromptPrepareStep requires a TASK entry.")
        if isinstance(tasks, str):
            tasks = [tasks]

        out = transition.copy()
        out[PROMPT] = [task if task.endswith("\n") else f"{task}\n" for task in tasks]
        return out


@ProcessorStepRegistry.register("pad_state_step")
@dataclass
class PadStateStep(ProcessorStep):
    """Pad state vectors to pi0's fixed expert-side state width."""

    max_state_dim: int = 32

    def __call__(self, transition: Transition) -> Transition:
        state = transition.get(STATE)
        if state is None:
            raise ValueError("PadStateStep requires a STATE entry.")
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"STATE must be a torch.Tensor, got {type(state)!r}.")
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() != 2:
            raise ValueError(
                f"STATE must be 2-D (B, state_dim), got shape {tuple(state.shape)}."
            )

        state_dim = int(state.shape[-1])
        if state_dim > self.max_state_dim:
            raise ValueError(
                f"state_dim={state_dim} exceeds max_state_dim={self.max_state_dim}."
            )
        if state_dim == self.max_state_dim:
            padded = state
        else:
            padded = torch.nn.functional.pad(state, (0, self.max_state_dim - state_dim))

        out = transition.copy()
        out[STATE] = padded
        return out

    def get_config(self) -> dict[str, Any]:
        return {"max_state_dim": self.max_state_dim}


@ProcessorStepRegistry.register("pi0_new_line_processor")
@dataclass
class Pi0NewLineProcessor(Pi0PromptPrepareStep):
    """Compatibility alias for LeRobot pi0 checkpoints.

    LeRobot saves this step as ``pi0_new_line_processor``. It has the same
    inference behavior as ``Pi0PromptPrepareStep``: write newline-terminated
    task strings into ``PROMPT`` for the tokenizer.
    """


__all__ = ["PadStateStep", "Pi0NewLineProcessor", "Pi0PromptPrepareStep"]
