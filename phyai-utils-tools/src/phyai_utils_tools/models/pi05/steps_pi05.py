"""pi0.5 prompt-prep step — state discretization + prompt assembly.

:class:`StateTokenizerPrepareStep` is pi0.5-specific (the openpi/PaliGemma state
binning + prompt template), so it lives under ``models/pi05`` rather than the
generic ``processing/steps`` tree. Importing this module registers the step
under lerobot's name ``pi05_prepare_state_tokenizer_processor_step`` so the
lerobot ``policy_preprocessor.json`` loads. The generic tokenization that
consumes the prompt it produces is
:class:`~phyai_utils_tools.processing.steps.text_steps.TokenizerStep`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phyai_utils_tools.models.pi05.ops_pi05 import (
    STATE_NUM_BINS,
    build_prompts,
    discretize_state,
)
from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import PROMPT, STATE, TASK, Transition


@ProcessorStepRegistry.register("pi05_prepare_state_tokenizer_processor_step")
@dataclass
class StateTokenizerPrepareStep(ProcessorStep):
    """Discretize ``STATE`` and build the pi0.5 ``PROMPT`` strings.

    Reads ``STATE`` (``(B, state_dim)``, assumed already normalized to
    ``[-1, 1]``) and ``TASK`` (``list[str]``), writes ``PROMPT`` (``list[str]``)
    of the form ``"Task: <task>, State: <bins>;\\nAction: "``. Mirrors openpi's
    PaligemmaTokenizer prompt assembly. If ``STATE`` is absent, the prompt uses
    the task text alone (no state bins).

    ``prompt_mode="instruction"`` emits only cleaned task text and ignores
    state. A token suffix can be supplied by the following TokenizerStep.

    Registry name + default empty ``get_config`` match lerobot's
    ``pi05_prepare_state_tokenizer_processor_step`` (it persists no config;
    ``max_state_dim`` / ``num_bins`` are construction-time defaults).
    """

    num_bins: int = STATE_NUM_BINS
    max_state_dim: int = 32
    prompt_mode: str = "state"

    def __call__(self, transition: Transition) -> Transition:
        tasks = transition.get(TASK)
        if tasks is None:
            raise ValueError("StateTokenizerPrepareStep requires a TASK entry.")
        if isinstance(tasks, str):
            tasks = [tasks]

        out = transition.copy()
        if self.prompt_mode == "instruction":
            out[PROMPT] = [
                t.strip().replace("_", " ").replace("\n", " ") for t in tasks
            ]
            return out
        if self.prompt_mode != "state":
            raise ValueError(f"Unknown pi0.5 prompt mode: {self.prompt_mode!r}")
        state = transition.get(STATE)
        if state is not None:
            discretized = discretize_state(state, num_bins=self.num_bins)
            out[PROMPT] = build_prompts(list(tasks), discretized)
        else:
            out[PROMPT] = [
                f"Task: {t.strip().replace('_', ' ').replace(chr(10), ' ')};\nAction: "
                for t in tasks
            ]
        return out

    def get_config(self) -> dict[str, Any]:
        # Preserve lerobot's empty config for the default state prompt.
        return {} if self.prompt_mode == "state" else {"prompt_mode": self.prompt_mode}


__all__ = ["StateTokenizerPrepareStep"]
