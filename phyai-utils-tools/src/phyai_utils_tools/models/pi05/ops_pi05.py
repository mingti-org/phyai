"""pi0.5 text/state ops — state binning + prompt assembly (openpi/PaliGemma).

These encode pi0.5's *specific* state-tokenization convention (mirrors openpi's
``PaligemmaTokenizer.tokenize()``): a ``[-1, 1]``-normalized state vector is
discretized into 256 integer bins and embedded in the prompt template
``"Task: <task>, State: <bins>;\\nAction: "``. The bin scheme and the template
are pi0.5-specific, so they live under ``models/pi05`` rather than the generic
``processing`` tree. The generic HF tokenization that consumes the prompt is
:class:`~phyai_utils_tools.processing.steps.text_steps.TokenizerStep`.
"""

from __future__ import annotations

import numpy as np
import torch

STATE_NUM_BINS = 256


def discretize_state(state: torch.Tensor, num_bins: int = STATE_NUM_BINS) -> np.ndarray:
    """Discretize a ``[-1, 1]``-normalized state vector into ``num_bins`` bins.

    Mirrors openpi's PaligemmaTokenizer state binning, including the edge
    indices returned by ``np.digitize`` for values outside ``[-1, 1]``.
    Returns an integer array with the same shape as ``state``.
    Preserve float64 normalization at bin boundaries;
    promote lower-precision input to float32 for NumPy compatibility.
    """
    dtype = torch.float64 if state.dtype == torch.float64 else torch.float32
    state_np = state.detach().to(dtype).cpu().numpy()
    bins = np.linspace(-1.0, 1.0, num_bins + 1)[:-1]
    return np.digitize(state_np, bins=bins) - 1


def build_prompts(tasks: list[str], discretized_states: np.ndarray) -> list[str]:
    """Assemble pi0.5 prompt strings from tasks + binned states.

    Produces ``"Task: <cleaned_task>, State: <bin_0> <bin_1> ...;\\nAction: "``
    per sample. The task text is cleaned (strip, ``_``/newline -> space).
    """
    if len(tasks) != len(discretized_states):
        raise ValueError(
            f"len(tasks)={len(tasks)} must match "
            f"len(discretized_states)={len(discretized_states)}"
        )
    prompts: list[str] = []
    for task, state_bins in zip(tasks, discretized_states):
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, state_bins))
        prompts.append(f"Task: {cleaned}, State: {state_str};\nAction: ")
    return prompts


__all__ = ["STATE_NUM_BINS", "build_prompts", "discretize_state"]
