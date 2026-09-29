"""DreamZero policy wrapper around a caller-owned inference function."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from phyai_utils_tools.models.dreamzero.processor_dreamzero import (
    DreamZeroActionOutput,
    DreamZeroProcessedInputs,
    DreamZeroProcessor,
)


@dataclass
class DreamZeroPolicyResult:
    """Result of one DreamZero policy call."""

    action: Any
    postprocessed: DreamZeroActionOutput
    model_output: Any
    processed: DreamZeroProcessedInputs


class DreamZeroPolicy:
    """Compose raw-observation preprocessing, inference, and action postprocess.

    ``infer`` is intentionally caller-owned. It can be ``Engine.step``, a mock
    for tests, or a distributed RPC shim; this package stays independent from
    the main ``phyai`` engine package.
    """

    def __init__(
        self,
        *,
        processor: DreamZeroProcessor,
        infer: Callable[[DreamZeroProcessedInputs], Any],
    ) -> None:
        self.processor = processor
        self.infer = infer

    def act(self, observation: Any) -> DreamZeroPolicyResult:
        """Run one policy step from raw observation to executable action."""
        processed = self.processor.preprocess(observation)
        model_output = self.infer(processed)
        postprocessed = self.processor.postprocess(model_output, obs=observation)
        return DreamZeroPolicyResult(
            action=postprocessed.action,
            postprocessed=postprocessed,
            model_output=model_output,
            processed=processed,
        )

    __call__ = act


__all__ = ["DreamZeroPolicy", "DreamZeroPolicyResult"]
