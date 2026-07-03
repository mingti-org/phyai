"""GR00T-N1.7 single-GPU scheduler boundary.

The engine is strict about inputs: image transform, Qwen3-VL
patchify, tokenization, and state/action normalization are the **caller's**
responsibility (see ``phyai_utils_tools.models.gr00t.GR00TProcessor``). The
scheduler consumes the already-prepared model-input tensors and returns the raw
normalized action chunk; the caller decodes it back to physical units. This
keeps the engine free of any processor / tokenizer dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.models.gr00t_n17.model_runner_gr00t_n17 import (
    GR00TN17ActionHeadRunner,
    GR00TN17BackboneRunner,
)
from phyai.models.gr00t_n17.modeling_gr00t_n17 import (
    GR00TN17Model,
)
from phyai.runtime.schedule import Scheduler


@dataclass(frozen=True)
class GR00TN17Request:
    """One GR00T-N1.7 inference request — prepared model-input tensors.

    ``tensors`` is exactly the dict produced by the out-of-engine processor
    (``GR00TProcessor.process_observation(obs).tensors``): ``state`` /
    ``embodiment_id`` / ``action_mask`` plus the VLM tensors (``pixel_values``,
    ``input_ids``, ``attention_mask``, ``image_grid_thw``, ``mm_token_type_ids``).
    ``noise`` optionally seeds the flow-matching sampler for deterministic runs.
    """

    tensors: dict[str, torch.Tensor]
    noise: torch.Tensor | None = None


class GR00TN17WS1Scheduler(Scheduler):
    """Single-rank GR00T-N1.7 inference scheduler.

    The scheduler owns runner order and the denoising request lifecycle. It does
    not own preprocessing: it receives prepared tensors and emits the normalized
    action chunk.
    """

    def __init__(
        self,
        model: GR00TN17Model,
        *,
        max_batch_size: int = 1,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")
        self.model = model
        self.model.eval()
        self.max_batch_size = int(max_batch_size)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.use_cuda_graph = bool(use_cuda_graph)
        self.backbone_runner = GR00TN17BackboneRunner(model)
        self.action_head_runner = GR00TN17ActionHeadRunner(
            model,
            max_batch_size=self.max_batch_size,
            device=self.device,
            use_cuda_graph=self.use_cuda_graph,
        )

    def setup(self) -> None:
        self.backbone_runner.setup()
        self.action_head_runner.setup()

    @torch.no_grad()
    def step(self, request: GR00TN17Request) -> torch.Tensor:
        """Run backbone + action-head denoising; return the normalized action.

        Returns the ``(B, action_horizon, max_action_dim)`` normalized action
        chunk. The caller maps it back to physical units via the processor's
        ``decode_action`` (which needs the per-request ``raw_state``).
        """
        tensors = request.tensors
        batch = tensors["state"].shape[0]
        if batch > self.max_batch_size:
            raise ValueError(
                "GR00T-N1.7 request batch exceeds scheduler max_batch_size: "
                f"{batch} > {self.max_batch_size}."
            )
        backbone_inputs, action_inputs = self.model.prepare_input(
            tensors, device=self.device
        )
        backbone_output = self.backbone_runner.forward(backbone_inputs)
        normalized_action = self.action_head_runner.forward(
            backbone_output, action_inputs, noise=request.noise
        )
        return normalized_action

    def close(self) -> None:
        self.backbone_runner.close()
        self.action_head_runner.close()
        return None


__all__ = ["GR00TN17Request", "GR00TN17WS1Scheduler"]
