"""Map AgileX observations and Enactive pi0.5 actions without trajectory clipping."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from phyai_robot import ActionChunk

from ..policies import Pi05PolicyRequest
from ..robots.agilex import action_from_target, state_from_observation
from ..robots.agilex_protocol import VIEWS


class AgilexPi05Adapter:
    def __init__(self, checkpoint: Path, *, task: str, action_hz: float = 30.0):
        import torch
        from phyai_utils_tools.models.pi05 import PI05Processor

        config = json.loads((checkpoint / "config.json").read_text())
        self.horizon = int(config["chunk_size"])
        self.task, self.action_hz, self.torch = task, action_hz, torch
        # Match examples/pi05/run_enactive.py exactly. The exported processor
        # owns the prompt, image resize, quantiles, and joint-only delta mask.
        self.processor = PI05Processor.from_pretrained(
            checkpoint,
            tokenizer_name=str(checkpoint),
            image_resize_backend="pil",
            normalize_pixels=True,
            action_dim=14,
            params_dtype=torch.float32,
        )

    def to_request(self, observation):
        images = [
            self.torch.from_numpy(observation.samples[f"{view}_camera"].value.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            for view in VIEWS
        ]
        return Pi05PolicyRequest(
            self.processor.preprocess(
                {
                    "images": images,
                    "task": self.task,
                    "state": self.torch.from_numpy(
                        state_from_observation(observation)
                    ).unsqueeze(0),
                }
            )
        )

    def to_actions(self, result, observation):
        # Add the paired snapshot state to joint deltas only; grippers are
        # absolute meters. The processor reads this mask from the checkpoint.
        absolute = (
            self.processor.postprocess(
                {
                    "action": result,
                    "state": state_from_observation(observation),
                }
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        if absolute.shape != (1, self.horizon, 14) or not np.isfinite(absolute).all():
            raise ValueError(
                f"Expected finite [1, {self.horizon}, 14] actions, got {absolute.shape}"
            )
        return ActionChunk(
            tuple(action_from_target(row) for row in absolute[0]),
            1.0 / self.action_hz,
        )
