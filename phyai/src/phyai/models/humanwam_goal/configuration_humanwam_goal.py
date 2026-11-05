"""Inference architecture and sampling configuration for HumanWAM-Goal."""

import math
from dataclasses import dataclass

from phyai.models.configuration import PretrainedConfig
from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3Config


@dataclass(frozen=True)
class HumanWAMGoalConfig(Cosmos3Config):
    action_gen: bool = True
    action_dim: int = 64
    joint_dim: int = 23
    shared_dim: int = 19
    horizon: int = 48

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.action_gen or self.sound_gen:
            raise ValueError(
                "HumanWAM-Goal requires visual and action generation without sound"
            )
        if min(self.joint_dim, self.shared_dim, self.horizon) < 1:
            raise ValueError("Action dimensions and horizon must be positive")
        if self.joint_dim + self.shared_dim > self.action_dim:
            raise ValueError("Shared and joint channels exceed the action width")
        if not self.qk_norm_for_text or not self.qk_norm_for_diffusion:
            raise ValueError(
                "HumanWAM-Goal requires text and generator Q/K normalization"
            )


@dataclass(frozen=True)
class HumanWAMGoalSamplingConfig(PretrainedConfig):
    num_steps: int = 10
    flow_shift: float = 2.0
    prefix_length: int = 7
    corrector_weights: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.num_steps != 10:
            raise ValueError("HumanWAM-Goal inference uses ten denoising steps")
        if not math.isfinite(self.flow_shift) or self.flow_shift <= 0:
            raise ValueError("flow_shift must be positive")
        if self.prefix_length < 0:
            raise ValueError("prefix_length must be nonnegative")

        if self.corrector_weights is not None:
            weights = tuple(
                tuple(float(x) for x in row) for row in self.corrector_weights
            )
            if len(weights) != self.num_steps or any(
                len(row) != min(index, 2) or not all(math.isfinite(x) for x in row)
                for index, row in enumerate(weights)
            ):
                raise ValueError(
                    "UniPC corrector_weights must contain 0, 1, then 2 finite coefficients per step"
                )
            object.__setattr__(self, "corrector_weights", weights)
