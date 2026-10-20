"""MOSS-Image's shifted, nearest-training-grid Euler sampler."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class MOSSImageSamplerConfig:
    num_train_timesteps: int = 1000
    shift: float = 5.0
    training_shift: float = 1.0

    def __post_init__(self) -> None:
        if type(self.num_train_timesteps) is not int or self.num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be a positive integer")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.shift, self.training_shift)
        ):
            raise ValueError("shift and training_shift must be finite and positive")

    @classmethod
    def from_json(cls, path: str | Path) -> MOSSImageSamplerConfig:
        with Path(path).open() as stream:
            data = json.load(stream)
        for name in (
            "invert_sigmas",
            "stochastic_sampling",
            "use_dynamic_shifting",
            "use_beta_sigmas",
            "use_exponential_sigmas",
            "use_karras_sigmas",
        ):
            if data.get(name, False):
                raise ValueError(f"MOSS-Image does not use {name}")
        if data.get("shift_terminal") is not None:
            raise ValueError("MOSS-Image does not use shift_terminal")
        return cls(
            **{
                name: data[name]
                for name in ("num_train_timesteps", "shift", "training_shift")
                if name in data
            }
        )


class MOSSImageEulerSampler:
    def __init__(self, config: MOSSImageSamplerConfig | None = None) -> None:
        self.config = config or MOSSImageSamplerConfig()

    def timesteps(
        self, num_inference_steps: int, *, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        total = self.config.num_train_timesteps

        def grid(count: int, shift: float) -> torch.Tensor:
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count, dtype=torch.float32, device="cpu")
            return shift * values / (1 + (shift - 1) * values)

        requested = grid(num_inference_steps, self.config.shift)
        training = grid(total, self.config.training_shift).flip(0)
        upper = torch.searchsorted(training, requested).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where(
            (requested - training[lower]).abs() <= (training[upper] - requested).abs(),
            lower,
            upper,
        )
        sigmas = torch.unique_consecutive(training[nearest]).to(device)
        timesteps = sigmas * total
        return timesteps, torch.cat((sigmas, sigmas.new_zeros(1)))

    @staticmethod
    def step(
        velocity: torch.Tensor,
        latents: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        return latents.float() + (sigma_next - sigma) * velocity.float()


__all__ = ["MOSSImageSamplerConfig", "MOSSImageEulerSampler"]
