"""The policy and standalone-video flow schedules released with VPP2."""

from dataclasses import dataclass
import math

import numpy as np
import torch


def validate_schedule(steps: int, shift: float, training_steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("num_inference_steps must be a positive integer.")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("sigma_shift must be finite and positive.")
    if training_steps <= 0:
        raise ValueError("num_train_timesteps must be positive.")


@dataclass(frozen=True)
class VPP2FlowSchedule:
    timesteps: torch.Tensor
    deltas: torch.Tensor


class VPP2EulerSampler:
    """Continuous shifted flow with the reference's low-precision updates."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 3.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

    def schedule(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        shift: float | None = None,
    ) -> VPP2FlowSchedule:
        shift = self.shift if shift is None else shift
        validate_schedule(num_inference_steps, shift, self.num_train_timesteps)
        uniform = torch.linspace(
            1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32
        )
        sigma = shift * uniform / (1.0 + (shift - 1.0) * uniform)
        return VPP2FlowSchedule(
            timesteps=(sigma[:-1] * float(self.num_train_timesteps)).to(dtype),
            deltas=(sigma[1:] - sigma[:-1]).to(dtype),
        )

    @staticmethod
    def step(
        prediction: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        delta = delta.to(device=sample.device, dtype=sample.dtype)
        if delta.ndim > 0:
            delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        # Rounding the multiply before the add is part of policy inference.
        return sample + prediction * delta


class VPP2DiscreteFlowSampler:
    """The standalone video command's integer 999-to-0 flow interpolation."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 3.0):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

    def schedule(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str,
        shift: float | None = None,
    ) -> torch.Tensor:
        shift = self.shift if shift is None else shift
        validate_schedule(num_inference_steps, shift, self.num_train_timesteps)
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError("Discrete video steps cannot exceed training timesteps.")
        # NumPy's float64 arange and ties-to-even rounding match the release.
        sigma = np.arange(1.0, 0, -1.0 / num_inference_steps)
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        return torch.from_numpy(
            np.round(sigma * self.num_train_timesteps - 1).astype(np.int64)
        ).to(device)

    def step(
        self,
        prediction: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor | int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        a = 1 - timestep / self.num_train_timesteps
        b = timestep / self.num_train_timesteps
        clean = (sample - b * prediction) / (a + b)
        noise = (sample + a * prediction) / (a + b)
        return (1 - next_timestep / self.num_train_timesteps) * clean + (
            next_timestep / self.num_train_timesteps
        ) * noise
