"""Flow-matching Euler integration used by Qwen-Image 2.1."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class FlowMatchEulerConfig(PretrainedConfig):
    num_train_timesteps: int = 1000
    shift: float = 1.0
    use_dynamic_shifting: bool = True
    base_shift: float = 0.5
    max_shift: float = 0.9
    base_image_seq_len: int = 256
    max_image_seq_len: int = 8192
    shift_terminal: float | None = 0.02
    time_shift_type: str = "exponential"
    invert_sigmas: bool = False
    stochastic_sampling: bool = False
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False
    use_beta_sigmas: bool = False

    def __post_init__(self) -> None:
        if self.num_train_timesteps < 1 or self.shift <= 0:
            raise ValueError("num_train_timesteps and shift must be positive.")
        if self.max_image_seq_len <= self.base_image_seq_len:
            raise ValueError("max_image_seq_len must exceed base_image_seq_len.")
        if self.time_shift_type not in ("exponential", "linear"):
            raise ValueError("time_shift_type must be exponential or linear.")
        if self.shift_terminal is not None and not 0 <= self.shift_terminal < 1:
            raise ValueError("shift_terminal must be in [0, 1).")
        if (
            self.use_karras_sigmas
            or self.use_exponential_sigmas
            or self.use_beta_sigmas
        ):
            raise ValueError("Qwen-Image 2.1 uses the flow-matching sigma schedule.")
        if self.stochastic_sampling:
            raise ValueError("Qwen-Image 2.1 uses deterministic Euler sampling.")


@dataclass(frozen=True)
class FlowMatchSchedule:
    timesteps: torch.Tensor
    sigmas: torch.Tensor


class FlowMatchEulerSampler:
    def __init__(self, config: FlowMatchEulerConfig | None = None) -> None:
        self.config = config or FlowMatchEulerConfig()

    def build_schedule(
        self,
        num_inference_steps: int,
        image_seq_len: int,
        *,
        device: torch.device | str,
        sigmas: list[float] | tuple[float, ...] | None = None,
    ) -> FlowMatchSchedule:
        if num_inference_steps < 1 or image_seq_len < 1:
            raise ValueError("num_inference_steps and image_seq_len must be positive.")
        if sigmas is None:
            values = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        else:
            if not sigmas:
                raise ValueError("sigmas cannot be empty.")
            values = np.asarray(sigmas)
        values = values.astype(np.float32)
        if not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(values > 1):
            raise ValueError("sigmas must be finite values in (0, 1].")
        if np.any(np.diff(values) > 0):
            raise ValueError("sigmas must be nonincreasing.")
        cfg = self.config
        if cfg.use_dynamic_shifting:
            slope = (cfg.max_shift - cfg.base_shift) / (
                cfg.max_image_seq_len - cfg.base_image_seq_len
            )
            mu = image_seq_len * slope + cfg.base_shift - slope * cfg.base_image_seq_len
            shift = math.exp(mu) if cfg.time_shift_type == "exponential" else mu
            values = shift / (shift + (1 / values - 1))
        else:
            values = cfg.shift * values / (1 + (cfg.shift - 1) * values)
        if cfg.shift_terminal:
            one_minus = 1 - values
            scale = one_minus[-1] / (1 - cfg.shift_terminal)
            if scale == 0:
                raise ValueError("A terminal-shifted schedule needs a sigma below 1.")
            values = 1 - one_minus / scale
        tensor = torch.from_numpy(values).to(device=device, dtype=torch.float32)
        if cfg.invert_sigmas:
            tensor = 1 - tensor
        timesteps = tensor * cfg.num_train_timesteps
        terminal = tensor.new_ones(1) if cfg.invert_sigmas else tensor.new_zeros(1)
        return FlowMatchSchedule(timesteps, torch.cat([tensor, terminal]))

    @staticmethod
    def step(
        model_output: torch.Tensor,
        sample: torch.Tensor,
        schedule: FlowMatchSchedule,
        step_index: int,
    ) -> torch.Tensor:
        dt = schedule.sigmas[step_index + 1] - schedule.sigmas[step_index]
        # Keep the reference's scalar multiplication dtype before the fp32 sum.
        return (sample.float() + dt * model_output).to(model_output.dtype)


__all__ = ["FlowMatchEulerConfig", "FlowMatchEulerSampler", "FlowMatchSchedule"]
