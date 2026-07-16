"""Single-GPU MiniCPM-GR00T clean-action inference scheduler."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.models.minicpm_gr00t.model_runner_minicpm_gr00t import (
    MiniCPMGR00TModelRunner,
)
from phyai.runtime.schedule import Scheduler


@dataclass
class MiniCPMGR00TRequest:
    """Canonical tensors produced by the MiniCPM-V 4.6 processor."""

    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    target_sizes: torch.Tensor
    state: torch.Tensor
    attention_mask: torch.Tensor | None = None
    noise: torch.Tensor | None = None


class MiniCPMGR00TWS1Scheduler(Scheduler):
    """Run one VLM prefill followed by four clean-action DiT predictions."""

    def __init__(
        self,
        runner: MiniCPMGR00TModelRunner,
        *,
        device: torch.device | str,
    ) -> None:
        self.runner = runner
        self.device = torch.device(device)
        self.config = runner.config

    def setup(self) -> None:
        self.runner.setup()

    @torch.inference_mode()
    def step(self, request: MiniCPMGR00TRequest) -> torch.Tensor:
        cfg = self.config.action
        vlm_hidden_states = self.runner.encode_vlm(
            input_ids=request.input_ids,
            pixel_values=request.pixel_values,
            target_sizes=request.target_sizes,
            attention_mask=request.attention_mask,
        )
        batch_size = request.input_ids.shape[0]
        expected_noise_shape = (
            batch_size,
            cfg.action_horizon,
            cfg.action_dim,
        )
        if request.noise is None:
            noise = torch.randn(
                expected_noise_shape,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            if tuple(request.noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"noise has shape {tuple(request.noise.shape)}; expected "
                    f"{expected_noise_shape}."
                )
            noise = request.noise.to(device=self.device, dtype=torch.float32)

        actions = torch.zeros_like(noise)
        num_steps = cfg.num_inference_steps
        for step in range(num_steps, 0, -1):
            time_continuous = step / float(num_steps)
            timestep_value = min(
                int(time_continuous * cfg.dit.num_timestep_buckets),
                cfg.dit.num_timestep_buckets - 1,
            )
            timestep = torch.full(
                (batch_size,),
                timestep_value,
                dtype=torch.int64,
                device=self.device,
            )
            noisy_actions = time_continuous * noise + (1.0 - time_continuous) * actions
            actions = self.runner.predict_clean_action(
                vlm_hidden_states=vlm_hidden_states,
                state=request.state,
                noisy_actions=noisy_actions,
                timestep=timestep,
            )
        return actions

    def close(self) -> None:
        self.runner.close()


__all__ = ["MiniCPMGR00TRequest", "MiniCPMGR00TWS1Scheduler"]
