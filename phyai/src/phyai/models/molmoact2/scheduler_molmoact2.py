"""MolmoAct2 Euler action flow and greedy token generation."""

from dataclasses import dataclass

import torch

from phyai.runtime.schedule import Scheduler
from phyai.models.molmoact2.model_runner_molmoact2 import MolmoAct2Runner


@dataclass
class MolmoAct2Request:
    inputs: dict[str, torch.Tensor]
    action_horizon: int | None = None
    num_steps: int | None = None
    seed: int = 0
    noise: torch.Tensor | None = None


@dataclass
class MolmoAct2GenerationRequest:
    inputs: dict[str, torch.Tensor]
    max_new_tokens: int = 128
    eos_token_id: int | None = None


class MolmoAct2Scheduler(Scheduler):
    def __init__(self, runner: MolmoAct2Runner) -> None:
        self.runner = runner

    def setup(self) -> None:
        self.runner.setup()

    @torch.inference_mode()
    def step(
        self, request: MolmoAct2Request | MolmoAct2GenerationRequest
    ) -> torch.Tensor:
        self.runner.reset()
        try:
            inputs = {
                name: tensor.to(self.runner.device)
                for name, tensor in request.inputs.items()
            }
            if "input_ids" not in inputs or inputs["input_ids"].ndim != 2:
                raise ValueError(
                    "inputs must include input_ids with shape (batch, sequence)."
                )
            if isinstance(request, MolmoAct2GenerationRequest):
                return self.generate(request, inputs)
            config = self.runner.model.config
            if config.action_mode not in ("continuous", "both"):
                raise ValueError("The checkpoint does not support continuous actions.")
            steps = (
                config.flow_matching_num_steps
                if request.num_steps is None
                else request.num_steps
            )
            horizon = (
                config.max_action_horizon
                if request.action_horizon is None
                else request.action_horizon
            )
            if not isinstance(steps, int) or steps < 1:
                raise ValueError("num_steps must be a positive integer.")
            if (
                not isinstance(horizon, int)
                or not 1 <= horizon <= config.max_action_horizon
            ):
                raise ValueError(
                    "action_horizon must be between 1 and the checkpoint maximum."
                )
            pad = inputs.pop("action_dim_is_pad", None)
            batch = inputs["input_ids"].shape[0]
            shape = (batch, horizon, config.max_action_dim)
            if pad is not None:
                if pad.ndim == 1:
                    pad = pad[None, :]
                if pad.ndim == 2 and pad.shape[0] == 1:
                    pad = pad.expand(batch, -1)
                if pad.shape not in ((batch, config.max_action_dim), shape):
                    raise ValueError(
                        "action_dim_is_pad must have shape (B,D) or (B,T,D)."
                    )
                pad = pad.bool()
                if pad.ndim == 2:
                    pad = pad[:, None, :]
            if not config.mask_action_dim_padding:
                pad = None
            if request.noise is None:
                generator = torch.Generator(device=self.runner.device).manual_seed(
                    request.seed
                )
                trajectory = torch.randn(
                    shape,
                    device=self.runner.device,
                    dtype=self.runner.dtype,
                    generator=generator,
                )
            else:
                if request.noise.shape != shape:
                    raise ValueError(f"noise must have shape {shape}.")
                trajectory = request.noise.to(
                    device=self.runner.device, dtype=self.runner.dtype
                ).clone()
            if pad is not None:
                trajectory.masked_fill_(pad, 0)
            self.runner.prepare_action(inputs, action_horizon=horizon, num_steps=steps)
            for index in range(steps):
                velocity = self.runner.forward(trajectory, step_index=index)
                if pad is not None:
                    velocity = velocity.masked_fill(pad, 0)
                trajectory = trajectory + (1.0 / steps) * velocity
                if pad is not None:
                    trajectory.masked_fill_(pad, 0)
            return trajectory
        finally:
            self.runner.reset()

    def generate(
        self, request: MolmoAct2GenerationRequest, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if request.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        inputs.pop("action_dim_is_pad", None)
        mask = inputs.get("attention_mask")
        if mask is not None and mask.ndim != 2:
            raise ValueError(
                "Token generation requires a two-dimensional padding mask."
            )
        position_ids = inputs.get("position_ids")
        eos = (
            self.runner.model.config.eos_token_id
            if request.eos_token_id is None
            else request.eos_token_id
        )
        output = self.runner.prefill(inputs, output_logits=True)
        finished = torch.zeros(
            inputs["input_ids"].shape[0], device=self.runner.device, dtype=torch.bool
        )
        generated = []
        for index in range(request.max_new_tokens):
            token = output.logits[:, -1].argmax(-1)
            token = torch.where(finished, eos, token)
            generated.append(token)
            finished |= token == eos
            if bool(finished.all()) or index + 1 == request.max_new_tokens:
                break
            if mask is not None:
                mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
            if position_ids is not None:
                position_ids = position_ids[:, -1:] + 1
            output = self.runner.decode(token[:, None], mask, position_ids)
        return torch.stack(generated, dim=1)

    def close(self) -> None:
        self.runner.close()
