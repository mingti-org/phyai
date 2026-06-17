"""Experimental native WALL-OSS-0.5 modules.

This file starts with the action processor because it is the smallest useful
unit with stable checkpoint keys. Full decoder and joint-attention native
implementation will be added after action-processor parity is established.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from phyai.layers.linear import ReplicatedLinear

from .configuration_walloss05_native import WallOSS05NativeConfig


class WallOSS05SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding used by the WALL-OSS action processor."""

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        if half_dim <= 1:
            emb = x[:, None]
            return torch.cat([emb, emb], dim=-1)[:, : self.dim]

        scale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -scale)
        emb = x.float()[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


def _linear_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = module(x)
    if isinstance(out, tuple):
        return out[0]
    return out


class WallOSS05ActionProcessorNative(nn.Module):
    """Native PHYAI action processor for WALL-OSS-0.5 flow action path."""

    def __init__(
        self,
        config: WallOSS05NativeConfig,
        *,
        params_dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if config.action_dim_internal <= 0:
            raise ValueError("config.dof_config must be overlaid before constructing action processor")

        self.config = config
        self.action_dim = config.action_dim_internal
        self.propri_dim = config.propri_dim_internal
        self.action_hidden_size = config.action_hidden_size
        self.state_hidden_size = config.state_hidden_size
        self.hidden_size = config.hidden_size
        self.proj_with_mask = config.proj_with_mask
        self.use_adarms = config.use_adarms

        self.s = float(config.noise_scheduler.get("s", 0.999))
        self.time_shift = float(config.noise_scheduler.get("time_shift", 1.0))
        self.time_embed = WallOSS05SinusoidalPosEmb(self.action_hidden_size)

        action_in = self.action_dim * 2 if self.proj_with_mask else self.action_dim

        self.w1 = ReplicatedLinear(
            action_in,
            self.action_hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="action_preprocessor.w1",
        )

        if self.use_adarms:
            raise NotImplementedError("WALL-OSS-0.5 baseline uses use_adarms=False")
        self.w2 = ReplicatedLinear(
            self.action_hidden_size * 2,
            self.action_hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="action_preprocessor.w2",
        )
        self.w3 = ReplicatedLinear(
            self.action_hidden_size,
            self.action_hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="action_preprocessor.w3",
        )
        self.act_fn = nn.SiLU()

        self.action_proj_back = ReplicatedLinear(
            self.action_hidden_size,
            self.action_dim,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="action_preprocessor.action_proj_back",
        )

    def get_inference_times(
        self,
        num_steps: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
        if self.time_shift != 1.0:
            times = (self.time_shift * times) / (1 + (self.time_shift - 1) * times)
        return times * self.s

    def step(
        self,
        timestep: torch.Tensor,
        noisy_action: torch.Tensor,
        dof_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if dof_mask is not None and self.proj_with_mask:
            noisy_action = torch.cat([noisy_action, dof_mask], dim=-1)

        time_embed = self.time_embed(timestep).to(dtype=noisy_action.dtype)
        action_embed = _linear_forward(self.w1, noisy_action)

        time_embed = time_embed.unsqueeze(1).repeat(1, action_embed.shape[1], 1)
        time_embed = time_embed.to(device=action_embed.device, dtype=action_embed.dtype)
        concat_embed = torch.cat([action_embed, time_embed], dim=-1)
        concat_embed = _linear_forward(self.w2, concat_embed)
        embed = _linear_forward(self.w3, self.act_fn(concat_embed))

        if self.action_hidden_size < self.hidden_size:
            padding_size = self.hidden_size - self.action_hidden_size
            padding = torch.zeros(
                (embed.shape[0], embed.shape[1], padding_size),
                device=embed.device,
                dtype=embed.dtype,
            )
            embed = torch.cat([embed, padding], dim=-1)

        return embed, None



def _activation_fn(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation {name!r}")


class WallOSS05BlockSparseMLPNative(nn.Module):
    """Single expert MLP used by WALL-OSS-0.5 SparseMoeBlock."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        params_dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        prefix: str,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.hidden_act = hidden_act

        self.gate_up_proj = ReplicatedLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = ReplicatedLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = _activation_fn(hidden_act)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        gate_up_out = _linear_forward(self.gate_up_proj, hidden_state)
        gate_out, up_out = gate_up_out.split(
            [self.intermediate_size, self.intermediate_size],
            dim=-1,
        )
        act_out = self.act_fn(gate_out) * up_out
        return _linear_forward(self.down_proj, act_out)


class WallOSS05SparseMoeBlockNative(nn.Module):
    """Expert-routed MLP block matching Wall-X SparseMoeBlock for mot_opt=True."""

    def __init__(
        self,
        config: WallOSS05NativeConfig,
        *,
        layer_idx: int,
        params_dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.dim_inputs = tuple(int(v) for v in config.dim_inputs)
        self.permuted = bool(config.mot_opt)

        if not self.permuted:
            raise NotImplementedError("Initial native SparseMoeBlock only supports mot_opt=True")

        if not config.experts:
            raise ValueError("config.experts must be present for SparseMoeBlock")

        self.experts = nn.ModuleList()
        for idx in range(self.num_experts):
            expert_cfg = config.experts[idx]
            self.experts.append(
                WallOSS05BlockSparseMLPNative(
                    hidden_size=int(expert_cfg["hidden_size"]),
                    intermediate_size=int(expert_cfg["intermediate_size"]),
                    hidden_act=str(expert_cfg["hidden_act"]),
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"model.layers.{layer_idx}.moe.experts.{idx}",
                )
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts_indices: torch.Tensor | None,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> torch.Tensor:
        del experts_indices

        permuted_inputs = hidden_states
        final_output = torch.zeros_like(permuted_inputs)

        for expert_idx, expert in enumerate(self.experts):
            start = int(start_indices[expert_idx].item())
            end = int(end_indices[expert_idx].item())
            if start == end:
                continue

            dim_input = self.dim_inputs[expert_idx]
            expert_input = permuted_inputs[start:end, :dim_input]
            partial_output = expert(expert_input)
            final_output[start:end, :dim_input] = partial_output[:, :dim_input]

        return final_output


def walloss05_native_weight_remap(key: str) -> str | None:
    """Initial remap for the action processor subset."""
    if key.startswith("action_preprocessor."):
        return key
    if key.startswith("model.layers.") and ".moe.experts." in key:
        return key
    return None


__all__ = [
    "WallOSS05ActionProcessorNative",
    "WallOSS05BlockSparseMLPNative",
    "WallOSS05SparseMoeBlockNative",
    "WallOSS05SinusoidalPosEmb",
    "walloss05_native_weight_remap",
]
