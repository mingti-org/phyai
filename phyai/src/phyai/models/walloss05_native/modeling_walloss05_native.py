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




class WallOSS05Qwen2RMSNormNative(nn.Module):
    """Exact non-adaptive Qwen2 RMSNorm used by WALL-OSS-0.5.

    PR26's phyai.layers.RMSNorm currently exposes flashinfer / phyai-kernel
    backends only in this environment. This thin native layer mirrors the
    official Wall-X Qwen2RMSNorm ordinary path exactly and can later be swapped
    to phyai.layers.RMSNorm once a torch fallback backend is available.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.variance_epsilon = float(eps)
        self.prefix = prefix
        self.weight = nn.Parameter(torch.ones(self.hidden_size, dtype=dtype, device=device))
        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        if cond is not None:
            raise NotImplementedError("WALL-OSS-0.5 baseline uses use_adarms=False")

        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.variance_epsilon)
        normed = self.weight.to(torch.float32) * normed
        return normed.to(input_dtype), None


class WallOSS05NormMoeNative(nn.Module):
    """Expert-wise RMSNorm for WALL-OSS-0.5 norm_moe=True, mot_opt=True."""

    def __init__(
        self,
        config: WallOSS05NativeConfig,
        *,
        layer_idx: int,
        kind: str,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if kind not in {"input", "post_attention"}:
            raise ValueError(f"unknown norm kind {kind!r}")

        if not config.norm_moe:
            raise NotImplementedError("Initial native NormMoe only supports norm_moe=True")
        if not config.mot_opt:
            raise NotImplementedError("Initial native NormMoe only supports mot_opt=True")
        if config.use_adarms:
            raise NotImplementedError("WALL-OSS-0.5 baseline uses use_adarms=False")

        self.dim_inputs = tuple(int(v) for v in config.dim_inputs)
        self.num_experts = config.num_experts

        if kind == "input":
            prefix_base = f"model.layers.{layer_idx}.input_layernorms"
        else:
            prefix_base = f"model.layers.{layer_idx}.post_attention_layernorms"

        self.norms = nn.ModuleList(
            [
                WallOSS05Qwen2RMSNormNative(
                    self.dim_inputs[i],
                    eps=config.rms_norm_eps,
                    dtype=dtype,
                    device=device,
                    prefix=f"{prefix_base}.{i}",
                )
                for i in range(self.num_experts)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
        adarms_conds: list[torch.Tensor | None] | None = None,
    ) -> tuple[torch.Tensor, None, None]:
        if adarms_conds is None:
            adarms_conds = [None] * self.num_experts

        new_hidden_states = torch.zeros_like(hidden_states)

        for expert_idx, expert_norm in enumerate(self.norms):
            start = int(start_indices[expert_idx].item())
            end = int(end_indices[expert_idx].item())
            if start == end:
                continue

            dim_input = self.dim_inputs[expert_idx]
            selected = hidden_states[start:end]
            input_slice = selected[:, :dim_input]
            cond = adarms_conds[expert_idx]

            processed, gate = expert_norm(input_slice, cond)
            if gate is not None:
                raise NotImplementedError("Unexpected adaptive RMSNorm gate in baseline path")

            new_hidden_states[start:end, :dim_input] = processed.to(hidden_states.dtype)

        return new_hidden_states, None, None




class WallOSS05DecoderFFNBlockNative(nn.Module):
    """Non-attention FFN/MoE subpath of a WALL-OSS-0.5 decoder layer.

    This mirrors the post-attention part of Qwen2_5_VLDecoderLayer_with_MoE:
    residual -> post_attention_norm_moe -> sparse_moe -> residual add.
    Attention and KV-cache logic are intentionally out of scope for this block.
    """

    def __init__(
        self,
        config: WallOSS05NativeConfig,
        *,
        layer_idx: int,
        params_dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.post_attention_norm = WallOSS05NormMoeNative(
            config,
            layer_idx=layer_idx,
            kind="post_attention",
            dtype=torch.float32,
            device=device,
        )
        self.moe = WallOSS05SparseMoeBlockNative(
            config,
            layer_idx=layer_idx,
            params_dtype=params_dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
        adarms_conds: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, gate, gate_mask = self.post_attention_norm(
            hidden_states,
            start_indices,
            end_indices,
            adarms_conds=adarms_conds,
        )
        if gate is not None or gate_mask is not None:
            raise NotImplementedError("WALL-OSS-0.5 baseline FFN block expects no adaptive RMSNorm gate")

        hidden_states = self.moe(
            hidden_states,
            experts_indices=None,
            start_indices=start_indices,
            end_indices=end_indices,
        )
        return residual + hidden_states




class WallOSS05JointAttentionProjectionNative(nn.Module):
    """Projection-only native subset of WALL-OSS-0.5 JointQwen2VLAttention.

    This covers qkv_proj_experts and o_proj_experts only. RoPE, attention core,
    KV cache, and mask handling are intentionally out of scope for this module.
    """

    def __init__(
        self,
        config: WallOSS05NativeConfig,
        *,
        layer_idx: int,
        params_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.dim_inputs = tuple(int(v) for v in config.dim_inputs)

        qkv_out_features = (
            self.num_heads * self.head_dim
            + 2 * self.num_key_value_heads * self.head_dim
        )

        self.qkv_proj_experts = nn.ModuleList(
            [
                ReplicatedLinear(
                    dim_input,
                    qkv_out_features,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"model.layers.{layer_idx}.self_attn.qkv_proj_experts.{expert_idx}",
                )
                for expert_idx, dim_input in enumerate(self.dim_inputs)
            ]
        )

        self.o_proj_experts = nn.ModuleList(
            [
                ReplicatedLinear(
                    self.num_heads * self.head_dim,
                    dim_input,
                    bias=False,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"model.layers.{layer_idx}.self_attn.o_proj_experts.{expert_idx}",
                )
                for expert_idx, dim_input in enumerate(self.dim_inputs)
            ]
        )

    def project_qkv_permuted(
        self,
        hidden_states: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_tokens, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(f"hidden dim {hidden_dim} != expected {self.hidden_size}")

        device = hidden_states.device
        dtype = hidden_states.dtype
        kv_dim = self.num_key_value_heads * self.head_dim

        q_buffer = torch.zeros(total_tokens, self.hidden_size, device=device, dtype=dtype)
        k_buffer = torch.zeros(total_tokens, kv_dim, device=device, dtype=dtype)
        v_buffer = torch.zeros(total_tokens, kv_dim, device=device, dtype=dtype)

        for expert_idx, qkv_proj in enumerate(self.qkv_proj_experts):
            start = int(start_indices[expert_idx].item())
            end = int(end_indices[expert_idx].item())
            if start == end:
                continue

            dim_input = self.dim_inputs[expert_idx]
            expert_input = hidden_states[start:end, :dim_input]
            if expert_input.dtype != qkv_proj.weight.dtype:
                expert_input = expert_input.to(qkv_proj.weight.dtype)

            qkv_out = _linear_forward(qkv_proj, expert_input)
            q_out, k_out, v_out = torch.split(
                qkv_out,
                [self.hidden_size, kv_dim, kv_dim],
                dim=-1,
            )

            q_buffer[start:end] = q_out.to(dtype)
            k_buffer[start:end] = k_out.to(dtype)
            v_buffer[start:end] = v_out.to(dtype)

        return q_buffer, k_buffer, v_buffer

    def project_output_permuted(
        self,
        attn_output: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> torch.Tensor:
        total_tokens, hidden_dim = attn_output.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(f"attention hidden dim {hidden_dim} != expected {self.hidden_size}")

        device = attn_output.device
        dtype = attn_output.dtype
        output_buffer = torch.zeros(total_tokens, self.hidden_size, device=device, dtype=dtype)

        for expert_idx, o_proj in enumerate(self.o_proj_experts):
            start = int(start_indices[expert_idx].item())
            end = int(end_indices[expert_idx].item())
            if start == end:
                continue

            dim_input = self.dim_inputs[expert_idx]
            expert_input = attn_output[start:end]
            if expert_input.dtype != o_proj.weight.dtype:
                expert_input = expert_input.to(o_proj.weight.dtype)

            projected = _linear_forward(o_proj, expert_input)
            output_buffer[start:end, :dim_input] = projected[:, :dim_input].to(dtype)

        return output_buffer




def walloss05_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Official WALL-X/Qwen2.5-VL rotate-half: [x1, x2] -> [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class WallOSS05MRoPENative(nn.Module):
    """Strict official-compatible M-RoPE used by WALL-OSS-0.5 JointQwen2VLAttention.

    Mirrors wall_x.model.core.ops.rope.MRoPEOp._pytorch_fallback:
      1. cos/sin half-dim -> full-dim by cat((cos, cos), -1)
      2. split by mrope_section + mrope_section
      3. select temporal/height/width pieces with i % 3
      4. apply NeoX-style rotate_half
    """

    def __init__(self, mrope_section: list[int] | tuple[int, ...]):
        super().__init__()
        section = tuple(int(v) for v in mrope_section)
        if len(section) != 3:
            raise ValueError(f"mrope_section must have 3 entries, got {section}")
        if any(v <= 0 for v in section):
            raise ValueError(f"mrope_section entries must be positive, got {section}")
        self.mrope_section = section

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_dtype = query_states.dtype
        k_dtype = key_states.dtype

        cos = cos.float()
        sin = sin.float()

        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)

        mrope_section_doubled = list(self.mrope_section) + list(self.mrope_section)

        cos_split = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(mrope_section_doubled, dim=-1))],
            dim=-1,
        ).unsqueeze(2)
        sin_split = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(mrope_section_doubled, dim=-1))],
            dim=-1,
        ).unsqueeze(2)

        q_embed = (query_states.float() * cos_split) + (
            walloss05_rotate_half(query_states.float()) * sin_split
        )
        k_embed = (key_states.float() * cos_split) + (
            walloss05_rotate_half(key_states.float()) * sin_split
        )

        return q_embed.to(q_dtype), k_embed.to(k_dtype)


def walloss05_native_weight_remap(key: str) -> str | None:
    """Initial remap for the action processor subset."""
    if key.startswith("action_preprocessor."):
        return key
    if key.startswith("model.layers.") and ".moe.experts." in key:
        return key
    if key.startswith("model.layers.") and (".input_layernorms." in key or ".post_attention_layernorms." in key):
        return key
    if key.startswith("model.layers.") and (".self_attn.qkv_proj_experts." in key or ".self_attn.o_proj_experts." in key):
        return key
    return None


__all__ = [
    "WallOSS05ActionProcessorNative",
    "WallOSS05BlockSparseMLPNative",
    "WallOSS05DecoderFFNBlockNative",
    "WallOSS05JointAttentionProjectionNative",
    "WallOSS05MRoPENative",
    "WallOSS05NormMoeNative",
    "WallOSS05Qwen2RMSNormNative",
    "WallOSS05SparseMoeBlockNative",
    "WallOSS05SinusoidalPosEmb",
    "walloss05_native_weight_remap",
]
