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
        self.weight = nn.Parameter(torch.ones(self.hidden_size, dtype=dtype, device=device), requires_grad=False)
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




class WallOSS05AttentionCoreNative(nn.Module):
    """Eager/SDPA attention core for WALL-OSS-0.5 JointQwen2VLAttention.

    Inputs are expected after qkv projection and M-RoPE:
      query_states: [B, S, num_heads, head_dim]
      key_states:   [B, S, num_kv_heads, head_dim]
      value_states: [B, S, num_kv_heads, head_dim]

    This module covers transpose, repeat_kv, mask handling, SDPA, and reshape.
    Projection, RoPE, output projection, and KV cache are intentionally out of scope.
    """

    def __init__(self, config: WallOSS05NativeConfig):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = True

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Match official JointQwen2VLAttention.repeat_kv layout.

        Input:  [B, S, num_kv_heads, D]
        Output: [B, S, num_kv_heads * n_rep, D]
        """
        if n_rep == 1:
            return hidden_states

        batch, slen, num_key_value_heads, head_dim = hidden_states.shape
        hidden_states = hidden_states.unsqueeze(3)
        hidden_states = hidden_states.expand(
            batch,
            slen,
            num_key_value_heads,
            n_rep,
            head_dim,
        )
        return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

    @staticmethod
    def _prepare_causal_mask(
        attention_mask: torch.Tensor | None,
        *,
        bsz: int,
        q_len: int,
        key_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        causal_mask = attention_mask
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                _, seq_len = attention_mask.shape
                causal_mask = attention_mask.view(bsz, 1, 1, seq_len).expand(
                    bsz, 1, q_len, seq_len
                )
            elif len(attention_mask.shape) == 3:
                causal_mask = attention_mask.unsqueeze(1)
            elif len(attention_mask.shape) == 4:
                causal_mask = attention_mask
            else:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

            causal_mask = causal_mask.to(torch.bool)

        if q_len == 1:
            causal_mask = torch.ones(
                bsz,
                1,
                1,
                key_len,
                device=device,
                dtype=dtype,
            ).contiguous()
            causal_mask = causal_mask.to(torch.bool)

        return causal_mask

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        projection_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        bsz, q_len, num_heads, head_dim = query_states.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                f"query shape mismatch: got heads={num_heads}, head_dim={head_dim}, "
                f"expected heads={self.num_heads}, head_dim={self.head_dim}"
            )

        if key_states.shape[:2] != (bsz, q_len) or value_states.shape[:2] != (bsz, q_len):
            raise ValueError("key/value states must have same batch and sequence length as query")

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if projection_dtype is not None:
            if query_states.dtype != projection_dtype:
                query_states = query_states.to(projection_dtype)
            if key_states.dtype != projection_dtype:
                key_states = key_states.to(projection_dtype)
            if value_states.dtype != projection_dtype:
                value_states = value_states.to(projection_dtype)

        causal_mask = self._prepare_causal_mask(
            attention_mask,
            bsz=bsz,
            q_len=q_len,
            key_len=key_states.shape[2],
            device=query_states.device,
            dtype=query_states.dtype,
        )

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        if projection_dtype is not None and attn_output.dtype != projection_dtype:
            attn_output = attn_output.to(projection_dtype)

        return attn_output




def walloss05_permute_by_expert(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable PyTorch equivalent of official permute(tokens, expert_indices)."""
    if expert_indices.dim() == 1:
        expert_indices = expert_indices.view(-1, 1)
    expand_factor = expert_indices.size(1)
    flatten_indices = expert_indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // expand_factor)
    return permuted_tokens, sorted_indices


def walloss05_unpermute_by_map(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stable PyTorch equivalent of official unpermute(permuted, sorted_indices, probs)."""
    if probs is not None:
        merge_factor = probs.size(1)
    else:
        merge_factor = 1

    unpermuted_tokens = torch.zeros_like(permuted_tokens)
    unpermuted_tokens.index_copy_(0, sorted_indices.long(), permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(
        -1,
        merge_factor,
        permuted_tokens.size(-1),
    )

    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)

    return unpermuted_tokens.sum(dim=1)


class WallOSS05JointAttentionNative(nn.Module):
    """No-cache, mot_opt native JointQwen2VLAttention path for WALL-OSS-0.5."""

    def __init__(
        self,
        config: WallOSS05NativeConfig,
        *,
        layer_idx: int,
        params_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if not config.mot_opt:
            raise NotImplementedError("Initial native JointAttention supports mot_opt=True only")

        rope_scaling = config.rope_scaling or {}
        mrope_section = rope_scaling.get("mrope_section")
        if mrope_section is None:
            raise ValueError("config.rope_scaling['mrope_section'] is required")

        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.dim_inputs = tuple(int(v) for v in config.dim_inputs)

        self.projections = WallOSS05JointAttentionProjectionNative(
            config,
            layer_idx=layer_idx,
            params_dtype=params_dtype,
            device=device,
        )
        self.mrope = WallOSS05MRoPENative(mrope_section)
        self.core = WallOSS05AttentionCoreNative(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        token_types: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
        row_id_map: torch.Tensor,
        probs: torch.Tensor | None,
        orig_shape: tuple[int, int, int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        projection_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = orig_shape
        if projection_dtype is None:
            projection_dtype = self.projections.qkv_proj_experts[0].weight.dtype

        q_buffer, k_buffer, v_buffer = self.projections.project_qkv_permuted(
            hidden_states,
            start_indices,
            end_indices,
        )

        q_unpermuted = walloss05_unpermute_by_map(q_buffer, row_id_map, probs)
        k_unpermuted = walloss05_unpermute_by_map(k_buffer, row_id_map, probs)
        v_unpermuted = walloss05_unpermute_by_map(v_buffer, row_id_map, probs)

        query_states = q_unpermuted.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = k_unpermuted.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = v_unpermuted.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = self.mrope(
            query_states.contiguous(),
            key_states.contiguous(),
            cos[..., : (cos.size(3) // 2)].contiguous().float(),
            sin[..., : (sin.size(3) // 2)].contiguous().float(),
        )

        attn_output = self.core(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            projection_dtype=projection_dtype,
        )

        flat_attn_output = attn_output.view(-1, self.hidden_size)
        flat_expert_indices = token_types.reshape(-1)
        permuted_attn_output, _ = walloss05_permute_by_expert(
            flat_attn_output,
            flat_expert_indices,
        )

        return self.projections.project_output_permuted(
            permuted_attn_output,
            start_indices,
            end_indices,
        )




class WallOSS05DecoderLayerNative(nn.Module):
    """No-cache, mot_opt decoder layer for WALL-OSS-0.5.

    Covers:
      input_norm_moe -> joint_attention -> residual add ->
      post_attention_norm_moe -> sparse_moe -> residual add.
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
        self.input_norm = WallOSS05NormMoeNative(
            config,
            layer_idx=layer_idx,
            kind="input",
            dtype=torch.float32,
            device=device,
        )
        self.self_attn = WallOSS05JointAttentionNative(
            config,
            layer_idx=layer_idx,
            params_dtype=params_dtype,
            device=device,
        )
        self.ffn = WallOSS05DecoderFFNBlockNative(
            config,
            layer_idx=layer_idx,
            params_dtype=params_dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        token_types: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
        row_id_map: torch.Tensor,
        probs: torch.Tensor | None,
        orig_shape: tuple[int, int, int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        adarms_conds: list[torch.Tensor | None] | None = None,
        projection_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, gate, gate_mask = self.input_norm(
            hidden_states,
            start_indices,
            end_indices,
            adarms_conds=adarms_conds,
        )
        if gate is not None or gate_mask is not None:
            raise NotImplementedError("WALL-OSS-0.5 baseline expects no adaptive RMSNorm gate")

        hidden_states = self.self_attn(
            hidden_states,
            token_types=token_types,
            start_indices=start_indices,
            end_indices=end_indices,
            row_id_map=row_id_map,
            probs=probs,
            orig_shape=orig_shape,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            projection_dtype=projection_dtype,
        )
        hidden_states = residual + hidden_states

        hidden_states = self.ffn(
            hidden_states,
            start_indices,
            end_indices,
            adarms_conds=adarms_conds,
        )
        return hidden_states




class WallOSS05DecoderModelNative(nn.Module):
    """Decoder-only native skeleton for WALL-OSS-0.5.

    This stacks all WALL-OSS-0.5 decoder layers and final expert-wise RMSNorms.
    Embeddings, vision tower, action generation loop, and processor/runtime logic
    are intentionally handled outside this skeleton.
    """

    def __init__(
        self,
        config: WallOSS05NativeConfig,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                WallOSS05DecoderLayerNative(
                    config,
                    layer_idx=layer_idx,
                    params_dtype=params_dtype,
                    device=device,
                )
                for layer_idx in range(int(config.num_hidden_layers))
            ]
        )
        self.norms = nn.ModuleList(
            [
                WallOSS05Qwen2RMSNormNative(
                    int(config.dim_inputs[expert_idx]),
                    eps=float(config.rms_norm_eps),
                    dtype=torch.float32,
                    device=device,
                    prefix=f"model.norms.{expert_idx}",
                )
                for expert_idx in range(int(config.num_experts))
            ]
        )

    def final_norm(
        self,
        hidden_states: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
    ) -> torch.Tensor:
        new_hidden_states = torch.zeros_like(hidden_states)
        for expert_idx, norm in enumerate(self.norms):
            start = int(start_indices[expert_idx].item())
            end = int(end_indices[expert_idx].item())
            if start == end:
                continue
            dim_input = int(self.config.dim_inputs[expert_idx])
            processed, gate = norm(hidden_states[start:end, :dim_input], None)
            if gate is not None:
                raise NotImplementedError("WALL-OSS-0.5 baseline expects no final adaptive norm gate")
            new_hidden_states[start:end, :dim_input] = processed.to(hidden_states.dtype)
        return new_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        token_types: torch.Tensor,
        start_indices: torch.Tensor,
        end_indices: torch.Tensor,
        row_id_map: torch.Tensor,
        probs: torch.Tensor | None,
        orig_shape: tuple[int, int, int],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        adarms_conds: list[torch.Tensor | None] | None = None,
        projection_dtype: torch.dtype | None = None,
        apply_final_norm: bool = True,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                token_types=token_types,
                start_indices=start_indices,
                end_indices=end_indices,
                row_id_map=row_id_map,
                probs=probs,
                orig_shape=orig_shape,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                adarms_conds=adarms_conds,
                projection_dtype=projection_dtype,
            )

        if apply_final_norm:
            hidden_states = self.final_norm(hidden_states, start_indices, end_indices)

        return hidden_states


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
    if key.startswith("model.norms."):
        return key
    return None


__all__ = [
    "WallOSS05ActionProcessorNative",
    "WallOSS05AttentionCoreNative",
    "WallOSS05BlockSparseMLPNative",
    "WallOSS05DecoderFFNBlockNative",
    "WallOSS05DecoderLayerNative",
    "WallOSS05DecoderModelNative",
    "WallOSS05JointAttentionNative",
    "WallOSS05JointAttentionProjectionNative",
    "WallOSS05MRoPENative",
    "WallOSS05NormMoeNative",
    "WallOSS05Qwen2RMSNormNative",
    "WallOSS05SparseMoeBlockNative",
    "WallOSS05SinusoidalPosEmb",
    "walloss05_native_weight_remap",
]
