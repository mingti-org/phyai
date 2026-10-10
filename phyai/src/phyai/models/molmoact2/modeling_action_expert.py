"""Stateless MolmoAct2 flow-matching action expert."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers.activation import SiLU
from phyai.layers.mlp import DenseMLP
from phyai.layers.modulation import AffineModulation
from phyai.layers.linear import ReplicatedLinear
from phyai.layers.attention import AttnCtx, AttnMask, Attention
from phyai.layers.layer_norm import RMSNorm
from phyai.layers.rotary_embedding import RotaryEmbedding
from phyai.models.molmoact2.configuration_molmoact2 import MolmoAct2ActionExpertConfig


@dataclass(frozen=True)
class ActionExpertContext:
    kv_contexts: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    cross_mask: AttnMask | None
    self_mask: AttnMask | None
    valid_action: torch.Tensor | None
    rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
    self_attn_ctx: AttnCtx | None = None
    cross_attn_ctx: AttnCtx | None = None


@dataclass(frozen=True)
class ActionExpertStepModulation:
    conditioning: torch.Tensor
    block_modulations: tuple[tuple[torch.Tensor, ...], ...]
    final_modulation: tuple[torch.Tensor, torch.Tensor]


class ActionExpertSelfAttention(nn.Module):
    def __init__(
        self,
        config: MolmoAct2ActionExpertConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str | None,
        norm_backend: str | None,
        device: torch.device | str | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        norm_kwargs = dict(
            eps=config.qk_norm_eps,
            elementwise_affine=False,
            dtype=params_dtype,
            backend=norm_backend,
            device=device,
        )
        self.q_norm = RMSNorm(self.head_dim, **norm_kwargs) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, **norm_kwargs) if config.qk_norm else None
        self.qkv = ReplicatedLinear(
            config.hidden_size,
            3 * config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.qkv",
        )
        self.out_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = Attention(
            config.num_heads,
            self.head_dim,
            causal=config.causal_attn,
            backend=attn_backend,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: AttnMask | None = None,
        attn_ctx: AttnCtx | None = None,
        rope: RotaryEmbedding | None = None,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv, _ = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if rope is not None:
            if rope_cache is None:
                raise ValueError("RoPE requires the action context's cos/sin tensors")
            q, k = rope.apply(q, k, *rope_cache)
        out = self.attn(
            q,
            k,
            v.contiguous(),
            ctx=attn_ctx,
            mask=attn_mask if attn_ctx is None else None,
        )
        out = out.reshape(batch_size, seq_len, self.hidden_size)
        out, _ = self.out_proj(out)
        return out


class ActionExpertCrossAttention(nn.Module):
    def __init__(
        self,
        config: MolmoAct2ActionExpertConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str | None,
        norm_backend: str | None,
        device: torch.device | str | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        norm_kwargs = dict(
            eps=config.qk_norm_eps,
            elementwise_affine=False,
            dtype=params_dtype,
            backend=norm_backend,
            device=device,
        )
        self.q_norm = RMSNorm(self.head_dim, **norm_kwargs) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, **norm_kwargs) if config.qk_norm else None
        self.q_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.q_proj",
        )
        self.out_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.out_proj",
        )
        self.attn = Attention(
            config.num_heads, self.head_dim, causal=False, backend=attn_backend
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_k: torch.Tensor,
        kv_v: torch.Tensor,
        attn_mask: AttnMask | None = None,
        attn_ctx: AttnCtx | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q, _ = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        out = self.attn(
            q,
            kv_k,
            kv_v,
            ctx=attn_ctx,
            mask=attn_mask if attn_ctx is None else None,
        )
        out = out.reshape(batch_size, seq_len, self.hidden_size)
        out, _ = self.out_proj(out)
        return out


class ActionExpertModulation(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_chunks: int,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.act = SiLU()
        self.linear = ReplicatedLinear(
            hidden_size,
            num_chunks * hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear",
        )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        out, _ = self.linear(self.act(conditioning))
        return out


class ActionExpertBlock(nn.Module):
    def __init__(
        self,
        config: MolmoAct2ActionExpertConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str | None,
        norm_backend: str | None,
        device: torch.device | str | None,
        prefix: str,
    ) -> None:
        super().__init__()
        norm_kwargs = dict(
            elementwise_affine=False,
            dtype=params_dtype,
            backend=norm_backend,
            device=device,
        )
        self.self_norm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.cross_norm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.ff_norm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.modulate = AffineModulation()
        attention_kwargs = dict(
            params_dtype=params_dtype,
            norm_backend=norm_backend,
            device=device,
            attn_backend=attn_backend,
        )
        self.self_attn = ActionExpertSelfAttention(
            config, **attention_kwargs, prefix=f"{prefix}.self_attn"
        )
        self.cross_attn = ActionExpertCrossAttention(
            config, **attention_kwargs, prefix=f"{prefix}.cross_attn"
        )
        inner_dim = int(config.hidden_size * config.mlp_ratio)
        if config.ffn_multiple_of > 0:
            inner_dim = (
                math.ceil(inner_dim / config.ffn_multiple_of) * config.ffn_multiple_of
            )
        self.mlp = DenseMLP(
            config.hidden_size,
            inner_dim,
            activation="silu",
            gated=True,
            bias=True,
            cast_before_multiply=True,
            sequence_parallel=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.mlp",
        )
        self.modulation = ActionExpertModulation(
            config.hidden_size,
            9,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.modulation",
        )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        cross_kv: tuple[torch.Tensor, torch.Tensor],
        self_attn_mask: AttnMask | None = None,
        attn_mask: AttnMask | None = None,
        self_attn_ctx: AttnCtx | None = None,
        cross_attn_ctx: AttnCtx | None = None,
        modulation: tuple[torch.Tensor, ...] | None = None,
        rope: RotaryEmbedding | None = None,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if modulation is None:
            modulation = self.modulation(conditioning).chunk(9, dim=1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            self.modulate(self.self_norm(x), shift_msa, scale_msa),
            attn_mask=self_attn_mask,
            attn_ctx=self_attn_ctx,
            rope=rope,
            rope_cache=rope_cache,
        )
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            self.modulate(self.cross_norm(x), shift_mca, scale_mca),
            kv_k=cross_kv[0],
            kv_v=cross_kv[1],
            attn_mask=attn_mask,
            attn_ctx=cross_attn_ctx,
        )
        return x + gate_mlp.unsqueeze(1) * self.mlp(
            self.modulate(self.ff_norm(x), shift_mlp, scale_mlp)
        )


class ActionExpertFinalLayer(nn.Module):
    def __init__(
        self,
        config: MolmoAct2ActionExpertConfig,
        *,
        params_dtype: torch.dtype,
        norm_backend: str | None,
        device: torch.device | str | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(
            config.hidden_size,
            elementwise_affine=False,
            dtype=params_dtype,
            backend=norm_backend,
            device=device,
        )
        self.modulate = AffineModulation()
        self.modulation = ActionExpertModulation(
            config.hidden_size,
            2,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.modulation",
        )
        self.linear = ReplicatedLinear(
            config.hidden_size,
            config.max_action_dim,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear",
        )

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if modulation is None:
            modulation = self.modulation(conditioning).chunk(2, dim=1)
        shift, scale = modulation
        out, _ = self.linear(self.modulate(self.norm(x), shift, scale))
        return out


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim > 1:
            timesteps = timesteps.reshape(timesteps.shape[0], -1)[:, 0]
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = timesteps[:, None] * freq[None, :]
        out = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return F.pad(out, (0, 1)) if self.dim % 2 else out


class ActionExpert(nn.Module):
    def __init__(
        self,
        config: MolmoAct2ActionExpertConfig,
        *,
        llm_dim: int,
        llm_kv_dim: int,
        llm_num_layers: int,
        params_dtype: torch.dtype = torch.bfloat16,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "action_expert",
    ) -> None:
        super().__init__()
        if config.num_layers != llm_num_layers:
            raise ValueError("The action expert requires one block per language layer")
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_dim = llm_dim
        self.llm_kv_dim = llm_kv_dim
        self.action_head_dim = config.hidden_size // config.num_heads
        self.time_embed = nn.ModuleList(
            [
                SinusoidalTimeEmbedding(config.timestep_embed_dim),
                ReplicatedLinear(
                    config.timestep_embed_dim,
                    config.hidden_size,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.time_embed.1",
                ),
                SiLU(),
                ReplicatedLinear(
                    config.hidden_size,
                    config.hidden_size,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.time_embed.3",
                ),
            ]
        )
        self.action_embed = ReplicatedLinear(
            config.max_action_dim,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.action_embed",
        )
        self.context_k_proj = ReplicatedLinear(
            llm_kv_dim,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.context_k_proj",
        )
        self.context_v_proj = ReplicatedLinear(
            llm_kv_dim,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.context_v_proj",
        )
        self.context_norm = (
            RMSNorm(
                config.hidden_size,
                elementwise_affine=False,
                dtype=params_dtype,
                backend=norm_backend,
                device=device,
            )
            if config.context_layer_norm
            else nn.Identity()
        )
        self.rope = (
            RotaryEmbedding(
                self.action_head_dim,
                max_position_embeddings=config.max_action_horizon,
                device=device,
            )
            if config.rope
            else None
        )
        self.blocks = nn.ModuleList(
            ActionExpertBlock(
                config,
                params_dtype=params_dtype,
                attn_backend=attn_backend,
                norm_backend=norm_backend,
                device=device,
                prefix=f"{prefix}.blocks.{index}",
            )
            for index in range(config.num_layers)
        )
        self.final_layer = ActionExpertFinalLayer(
            config,
            params_dtype=params_dtype,
            norm_backend=norm_backend,
            device=device,
            prefix=f"{prefix}.final_layer",
        )

    def time_conditioning(self, timesteps: torch.Tensor) -> torch.Tensor:
        conditioning = self.time_embed[0](timesteps)
        conditioning = conditioning.to(dtype=self.time_embed[1].weight.dtype)
        conditioning, _ = self.time_embed[1](conditioning)
        conditioning = self.time_embed[2](conditioning)
        conditioning, _ = self.time_embed[3](conditioning)
        return conditioning

    def project_kv_tensor(
        self, x: torch.Tensor, proj: ReplicatedLinear
    ) -> torch.Tensor:
        flat, _ = proj(x)
        flat = self.context_norm(flat)
        return flat.view(*flat.shape[:2], self.config.num_heads, self.action_head_dim)

    def prepare_context(
        self,
        *,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        encoder_attention_mask: torch.Tensor | None = None,
        action_attention_mask: torch.Tensor | None = None,
        state_embeddings: torch.Tensor | None = None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ActionExpertContext:
        if state_embeddings is not None:
            raise ValueError(
                "MolmoAct2 uses discrete state tokens, not state embeddings"
            )
        if len(encoder_kv_states) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} encoder KV layers")
        if not 1 <= seq_len <= self.config.max_action_horizon:
            raise ValueError("Action length must be within the configured horizon")
        kv_contexts = []
        for block, (key, value) in zip(self.blocks, encoder_kv_states, strict=True):
            if (
                key.ndim != 3
                or key.shape != value.shape
                or key.shape[0] != batch_size
                or key.shape[-1] != self.llm_kv_dim
            ):
                raise ValueError(
                    "Encoder KV tensors must have shape (B, S, llm_kv_dim)"
                )
            key = self.project_kv_tensor(key, self.context_k_proj)
            value = self.project_kv_tensor(value, self.context_v_proj)
            if block.cross_attn.k_norm is not None:
                key = block.cross_attn.k_norm(key)
            kv_contexts.append((key, value))
        rope_cache = None
        if self.rope is not None:
            positions = torch.arange(seq_len, device=device)
            cos, sin = self.rope.get_cos_sin(positions)
            rope_cache = (cos.to(dtype=dtype), sin.to(dtype=dtype))
        valid_action = None
        self_mask = None
        if action_attention_mask is not None:
            if action_attention_mask.shape != (batch_size, seq_len):
                raise ValueError("Action mask must have shape (batch, action length)")
            valid = action_attention_mask.to(device=device, dtype=torch.bool)
            self_mask = AttnMask.from_key_mask(valid)
            valid_action = valid.to(dtype=dtype).unsqueeze(-1)
        cross_mask = (
            None
            if encoder_attention_mask is None
            else AttnMask.from_key_mask(encoder_attention_mask.to(device=device))
        )
        return ActionExpertContext(
            kv_contexts=tuple(kv_contexts),
            cross_mask=cross_mask,
            self_mask=self_mask,
            valid_action=valid_action,
            rope_cache=rope_cache,
        )

    def prepare_modulation_cache(
        self, timesteps: Sequence[torch.Tensor]
    ) -> tuple[ActionExpertStepModulation, ...]:
        result = []
        for timestep in timesteps:
            conditioning = self.time_conditioning(timestep)
            blocks = tuple(
                block.modulation(conditioning).chunk(9, dim=1) for block in self.blocks
            )
            shift, scale = self.final_layer.modulation(conditioning).chunk(2, dim=1)
            result.append(
                ActionExpertStepModulation(conditioning, blocks, (shift, scale))
            )
        return tuple(result)

    def forward_with_context(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: ActionExpertContext,
        modulation: ActionExpertStepModulation | None = None,
    ) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.config.max_action_dim:
            raise ValueError("Actions must have shape (B, horizon, max_action_dim)")
        if not 1 <= actions.shape[1] <= self.config.max_action_horizon:
            raise ValueError("Action length must be within the configured horizon")
        conditioning = (
            self.time_conditioning(timesteps)
            if modulation is None
            else modulation.conditioning
        )
        block_modulations = (
            (None,) * len(self.blocks)
            if modulation is None
            else modulation.block_modulations
        )
        x, _ = self.action_embed(actions)
        if context.valid_action is not None:
            x = x * context.valid_action
        for block, kv_context, block_modulation in zip(
            self.blocks, context.kv_contexts, block_modulations, strict=True
        ):
            x = block(
                x,
                conditioning,
                cross_kv=kv_context,
                self_attn_mask=context.self_mask,
                attn_mask=context.cross_mask,
                self_attn_ctx=context.self_attn_ctx,
                cross_attn_ctx=context.cross_attn_ctx,
                modulation=block_modulation,
                rope=self.rope,
                rope_cache=context.rope_cache,
            )
            if context.valid_action is not None:
                x = x * context.valid_action
        final_modulation = None if modulation is None else modulation.final_modulation
        out = self.final_layer(x, conditioning, modulation=final_modulation)
        return out if context.valid_action is None else out * context.valid_action

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: ActionExpertContext,
        modulation: ActionExpertStepModulation | None = None,
    ) -> torch.Tensor:
        return self.forward_with_context(
            actions, timesteps, context=context, modulation=modulation
        )


__all__ = ["ActionExpert", "ActionExpertContext", "ActionExpertStepModulation"]
