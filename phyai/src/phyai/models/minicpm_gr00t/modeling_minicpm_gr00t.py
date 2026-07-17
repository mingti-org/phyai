"""Pure MiniCPM-V 4.6 GR00T architecture without runtime orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.engine_config import get_engine_config, resolve_engine_defaults
from phyai.layers.attention.attention.layer import Attention
from phyai.layers.attention.gdn import GatedDeltaNet
from phyai.layers.conv import Conv1d, Conv2d
from phyai.layers.layer_norm import GemmaRMSNorm, LayerNorm
from phyai.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.layers.mlp.dense_mlp import DenseMLP
from phyai.layers.rotary_embedding import RotaryEmbedding
from phyai.layers.vocab_embedding import VocabParallelEmbedding
from phyai.models.minicpm_gr00t.configuration_minicpm_gr00t import (
    MiniCPMGR00TActionConfig,
    MiniCPMGR00TConfig,
    MiniCPMGR00TDiTConfig,
    MiniCPMGR00TTextConfig,
    MiniCPMGR00TVisionConfig,
)
from phyai.weights.shards import replicated


if TYPE_CHECKING:
    from phyai.layers.attention import ARAttnCtx, GatedDeltaNetCtx


def minicpm_gr00t_weight_remap(name: str) -> str | None:
    """Map training-checkpoint names onto PHYAI DenseMLP placement names."""
    name = name.removeprefix("module.")
    if name.startswith("vlm.llm.lm_head."):
        return None
    name = name.replace(
        "vlm.resampler.mlp.0.mlp.0.",
        "vlm.resampler.mlp.0.mlp.fc1.",
    )
    name = name.replace(
        "vlm.resampler.mlp.0.mlp.2.",
        "vlm.resampler.mlp.0.mlp.fc2.",
    )
    name = name.replace(".ff.net.0.proj.", ".ff.fc1.")
    return name.replace(".ff.net.2.", ".ff.fc2.")


def _fp32_norm_backend(norm_backend: str, dtype: torch.dtype) -> str:
    if norm_backend == "flashinfer" and dtype == torch.float32:
        return "phyai-kernel"
    return norm_backend


def _attention_backend_for_head_dim(attn_backend: str, head_dim: int) -> str:
    if attn_backend == "flashinfer" and head_dim == 72:
        return "sdpa"
    return attn_backend


class MiniCPMGR00TQwenRMSNormGated(nn.Module):
    """Qwen3.5 head-wise RMSNorm followed by a SiLU gate."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.weight.hf_keys = [(f"{prefix}.weight", None)]
        self.weight.weight_loader = replicated()

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if hidden_states.is_cuda:
            from phyai_kernel import rmsnorm_silu_mul

            return rmsnorm_silu_mul(
                hidden_states.contiguous(),
                gate.contiguous(),
                self.weight,
                self.eps,
            )
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        normalized = self.weight * normalized.to(input_dtype)
        return (normalized * F.silu(gate.float())).to(input_dtype)


class MiniCPMGR00TQwenGatedDeltaNet(nn.Module):
    """Qwen3.5 projection, causal-convolution, and GatedDeltaNet stack."""

    def __init__(
        self,
        config: MiniCPMGR00TTextConfig,
        *,
        params_dtype: torch.dtype,
        gdn_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [
                self.conv_dim,
                self.value_dim,
                self.num_value_heads,
                self.num_value_heads,
            ],
            bias=False,
            params_dtype=params_dtype,
            device=device,
            hf_legs=("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"),
            prefix=f"{prefix}.in_proj",
        )
        self.conv1d = Conv1d(
            self.conv_dim,
            self.conv_dim,
            config.linear_conv_kernel_dim,
            padding=config.linear_conv_kernel_dim - 1,
            groups=self.conv_dim,
            bias=False,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.conv1d",
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_value_heads, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.A_log.hf_keys = [(f"{prefix}.A_log", None)]
        self.A_log.weight_loader = replicated()
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_value_heads, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        self.dt_bias.hf_keys = [(f"{prefix}.dt_bias", None)]
        self.dt_bias.weight_loader = replicated()
        self.gdn = GatedDeltaNet(
            self.num_value_heads,
            self.value_head_dim,
            num_key_heads=self.num_value_heads,
            num_value_heads=self.num_value_heads,
            backend=gdn_backend,
        )
        self.norm = MiniCPMGR00TQwenRMSNormGated(
            self.value_head_dim,
            config.rms_norm_eps,
            device=device,
            prefix=f"{prefix}.norm",
        )
        self.out_proj = ReplicatedLinear(
            self.value_dim,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.out_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        gdn_ctx: GatedDeltaNetCtx | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None and hidden_states.shape[0] > 1:
            hidden_states = hidden_states * attention_mask[..., None]
        batch_size, seq_len, _ = hidden_states.shape
        projected, _ = self.in_proj(hidden_states)
        mixed_qkv, z, a, b = projected.split(
            (
                self.conv_dim,
                self.value_dim,
                self.num_value_heads,
                self.num_value_heads,
            ),
            dim=-1,
        )
        if mixed_qkv.is_cuda:
            from phyai_kernel import causal_conv1d_silu_split_qkv

            query, key, value = causal_conv1d_silu_split_qkv(
                mixed_qkv,
                self.conv1d.weight,
                (self.key_dim, self.key_dim, self.value_dim),
            )
        else:
            mixed_qkv = self.conv1d(mixed_qkv.transpose(1, 2))[:, :, :seq_len]
            mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)
            query, key, value = torch.split(
                mixed_qkv,
                (self.key_dim, self.key_dim, self.value_dim),
                dim=-1,
            )
        query = query.reshape(
            batch_size, seq_len, self.num_key_heads, self.key_head_dim
        )
        key = key.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(
            batch_size, seq_len, self.num_value_heads, self.value_head_dim
        )
        if self.num_value_heads != self.num_key_heads:
            repeats = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        output = self.gdn(
            query,
            key,
            value,
            a.contiguous(),
            b.contiguous(),
            self.A_log,
            self.dt_bias,
            ctx=gdn_ctx,
        )
        output = self.norm(
            output.reshape(-1, self.value_head_dim),
            z.contiguous().reshape(-1, self.value_head_dim),
        ).reshape(batch_size, seq_len, self.value_dim)
        output, _ = self.out_proj(output)
        return output


class MiniCPMGR00TQwenAttention(nn.Module):
    """Gated Qwen3.5 causal full attention."""

    def __init__(
        self,
        config: MiniCPMGR00TTextConfig,
        *,
        rotary_emb: RotaryEmbedding,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_emb = rotary_emb
        query_gate_size = self.num_heads * self.head_dim * 2
        key_value_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [query_gate_size, key_value_size, key_value_size],
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            hf_legs=("q_proj", "k_proj", "v_proj"),
            prefix=f"{prefix}.qkv_proj",
        )
        self.q_norm = GemmaRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.q_norm",
        )
        self.k_norm = GemmaRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.k_norm",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_kv_heads,
            causal=True,
            backend=attn_backend,
        )
        self.o_proj = ReplicatedLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: ARAttnCtx | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        projected, _ = self.qkv_proj(hidden_states)
        query_gate, key, value = projected.split(
            (
                self.num_heads * self.head_dim * 2,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ),
            dim=-1,
        )
        query, gate = query_gate.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim * 2,
        ).chunk(2, dim=-1)
        query = self.q_norm(query)
        key = self.k_norm(
            key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        )
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        query, key = self.rotary_emb.apply(query, key, cos, sin)
        output = self.attn(query, key, value, ctx=attn_ctx)
        output = output.reshape(batch_size, seq_len, -1)
        output = output * torch.sigmoid(gate.reshape(batch_size, seq_len, -1))
        output, _ = self.o_proj(output)
        return output


class MiniCPMGR00TQwenDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MiniCPMGR00TTextConfig,
        layer_idx: int,
        *,
        rotary_emb: RotaryEmbedding,
        params_dtype: torch.dtype,
        attn_backend: str,
        gdn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = MiniCPMGR00TQwenGatedDeltaNet(
                config,
                params_dtype=params_dtype,
                gdn_backend=gdn_backend,
                device=device,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = MiniCPMGR00TQwenAttention(
                config,
                rotary_emb=rotary_emb,
                params_dtype=params_dtype,
                attn_backend=attn_backend,
                norm_backend=norm_backend,
                device=device,
                prefix=f"{prefix}.self_attn",
            )
        self.mlp = DenseMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="silu",
            gated=True,
            bias=False,
            params_dtype=params_dtype,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.input_layernorm",
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.post_attention_layernorm",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attn_ctx: ARAttnCtx | None = None,
        gdn_ctx: GatedDeltaNetCtx | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            normalized = self.input_layernorm(hidden_states)
        else:
            normalized, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                normalized,
                attention_mask=attention_mask,
                gdn_ctx=gdn_ctx,
            )
        else:
            hidden_states = self.self_attn(
                normalized,
                cos=cos,
                sin=sin,
                attn_ctx=attn_ctx,
            )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        return self.mlp(hidden_states), residual


class MiniCPMGR00TTextModel(nn.Module):
    """Qwen3.5 hybrid text model with runner-provided RoPE tensors."""

    def __init__(
        self,
        config: MiniCPMGR00TTextConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        gdn_backend: str = "flashinfer",
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "vlm.llm.model",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        if device is None:
            device = get_engine_config().device.target
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.embed_tokens",
        )
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_type=config.rope_type,
            partial_rotary_factor=config.partial_rotary_factor,
            backend="eager",
            device=device,
        )
        del self.rotary_emb.cos_sin_cache
        self.layers = nn.ModuleList(
            [
                MiniCPMGR00TQwenDecoderLayer(
                    config,
                    layer_idx,
                    rotary_emb=self.rotary_emb,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    gdn_backend=gdn_backend,
                    norm_backend=norm_backend,
                    device=device,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm",
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attn_ctx: ARAttnCtx | None = None,
        gdn_ctxs: tuple[GatedDeltaNetCtx | None, ...] | None = None,
    ) -> torch.Tensor:
        if gdn_ctxs is not None and len(gdn_ctxs) != len(self.layers):
            raise ValueError(
                f"gdn_ctxs has {len(gdn_ctxs)} entries; expected {len(self.layers)}."
            )
        hidden_states = inputs_embeds
        residual = None
        for layer_idx, layer in enumerate(self.layers):
            gdn_ctx = None if gdn_ctxs is None else gdn_ctxs[layer_idx]
            hidden_states, residual = layer(
                hidden_states,
                residual=residual,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                attn_ctx=attn_ctx,
                gdn_ctx=gdn_ctx,
            )
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class MiniCPMGR00TVisionAttention(nn.Module):
    """SigLIP MHA using independent checkpoint projections."""

    def __init__(
        self,
        config: MiniCPMGR00TVisionConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.qkv_proj",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            causal=False,
            backend=attn_backend,
        )
        self.out_proj = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.out_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        projected, _ = self.qkv_proj(hidden_states)
        query, key, value = projected.split(self.hidden_size, dim=-1)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)
        if cu_seqlens is None:
            output = self.attn(query, key, value)
        else:
            if batch_size != 1:
                raise ValueError("Ragged vision attention requires batch_size=1.")
            output = self.attn(
                query.squeeze(0),
                key.squeeze(0),
                value.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ).unsqueeze(0)
        output, _ = self.out_proj(output.reshape(batch_size, seq_len, -1))
        return output


class MiniCPMGR00TVisionLayer(nn.Module):
    def __init__(
        self,
        config: MiniCPMGR00TVisionConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_norm1 = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer_norm1",
        )
        self.self_attn = MiniCPMGR00TVisionAttention(
            config,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            device=device,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm2 = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer_norm2",
        )
        self.mlp = DenseMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="gelu_pytorch_tanh",
            gated=False,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states),
            cu_seqlens=cu_seqlens,
        )
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class MiniCPMGR00TVisionEmbeddings(nn.Module):
    """Patch projection plus runner-indexed learned position embeddings."""

    def __init__(
        self,
        config: MiniCPMGR00TVisionConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_embedding = Conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
            padding=0,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.patch_embedding",
        )
        self.position_embedding = VocabParallelEmbedding(
            config.num_position_embeddings,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.position_embedding",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        position_embeds = self.position_embedding(position_ids)
        if position_embeds.ndim == 2:
            position_embeds = position_embeds.unsqueeze(0)
        if position_embeds.shape != patch_embeds.shape:
            raise ValueError(
                f"position embeddings shape {tuple(position_embeds.shape)} must "
                f"match patch embeddings shape {tuple(patch_embeds.shape)}."
            )
        return patch_embeds + position_embeds


class MiniCPMGR00TVisionWindowMerger(nn.Module):
    """Window attention and 2x2 ViT-MLP merge with runner-owned indexing."""

    def __init__(
        self,
        config: MiniCPMGR00TVisionConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str = "vlm.vit_merger",
    ) -> None:
        super().__init__()
        self.layer_norm1 = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer_norm1",
        )
        self.self_attn = MiniCPMGR00TVisionAttention(
            config,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            device=device,
            prefix=f"{prefix}.self_attn",
        )
        self.pre_norm = LayerNorm(
            config.merged_hidden_size,
            eps=1e-6,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.pre_norm",
        )
        self.linear_1 = ColumnParallelLinear(
            config.merged_hidden_size,
            config.merger_intermediate_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_1",
        )
        self.linear_2 = RowParallelLinear(
            config.merger_intermediate_size,
            config.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_2",
        )

    def attend_windows(
        self,
        window_hidden_states: torch.Tensor,
        window_cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend already window-ordered tokens; the runner restores order."""
        attention_output = self.self_attn(
            self.layer_norm1(window_hidden_states),
            cu_seqlens=window_cu_seqlens,
        )
        return window_hidden_states + attention_output

    def downsample(
        self,
        grouped_hidden_states: torch.Tensor,
        pooled_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Merge runner-grouped 2x2 tokens and add their mean residual."""
        hidden_states, _ = self.linear_1(self.pre_norm(grouped_hidden_states))
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states + pooled_residual


class MiniCPMGR00TVisionResampler(nn.Module):
    """Final 2x2 downsample MLP from SigLIP width to Qwen width."""

    def __init__(
        self,
        config: MiniCPMGR00TVisionConfig,
        *,
        params_dtype: torch.dtype,
        norm_backend: str,
        device: torch.device | str,
        prefix: str = "vlm.resampler.mlp.0",
    ) -> None:
        super().__init__()
        self.pre_norm = LayerNorm(
            config.merged_hidden_size,
            eps=1e-6,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.pre_norm",
        )
        self.mlp = DenseMLP(
            config.merged_hidden_size,
            config.merged_hidden_size,
            activation="gelu",
            gated=False,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.mlp",
        )
        if self.mlp.fc2.out_features != config.merged_hidden_size:
            raise AssertionError("DenseMLP topology was unexpectedly changed.")
        self.output = RowParallelLinear(
            config.merged_hidden_size,
            config.output_hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.mlp.fc2",
        )
        del self.mlp.fc2

    def forward(self, grouped_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(grouped_hidden_states)
        hidden_states, _ = self.mlp.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states, _ = self.output(hidden_states)
        return hidden_states


class MiniCPMGR00TVisionModel(nn.Module):
    """SigLIP stages split around the runner-managed window permutation."""

    def __init__(
        self,
        config: MiniCPMGR00TVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "vlm.vpm",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        if device is None:
            device = get_engine_config().device.target
        vision_attn_backend = _attention_backend_for_head_dim(
            attn_backend, config.head_dim
        )
        self.config = config
        self.embeddings = MiniCPMGR00TVisionEmbeddings(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.embeddings",
        )
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [
                MiniCPMGR00TVisionLayer(
                    config,
                    params_dtype=params_dtype,
                    attn_backend=vision_attn_backend,
                    norm_backend=norm_backend,
                    device=device,
                    prefix=f"{prefix}.encoder.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.post_layernorm",
        )

    def embed(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embeddings(pixel_values, position_ids)

    def encode_prefix(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_idx in range(self.config.insert_layer_id + 1):
            hidden_states = self.encoder.layers[layer_idx](
                hidden_states,
                cu_seqlens=cu_seqlens,
            )
        return hidden_states

    def encode_suffix(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer_idx in range(
            self.config.insert_layer_id + 1,
            self.config.num_hidden_layers,
        ):
            hidden_states = self.encoder.layers[layer_idx](
                hidden_states,
                cu_seqlens=cu_seqlens,
            )
        return hidden_states

    def finalize(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.post_layernorm(hidden_states)


class MiniCPMGR00TVLM(nn.Module):
    """MiniCPM-V parameter container with explicit vision and text stages."""

    def __init__(
        self,
        config: MiniCPMGR00TConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        gdn_backend: str,
        norm_backend: str,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.llm = nn.Module()
        self.llm.model = MiniCPMGR00TTextModel(
            config.text,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            gdn_backend=gdn_backend,
            norm_backend=norm_backend,
            device=device,
            prefix="vlm.llm.model",
        )
        self.vpm = MiniCPMGR00TVisionModel(
            config.vision,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            device=device,
            prefix="vlm.vpm",
        )
        vision_attn_backend = _attention_backend_for_head_dim(
            attn_backend, config.vision.head_dim
        )
        self.vit_merger = MiniCPMGR00TVisionWindowMerger(
            config.vision,
            params_dtype=params_dtype,
            attn_backend=vision_attn_backend,
            norm_backend=norm_backend,
            device=device,
        )
        self.resampler = nn.Module()
        self.resampler.mlp = nn.ModuleList(
            [
                MiniCPMGR00TVisionResampler(
                    config.vision,
                    params_dtype=params_dtype,
                    norm_backend=norm_backend,
                    device=device,
                )
            ]
        )


class MiniCPMGR00TAdaLayerNorm(nn.Module):
    """Standard LayerNorm modulated by timestep scale and shift."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        params_dtype: torch.dtype,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.linear = ReplicatedLinear(
            hidden_size,
            hidden_size * 2,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear",
        )
        self.norm = LayerNorm(
            hidden_size,
            eps=eps,
            backend=norm_backend,
            bias=False,
            dtype=params_dtype,
            device=device,
            prefix="",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        modulation, _ = self.linear(F.silu(timestep_embedding))
        scale, shift = modulation.chunk(2, dim=-1)
        return self.norm(hidden_states) * (1.0 + scale[:, None]) + shift[:, None]


class MiniCPMGR00TDiTAttention(nn.Module):
    """Self/cross attention with independent q/k/v checkpoint projections."""

    def __init__(
        self,
        config: MiniCPMGR00TDiTConfig,
        context_dim: int,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.self_attention = context_dim == self.hidden_size
        if self.self_attention:
            self.to_qkv = QKVParallelLinear(
                self.hidden_size,
                self.head_dim,
                self.num_heads,
                bias=True,
                params_dtype=params_dtype,
                device=device,
                hf_legs={"q": "to_q", "k": "to_k", "v": "to_v"},
                prefix=f"{prefix}.to_qkv",
            )
        else:
            self.to_q = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                params_dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.to_q",
            )
            self.to_kv = MergedColumnParallelLinear(
                context_dim,
                [self.hidden_size, self.hidden_size],
                bias=True,
                params_dtype=params_dtype,
                device=device,
                hf_legs=("to_k", "to_v"),
                prefix=f"{prefix}.to_kv",
            )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            causal=False,
            backend=attn_backend,
        )
        self.to_out = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.to_out.0",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_len, _ = hidden_states.shape
        key_len = context.shape[1]
        if self.self_attention:
            projected, _ = self.to_qkv(hidden_states)
            query, key, value = projected.split(self.hidden_size, dim=-1)
        else:
            query, _ = self.to_q(hidden_states)
            projected, _ = self.to_kv(context)
            key, value = projected.split(self.hidden_size, dim=-1)
        query = query.view(batch_size, query_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, key_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, key_len, self.num_heads, self.head_dim)
        output = self.attn(query, key, value)
        output, _ = self.to_out(output.reshape(batch_size, query_len, -1))
        return output


class MiniCPMGR00TDiTBlock(nn.Module):
    def __init__(
        self,
        config: MiniCPMGR00TDiTConfig,
        layer_idx: int,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.cross_attention = layer_idx % 2 == 0
        context_dim = (
            config.cross_attention_dim if self.cross_attention else config.hidden_size
        )
        self.norm1 = MiniCPMGR00TAdaLayerNorm(
            config.hidden_size,
            config.norm_eps,
            params_dtype=params_dtype,
            norm_backend=norm_backend,
            device=device,
            prefix=f"{prefix}.norm1",
        )
        self.attn1 = MiniCPMGR00TDiTAttention(
            config,
            context_dim,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            device=device,
            prefix=f"{prefix}.attn1",
        )
        self.norm3 = LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            backend=norm_backend,
            bias=False,
            dtype=params_dtype,
            device=device,
            prefix="",
        )
        self.ff = DenseMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="gelu_pytorch_tanh",
            gated=False,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.ff",
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.ff_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(hidden_states, timestep_embedding)
        context = encoder_hidden_states if self.cross_attention else normalized
        hidden_states = hidden_states + self.attention_dropout(
            self.attn1(normalized, context)
        )
        return hidden_states + self.ff_dropout(self.ff(self.norm3(hidden_states)))


class MiniCPMGR00TTimestepEncoder(nn.Module):
    """Learned half of timestep encoding; sinusoidal input comes from the runner."""

    def __init__(
        self,
        config: MiniCPMGR00TDiTConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.input_dim = config.timestep_input_dim
        self.linear_1 = ReplicatedLinear(
            config.timestep_input_dim,
            config.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_1",
        )
        self.linear_2 = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, timestep_sinusoid: torch.Tensor) -> torch.Tensor:
        if timestep_sinusoid.shape[-1] != self.input_dim:
            raise ValueError(
                f"timestep_sinusoid last dim={timestep_sinusoid.shape[-1]} must "
                f"equal {self.input_dim}."
            )
        hidden_states, _ = self.linear_1(timestep_sinusoid)
        hidden_states = F.silu(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class MiniCPMGR00TDiT(nn.Module):
    """DiT-B with alternating VLM cross-attention and action self-attention."""

    def __init__(
        self,
        config: MiniCPMGR00TDiTConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str = "action_head.model",
    ) -> None:
        super().__init__()
        self.timestep_encoder = nn.Module()
        self.timestep_encoder.timestep_embedder = MiniCPMGR00TTimestepEncoder(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.timestep_encoder.timestep_embedder",
        )
        self.transformer_blocks = nn.ModuleList(
            [
                MiniCPMGR00TDiTBlock(
                    config,
                    layer_idx,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    device=device,
                    prefix=f"{prefix}.transformer_blocks.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm_out = LayerNorm(
            config.hidden_size,
            eps=config.output_norm_eps,
            backend=norm_backend,
            bias=False,
            dtype=params_dtype,
            device=device,
            prefix="",
        )
        self.proj_out_1 = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size * 2,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.proj_out_1",
        )
        self.proj_out_2 = ReplicatedLinear(
            config.hidden_size,
            config.output_dim,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.proj_out_2",
        )

    def encode_timestep(self, timestep_sinusoid: torch.Tensor) -> torch.Tensor:
        return self.timestep_encoder.timestep_embedder(timestep_sinusoid)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_sinusoid: torch.Tensor,
    ) -> torch.Tensor:
        timestep_embedding = self.encode_timestep(timestep_sinusoid)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_embedding,
            )
        modulation, _ = self.proj_out_1(F.silu(timestep_embedding))
        shift, scale = modulation.chunk(2, dim=-1)
        hidden_states = (
            self.norm_out(hidden_states) * (1.0 + scale[:, None]) + shift[:, None]
        )
        hidden_states, _ = self.proj_out_2(hidden_states)
        return hidden_states


class MiniCPMGR00TActionEncoder(nn.Module):
    """Boundary MLP split so the runner owns action/proprio/time concatenation."""

    def __init__(
        self,
        config: MiniCPMGR00TActionConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str = "action_head.action_encoder",
    ) -> None:
        super().__init__()
        hidden_size = config.dit.hidden_size
        self.layer1 = ReplicatedLinear(
            config.action_encoder_input_dim,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer1",
        )
        self.layer2 = ReplicatedLinear(
            hidden_size * 2,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer2",
        )
        self.layer3 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer3",
        )

    def project_action_proprio(self, action_proprio: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.layer1(action_proprio)
        return hidden_states

    def forward(self, projected_action_and_time: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.layer2(projected_action_and_time)
        hidden_states = F.silu(hidden_states)
        hidden_states, _ = self.layer3(hidden_states)
        return hidden_states


class MiniCPMGR00TActionDecoder(nn.Module):
    def __init__(
        self,
        config: MiniCPMGR00TActionConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str = "action_head.action_decoder",
    ) -> None:
        super().__init__()
        self.layer1 = ReplicatedLinear(
            config.dit.output_dim,
            config.dit.output_dim,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer1",
        )
        self.layer2 = ReplicatedLinear(
            config.dit.output_dim,
            config.action_dim,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.layer1(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states, _ = self.layer2(hidden_states)
        return hidden_states


class MiniCPMGR00TActionHead(nn.Module):
    """Action boundary layers and DiT with no sampling or denoise schedule."""

    def __init__(
        self,
        config: MiniCPMGR00TActionConfig,
        *,
        params_dtype: torch.dtype = torch.float32,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str = "action_head",
    ) -> None:
        super().__init__()
        action_attn_backend = (
            "sdpa"
            if attn_backend == "flashinfer" and params_dtype == torch.float32
            else attn_backend
        )
        action_norm_backend = _fp32_norm_backend(norm_backend, params_dtype)
        self.config = config
        self.model = MiniCPMGR00TDiT(
            config.dit,
            params_dtype=params_dtype,
            attn_backend=action_attn_backend,
            norm_backend=action_norm_backend,
            device=device,
            prefix=f"{prefix}.model",
        )
        self.action_encoder = MiniCPMGR00TActionEncoder(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.action_encoder",
        )
        self.action_decoder = MiniCPMGR00TActionDecoder(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.action_decoder",
        )
        self.future_tokens = VocabParallelEmbedding(
            config.num_future_tokens,
            config.dit.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.future_tokens",
        )
        self.position_embedding = VocabParallelEmbedding(
            config.max_position_embeddings,
            config.dit.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.position_embedding",
        )

    def project_action_proprio(self, action_proprio: torch.Tensor) -> torch.Tensor:
        return self.action_encoder.project_action_proprio(action_proprio)

    def encode_action(self, projected_action_and_time: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(projected_action_and_time)

    def embed_future(self, future_token_ids: torch.Tensor) -> torch.Tensor:
        return self.future_tokens(future_token_ids)

    def embed_action_positions(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.position_embedding(position_ids)

    def run_dit(
        self,
        hidden_states: torch.Tensor,
        vlm_hidden_states: torch.Tensor,
        timestep_sinusoid: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            hidden_states,
            vlm_hidden_states,
            timestep_sinusoid,
        )

    def decode_action(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.action_decoder(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_hidden_states: torch.Tensor,
        timestep_sinusoid: torch.Tensor,
    ) -> torch.Tensor:
        return self.decode_action(
            self.run_dit(hidden_states, vlm_hidden_states, timestep_sinusoid)
        )


class MiniCPMGR00TModel(nn.Module):
    """Full parameter container exposing runner-oriented architecture stages."""

    def __init__(
        self,
        config: MiniCPMGR00TConfig,
        *,
        vlm_params_dtype: torch.dtype = torch.bfloat16,
        action_params_dtype: torch.dtype = torch.float32,
        attn_backend: str | None = None,
        gdn_backend: str = "flashinfer",
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        _, attn_backend, norm_backend = resolve_engine_defaults(
            vlm_params_dtype,
            attn_backend,
            norm_backend,
        )
        if device is None:
            device = get_engine_config().device.target
        self.config = config
        self.vlm = MiniCPMGR00TVLM(
            config,
            params_dtype=vlm_params_dtype,
            attn_backend=attn_backend,
            gdn_backend=gdn_backend,
            norm_backend=norm_backend,
            device=device,
        )
        self.action_head = MiniCPMGR00TActionHead(
            config.action,
            params_dtype=action_params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            device=device,
        )

    def embed_vision(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.vlm.vpm.embed(pixel_values, position_ids)

    def encode_vision_prefix(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.vlm.vpm.encode_prefix(hidden_states, cu_seqlens=cu_seqlens)

    def attend_vision_windows(
        self,
        window_hidden_states: torch.Tensor,
        window_cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.vlm.vit_merger.attend_windows(
            window_hidden_states,
            window_cu_seqlens,
        )

    def merge_vision_windows(
        self,
        grouped_hidden_states: torch.Tensor,
        pooled_residual: torch.Tensor,
    ) -> torch.Tensor:
        return self.vlm.vit_merger.downsample(
            grouped_hidden_states,
            pooled_residual,
        )

    def encode_vision_suffix(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.vlm.vpm.encode_suffix(hidden_states, cu_seqlens=cu_seqlens)

    def finish_vision(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.vlm.vpm.finalize(hidden_states)

    def downsample_vision(
        self,
        grouped_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.vlm.resampler.mlp[0](grouped_hidden_states)

    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.vlm.llm.model.embed(input_ids)

    def encode_text(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attn_ctx: ARAttnCtx | None = None,
        gdn_ctxs: tuple[GatedDeltaNetCtx | None, ...] | None = None,
    ) -> torch.Tensor:
        return self.vlm.llm.model(
            inputs_embeds,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            attn_ctx=attn_ctx,
            gdn_ctxs=gdn_ctxs,
        )

    def project_action_proprio(self, action_proprio: torch.Tensor) -> torch.Tensor:
        return self.action_head.project_action_proprio(action_proprio)

    def encode_action(self, projected_action_and_time: torch.Tensor) -> torch.Tensor:
        return self.action_head.encode_action(projected_action_and_time)

    def run_dit(
        self,
        hidden_states: torch.Tensor,
        vlm_hidden_states: torch.Tensor,
        timestep_sinusoid: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_head.run_dit(
            hidden_states,
            vlm_hidden_states,
            timestep_sinusoid,
        )

    def decode_action(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.action_head.decode_action(hidden_states)

    def forward(
        self,
        dit_hidden_states: torch.Tensor,
        vlm_hidden_states: torch.Tensor,
        timestep_sinusoid: torch.Tensor,
    ) -> torch.Tensor:
        return self.action_head(
            dit_hidden_states,
            vlm_hidden_states,
            timestep_sinusoid,
        )


__all__ = [
    "MiniCPMGR00TActionDecoder",
    "MiniCPMGR00TActionEncoder",
    "MiniCPMGR00TActionHead",
    "MiniCPMGR00TAdaLayerNorm",
    "MiniCPMGR00TDiT",
    "MiniCPMGR00TDiTAttention",
    "MiniCPMGR00TDiTBlock",
    "MiniCPMGR00TModel",
    "MiniCPMGR00TQwenAttention",
    "MiniCPMGR00TQwenDecoderLayer",
    "MiniCPMGR00TQwenGatedDeltaNet",
    "MiniCPMGR00TTextModel",
    "MiniCPMGR00TTimestepEncoder",
    "MiniCPMGR00TVisionAttention",
    "MiniCPMGR00TVisionLayer",
    "MiniCPMGR00TVisionModel",
    "MiniCPMGR00TVisionResampler",
    "MiniCPMGR00TVisionWindowMerger",
    "MiniCPMGR00TVLM",
    "minicpm_gr00t_weight_remap",
]
