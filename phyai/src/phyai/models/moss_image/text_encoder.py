"""Stateless Qwen3.5 prompt encoding for MOSS-Image."""

from __future__ import annotations

import torch
import torch.nn as nn

from phyai.engine_config import get_engine_config
from phyai.layers.activation import SiLU
from phyai.layers.attention.gdn import GatedDeltaNet
from phyai.layers.attention.nocache.layer import Attention
from phyai.layers.conv import CausalConv1d
from phyai.layers.layer_norm import GemmaRMSNorm, RMSNorm
from phyai.layers.linear import ReplicatedLinear
from phyai.layers.mlp import DenseMLP
from phyai.layers.rotary_embedding import (
    RotaryEmbedding,
    compute_qwen3vl_mrope_cos_sin_from_inv_freq,
)
from phyai.layers.vocab_embedding import VocabParallelEmbedding
from phyai.models.moss_image.configuration_text_encoder import MOSSImageTextConfig
from phyai.weights.shards import replicated


def moss_image_text_weight_remap(name: str) -> str | None:
    if name.startswith("language_model."):
        return name
    return None


class MOSSImageGatedDeltaNet(nn.Module):
    """Qwen3.5 projections, causal convolution, and GDN core."""

    def __init__(
        self,
        config: MOSSImageTextConfig,
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
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = ReplicatedLinear(
            config.hidden_size,
            self.conv_dim,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.in_proj_qkv",
        )
        self.in_proj_z = ReplicatedLinear(
            config.hidden_size,
            self.value_dim,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.in_proj_z",
        )
        self.in_proj_b = ReplicatedLinear(
            config.hidden_size,
            self.num_value_heads,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.in_proj_b",
        )
        self.in_proj_a = ReplicatedLinear(
            config.hidden_size,
            self.num_value_heads,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.in_proj_a",
        )
        self.conv1d = CausalConv1d(
            self.conv_dim,
            self.conv_kernel_size,
            split_sizes=(self.key_dim, self.key_dim, self.value_dim),
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.conv1d",
            kernel_role="qwen3_5.causal_conv",
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
            kernel_role="qwen3_5.gdn",
        )
        self.norm = RMSNorm(
            self.value_head_dim,
            config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm",
            cast_before_affine=True,
            kernel_role="qwen3_5.gated_norm",
        )
        self.gate_activation = SiLU()
        self.out_proj = ReplicatedLinear(
            self.value_dim,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.out_proj",
        )

    def _apply(self, fn, recurse=True):
        a_log = self.A_log.detach()
        metadata = vars(self.A_log).copy()
        super()._apply(fn, recurse=recurse)
        vars(self.A_log).update(metadata)
        if a_log.is_meta:
            self.A_log.data = self.A_log.data.float()
        else:
            self.A_log.data = a_log.to(device=self.A_log.device, dtype=torch.float32)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        query, key, value = self.conv1d(mixed_qkv)
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
        a, _ = self.in_proj_a(hidden_states)
        b, _ = self.in_proj_b(hidden_states)
        core_output = self.gdn(query, key, value, a, b, self.A_log, self.dt_bias)
        z, _ = self.in_proj_z(hidden_states)
        normalized = self.norm(core_output.reshape(-1, self.value_head_dim))
        gate = self.gate_activation(z.reshape(-1, self.value_head_dim).float())
        core_output = (normalized.float() * gate).to(hidden_states.dtype)
        core_output = core_output.reshape(batch_size, seq_len, self.value_dim)
        output, _ = self.out_proj(core_output)
        return output


class MOSSImageAttention(nn.Module):
    """Gated Qwen3.5 full attention."""

    def __init__(
        self,
        config: MOSSImageTextConfig,
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
        self.q_proj = ReplicatedLinear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ReplicatedLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ReplicatedLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.v_proj",
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
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_gate, _ = self.q_proj(hidden_states)
        query, gate = query_gate.view(
            batch_size, seq_len, self.num_heads, self.head_dim * 2
        ).chunk(2, dim=-1)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)
        query = self.q_norm(query)
        key = self.k_norm(
            key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        )
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        query, key = self.rotary_emb.apply(query, key, cos, sin)
        output = self.attn(query, key, value)
        output = output.reshape(batch_size, seq_len, -1)
        output = output * torch.sigmoid(gate.reshape(batch_size, seq_len, -1))
        output, _ = self.o_proj(output)
        return output


class MOSSImageDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MOSSImageTextConfig,
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
            self.linear_attn = MOSSImageGatedDeltaNet(
                config,
                params_dtype=params_dtype,
                gdn_backend=gdn_backend,
                device=device,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = MOSSImageAttention(
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
            cast_before_multiply=True,
            sequence_parallel=False,
            params_dtype=params_dtype,
            device=device,
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
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(
                hidden_states,
                cos=cos,
                sin=sin,
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class MOSSImageTextEncoder(nn.Module):
    """Qwen3.5 hybrid text decoder for stateless prefill."""

    def __init__(
        self,
        config: MOSSImageTextConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        gdn_backend: str = "flashinfer",
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "language_model",
    ) -> None:
        super().__init__()
        params_dtype = torch.bfloat16 if params_dtype is None else params_dtype
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
        self.rotary_emb = self.build_rotary_embedding(device)
        self.layers = nn.ModuleList(
            [
                MOSSImageDecoderLayer(
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

    def build_rotary_embedding(self, device):
        return RotaryEmbedding(
            self.config.head_dim,
            1,
            rope_theta=self.config.rope_theta,
            rope_type=self.config.rope_type,
            partial_rotary_factor=self.config.partial_rotary_factor,
            backend="eager",
            device=device,
        )

    def _apply(self, fn, recurse=True):
        inv_freq = self.rotary_emb.inv_freq
        cache = self.rotary_emb.cos_sin_cache
        super()._apply(fn, recurse=recurse)
        device = self.rotary_emb.inv_freq.device
        if inv_freq.is_meta:
            materialized = self.build_rotary_embedding(device)
            self.rotary_emb.inv_freq = materialized.inv_freq
            self.rotary_emb.cos_sin_cache = materialized.cos_sin_cache
        else:
            self.rotary_emb.inv_freq = inv_freq.to(device=device, dtype=torch.float32)
            self.rotary_emb.cos_sin_cache = cache.to(device=device, dtype=torch.float32)
        return self

    def get_position_embeddings(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, *position_ids.shape)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        if position_ids.shape != (3, batch_size, seq_len):
            raise ValueError(
                f"position_ids must have shape (3, B, S) or (4, B, S), got "
                f"{tuple(position_ids.shape)}."
            )
        cos, sin = compute_qwen3vl_mrope_cos_sin_from_inv_freq(
            position_ids,
            self.rotary_emb.inv_freq,
            self.config.mrope_section,
        )
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "Encode each unpadded prompt separately, with input_ids [1, S]"
            )
        if input_ids.shape[1] == 0:
            raise ValueError("A prompt must contain at least one token")
        inputs_embeds = self.embed_tokens(input_ids)
        cos, sin = self.get_position_embeddings(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
            )
        return self.norm(hidden_states)
