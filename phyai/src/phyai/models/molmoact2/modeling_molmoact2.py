"""Stateless MolmoAct2 language transformer with explicit key/value outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from phyai.layers.mlp import DenseMLP
from phyai.engine_config import get_engine_config
from phyai.layers.linear import ReplicatedLinear
from phyai.weights.shards import replicated
from phyai.layers.attention import AttnMask, Attention
from phyai.layers.layer_norm import RMSNorm
from phyai.layers.rotary_embedding import RotaryEmbedding
from phyai.models.molmoact2.vision_molmoact2 import MolmoAct2VisionBackbone
from phyai.models.molmoact2.configuration_molmoact2 import MolmoAct2TextConfig

LayerKV = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = tuple[LayerKV, ...]


@dataclass(frozen=True)
class MolmoAct2TextOutput:
    last_hidden_state: torch.Tensor
    past_key_values: PastKeyValues
    hidden_states: tuple[torch.Tensor, ...] | None = None


class MolmoAct2Embedding(nn.Module):
    def __init__(self, config: MolmoAct2TextConfig, *, params_dtype, device, prefix):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.additional_vocab_size = config.additional_vocab_size
        if config.additional_vocab_size is None:
            self.weight = nn.Parameter(
                torch.empty(
                    config.vocab_size,
                    config.hidden_size,
                    dtype=params_dtype,
                    device=device,
                ),
                requires_grad=False,
            )
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()
        else:
            for name, count in (
                ("embedding", config.vocab_size),
                ("new_embedding", config.additional_vocab_size),
            ):
                parameter = nn.Parameter(
                    torch.empty(
                        count, config.hidden_size, dtype=params_dtype, device=device
                    ),
                    requires_grad=False,
                )
                parameter.hf_keys = [(f"{prefix}.{name}", None)]
                parameter.weight_loader = replicated()
                self.register_parameter(name, parameter)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.masked_fill(input_ids == -1, 0)
        if self.additional_vocab_size is None:
            return F.embedding(input_ids, self.weight)
        base = F.embedding(input_ids.clamp_max(self.vocab_size - 1), self.embedding)
        if self.additional_vocab_size == 0:
            return base
        extra = F.embedding(
            (input_ids - self.vocab_size).clamp_min(0), self.new_embedding
        )
        return torch.where((input_ids < self.vocab_size)[..., None], base, extra)


class MolmoAct2Attention(nn.Module):
    def __init__(
        self,
        config: MolmoAct2TextConfig,
        *,
        params_dtype,
        device,
        prefix,
        attn_backend=None,
        norm_backend=None,
    ):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.fused_dims = (
            config.num_attention_heads * config.head_dim,
            config.num_key_value_heads * config.head_dim,
            config.num_key_value_heads * config.head_dim,
        )
        self.att_proj = ReplicatedLinear(
            config.hidden_size,
            sum(self.fused_dims),
            bias=config.qkv_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.att_proj",
        )
        self.attn_out = ReplicatedLinear(
            self.fused_dims[0],
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.attn_out",
        )
        self.q_norm = self.k_norm = None
        if config.use_qk_norm:
            self.q_norm = RMSNorm(
                config.head_dim
                if config.qk_norm_type == "qwen3"
                else self.fused_dims[0],
                eps=config.layer_norm_eps,
                dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.q_norm",
                backend=norm_backend,
                cast_before_affine=True,
            )
            self.k_norm = RMSNorm(
                config.head_dim
                if config.qk_norm_type == "qwen3"
                else self.fused_dims[1],
                eps=config.layer_norm_eps,
                dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.k_norm",
                backend=norm_backend,
                cast_before_affine=True,
            )
        self.attention = Attention(
            config.num_attention_heads,
            config.head_dim,
            num_kv_heads=config.num_key_value_heads,
            causal=True,
            backend=attn_backend,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        rope: RotaryEmbedding,
        attention_mask: AttnMask | torch.Tensor | None = None,
        past_key_value: LayerKV | None = None,
    ) -> tuple[torch.Tensor, LayerKV]:
        batch, tokens, _ = hidden_states.shape
        query, key, value = self.att_proj(hidden_states)[0].split(
            self.fused_dims, dim=-1
        )
        if self.q_norm is not None and self.config.qk_norm_type != "qwen3":
            query, key = self.q_norm(query), self.k_norm(key)
        query = query.reshape(
            batch, tokens, self.config.num_attention_heads, self.head_dim
        )
        key = key.reshape(batch, tokens, self.config.num_key_value_heads, self.head_dim)
        value = value.reshape_as(key)
        if self.q_norm is not None and self.config.qk_norm_type == "qwen3":
            query, key = self.q_norm(query), self.k_norm(key)
        query, key = rope.apply(query, key, *position_embeddings)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0].transpose(1, 2), key), dim=1)
            value = torch.cat((past_key_value[1].transpose(1, 2), value), dim=1)
        present = (key.transpose(1, 2), value.transpose(1, 2))
        if isinstance(attention_mask, torch.Tensor):
            # Image-to-image visibility can cross separate image blocks, which
            # the shared segment mask cannot represent. SDPA accepts this bias.
            repeats = self.config.num_attention_heads // self.config.num_key_value_heads
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                present[0].repeat_interleave(repeats, dim=1),
                present[1].repeat_interleave(repeats, dim=1),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
        else:
            output = self.attention(query, key, value, mask=attention_mask)
        return self.attn_out(output.reshape(batch, tokens, -1))[0], present


def configure_text_mlp_weights(mlp: DenseMLP, prefix: str) -> None:
    """Load the checkpoint's up-then-gate projection into the shared gate-then-up layout."""
    weight = mlp.gate_up_proj.weight
    load_leg = weight.weight_loader

    def load_ff_proj(parameter, loaded, shard_id=None):
        up, gate = loaded.chunk(2, dim=0)
        load_leg(parameter, gate, 0)
        load_leg(parameter, up, 1)

    weight.hf_keys = [(f"{prefix}.ff_proj.weight", None)]
    weight.weight_loader = load_ff_proj
    mlp.down_proj.weight.hf_keys = [(f"{prefix}.ff_out.weight", None)]


class MolmoAct2DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        *,
        params_dtype,
        device,
        prefix,
        attn_backend=None,
        norm_backend=None,
    ):
        super().__init__()
        self.norm_after = config.norm_after
        self.self_attn = MolmoAct2Attention(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.self_attn",
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        for name in ("attn_norm", "ff_norm"):
            setattr(
                self,
                name,
                RMSNorm(
                    config.hidden_size,
                    eps=config.layer_norm_eps,
                    dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.{name}",
                    backend=norm_backend,
                    cast_before_affine=True,
                ),
            )
        self.mlp = DenseMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="silu",
            gated=True,
            bias=False,
            cast_before_multiply=True,
            sequence_parallel=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.mlp",
        )
        configure_text_mlp_weights(self.mlp, f"{prefix}.mlp")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        rope: RotaryEmbedding,
        attention_mask: AttnMask | torch.Tensor | None = None,
        past_key_value: LayerKV | None = None,
    ) -> tuple[torch.Tensor, LayerKV]:
        residual = hidden_states
        if not self.norm_after:
            hidden_states = self.attn_norm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states, position_embeddings, rope, attention_mask, past_key_value
        )
        if self.norm_after:
            hidden_states = self.attn_norm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        if not self.norm_after:
            hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.norm_after:
            hidden_states = self.ff_norm(hidden_states)
        return residual + hidden_states, present


class MolmoAct2TextModel(nn.Module):
    def __init__(
        self,
        config: MolmoAct2TextConfig,
        *,
        params_dtype=torch.bfloat16,
        device=None,
        prefix="model.transformer",
        attn_backend=None,
        norm_backend=None,
    ):
        super().__init__()
        device = device or get_engine_config().device.target
        self.config = config
        self.wte = MolmoAct2Embedding(
            config, params_dtype=params_dtype, device=device, prefix=f"{prefix}.wte"
        )
        self.emb_drop = nn.Identity()
        self.blocks = nn.ModuleList(
            MolmoAct2DecoderLayer(
                config,
                params_dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.blocks.{index}",
                attn_backend=attn_backend,
                norm_backend=norm_backend,
            )
            for index in range(config.num_hidden_layers)
        )
        self.ln_f = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.ln_f",
            backend=norm_backend,
            cast_before_affine=True,
        )
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_type=config.rope_type,
            device=device,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        output_hidden_states: bool = False,
    ) -> MolmoAct2TextOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        hidden = self.wte(input_ids) if inputs_embeds is None else inputs_embeds
        batch, tokens = hidden.shape[:2]
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError(
                "past_key_values must contain one key/value pair per layer"
            )
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + tokens, device=hidden.device
            )[None].expand(batch, -1)
        position_embeddings = tuple(
            value.to(hidden.dtype)
            for value in self.rotary_emb.get_cos_sin(position_ids)
        )
        mask: AttnMask | torch.Tensor | None = None
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                mask = attention_mask.to(device=hidden.device)
            elif attention_mask.ndim == 2:
                if attention_mask.shape != (batch, past_length + tokens):
                    raise ValueError(
                        "attention_mask must cover past and current tokens"
                    )
                mask = AttnMask.from_key_mask(attention_mask.to(device=hidden.device))
            else:
                raise ValueError("attention_mask must have rank 2 or 4")
        if (
            token_type_ids is not None
            and past_length == 0
            and not isinstance(mask, torch.Tensor)
        ):
            if token_type_ids.shape != (batch, tokens):
                raise ValueError("token_type_ids must match the prefill token shape")
            positions = torch.arange(tokens, device=hidden.device)
            allowed = positions[:, None] >= positions[None, :]
            image = token_type_ids.to(device=hidden.device, dtype=torch.bool)
            allowed = allowed[None] | (image[:, :, None] & image[:, None, :])
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, :].to(
                    device=hidden.device, dtype=torch.bool
                )
            mask = torch.where(
                allowed[:, None],
                hidden.new_zeros(()),
                hidden.new_full((), torch.finfo(hidden.dtype).min),
            )
        present = []
        all_hidden = [] if output_hidden_states else None
        for index, block in enumerate(self.blocks):
            if all_hidden is not None:
                all_hidden.append(hidden)
            hidden, layer_kv = block(
                hidden,
                position_embeddings,
                self.rotary_emb,
                mask,
                None if past_key_values is None else past_key_values[index],
            )
            present.append(layer_kv)
        hidden = self.ln_f(hidden)
        if all_hidden is not None:
            all_hidden.append(hidden)
        return MolmoAct2TextOutput(
            hidden, tuple(present), None if all_hidden is None else tuple(all_hidden)
        )


__all__ = [
    "LayerKV",
    "PastKeyValues",
    "MolmoAct2TextOutput",
    "MolmoAct2TextModel",
    "MolmoAct2VisionBackbone",
]
