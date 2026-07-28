"""Stateless MiniCPM-RobotTrack policy model."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from phyai.engine_config import get_engine_config, resolve_engine_defaults
from phyai.layers.attention.attention.layer import Attention
from phyai.layers.layer_norm import LayerNorm, RMSNorm
from phyai.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)
from phyai.layers.rotary_embedding import (
    apply_rotary_pos_emb,
    compute_cos_sin_from_inv_freq,
)
from phyai.models.minicpm_robot_track.configuration_minicpm_robot_track import (
    MiniCPM4TrackConfig,
    MiniCPMRobotTrackConfig,
)
from phyai.weights.shards import replicated


def attach_replicated_weight(parameter: nn.Parameter, key: str) -> None:
    """Attach PhyAI's checkpoint placement metadata to a plain parameter."""

    parameter.hf_keys = [(key, None)]
    parameter.weight_loader = replicated()


def minicpm_robot_track_weight_remap(name: str) -> str | None:
    """Drop the checkpoint buffer because PhyAI derives the scale from config.

    ``load_pretrained`` indexes parameters rather than buffers, so the official
    checkpoint's ``output_scale`` key must be removed for strict loading.
    """

    if name == "output_scale":
        return None
    return name


class MiniCPM4MLP(nn.Module):
    def __init__(
        self,
        config: MiniCPM4TrackConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = ReplicatedLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class MiniCPM4Attention(nn.Module):
    def __init__(
        self,
        config: MiniCPM4TrackConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.qkv_proj",
        )
        if self.qkv_proj.tp_size != 1:
            raise NotImplementedError(
                "MiniCPM-RobotTrack currently supports world_size=1 only."
            )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_kv_heads,
            causal=True,
            backend=attn_backend,
        )
        self.o_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        query_dtype = query.dtype
        query, key = apply_rotary_pos_emb(
            query.float(),
            key.float(),
            cos.float(),
            sin.float(),
            unsqueeze_dim=2,
        )
        query = query.to(query_dtype)
        key = key.to(query_dtype)
        output = self.attn(query, key, value)
        output = output.reshape(batch_size, seq_len, -1)
        output, _ = self.o_proj(output)
        return output


class MiniCPM4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: MiniCPM4TrackConfig,
        layer_idx: int,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        layer_prefix = f"{prefix}.{layer_idx}"
        self.self_attn = MiniCPM4Attention(
            config,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            device=device,
            prefix=f"{layer_prefix}.self_attn",
        )
        self.mlp = MiniCPM4MLP(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{layer_prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{layer_prefix}.input_layernorm",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{layer_prefix}.post_attention_layernorm",
        )
        self.residual_scale = config.scale_depth / math.sqrt(config.num_hidden_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + hidden_states * self.residual_scale
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states * self.residual_scale


class MiniCPM4Backbone(nn.Module):
    def __init__(
        self,
        config: MiniCPM4TrackConfig,
        *,
        input_seq_length: int,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str = "backbone",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=params_dtype,
            device=device,
        )
        self.embed_tokens.requires_grad_(False)
        attach_replicated_weight(
            self.embed_tokens.weight, f"{prefix}.embed_tokens.weight"
        )
        self.layers = nn.ModuleList(
            [
                MiniCPM4DecoderLayer(
                    config,
                    layer_idx,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    device=device,
                    prefix=f"{prefix}.layers",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm",
        )

        factor = (
            config.long_factor
            if input_seq_length > config.original_max_position_embeddings
            else config.short_factor
        )
        factor_tensor = torch.tensor(factor, dtype=torch.float32, device=device)
        base_inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device)
                / config.head_dim
            )
        )
        scale = config.max_position_embeddings / config.original_max_position_embeddings
        attention_scaling = math.sqrt(
            1 + math.log(scale) / math.log(config.original_max_position_embeddings)
        )
        self.register_buffer(
            "rope_inv_freq", base_inv_freq / factor_tensor, persistent=False
        )
        self.attention_scaling = attention_scaling
        self.register_buffer(
            "position_ids",
            torch.arange(input_seq_length, dtype=torch.long, device=device)[None],
            persistent=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        position_ids = self.position_ids.expand(batch_size, -1)
        cos, sin = compute_cos_sin_from_inv_freq(
            position_ids,
            self.rope_inv_freq,
            attention_scaling=self.attention_scaling,
        )
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return hidden_states


class VisionProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.input_norm = LayerNorm(
            input_dim,
            backend=norm_backend,
            dtype=torch.float32,
            device=device,
            prefix=f"{prefix}.0",
        )
        self.input_proj = ReplicatedLinear(
            input_dim,
            hidden_dim,
            params_dtype=torch.float32,
            device=device,
            prefix=f"{prefix}.1",
        )
        self.output_proj = ReplicatedLinear(
            hidden_dim,
            hidden_dim,
            params_dtype=torch.float32,
            device=device,
            prefix=f"{prefix}.3",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_norm(features.float())
        hidden_states, _ = self.input_proj(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states, _ = self.output_proj(hidden_states)
        return hidden_states


class TemporalMarkerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        max_time_steps: int,
        *,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.time_embedding = nn.Embedding(
            max_time_steps, hidden_dim, dtype=torch.float32, device=device
        )
        self.stream_embedding = nn.Embedding(
            2, hidden_dim, dtype=torch.float32, device=device
        )
        self.camera_embedding = nn.Embedding(
            1, hidden_dim, dtype=torch.float32, device=device
        )
        self.requires_grad_(False)
        attach_replicated_weight(
            self.time_embedding.weight, f"{prefix}.time_embedding.weight"
        )
        attach_replicated_weight(
            self.stream_embedding.weight, f"{prefix}.stream_embedding.weight"
        )
        attach_replicated_weight(
            self.camera_embedding.weight, f"{prefix}.camera_embedding.weight"
        )

    def forward(self, time_indices: torch.Tensor, stream_id: int) -> torch.Tensor:
        return (
            self.time_embedding(time_indices)
            + self.stream_embedding.weight[stream_id]
            + self.camera_embedding.weight[0]
        )


class FunnelTrajectoryHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_waypoints: int,
        action_dim: int,
        dropout: float,
        use_tanh: bool,
        *,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_waypoints = num_waypoints
        self.action_dim = action_dim
        self.use_tanh = use_tanh
        self.input_norm = LayerNorm(
            hidden_dim,
            backend=norm_backend,
            dtype=torch.float32,
            device=device,
            prefix=f"{prefix}.0",
        )
        widths = (hidden_dim, 4096, 1024, 512, 256, 128)
        checkpoint_indices = (1, 4, 7, 10, 13)
        self.hidden_layers = nn.ModuleList(
            [
                ReplicatedLinear(
                    widths[index],
                    widths[index + 1],
                    params_dtype=torch.float32,
                    device=device,
                    prefix=f"{prefix}.{checkpoint_indices[index]}",
                )
                for index in range(len(widths) - 1)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = LayerNorm(
            128,
            backend=norm_backend,
            dtype=torch.float32,
            device=device,
            prefix=f"{prefix}.16",
        )
        self.output_proj = ReplicatedLinear(
            128,
            num_waypoints * action_dim,
            params_dtype=torch.float32,
            device=device,
            prefix=f"{prefix}.17",
        )

    def forward(self, control_state: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_norm(control_state.float())
        for layer in self.hidden_layers:
            hidden_states, _ = layer(hidden_states)
            hidden_states = self.dropout(F.gelu(hidden_states))
        hidden_states = self.output_norm(hidden_states)
        trajectory, _ = self.output_proj(hidden_states)
        if self.use_tanh:
            trajectory = torch.tanh(trajectory)
        return trajectory.view(-1, self.num_waypoints, self.action_dim)


class MiniCPMRobotTrackModel(nn.Module):
    """Fixed-length RobotTrack forward graph with trailing causal padding."""

    def __init__(
        self,
        config: MiniCPMRobotTrackConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        float32_norm_backend: str = "phyai-kernel",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        if device is None:
            device = get_engine_config().device.target
        self.config = config
        self.params_dtype = params_dtype
        self.backbone = MiniCPM4Backbone(
            config.backbone,
            input_seq_length=config.input_seq_length,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            device=device,
        )
        self.vision_projector = VisionProjector(
            config.vision_feature_dim,
            config.backbone.hidden_size,
            norm_backend=float32_norm_backend,
            device=device,
            prefix="vision_projector.layers",
        )
        self.temporal_markers = TemporalMarkerEncoder(
            config.backbone.hidden_size,
            config.max_time_steps,
            device=device,
            prefix="temporal_markers",
        )
        self.control_query = nn.Parameter(
            torch.empty(
                1,
                1,
                config.backbone.hidden_size,
                dtype=torch.float32,
                device=device,
            ),
            requires_grad=False,
        )
        attach_replicated_weight(self.control_query, "control_query")
        self.trajectory_head = FunnelTrajectoryHead(
            config.backbone.hidden_size,
            config.num_waypoints,
            config.action_dim,
            config.trajectory_dropout,
            config.use_tanh_actions,
            norm_backend=float32_norm_backend,
            device=device,
            prefix="trajectory_head.layers",
        )
        output_scale = torch.ones(
            1, 1, config.action_dim, dtype=torch.float32, device=device
        )
        output_scale[..., :2] = config.xy_scale
        self.register_buffer("output_scale", output_scale)
        self.register_buffer(
            "sequence_positions",
            torch.arange(config.input_seq_length, device=device)[None],
            persistent=False,
        )

    def build_sequence(
        self,
        input_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        coarse_tokens: torch.Tensor,
        coarse_time_indices: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_time_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        batch_size = input_ids.shape[0]
        text = self.backbone.embed_tokens(input_ids)

        history = self.vision_projector(coarse_tokens)
        history = history.view(
            batch_size,
            config.history_frames,
            config.coarse_tokens_per_frame,
            config.backbone.hidden_size,
        )
        history_times = coarse_time_indices.view(
            batch_size,
            config.history_frames,
            config.coarse_tokens_per_frame,
        )[:, :, 0]
        history_markers = self.temporal_markers(history_times, stream_id=0)
        history = torch.cat((history_markers[:, :, None], history), dim=2)
        history = history.flatten(1, 2)

        current = self.vision_projector(fine_tokens)
        current_times = fine_time_indices[:, :1]
        current_marker = self.temporal_markers(current_times, stream_id=1)
        current = torch.cat((current_marker, current), dim=1)
        control_query = self.control_query.expand(batch_size, -1, -1)

        # Start from [text capacity, visual/control]. Move unused text slots to
        # the tail so the control token sees the same compact causal prefix as
        # the variable-length reference while the physical shape stays fixed.
        base = torch.cat((text.float(), history, current, control_query), dim=1)
        positions = self.sequence_positions.expand(batch_size, -1)
        lengths = text_lengths[:, None]
        visual_end = lengths + config.fixed_non_text_tokens
        base_indices = torch.where(
            positions < lengths,
            positions,
            torch.where(
                positions < visual_end,
                config.text_capacity + positions - lengths,
                positions - config.fixed_non_text_tokens,
            ),
        )
        sequence = torch.gather(
            base,
            1,
            base_indices[:, :, None].expand(-1, -1, base.shape[-1]),
        )
        control_positions = text_lengths + config.fixed_non_text_tokens - 1
        return sequence.to(self.params_dtype), control_positions

    def forward(
        self,
        input_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        coarse_tokens: torch.Tensor,
        coarse_time_indices: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_time_indices: torch.Tensor,
    ) -> torch.Tensor:
        sequence, control_positions = self.build_sequence(
            input_ids,
            text_lengths,
            coarse_tokens,
            coarse_time_indices,
            fine_tokens,
            fine_time_indices,
        )
        hidden_states = self.backbone(sequence)
        gather_indices = control_positions[:, None, None].expand(
            -1, 1, hidden_states.shape[-1]
        )
        control_state = torch.gather(hidden_states, 1, gather_indices).squeeze(1)
        control_state = self.backbone.norm(control_state)
        normalized = self.trajectory_head(control_state)
        return normalized * self.output_scale


__all__ = [
    "MiniCPMRobotTrackModel",
    "minicpm_robot_track_weight_remap",
]
