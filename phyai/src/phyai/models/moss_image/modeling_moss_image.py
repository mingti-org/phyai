"""Stateless MOSS-Image text refinement and joint image/text transformer."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from phyai.engine_config import get_engine_config
from phyai.layers.activation import SiLU
from phyai.layers.attention import Attention, AttnCtx
from phyai.layers.layer_norm import RMSNorm
from phyai.layers.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from phyai.layers.mlp import DenseMLP
from phyai.layers.modulation import AffineModulation
from phyai.layers.rotary_embedding import (
    ROPE_INV_FREQ_FNS,
    RotaryEmbedding,
    compute_cos_sin_from_inv_freq,
)
from phyai.models.moss_image.configuration_moss_image import MOSSImageConfig


@dataclass(frozen=True)
class MOSSImageConditionSample:
    text: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@dataclass(frozen=True)
class MOSSImagePreparedCondition:
    samples: tuple[MOSSImageConditionSample, ...]
    height: int
    width: int


def validate_skip_layers(layers: Iterable[int] | None, depth: int) -> frozenset[int]:
    indices = tuple(layers) if layers is not None else ()
    if any(type(index) is not int or not 0 <= index < depth for index in indices):
        raise ValueError(f"skip_layers must contain integer indices in [0, {depth})")
    return frozenset(indices)


def moss_image_weight_remap(name: str) -> str | None:
    if name == "rope.bands":
        return None
    return name.replace(".ffn_down.", ".down_proj.")


def apply_gated_residual(
    hidden: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return hidden + update * gate[:, None]


def bind_packed_weight(
    parameter: nn.Parameter,
    checkpoint_key: str,
    shard_ids: tuple[int | str, ...],
) -> None:
    """Send packed checkpoint legs through the shared TP weight loader."""
    shard_loader = parameter.weight_loader

    def load(param, tensor, shard_id=None):
        if tensor.shape[0] % len(shard_ids):
            raise ValueError(f"Invalid packed weight shape for {checkpoint_key}")
        for value, leg in zip(tensor.chunk(len(shard_ids), dim=0), shard_ids):
            shard_loader(param, value, leg)

    parameter.hf_keys = [(checkpoint_key, None)]
    parameter.weight_loader = load


class MOSSImageAttention(nn.Module):
    def __init__(
        self,
        config: MOSSImageConfig,
        *,
        prefix: str,
        text_refinement: bool,
        params_dtype: torch.dtype,
        device: torch.device | str,
        attention_backend: str | None,
    ) -> None:
        super().__init__()
        qkv_prefix = f"{prefix}.qkv" if text_refinement else f"{prefix}.proj.qkv"
        self.qkv = QKVParallelLinear(
            config.dim,
            config.head_dim,
            config.num_heads,
            bias=text_refinement,
            params_dtype=params_dtype,
            device=device,
            prefix=qkv_prefix,
            hf_legs={"q": "q", "k": "k", "v": "v"},
        )
        if config.num_heads % self.qkv.tp_size:
            raise ValueError("attention TP size must divide num_heads")
        self.heads = config.num_heads // self.qkv.tp_size
        self.head_dim = config.head_dim
        if text_refinement:
            bind_packed_weight(self.qkv.weight, f"{qkv_prefix}.weight", ("q", "k", "v"))
            bind_packed_weight(self.qkv.bias, f"{qkv_prefix}.bias", ("q", "k", "v"))
        self.q_norm = RMSNorm(
            config.head_dim,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.q_norm",
            cast_before_affine=True,
        )
        self.k_norm = RMSNorm(
            config.head_dim,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.k_norm",
            cast_before_affine=True,
        )
        self.attention = Attention(
            self.heads,
            self.head_dim,
            causal=False,
            backend=attention_backend,
        )
        self.proj = RowParallelLinear(
            config.dim,
            config.dim,
            group="attention_tp",
            bias=text_refinement,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.proj" if text_refinement else f"{prefix}.proj.o",
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: AttnCtx | None = None,
    ) -> torch.Tensor:
        q, k, v = self.qkv(x)[0].chunk(3, dim=-1)
        shape = (*x.shape[:2], self.heads, self.head_dim)
        q = self.q_norm(q.reshape(shape))
        k = self.k_norm(k.reshape(shape))
        v = v.reshape(shape)
        q, k = rope.apply(q, k, cos, sin)
        q, k = q.to(v.dtype), k.to(v.dtype)
        if attn_ctx is not None and attn_ctx.layout.is_ragged():
            output = self.attention(
                q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), ctx=attn_ctx
            ).reshape(shape)
        else:
            output = self.attention(q, k, v, ctx=attn_ctx)
        return self.proj(output.flatten(2))[0]


class MOSSImageTextBlock(nn.Module):
    def __init__(self, config: MOSSImageConfig, *, prefix: str, **kwargs) -> None:
        super().__init__()
        norm_kwargs = {
            "dtype": kwargs["params_dtype"],
            "device": kwargs["device"],
            "cast_before_affine": True,
        }
        self.norm1 = RMSNorm(config.dim, prefix=f"{prefix}.norm1", **norm_kwargs)
        self.norm2 = RMSNorm(config.dim, prefix=f"{prefix}.norm2", **norm_kwargs)
        self.attention = MOSSImageAttention(
            config, prefix=prefix, text_refinement=True, **kwargs
        )
        self.ffn = DenseMLP(
            config.dim,
            config.ffn_dim,
            activation="situ",
            bias=True,
            sequence_parallel=False,
            params_dtype=kwargs["params_dtype"],
            device=kwargs["device"],
            prefix=prefix,
            gated_hf_legs=("ffn_gate", "ffn_up"),
        )

    def forward(self, x, rope, cos, sin):
        x = x + self.attention(self.norm1(x), rope, cos, sin)
        return x + self.ffn(self.norm2(x))


class MOSSImageTextProjection(nn.Module):
    def __init__(self, config: MOSSImageConfig, **kwargs) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(
            config.text_dim,
            config.dim,
            bias=True,
            params_dtype=kwargs["params_dtype"],
            device=kwargs["device"],
            prefix="text_embedder.proj",
        )
        self.blocks = nn.ModuleList(
            MOSSImageTextBlock(config, prefix=f"text_embedder.blocks.{i}", **kwargs)
            for i in range(2)
        )

    def forward(self, x, rope, cos, sin):
        x = self.proj(x.to(self.proj.weight.dtype))[0]
        for block in self.blocks:
            x = block(x, rope, cos, sin)
        return x


class MOSSImageBlock(nn.Module):
    def __init__(self, config: MOSSImageConfig, *, prefix: str, **kwargs) -> None:
        super().__init__()
        norm_kwargs = {
            "dtype": kwargs["params_dtype"],
            "device": kwargs["device"],
            "cast_before_affine": True,
        }
        self.norm1 = RMSNorm(config.dim, prefix=f"{prefix}.norm1", **norm_kwargs)
        self.norm2 = RMSNorm(config.dim, prefix=f"{prefix}.norm2", **norm_kwargs)
        self.attention = MOSSImageAttention(
            config, prefix=prefix, text_refinement=False, **kwargs
        )
        self.ffn = DenseMLP(
            config.dim,
            config.ffn_dim,
            activation="situ",
            bias=False,
            sequence_parallel=False,
            params_dtype=kwargs["params_dtype"],
            device=kwargs["device"],
            prefix=f"{prefix}.proj",
        )
        # The checkpoint stores up before gate; DenseMLP stores gate before up.
        bind_packed_weight(
            self.ffn.gate_up_proj.weight, f"{prefix}.proj.ffn_up_gate.weight", (1, 0)
        )
        self.modulate = AffineModulation()
        self.apply_gated_residual = apply_gated_residual

    def forward(self, x, rope, cos, sin, modulation, attn_ctx: AttnCtx | None = None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.modulate(self.norm1(x), shift_a, scale_a)
        y = y.to(self.attention.qkv.weight.dtype)
        x = self.apply_gated_residual(
            x, self.attention(y, rope, cos, sin, attn_ctx), gate_a
        )
        y = self.modulate(self.norm2(x), shift_f, scale_f)
        y = y.to(self.ffn.gate_up_proj.weight.dtype)
        return self.apply_gated_residual(x, self.ffn(y), gate_f)


class MOSSImageTransformer(nn.Module):
    def __init__(
        self,
        config: MOSSImageConfig,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if device is None:
            device = get_engine_config().device.target
        linear_kwargs = {"params_dtype": params_dtype, "device": device}
        block_kwargs = {**linear_kwargs, "attention_backend": attention_backend}
        self.video_embedder = ReplicatedLinear(
            config.latent_channels, config.dim, prefix="video_embedder", **linear_kwargs
        )
        self.text_embedder = MOSSImageTextProjection(config, **block_kwargs)
        self.rope = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=1,
            rope_theta=config.rope_base,
            device=device,
        )
        self.rope_axis = RotaryEmbedding(
            config.head_dim // 4,
            max_position_embeddings=1,
            rope_theta=config.rope_base,
            device="cpu",
        ).to(device)
        self.time_linear_1 = ReplicatedLinear(
            config.freq_dim, config.dim, prefix="time_embedding.0", **linear_kwargs
        )
        self.time_activation = SiLU()
        self.time_linear_2 = ReplicatedLinear(
            config.dim, config.dim, prefix="time_embedding.2", **linear_kwargs
        )
        self.shared_block_modulation = ReplicatedLinear(
            config.dim,
            6 * config.dim,
            prefix="shared_block_modulation",
            **linear_kwargs,
        )
        self.layers = nn.ModuleList(
            MOSSImageBlock(config, prefix=f"layers.{i}", **block_kwargs)
            for i in range(config.num_layers)
        )
        self.final_norm_video = RMSNorm(
            config.dim,
            dtype=params_dtype,
            device=device,
            prefix="final_norm_video",
            cast_before_affine=True,
        )
        self.final_linear_video = ReplicatedLinear(
            config.dim,
            config.latent_channels,
            bias=False,
            prefix="final_linear_video",
            **linear_kwargs,
        )
        self.video_head_modulation = ReplicatedLinear(
            config.dim, 2 * config.dim, prefix="video_head_modulation", **linear_kwargs
        )
        self.modulate = AffineModulation()
        self.register_buffer(
            "time_frequencies", self.make_time_frequencies(device), persistent=False
        )

    def make_time_frequencies(self, device):
        half = self.config.freq_dim // 2
        return 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)

    def enable_compile(self) -> None:
        """Fuse elementwise work while preserving BF16 operation boundaries."""
        options = {"emulate_precision_casts": True, "triton.cudagraphs": False}
        residual = torch.compile(
            apply_gated_residual, fullgraph=True, dynamic=True, options=options
        )
        for block in self.layers:
            block.apply_gated_residual = residual
        self.rope.apply = torch.compile(
            self.rope.apply, fullgraph=True, dynamic=True, options=options
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Derived frequencies must survive model.to(dtype) without precision loss.
        frequency_device = self.rope_axis.inv_freq.device
        frequency, _ = ROPE_INV_FREQ_FNS["default"](
            self.config.head_dim // 4, self.config.rope_base, device="cpu"
        )
        self.rope_axis.inv_freq = frequency.to(frequency_device)
        self.time_frequencies = self.make_time_frequencies(frequency_device)
        return self

    def rotary_coordinates(self, height: int, width: int, text_length: int, device):
        h, w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image = (
            torch.stack(
                (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
            )
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        return torch.cat((image, text), 0)

    def prepare_condition(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        height: int,
        width: int,
    ) -> MOSSImagePreparedCondition:
        if (
            encoder_hidden_states.ndim != 3
            or encoder_hidden_states.shape[-1] != self.config.text_dim
        ):
            raise ValueError("encoder_hidden_states must have shape [B, S, text_dim]")
        if height <= 0 or width <= 0:
            raise ValueError("latent height and width must be positive")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(
                encoder_hidden_states.shape[:2],
                device=encoder_hidden_states.device,
                dtype=torch.bool,
            )
        if mask.shape != encoder_hidden_states.shape[:2] or not bool(
            torch.all((mask == 0) | (mask == 1))
        ):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask")
        dtype = self.video_embedder.weight.dtype
        samples = []
        for index in range(encoder_hidden_states.shape[0]):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token")
            positions = self.rotary_coordinates(
                height, width, context.shape[1], context.device
            )
            axis_cos, axis_sin = compute_cos_sin_from_inv_freq(
                positions, self.rope_axis.inv_freq
            )
            bands = self.config.head_dim // 8
            cos_half = axis_cos[..., :bands].flatten(-2)
            sin_half = axis_sin[..., :bands].flatten(-2)
            cos = torch.cat((cos_half, cos_half), -1).unsqueeze(0).to(dtype)
            sin = torch.cat((sin_half, sin_half), -1).unsqueeze(0).to(dtype)
            text = self.text_embedder(
                context, self.rope, cos[:, height * width :], sin[:, height * width :]
            )
            samples.append(MOSSImageConditionSample(text, cos, sin))
        return MOSSImagePreparedCondition(tuple(samples), height, width)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | float,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        *,
        condition: MOSSImagePreparedCondition | None = None,
        skip_layers: Iterable[int] | None = None,
        attention_contexts: tuple[AttnCtx, ...] | None = None,
    ) -> torch.Tensor:
        skipped = validate_skip_layers(skip_layers, len(self.layers))
        if (
            hidden_states.ndim != 4
            or hidden_states.shape[1] != self.config.latent_channels
        ):
            raise ValueError("hidden_states must have shape [B, latent_channels, H, W]")
        batch, _, height, width = hidden_states.shape
        if batch < 1:
            raise ValueError("hidden_states must contain at least one sample")
        if attention_contexts is not None and len(attention_contexts) != batch:
            raise ValueError("attention_contexts must contain one context per sample")
        if condition is None:
            if encoder_hidden_states is None:
                raise ValueError("Supply text embeddings or a prepared condition")
            condition = self.prepare_condition(
                encoder_hidden_states, encoder_attention_mask, height, width
            )
        elif encoder_hidden_states is not None or encoder_attention_mask is not None:
            raise ValueError("Supply either text embeddings or a prepared condition")
        if len(condition.samples) != batch or (condition.height, condition.width) != (
            height,
            width,
        ):
            raise ValueError(
                "Prepared condition batch and latent dimensions must match"
            )
        times = torch.as_tensor(
            timestep, device=hidden_states.device, dtype=torch.float32
        ).reshape(-1)
        if times.numel() not in (1, batch):
            raise ValueError("timestep must be scalar or have one value per sample")
        phase = times.expand(batch).double()[:, None] * self.time_frequencies
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_linear_1(embedding.to(self.time_linear_1.weight.dtype))[0]
        time = self.time_linear_2(self.time_activation(time))[0].float()
        modulation = self.shared_block_modulation(
            time.to(self.shared_block_modulation.weight.dtype)
        )[0].reshape(batch, 6, -1)
        head = self.video_head_modulation(
            time.to(self.video_head_modulation.weight.dtype)
        )[0].reshape(batch, 2, -1)
        outputs = []
        for index, sample in enumerate(condition.samples):
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(
                video.to(self.video_embedder.weight.dtype).contiguous()
            )[0]
            x = torch.cat((video, sample.text), dim=1)
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skipped:
                    x = layer(
                        x,
                        self.rope,
                        sample.cos,
                        sample.sin,
                        modulation[index : index + 1],
                        None
                        if attention_contexts is None
                        else attention_contexts[index],
                    )
            x = self.final_norm_video(
                x[:, : height * width].to(self.final_norm_video.weight.dtype)
            )
            x = self.modulate(x, head[index : index + 1, 0], head[index : index + 1, 1])
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))[0]
            outputs.append(
                x.transpose(1, 2).reshape(1, self.config.latent_channels, height, width)
            )
        return torch.cat(outputs, dim=0)


__all__ = [
    "MOSSImageConfig",
    "MOSSImageConditionSample",
    "MOSSImagePreparedCondition",
    "MOSSImageTransformer",
    "moss_image_weight_remap",
    "validate_skip_layers",
]
