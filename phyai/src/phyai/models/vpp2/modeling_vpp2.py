"""Stateless Wan video and action experts for VPP2."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

import phyai.parallel as parallel
from phyai.engine_config import get_engine_config
from phyai.layers import LayerNormModulation, QKRMSNormRotary
from phyai.layers.activation import SiLU
from phyai.layers.attention.mask import AttnMask
from phyai.layers.attention.nocache.base import AttnCtx
from phyai.layers.attention.nocache.layer import Attention
from phyai.layers.axial_rotary_embedding import AxialRotaryEmbedding
from phyai.layers.conv import Conv3d
from phyai.layers.layer_norm import LayerNorm, RMSNorm
from phyai.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.layers.mlp import DenseMLP
from phyai.layers.modulation import AffineModulation
from phyai.models.vpp2.configuration_vpp2 import (
    VPP2ActionConfig,
    VPP2Config,
    VPP2VideoConfig,
)
from phyai.weights.shards import replicated


def join_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def vpp2_weight_remap(name: str) -> str | None:
    for prefix in ("module.", "model_state_dict.", "state_dict."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    name = name.replace(".ffn.0.", ".ffn.fc1.")
    return name.replace(".ffn.2.", ".ffn.fc2.")


def vpp2_video_weight_remap(name: str) -> str | None:
    for prefix in ("video_expert.", "dit.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return vpp2_weight_remap(name)


def vpp2_action_weight_remap(name: str) -> str | None:
    if name.startswith("proprio_encoder."):
        return None
    if name.startswith("action_expert."):
        name = name[len("action_expert.") :]
    return vpp2_weight_remap(name)


def modulation_parameter(shape, *, dtype, device, prefix: str) -> nn.Parameter:
    value = nn.Parameter(
        torch.zeros(shape, dtype=dtype, device=device), requires_grad=False
    )
    value.hf_keys = [(prefix, None)]
    value.weight_loader = replicated()
    return value


def vpp2_gated_residual(
    residual: torch.Tensor, gate: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    return residual + gate[:, None] * value


class VPP2EmbeddingMLP(nn.ModuleList):
    def __init__(self, input_dim, hidden_dim, *, activation, dtype, device, prefix):
        super().__init__(
            [
                ReplicatedLinear(
                    input_dim,
                    hidden_dim,
                    bias=True,
                    params_dtype=dtype,
                    device=device,
                    prefix=f"{prefix}.0",
                ),
                SiLU() if activation == "silu" else nn.GELU(approximate="tanh"),
                ReplicatedLinear(
                    hidden_dim,
                    hidden_dim,
                    bias=True,
                    params_dtype=dtype,
                    device=device,
                    prefix=f"{prefix}.2",
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self[2](self[1](self[0](x)[0]))[0]


class VPP2ImageProjection(nn.Module):
    def __init__(self, config, *, dtype, device, prefix):
        super().__init__()
        self.proj = nn.ModuleList(
            [
                LayerNorm(
                    config.image_dim,
                    eps=1e-5,
                    dtype=dtype,
                    device=device,
                    prefix=f"{prefix}.proj.0",
                ),
                ReplicatedLinear(
                    config.image_dim,
                    config.image_dim,
                    bias=True,
                    params_dtype=dtype,
                    device=device,
                    prefix=f"{prefix}.proj.1",
                ),
                nn.GELU(),
                ReplicatedLinear(
                    config.image_dim,
                    config.hidden_dim,
                    bias=True,
                    params_dtype=dtype,
                    device=device,
                    prefix=f"{prefix}.proj.3",
                ),
                LayerNorm(
                    config.hidden_dim,
                    eps=1e-5,
                    dtype=dtype,
                    device=device,
                    prefix=f"{prefix}.proj.4",
                ),
            ]
        )

    def forward(self, x):
        return self.proj[4](
            self.proj[3](self.proj[2](self.proj[1](self.proj[0](x))[0]))[0]
        )


@dataclass(frozen=True)
class VPP2CrossKV:
    key: torch.Tensor
    value: torch.Tensor
    image_key: torch.Tensor | None = None
    image_value: torch.Tensor | None = None


@dataclass(frozen=True)
class VPP2ExpertCondition:
    context: torch.Tensor
    context_mask: torch.Tensor | None
    cross_kv: tuple[VPP2CrossKV, ...] = ()


@dataclass(frozen=True)
class VPP2Condition:
    video: VPP2ExpertCondition
    action: VPP2ExpertCondition | None


@dataclass(frozen=True)
class VPP2VideoCache:
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    tokens_per_frame: int
    video_seq_len: int
    prediction: torch.Tensor | None = None
    action_kv_buffers: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None


class VPP2AttentionBase(nn.Module):
    def __init__(self, config, *, dtype, device, prefix, attn_backend):
        super().__init__()
        self.mesh = parallel.default_mesh()
        self.tp_size = self.mesh.group_size("attention_tp")
        self.tp_rank = self.mesh.group_rank("attention_tp")
        config.validate_tp_size(self.tp_size)
        self.local_dim = config.attn_hidden_dim // self.tp_size
        self.num_local_heads = config.num_heads // self.tp_size
        self.head_dim = config.attn_head_dim
        self.qk_norm = config.qk_norm
        self.preserve_qk_layout = False
        self.norm_q = (
            RMSNorm(
                config.attn_hidden_dim,
                eps=config.eps,
                dtype=dtype,
                device=device,
                prefix=f"{prefix}.norm_q",
                kernel_role="vpp2.qk_norm",
                cast_before_affine=True,
            )
            if config.qk_norm
            else nn.Identity()
        )
        self.norm_k = (
            RMSNorm(
                config.attn_hidden_dim,
                eps=config.eps,
                dtype=dtype,
                device=device,
                prefix=f"{prefix}.norm_k",
                kernel_role="vpp2.qk_norm",
                cast_before_affine=True,
            )
            if config.qk_norm
            else nn.Identity()
        )
        self.o = RowParallelLinear(
            config.attn_hidden_dim,
            config.hidden_dim,
            group="attention_tp",
            bias=True,
            input_is_parallel=True,
            params_dtype=dtype,
            device=device,
            prefix=f"{prefix}.o",
        )
        self.attn = Attention(
            self.num_local_heads,
            self.head_dim,
            causal=False,
            backend=attn_backend,
            kernel_role="vpp2.attention",
        )

    def normalize(self, x, norm):
        # Wan normalizes the whole projection, before splitting it into heads.
        if self.tp_size > 1 and self.qk_norm:
            x = parallel.all_gather(x, group="attention_tp", dim=-1, mesh=self.mesh)
            x = norm(x)
            x = x.narrow(-1, self.tp_rank * self.local_dim, self.local_dim)
        elif self.qk_norm and self.preserve_qk_layout:
            x = norm(x, preserve_3d_layout=True)
        else:
            x = norm(x)
        return x.unflatten(-1, (self.num_local_heads, self.head_dim))


class VPP2SelfAttention(VPP2AttentionBase):
    def __init__(self, config, *, dtype, device, prefix, attn_backend):
        super().__init__(
            config, dtype=dtype, device=device, prefix=prefix, attn_backend=attn_backend
        )
        self.qkv = QKVParallelLinear(
            config.hidden_dim,
            config.attn_head_dim,
            config.num_heads,
            bias=True,
            params_dtype=dtype,
            device=device,
            prefix=f"{prefix}.qkv",
            hf_legs={"q": "q", "k": "k", "v": "v"},
        )
        self.norm_rope = QKRMSNormRotary(
            self.head_dim, eps=config.eps, kernel_role="vpp2.qk_norm_rope"
        )
        self.fuse_qk_norm_rope = False

    def project(self, x, rope, frequencies):
        q, k, v = self.qkv(x)[0].split(self.local_dim, dim=-1)
        if self.fuse_qk_norm_rope and self.qk_norm:
            if self.tp_size > 1:
                gathered = parallel.all_gather(
                    torch.stack((q, k), dim=0),
                    group="attention_tp",
                    dim=-1,
                    mesh=self.mesh,
                )
                q, k = gathered[0], gathered[1]
            q, k = self.norm_rope(
                q,
                k,
                self.norm_q.weight,
                self.norm_k.weight,
                *frequencies,
                head_start=self.tp_rank * self.num_local_heads,
                local_heads=self.num_local_heads,
            )
            return q, k, v.unflatten(-1, (self.num_local_heads, self.head_dim))
        if self.tp_size > 1 and self.qk_norm:
            gathered = parallel.all_gather(
                torch.stack((q, k), dim=0), group="attention_tp", dim=-1, mesh=self.mesh
            )
            offset = self.tp_rank * self.local_dim
            q = self.norm_q(gathered[0]).narrow(-1, offset, self.local_dim)
            k = self.norm_k(gathered[1]).narrow(-1, offset, self.local_dim)
            q = q.unflatten(-1, (self.num_local_heads, self.head_dim))
            k = k.unflatten(-1, (self.num_local_heads, self.head_dim))
        else:
            q, k = self.normalize(q, self.norm_q), self.normalize(k, self.norm_k)
        q, k = rope.apply(q, k, *frequencies)
        return q, k, v.unflatten(-1, (self.num_local_heads, self.head_dim))


class VPP2CrossAttention(VPP2AttentionBase):
    def __init__(self, config, *, image, dtype, device, prefix, attn_backend):
        super().__init__(
            config, dtype=dtype, device=device, prefix=prefix, attn_backend=attn_backend
        )
        self.image_tokens = config.image_tokens if image else 0
        self.q = ColumnParallelLinear(
            config.hidden_dim,
            config.attn_hidden_dim,
            group="attention_tp",
            bias=True,
            params_dtype=dtype,
            device=device,
            prefix=f"{prefix}.q",
        )
        self.kv = MergedColumnParallelLinear(
            config.hidden_dim,
            [config.attn_hidden_dim] * 2,
            group="attention_tp",
            bias=True,
            params_dtype=dtype,
            device=device,
            prefix=f"{prefix}.kv",
            hf_legs=("k", "v"),
        )
        if image:
            self.kv_img = MergedColumnParallelLinear(
                config.hidden_dim,
                [config.attn_hidden_dim] * 2,
                group="attention_tp",
                bias=True,
                params_dtype=dtype,
                device=device,
                prefix=f"{prefix}.kv_img",
                hf_legs=("k_img", "v_img"),
            )
            self.norm_k_img = (
                RMSNorm(
                    config.attn_hidden_dim,
                    eps=config.eps,
                    dtype=dtype,
                    device=device,
                    prefix=f"{prefix}.norm_k_img",
                    kernel_role="vpp2.qk_norm",
                    cast_before_affine=True,
                )
                if config.qk_norm
                else nn.Identity()
            )

    def encode(self, context):
        text = context[:, self.image_tokens :]
        k, v = self.kv(text)[0].split(self.local_dim, dim=-1)
        key = self.normalize(k, self.norm_k)
        value = v.unflatten(-1, (self.num_local_heads, self.head_dim))
        if not self.image_tokens:
            return VPP2CrossKV(key, value)
        k_img, v_img = self.kv_img(context[:, : self.image_tokens])[0].split(
            self.local_dim, dim=-1
        )
        return VPP2CrossKV(
            key,
            value,
            self.normalize(k_img, self.norm_k_img),
            v_img.unflatten(-1, (self.num_local_heads, self.head_dim)),
        )

    def forward(self, x, condition, *, kv=None, attention_context=None):
        kv = self.encode(condition.context) if kv is None else kv
        query = self.normalize(self.q(x)[0], self.norm_q)
        mask = None
        if condition.context_mask is not None:
            mask = AttnMask(key_mask=condition.context_mask[:, self.image_tokens :])
        text = self.attn(
            query,
            kv.key,
            kv.value,
            attention_context,
            mask=mask if attention_context is None else None,
        )
        if kv.image_key is not None:
            text = text + self.attn(query, kv.image_key, kv.image_value)
        return self.o(text.flatten(-2))[0]


class VPP2DiTBlock(nn.Module):
    def __init__(self, config, *, image=False, dtype, device, prefix, attn_backend):
        super().__init__()
        self.self_attn = VPP2SelfAttention(
            config,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.self_attn",
            attn_backend=attn_backend,
        )
        self.cross_attn = VPP2CrossAttention(
            config,
            image=image,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.cross_attn",
            attn_backend=attn_backend,
        )
        self.norm1 = LayerNorm(
            config.hidden_dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.norm2 = LayerNorm(
            config.hidden_dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.norm3 = LayerNorm(
            config.hidden_dim,
            eps=config.eps,
            elementwise_affine=config.cross_attn_norm,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.norm3",
            kernel_role="vpp2.residual_norm",
        )
        self.ffn = DenseMLP(
            config.hidden_dim,
            config.ffn_dim,
            activation="gelu_tanh",
            gated=False,
            bias=True,
            params_dtype=dtype,
            device=device,
            sequence_parallel=False,
            prefix=f"{prefix}.ffn",
        )
        self.modulation = modulation_parameter(
            (1, 6, config.hidden_dim),
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.modulation",
        )
        self.modulate = AffineModulation()
        self.norm_modulate = LayerNormModulation(
            eps=config.eps, kernel_role="vpp2.norm_modulation"
        )
        self.fuse_norm_modulation = False
        self.fuse_prepare_norm_modulation = False
        self.fuse_gated_layernorm = False
        self.gated_residual = vpp2_gated_residual

    def prepare(self, x, time_modulation, rope, frequencies):
        shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = (
            self.modulation + time_modulation
        ).unbind(dim=1)
        if self.fuse_prepare_norm_modulation:
            normalized, _ = self.norm_modulate(x, shift, scale)
        else:
            normalized = self.modulate(self.norm1(x), shift, scale)
        q, k, v = self.self_attn.project(normalized, rope, frequencies)
        return q, k, v, (gate, mlp_shift, mlp_scale, mlp_gate)

    def finish(
        self, x, attended, modulation, condition, *, kv=None, attention_context=None
    ):
        gate, shift, scale, mlp_gate = modulation
        branch = self.self_attn.o(attended.flatten(-2))[0]
        if self.fuse_gated_layernorm:
            cross_input, x = self.norm3.forward_gated_residual(x, branch, gate)
        else:
            x = self.gated_residual(x, gate, branch)
            cross_input = self.norm3(x)
        cross = self.cross_attn(
            cross_input, condition, kv=kv, attention_context=attention_context
        )
        if self.fuse_norm_modulation:
            normalized, x = self.norm_modulate(x, shift, scale, residual=cross)
        else:
            x = x + cross
            normalized = self.modulate(self.norm2(x), shift, scale)
        return self.gated_residual(x, mlp_gate, self.ffn(normalized))


class VPP2VideoHead(nn.Module):
    def __init__(self, config, *, dtype, device, prefix):
        super().__init__()
        self.norm = LayerNorm(
            config.hidden_dim,
            eps=config.eps,
            elementwise_affine=False,
            dtype=dtype,
            device=device,
        )
        self.head = ReplicatedLinear(
            config.hidden_dim,
            config.out_dim * math.prod(config.patch_size),
            bias=True,
            params_dtype=dtype,
            device=device,
            prefix=f"{prefix}.head",
        )
        self.modulation = modulation_parameter(
            (1, 2, config.hidden_dim),
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.modulation",
        )
        self.modulate = AffineModulation()
        self.norm_modulate = LayerNormModulation(
            eps=config.eps, kernel_role="vpp2.norm_modulation"
        )
        self.fuse_norm_modulation = False

    def forward(self, x, t):
        shift, scale = (self.modulation + t[:, None]).unbind(dim=1)
        if self.fuse_norm_modulation:
            normalized, _ = self.norm_modulate(x, shift, scale)
        else:
            normalized = self.modulate(self.norm(x), shift, scale)
        return self.head(normalized)[0]


class VPP2Expert(nn.Module):
    def __init__(self, config, *, image, params_dtype, device, prefix, attn_backend):
        super().__init__()
        if device is None:
            device = get_engine_config().device.target
        self.config = config
        self.params_dtype = params_dtype
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.attn_head_dim = config.attn_head_dim
        self.num_layers = config.num_layers
        config.validate_tp_size(parallel.default_mesh().group_size("dense_tp"))
        self.text_embedding = VPP2EmbeddingMLP(
            config.text_dim,
            config.hidden_dim,
            activation="gelu",
            dtype=params_dtype,
            device=device,
            prefix=join_prefix(prefix, "text_embedding"),
        )
        self.time_embedding = VPP2EmbeddingMLP(
            config.freq_dim,
            config.hidden_dim,
            activation="silu",
            dtype=params_dtype,
            device=device,
            prefix=join_prefix(prefix, "time_embedding"),
        )
        self.time_projection = nn.ModuleList(
            [
                SiLU(),
                ReplicatedLinear(
                    config.hidden_dim,
                    config.hidden_dim * 6,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=join_prefix(prefix, "time_projection.1"),
                ),
            ]
        )
        self.blocks = nn.ModuleList(
            VPP2DiTBlock(
                config,
                image=image,
                dtype=params_dtype,
                device=device,
                prefix=join_prefix(prefix, f"blocks.{i}"),
                attn_backend=attn_backend,
            )
            for i in range(config.num_layers)
        )
        axes = (config.attn_head_dim,)
        if image:
            spatial = 2 * (config.attn_head_dim // 6)
            axes = (config.attn_head_dim - 2 * spatial, spatial, spatial)
        self.rope = AxialRotaryEmbedding(axes, device=device)
        self.register_buffer(
            "time_frequencies",
            10000
            ** (
                -torch.arange(config.freq_dim // 2, dtype=torch.float64, device=device)
                / (config.freq_dim // 2)
            ),
            persistent=False,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        half = self.config.freq_dim // 2
        self.time_frequencies = 10000 ** (
            -torch.arange(
                half, dtype=torch.float64, device=self.time_frequencies.device
            )
            / half
        )
        return self

    def post_load(self) -> None:
        """Match the reference's model cast before promoting LayerNorm affine values."""
        for module in self.modules():
            if isinstance(module, LayerNorm) and module.elementwise_affine:
                for parameter in module.parameters(recurse=False):
                    parameter.data.copy_(parameter.data.to(self.params_dtype))

    def embed_time(self, timestep, batch_size):
        if timestep.ndim != 1 or timestep.shape[0] not in (1, batch_size):
            raise ValueError("timestep must have shape [B] or [1]")
        phases = timestep.double().reshape(-1, 1) * self.time_frequencies
        embedding = torch.cat((phases.cos(), phases.sin()), dim=-1)
        # Match the reference's cast through the timestep dtype.
        if timestep.is_floating_point():
            embedding = embedding.to(timestep.dtype)
        t = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype))
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size, -1)
        return t, self.time_projection[1](self.time_projection[0](t))[0].unflatten(
            -1, (6, self.hidden_dim)
        )

    def encode_condition(
        self, context, context_mask=None, clip_fea=None, *, precompute_kv=True
    ):
        if context.ndim != 3 or context.shape[-1] != self.config.text_dim:
            raise ValueError("context must have shape [B,L,text_dim]")
        if context_mask is not None and context_mask.shape != context.shape[:2]:
            raise ValueError("context_mask must match the context batch and length")
        embedded = self.text_embedding(context.to(self.text_embedding[0].weight.dtype))
        if hasattr(self, "img_emb"):
            if clip_fea is None:
                clip_fea = embedded.new_zeros(
                    (embedded.shape[0], self.config.image_tokens, self.config.image_dim)
                )
            if clip_fea.shape != (
                embedded.shape[0],
                self.config.image_tokens,
                self.config.image_dim,
            ):
                raise ValueError(
                    "CLIP features must have shape [B,image_tokens,image_dim]"
                )
            image = self.img_emb(clip_fea.to(embedded.dtype))
            embedded = torch.cat((image, embedded), dim=1)
            if context_mask is not None:
                context_mask = torch.cat(
                    (
                        torch.ones(
                            image.shape[:2], device=image.device, dtype=torch.bool
                        ),
                        context_mask.bool(),
                    ),
                    dim=1,
                )
        kv = (
            tuple(block.cross_attn.encode(embedded) for block in self.blocks)
            if precompute_kv
            else ()
        )
        return VPP2ExpertCondition(embedded, context_mask, kv)


class VPP2ActionDiT(VPP2Expert):
    def __init__(
        self,
        config: VPP2ActionConfig,
        *,
        params_dtype=torch.bfloat16,
        device=None,
        prefix="",
        attn_backend=None,
    ):
        super().__init__(
            config,
            image=False,
            params_dtype=params_dtype,
            device=device,
            prefix=prefix,
            attn_backend=attn_backend,
        )
        device = self.time_embedding[0].weight.device
        self.action_dim = config.action_dim
        self.action_encoder = ReplicatedLinear(
            config.action_dim,
            config.hidden_dim,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=join_prefix(prefix, "action_encoder"),
        )
        self.head = ReplicatedLinear(
            config.hidden_dim,
            config.action_dim,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=join_prefix(prefix, "head"),
        )

    def pre_dit(
        self,
        action_tokens,
        timestep,
        context=None,
        context_mask=None,
        *,
        condition=None,
    ):
        if action_tokens.ndim != 3 or action_tokens.shape[-1] != self.action_dim:
            raise ValueError("action_tokens must have shape [B,T,action_dim]")
        length = action_tokens.shape[1]
        if length > self.config.max_position_embeddings:
            raise ValueError("Action sequence exceeds the configured RoPE limit")
        if condition is None:
            condition = self.encode_condition(
                context, context_mask, precompute_kv=False
            )
        t, t_mod = self.embed_time(timestep, action_tokens.shape[0])
        positions = torch.arange(length, device=action_tokens.device).reshape(-1, 1)
        return {
            "tokens": self.action_encoder(action_tokens)[0],
            "t": t,
            "t_mod": t_mod,
            "freqs": self.rope.get_cos_sin(positions),
            "condition": condition,
            "context": condition.context,
            "context_mask": condition.context_mask,
            "meta": {"seq_len": length, "batch_size": action_tokens.shape[0]},
        }

    def post_dit(self, tokens, pre_state=None):
        return self.head(tokens)[0]

    def forward(
        self,
        action_tokens,
        timestep,
        context=None,
        context_mask=None,
        *,
        condition=None,
        video_cache=None,
        attention_contexts=None,
    ):
        pre = self.pre_dit(
            action_tokens, timestep, context, context_mask, condition=condition
        )
        condition = pre["condition"]
        x = pre["tokens"]
        if video_cache is not None and len(video_cache.layers) != len(self.blocks):
            raise ValueError("Video cache must have one K/V pair per action block")
        if (
            video_cache is not None
            and video_cache.action_kv_buffers is not None
            and len(video_cache.action_kv_buffers) != len(self.blocks)
        ):
            raise ValueError("Action K/V buffers must have one pair per action block")
        for i, block in enumerate(self.blocks):
            q, k, v, mod = block.prepare(x, pre["t_mod"], self.rope, pre["freqs"])
            if video_cache is not None:
                video_k, video_v = video_cache.layers[i]
                if video_cache.action_kv_buffers is None:
                    k, v = (
                        torch.cat((video_k, k), dim=1),
                        torch.cat((video_v, v), dim=1),
                    )
                else:
                    combined_k, combined_v = video_cache.action_kv_buffers[i]
                    video_length = video_cache.video_seq_len
                    expected_shape = (
                        k.shape[0],
                        video_length + k.shape[1],
                        *k.shape[2:],
                    )
                    if (
                        combined_k.shape != expected_shape
                        or combined_v.shape != expected_shape
                        or video_k.shape[1] != video_length
                        or video_v.shape[1] != video_length
                    ):
                        raise ValueError(
                            "Action K/V buffer shape must match video and action tokens"
                        )
                    # The runner owns these buffers and initializes their video prefix.
                    combined_k[:, video_length:].copy_(k)
                    combined_v[:, video_length:].copy_(v)
                    k, v = combined_k, combined_v
            contexts = (
                (None, None) if attention_contexts is None else attention_contexts[i]
            )
            attended = block.self_attn.attn(q, k, v, contexts[0])
            x = block.finish(
                x,
                attended,
                mod,
                condition,
                kv=condition.cross_kv[i] if condition.cross_kv else None,
                attention_context=contexts[1],
            )
        return self.post_dit(x)


class VPP2VideoDiT(VPP2Expert):
    def __init__(
        self,
        config: VPP2VideoConfig,
        *,
        params_dtype=torch.bfloat16,
        device=None,
        prefix="",
        attn_backend=None,
    ):
        super().__init__(
            config,
            image=True,
            params_dtype=params_dtype,
            device=device,
            prefix=prefix,
            attn_backend=attn_backend,
        )
        device = self.time_embedding[0].weight.device
        self.patch_embedding = Conv3d(
            config.in_dim,
            config.hidden_dim,
            config.patch_size,
            stride=config.patch_size,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=join_prefix(prefix, "patch_embedding"),
        )
        self.head = VPP2VideoHead(
            config,
            dtype=params_dtype,
            device=device,
            prefix=join_prefix(prefix, "head"),
        )
        self.img_emb = VPP2ImageProjection(
            config,
            dtype=params_dtype,
            device=device,
            prefix=join_prefix(prefix, "img_emb"),
        )

    def build_condition(self, x, condition_latents=None):
        condition_latents = (
            x[:, :, :1] if condition_latents is None else condition_latents
        )
        expected = (x.shape[0], self.config.out_dim, *x.shape[-2:])
        if (
            condition_latents.ndim != 5
            or (
                condition_latents.shape[0],
                condition_latents.shape[1],
                *condition_latents.shape[-2:],
            )
            != expected
        ):
            raise ValueError(
                "Condition latents must match video batch, channels and spatial dimensions"
            )
        count = condition_latents.shape[2]
        if count > x.shape[2] or count < 1:
            raise ValueError(
                "Condition length must be between one and the video length"
            )
        mask = x.new_zeros((x.shape[0], 4, *x.shape[2:]))
        mask[:, :, :count] = 1
        latents = x.new_zeros((x.shape[0], self.config.out_dim, *x.shape[2:]))
        latents[:, :, :count] = condition_latents
        return torch.cat((mask, latents), dim=1)

    def build_attention_mask(self, grid):
        frames, height, width = grid
        per_frame = height * width
        mode = self.config.video_attention_mask_mode
        if mode in ("bidirectional", "three_frame") or frames == 1:
            return None
        if mode == "first_frame_causal":
            return AttnMask(
                segments=((per_frame, False), ((frames - 1) * per_frame, True))
            )
        return AttnMask(
            segments=tuple((per_frame, index > 0) for index in range(frames))
        )

    def pre_dit(
        self,
        x,
        timestep,
        context=None,
        context_mask=None,
        clip_fea=None,
        condition_latents=None,
        *,
        condition=None,
    ):
        if x.ndim != 5:
            raise ValueError("Video latents must have shape [B,C,T,H,W]")
        if any(
            size % patch for size, patch in zip(x.shape[2:], self.config.patch_size)
        ):
            raise ValueError("Latent dimensions must be divisible by patch_size")
        if x.shape[1] == self.config.out_dim:
            x = torch.cat((x, self.build_condition(x, condition_latents)), dim=1)
        if x.shape[1] != self.config.in_dim:
            raise ValueError("Incorrect conditioned video channel count")
        if condition is None:
            condition = self.encode_condition(
                context, context_mask, clip_fea, precompute_kv=False
            )
        patches = self.patch_embedding(x)
        grid = patches.shape[2:]
        if max(grid) > self.config.max_position_embeddings:
            raise ValueError("Video grid exceeds the configured RoPE limit")
        t, t_mod = self.embed_time(timestep, x.shape[0])
        axes = torch.meshgrid(
            *(torch.arange(size, device=x.device) for size in grid), indexing="ij"
        )
        positions = torch.stack(axes, dim=-1).reshape(-1, 3)
        return {
            "tokens": patches.flatten(2).transpose(1, 2).contiguous(),
            "t": t,
            "t_mod": t_mod,
            "freqs": self.rope.get_cos_sin(positions),
            "condition": condition,
            "context": condition.context,
            "context_mask": condition.context_mask,
            "meta": {
                "grid_size": tuple(grid),
                "tokens_per_frame": grid[1] * grid[2],
                "batch_size": x.shape[0],
            },
        }

    def post_dit(self, tokens, pre_state):
        f, h, w = pre_state["meta"]["grid_size"]
        pt, ph, pw = self.config.patch_size
        prediction = self.head(tokens, pre_state["t"])
        return (
            prediction.reshape(
                prediction.shape[0], f, h, w, pt, ph, pw, self.config.out_dim
            )
            .permute(0, 7, 1, 4, 2, 5, 3, 6)
            .reshape(prediction.shape[0], self.config.out_dim, f * pt, h * ph, w * pw)
        )

    def forward(
        self,
        x,
        timestep,
        context=None,
        context_mask=None,
        clip_fea=None,
        condition_latents=None,
        *,
        condition=None,
        cache_visible_frames=None,
        return_prediction=True,
    ):
        pre = self.pre_dit(
            x,
            timestep,
            context,
            context_mask,
            clip_fea,
            condition_latents,
            condition=condition,
        )
        condition = pre["condition"]
        hidden = pre["tokens"]
        mask = self.build_attention_mask(pre["meta"]["grid_size"])
        caches = []
        visible = (
            hidden.shape[1]
            if cache_visible_frames is None
            else min(
                hidden.shape[1], cache_visible_frames * pre["meta"]["tokens_per_frame"]
            )
        )
        for i, block in enumerate(self.blocks):
            q, k, v, mod = block.prepare(hidden, pre["t_mod"], self.rope, pre["freqs"])
            if cache_visible_frames is not None:
                caches.append(
                    (k[:, :visible].contiguous(), v[:, :visible].contiguous())
                )
            # The final video output is unnecessary when only its K/V is consumed.
            if i == len(self.blocks) - 1 and not return_prediction:
                break
            attended = block.self_attn.attn(q, k, v, mask=mask)
            hidden = block.finish(
                hidden,
                attended,
                mod,
                condition,
                kv=condition.cross_kv[i] if condition.cross_kv else None,
            )
        prediction = self.post_dit(hidden, pre) if return_prediction else None
        if cache_visible_frames is None:
            return prediction
        return VPP2VideoCache(
            tuple(caches), pre["meta"]["tokens_per_frame"], visible, prediction
        )


class VPP2Model(nn.Module):
    def __init__(
        self,
        config: VPP2Config,
        *,
        params_dtype=torch.bfloat16,
        device=None,
        attn_backend=None,
        with_action=True,
    ):
        super().__init__()
        self.config = config
        self.video_expert = VPP2VideoDiT(
            config.video,
            params_dtype=params_dtype,
            device=device,
            prefix="video_expert",
            attn_backend=attn_backend,
        )
        self.action_expert = (
            VPP2ActionDiT(
                config.action,
                params_dtype=params_dtype,
                device=device,
                prefix="action_expert",
                attn_backend=attn_backend,
            )
            if with_action
            else None
        )
        self.proprio_encoder = (
            None
            if config.proprio_dim is None or not with_action
            else ReplicatedLinear(
                config.proprio_dim,
                config.action.text_dim,
                bias=True,
                params_dtype=params_dtype,
                device=self.video_expert.patch_embedding.weight.device,
                prefix="proprio_encoder",
            )
        )

    def enable_pointwise_fusion(self) -> None:
        """Fuse gated residuals without dropping intermediate BF16 rounding."""
        fused = torch.compile(
            vpp2_gated_residual,
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True, "triton.cudagraphs": False},
        )
        for expert in (self.video_expert, self.action_expert):
            if expert is not None:
                for block in expert.blocks:
                    block.gated_residual = fused

    def enable_qk_norm_layout(self, enabled: bool = True) -> None:
        """Let shared RMSNorm read strided Q/K projections without copying them."""
        for module in self.modules():
            if isinstance(module, VPP2AttentionBase):
                module.preserve_qk_layout = enabled

    def enable_layernorm_modulation(
        self, enabled: bool = True, *, include_prepare: bool = True
    ) -> None:
        """Fuse MLP input normalization and optionally block/head input normalization."""
        for expert in (self.video_expert, self.action_expert):
            if expert is not None:
                for block in expert.blocks:
                    block.fuse_norm_modulation = enabled
                    block.fuse_prepare_norm_modulation = enabled and include_prepare
        self.video_expert.head.fuse_norm_modulation = enabled and include_prepare

    def enable_qk_norm_fusion(self, enabled: bool = True) -> None:
        """Fuse Q/K RMS normalization and gamma while preserving activation rounding."""
        for module in self.modules():
            if isinstance(module, RMSNorm) and module.kernel_role == "vpp2.qk_norm":
                module.fuse_cast_before_affine = enabled

    def enable_qk_norm_rope_fusion(self, enabled: bool = True) -> None:
        """Fuse full-width Q/K normalization, rounded gamma, and rotary embedding."""
        for module in self.modules():
            if isinstance(module, VPP2SelfAttention):
                module.fuse_qk_norm_rope = enabled

    def enable_gated_layernorm(self, enabled: bool = True) -> None:
        """Fuse the self-attention gated residual with cross-attention normalization."""
        for module in self.modules():
            if isinstance(module, VPP2DiTBlock):
                module.fuse_gated_layernorm = enabled

    def encode_condition(self, context, context_mask=None, proprio=None, clip_fea=None):
        if proprio is not None:
            if self.proprio_encoder is None:
                raise ValueError("This model has no proprioception encoder")
            if proprio.ndim != 2 or proprio.shape != (
                context.shape[0],
                self.config.proprio_dim,
            ):
                raise ValueError("proprio must have shape [B,proprio_dim]")
            if context_mask is not None:
                context_mask = torch.cat(
                    (
                        context_mask.bool(),
                        torch.ones(
                            (context.shape[0], 1),
                            device=context_mask.device,
                            dtype=torch.bool,
                        ),
                    ),
                    dim=1,
                )
            proprio = self.proprio_encoder(proprio[:, None].to(context.dtype))[0]
            context = torch.cat((context, proprio), dim=1)
        return VPP2Condition(
            self.video_expert.encode_condition(context, context_mask, clip_fea),
            self.action_expert.encode_condition(context, context_mask)
            if self.action_expert is not None
            else None,
        )

    def prefill_video(
        self,
        video,
        timestep,
        condition,
        condition_latents=None,
        action_seq_len=None,
        return_prediction=False,
    ):
        return self.video_expert(
            video,
            timestep,
            condition_latents=condition_latents,
            condition=condition.video,
            cache_visible_frames=self.config.action_visible_video_frames,
            return_prediction=return_prediction,
        )

    def forward_action(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        condition: VPP2Condition,
        video_cache: VPP2VideoCache,
        *,
        attention_contexts: tuple[tuple[AttnCtx, AttnCtx], ...] | None = None,
    ):
        if self.action_expert is None or condition.action is None:
            raise ValueError("Action inference requires an action checkpoint")
        return self.action_expert(
            action,
            timestep,
            condition=condition.action,
            video_cache=video_cache,
            attention_contexts=attention_contexts,
        )

    def forward_video(self, video, timestep, condition, condition_latents=None):
        return self.video_expert(
            video,
            timestep,
            condition=condition.video,
            condition_latents=condition_latents,
        )

    def forward(self, action, timestep, condition, video_cache):
        return self.forward_action(action, timestep, condition, video_cache)
