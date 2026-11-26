"""LingBot-VLA 2.0 text and action modules."""

from __future__ import annotations

import math
import itertools
from collections.abc import Iterable

import torch
from torch import nn
from phyai_kernel.triton.lingbot_norm import lingbot_adarmsnorm

import phyai.parallel as P
from phyai.layers.activation import SiLU
from phyai.layers.mlp import DenseMLP
from phyai.kernel.call import CallSite
from phyai.layers.quant import Bf16Spec
from phyai.engine_config import get_engine_config, resolve_params_dtype
from phyai.layers.linear import (
    ReplicatedLinear,
    QKVParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
)
from phyai.parallel.state import resolve_mesh
from phyai.weights.shards import _Leg, fused, sharded, replicated
from phyai.models.lingbot_v2.norm import LingBotV2RMSNorm
from phyai.layers.attention.paged import PagedAttnCtx, PagedAttention
from phyai.layers.rotary_embedding import (
    InterleavedMRotaryEmbedding,
)
from phyai.layers.vocab_embedding.layers import VocabParallelEmbedding
from phyai.models.lingbot_v2.vision import Qwen3VLVisionModel
from phyai.models.lingbot_v2.configuration_lingbotv2 import (
    LingBotVLA2Config,
    Qwen3VLTextConfig,
    LingBotV2MoEConfig,
    LingBotV2ExpertConfig,
    LingBotV2FlowMatchingConfig,
)


def _official_topk_router(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the released LingBot sigmoid top-k router exactly."""
    routing_scores = router_logits.sigmoid()
    scores_for_choice = routing_scores + correction_bias.unsqueeze(0)
    selected_experts = torch.topk(
        scores_for_choice,
        top_k,
        dim=-1,
    ).indices
    routing_weights = routing_scores.gather(1, selected_experts)
    routing_weights = routing_weights / (
        routing_weights.sum(dim=-1, keepdim=True) + 1e-20
    )
    if routed_scaling_factor != 1.0:
        routing_weights = routing_weights * routed_scaling_factor
    return routing_weights, selected_experts


class Qwen3VLTextBlock(nn.Module):
    """Qwen text block with FP32 Q/K/V and LingBot normalization."""

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        layer_idx: int,
        *,
        rope: InterleavedMRotaryEmbedding,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
        attn_backend: str | None,
        causal: bool,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.rope = rope
        attn_prefix = f"{prefix}.self_attn"
        self.input_norm = LingBotV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.input_layernorm",
        )
        self.pre_ff_norm = LingBotV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.post_attention_layernorm",
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{attn_prefix}.qkv_proj",
        )
        q_width, k_width, _ = self.qkv_proj.output_partition_sizes
        self.q_heads_local = q_width // config.head_dim
        self.kv_heads_local = k_width // config.head_dim
        self.q_norm = LingBotV2RMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{attn_prefix}.q_norm",
        )
        self.k_norm = LingBotV2RMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{attn_prefix}.k_norm",
        )
        self.attn = PagedAttention(
            num_heads=self.q_heads_local,
            num_kv_heads=self.kv_heads_local,
            head_dim=config.head_dim,
            layer_id=layer_idx,
            causal=causal,
            backend=attn_backend,
            kernel_role="prefix",
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            group="attention_tp",
            input_is_parallel=True,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{attn_prefix}.o_proj",
        )
        self.mlp = DenseMLP(
            config.hidden_size,
            config.intermediate_size,
            activation=config.hidden_act,
            gated=True,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.mlp",
        )

    def project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, _ = self.qkv_proj(self.input_norm(x))
        q, k, v = fused.split(self.qkv_proj.output_partition_sizes, dim=-1)
        leading = x.shape[:-1]
        return (
            q.reshape(*leading, self.q_heads_local, self.head_dim),
            k.reshape(*leading, self.kv_heads_local, self.head_dim),
            v.reshape(*leading, self.kv_heads_local, self.head_dim),
        )

    def prefix_kv(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce the final prefix K/V without attention or feed-forward output."""
        _, k, v = self.project_qkv(x)
        k = self.k_norm(k).float()
        _, k = self.rope.apply(k, k, cos, sin)
        return (
            k.reshape(-1, self.kv_heads_local, self.head_dim),
            v.reshape(-1, self.kv_heads_local, self.head_dim).float(),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: PagedAttnCtx,
    ) -> torch.Tensor:
        q, k, v = self.project_qkv(x)
        q, k = self.rope.apply(self.q_norm(q).float(), self.k_norm(k).float(), cos, sin)
        attn_out = self.attn(
            q.reshape(-1, self.q_heads_local, self.head_dim),
            k.reshape(-1, self.kv_heads_local, self.head_dim),
            v.reshape(-1, self.kv_heads_local, self.head_dim).float(),
            attn_ctx,
        )
        out, _ = self.o_proj(
            attn_out.reshape(*x.shape[:-1], -1).to(self.o_proj.weight.dtype)
        )
        hidden_states = x + out
        return hidden_states + self.mlp(self.pre_ff_norm(hidden_states))


class Qwen3VLTextModel(nn.Module):
    """Qwen3 prefix decoder with M-RoPE and DeepStack features."""

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "model.language_model",
        attn_causal: bool = True,
    ) -> None:
        super().__init__()
        params_dtype = resolve_params_dtype(params_dtype)
        device = device if device is not None else get_engine_config().device.target
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.embed_tokens",
        )
        self.rotary_emb = InterleavedMRotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            mrope_section=config.mrope_section,
            rope_theta=config.rope_theta,
            backend="eager",
            device=device,
        ).to(dtype=params_dtype)
        self.layers = nn.ModuleList(
            [
                Qwen3VLTextBlock(
                    config,
                    i,
                    rope=self.rotary_emb,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.layers.{i}",
                    attn_backend=attn_backend,
                    causal=attn_causal,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = LingBotV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm",
        )

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        layers: Iterable[Qwen3VLTextBlock],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: PagedAttnCtx,
        visual_token_indices: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        for layer_idx, layer in enumerate(layers):
            hidden_states = layer(hidden_states, cos=cos, sin=sin, attn_ctx=attn_ctx)
            if (
                deepstack_visual_embeds is not None
                and visual_token_indices is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                visual_features = deepstack_visual_embeds[layer_idx].to(hidden_states)
                flat = hidden_states.reshape(-1, hidden_states.shape[-1])
                flat.index_add_(0, visual_token_indices, visual_features)
                hidden_states = flat.view(hidden_states.shape)
        return hidden_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: PagedAttnCtx,
        visual_token_indices: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cos = cos.reshape(*inputs_embeds.shape[:-1], -1)
        sin = sin.reshape(*inputs_embeds.shape[:-1], -1)
        hidden_states = self._run_layers(
            inputs_embeds,
            self.layers,
            cos,
            sin,
            attn_ctx,
            visual_token_indices,
            deepstack_visual_embeds,
        )
        return self.norm(hidden_states)

    def prefill(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: PagedAttnCtx,
        visual_token_indices: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the prefix and return the final layer's K/V for the runner to store."""
        cos = cos.reshape(*inputs_embeds.shape[:-1], -1)
        sin = sin.reshape(*inputs_embeds.shape[:-1], -1)
        hidden_states = self._run_layers(
            inputs_embeds,
            itertools.islice(self.layers, len(self.layers) - 1),
            cos,
            sin,
            attn_ctx,
            visual_token_indices,
            deepstack_visual_embeds,
        )
        return self.layers[-1].prefix_kv(hidden_states, cos, sin)


class LingBotV2AdaRMSNorm(nn.Module):
    """Checkpoint-compatible ``RMSNorm + FiLM`` used by the action stream."""

    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        *,
        eps: float,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        self.weight.hf_keys = [(f"{prefix}.weight", None)]
        self.weight.weight_loader = replicated()
        self.gamma = ReplicatedLinear(
            cond_dim,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.gamma",
        )
        self.beta = ReplicatedLinear(
            cond_dim,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.beta",
        )

        # Zero modulation initializes this layer as a weighted RMSNorm.
        with torch.no_grad():
            self.gamma.weight.zero_()
            if self.gamma.bias is not None:
                self.gamma.bias.zero_()
            self.beta.weight.zero_()
            if self.beta.bias is not None:
                self.beta.bias.zero_()

    def compute_modulation(
        self, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute condition-dependent gamma and beta once."""
        gamma, _ = self.gamma(cond.to(self.gamma.weight.dtype))
        beta, _ = self.beta(cond.to(self.beta.weight.dtype))
        return gamma, beta

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: torch.Tensor,
        *,
        modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply RMS normalization modulated by the diffusion condition."""

        if modulation is None:
            gamma, beta = self.compute_modulation(cond)
        else:
            gamma, beta = modulation
        return lingbot_adarmsnorm(
            hidden_states, self.weight, gamma, beta, eps=self.variance_epsilon
        )


class LingBotV2TokenMoE(nn.Module):
    """LingBot MoE with FP32 routing math and BF16-stored parameters."""

    def __init__(
        self,
        hidden_size: int,
        config: LingBotV2MoEConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise ValueError(
                "LingBotV2TokenMoE requires bfloat16 expert parameters because "
                "FlashInfer CUTLASS fused MoE consumes bfloat16 expert weights."
            )
        if config.router_activation != "sigmoid" or not config.normalize_topk_prob:
            raise NotImplementedError(
                "The FlashInfer LingBot router requires sigmoid routing with "
                "normalized top-k probabilities."
            )
        if config.shared_expert_gate:
            raise NotImplementedError(
                "The released LingBot V2 has no shared-expert gate; enabling it "
                "requires a fused sigmoid-multiply kernel."
            )
        if config.router_bias:
            raise NotImplementedError("The released LingBot V2 router is bias-free.")
        if not config.use_router_correction_bias:
            raise NotImplementedError(
                "use_router_correction_bias=False is unsupported: LingBot V2 "
                "routing requires the checkpoint's expert correction bias."
            )

        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = ReplicatedLinear(
            hidden_size,
            config.num_experts,
            bias=config.router_bias,
            params_dtype=params_dtype,
            out_dtype=torch.float32,
            spec=Bf16Spec(),
            device=device,
            prefix=f"{prefix}.gate",
            kernel_role="router",
        )
        mesh = resolve_mesh("model")
        self.mesh = mesh
        self.tp_size = mesh.group_size("moe_tp")
        self.tp_rank = mesh.group_rank("moe_tp")
        if config.moe_intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size={config.moe_intermediate_size} must be "
                f"divisible by tp_size={self.tp_size}."
            )
        self.intermediate_size_per_partition = (
            config.moe_intermediate_size // self.tp_size
        )

        # Expert geometry is fixed at construction, so it belongs on the
        # call site; only the token count varies between calls.
        self._moe_call = CallSite(
            "moe",
            role="lingbot_v2.token_moe",
            dims={"hidden": hidden_size},
            attrs={"activation": "swiglu"},
        )

        # FlashInfer SwiGLU expects FC1 in [up, gate] order. Fuse and TP-shard
        # the public checkpoint's separate [E, I, H] tensors at load time.
        self.expert_gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * self.intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        local_intermediate = self.intermediate_size_per_partition
        self.expert_gate_up_proj.hf_keys = [
            (f"{prefix}.experts.up_proj", "up"),
            (f"{prefix}.experts.gate_proj", "gate"),
        ]
        self.expert_gate_up_proj.weight_loader = fused(
            fuse_dim=1,
            legs={
                "up": _Leg(
                    offset=0,
                    size=local_intermediate,
                    dim=1,
                    group="moe_tp",
                ),
                "gate": _Leg(
                    offset=local_intermediate,
                    size=local_intermediate,
                    dim=1,
                    group="moe_tp",
                ),
            },
            mesh=mesh,
        )
        self.expert_down_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                hidden_size,
                self.intermediate_size_per_partition,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.expert_down_proj.hf_keys = [(f"{prefix}.experts.down_proj", None)]
        self.expert_down_proj.weight_loader = sharded(
            dim=2,
            group="moe_tp",
            mesh=mesh,
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        self.e_score_correction_bias.hf_keys = [
            (f"{prefix}.e_score_correction_bias", None)
        ]
        self.e_score_correction_bias.weight_loader = replicated()

        self.shared_expert = DenseMLP(
            hidden_size,
            config.shared_expert_intermediate_size,
            activation="silu",
            gated=True,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.shared_expert",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route tokens through selected experts and the shared expert path."""

        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, original_shape[-1])
        router_logits, _ = self.gate(hidden_flat)
        routing_weights, selected_experts = _official_topk_router(
            router_logits,
            self.e_score_correction_bias.float(),
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        routing_weights = routing_weights.to(hidden_flat.dtype)
        selection = self._moe_call.select(
            device=hidden_flat.device,
            dtype={
                "input": hidden_flat.dtype,
                "weight": self.expert_gate_up_proj.dtype,
            },
            dims={"tokens": hidden_flat.shape[0]},
        )
        routed = selection.execute(
            hidden_flat,
            selected_experts,
            routing_weights,
            self.expert_gate_up_proj,
            self.expert_down_proj,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        if self.tp_size > 1:
            # Reduce only routed partials over moe_tp. DenseMLP handles the
            # shared branch's own dense_tp reduction; adding it here would
            # reduce that completed contribution a second time.
            routed = P.all_reduce(routed, group="moe_tp", mesh=self.mesh)

        shared = self.shared_expert(hidden_flat)
        return (routed + shared).reshape(original_shape)


class LingBotV2ExpertLayer(nn.Module):
    """Qwen2 action layer backed by paged diffusion attention."""

    def __init__(
        self,
        config: LingBotV2ExpertConfig,
        moe_config: LingBotV2MoEConfig,
        layer_idx: int,
        *,
        params_dtype: torch.dtype,
        attn_backend: str | None,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        attn_prefix = f"{prefix}.self_attn"
        self.input_layernorm = LingBotV2AdaRMSNorm(
            config.hidden_size,
            config.adarms_cond_dim,
            eps=config.rms_norm_eps,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.input_layernorm",
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            bias=config.qkv_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{attn_prefix}.qkv_proj",
        )
        q_width, k_width, _ = self.qkv_proj.output_partition_sizes
        self.q_heads_local = q_width // config.head_dim
        self.kv_heads_local = k_width // config.head_dim
        self.o_proj = RowParallelLinear(
            in_features=config.joint_attention_dim,
            out_features=config.hidden_size,
            group="attention_tp",
            bias=config.output_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{attn_prefix}.o_proj",
        )
        self.attn = PagedAttention(
            num_heads=self.q_heads_local,
            head_dim=config.head_dim,
            layer_id=layer_idx,
            num_kv_heads=self.kv_heads_local,
            causal=False,
            backend=attn_backend,
            kernel_role="expert",
        )
        self.post_attention_layernorm = LingBotV2AdaRMSNorm(
            config.hidden_size,
            config.adarms_cond_dim,
            eps=config.rms_norm_eps,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.post_attention_layernorm",
        )
        if layer_idx in moe_config.layer_indices:
            self.mlp: nn.Module = LingBotV2TokenMoE(
                config.hidden_size,
                moe_config,
                params_dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DenseMLP(
                config.hidden_size,
                config.intermediate_size,
                activation="silu",
                gated=True,
                bias=False,
                params_dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.mlp",
            )

    def _split_qkv(
        self, fused: torch.Tensor, leading: torch.Size
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_dim = self.q_heads_local * self.head_dim
        kv_dim = self.kv_heads_local * self.head_dim
        q, k, v = fused.split([q_dim, kv_dim, kv_dim], dim=-1)
        return (
            q.reshape(*leading, self.q_heads_local, self.head_dim),
            k.reshape(*leading, self.kv_heads_local, self.head_dim),
            v.reshape(*leading, self.kv_heads_local, self.head_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cond: torch.Tensor,
        rope: InterleavedMRotaryEmbedding,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: PagedAttnCtx,
        input_modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
        post_modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply joint diffusion attention, normalization, and the MLP path."""

        residual = hidden_states
        normed = self.input_layernorm(hidden_states, cond, modulation=input_modulation)
        fused, _ = self.qkv_proj(normed)
        q, k, v = self._split_qkv(fused, hidden_states.shape[:-1])
        # Official joint attention promotes every stream's Q/K/V before MRoPE.
        q, k, v = q.float(), k.float(), v.float()
        q, k = rope.apply(q, k, cos, sin)
        # Paged attention consumes flattened tokens; AdaNorm uses (B, S, H).
        q = q.reshape(-1, self.q_heads_local, self.head_dim)
        k = k.reshape(-1, self.kv_heads_local, self.head_dim)
        v = v.reshape(-1, self.kv_heads_local, self.head_dim)
        attn_out = self.attn(q, k, v, attn_ctx).reshape(
            *hidden_states.shape[:-1],
            -1,
        )
        attn_out = attn_out.to(self.o_proj.weight.dtype)
        attn_out, _ = self.o_proj(attn_out)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            hidden_states, cond, modulation=post_modulation
        )
        return residual + self.mlp(hidden_states)


class LingBotV2ExpertStack(nn.Module):
    """All action layers plus the released plain final RMSNorm."""

    DEFAULT_PREFIX = "model.qwenvl_with_expert.qwen_expert.model"

    def __init__(
        self,
        config: LingBotV2ExpertConfig,
        moe_config: LingBotV2MoEConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str | None,
        device: torch.device | str,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                LingBotV2ExpertLayer(
                    config,
                    moe_config,
                    layer_idx,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    device=device,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LingBotV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm",
        )


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    *,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    """Flow-time sinusoidal embedding used both as token input and AdaNorm cond."""

    if time.dim() != 1 or dimension % 2:
        raise ValueError("time must be 1-D and dimension must be even.")
    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=torch.float32, device=time.device
    )
    period = min_period * (max_period / min_period) ** fraction
    phase = time.float().unsqueeze(1) * (2.0 * math.pi / period).unsqueeze(0)
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


class LingBotV2ActionTimeHeads(nn.Module):
    """State/action projections and flow-time fusion heads."""

    def __init__(
        self,
        expert_config: LingBotV2ExpertConfig,
        flow_config: LingBotV2FlowMatchingConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str = "model",
    ) -> None:
        super().__init__()
        self.expert_hidden = expert_config.hidden_size
        self.flow_config = flow_config
        # Boundary projections consume and produce replicated residuals.
        replicated_specs = (
            ("state_proj", flow_config.max_state_dim, expert_config.hidden_size),
            ("action_in_proj", flow_config.max_action_dim, expert_config.hidden_size),
            ("action_out_proj", expert_config.hidden_size, flow_config.max_action_dim),
        )
        for name, in_features, out_features in replicated_specs:
            setattr(
                self,
                name,
                ReplicatedLinear(
                    in_features,
                    out_features,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.action_time_mlp_in = ColumnParallelLinear(
            expert_config.hidden_size * 2,
            expert_config.hidden_size,
            gather_output=False,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.action_time_mlp_in",
        )
        self.action_time_mlp_out = RowParallelLinear(
            expert_config.hidden_size,
            expert_config.hidden_size,
            input_is_parallel=True,
            reduce_results=True,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.action_time_mlp_out",
        )
        self.activation = SiLU()
        fraction = torch.linspace(
            0.0, 1.0, self.expert_hidden // 2, dtype=torch.float32, device=device
        )
        min_period = flow_config.time_embedding_min_period
        max_period = flow_config.time_embedding_max_period
        period = min_period * (max_period / min_period) ** fraction
        self.register_buffer(
            "time_frequencies", 2.0 * math.pi / period, persistent=False
        )

    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        """Project padded model state into the action expert width."""

        out, _ = self.state_proj(state.to(self.state_proj.weight.dtype))
        return out

    def time_condition(self, time: torch.Tensor) -> torch.Tensor:
        phase = time.float().unsqueeze(1) * self.time_frequencies.unsqueeze(0)
        return torch.cat([phase.sin(), phase.cos()], dim=-1).to(
            self.action_in_proj.weight.dtype
        )

    def embed_action(self, actions: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        action_emb, _ = self.action_in_proj(
            actions.to(self.action_in_proj.weight.dtype)
        )
        cond_tokens = cond.unsqueeze(1).expand(-1, action_emb.shape[1], -1)
        fused = torch.cat([action_emb, cond_tokens], dim=-1)
        fused, _ = self.action_time_mlp_in(fused)
        fused, _ = self.action_time_mlp_out(self.activation(fused))
        return fused

    def embed_action_time(
        self, actions: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed noisy actions and diffusion time into expert inputs."""
        cond = self.time_condition(time)
        return self.embed_action(actions, cond), cond

    def project_action(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project expert hidden states back to action coordinates."""

        out, _ = self.action_out_proj(
            hidden_states.to(self.action_out_proj.weight.dtype)
        )
        return out


class LingBotV2DualQuery(nn.Module):
    """Build the eight current and eight future learned prefix queries."""

    def __init__(
        self,
        config,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str = "model",
    ) -> None:
        super().__init__()
        released_layout = {
            "use_future_depth": True,
            "use_future_video": True,
            "use_future_video_patch": True,
            "use_current_video_patch": True,
            "share_future_depth_query": True,
            "use_shared_future_task_proj": True,
            "use_current_shared_task_proj": True,
            "use_future_video_cls": False,
        }
        unsupported = [
            f"{name}={getattr(config, name)!r} (expected {expected!r})"
            for name, expected in released_layout.items()
            if getattr(config, name) != expected
        ]
        if unsupported:
            raise ValueError(
                "LingBotV2DualQuery supports only the released four-seed-table "
                "shared-projection layout; unsupported settings: "
                + ", ".join(unsupported)
            )
        self.config = config
        names = (
            "depth_align_embs",
            "current_video_align_embs",
            "future_depth_align_embs",
            "future_video_align_embs",
        )
        for name in names:
            parameter = nn.Parameter(
                torch.empty(
                    config.num_query_seeds,
                    config.query_hidden_size,
                    dtype=params_dtype,
                    device=device,
                ),
                requires_grad=False,
            )
            parameter.hf_keys = [(f"{prefix}.{name}", None)]
            parameter.weight_loader = replicated()
            setattr(self, name, parameter)
        self.current_shared_task_proj = RowParallelLinear(
            config.query_hidden_size * 2,
            config.query_hidden_size,
            input_is_parallel=False,
            reduce_results=True,
            bias=config.fusion_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.current_shared_task_proj",
        )
        self.future_shared_task_proj = RowParallelLinear(
            config.query_hidden_size * 2,
            config.query_hidden_size,
            input_is_parallel=False,
            reduce_results=True,
            bias=config.fusion_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.future_shared_task_proj",
        )

    def _pool(self, seeds: torch.Tensor) -> torch.Tensor:
        return seeds.view(
            self.config.num_task_tokens,
            self.config.seeds_per_task_token,
            self.config.query_hidden_size,
        ).mean(dim=1)

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return broadcast current and future learned task-query tokens."""

        current_depth = self._pool(self.depth_align_embs)
        current_video = self._pool(self.current_video_align_embs)
        current, _ = self.current_shared_task_proj(
            torch.cat([current_depth, current_video], dim=-1)
        )
        future_depth = self._pool(self.future_depth_align_embs)
        future_video = self._pool(self.future_video_align_embs)
        future, _ = self.future_shared_task_proj(
            torch.cat([future_depth, future_video], dim=-1)
        )
        return (
            current.unsqueeze(0).expand(batch_size, -1, -1),
            future.unsqueeze(0).expand(batch_size, -1, -1),
        )


def _vision_mrope_positions(
    start: int, grid_thw: torch.Tensor, merge_size: int
) -> torch.Tensor:
    t, h, w = (int(value) for value in grid_thw.tolist())
    h //= merge_size
    w //= merge_size
    temporal = torch.arange(t, device=grid_thw.device).repeat_interleave(h * w) + start
    height = (
        torch.arange(h, device=grid_thw.device).repeat_interleave(w).repeat(t) + start
    )
    width = torch.arange(w, device=grid_thw.device).repeat(h * t) + start
    return torch.stack([temporal, height, width], dim=0)


def build_lingbot_v2_mrope_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    image_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build CPU ``(3, B, S)`` positions from host-side prefix metadata."""
    if any(t.device.type != "cpu" for t in (input_ids, attention_mask, image_grid_thw)):
        raise ValueError("prefix position metadata must be on CPU")
    position_ids = torch.zeros(
        3, *input_ids.shape, dtype=torch.int64, device=input_ids.device
    )
    grid_index = 0
    for batch_index in range(input_ids.shape[0]):
        valid = attention_mask[batch_index].bool()
        ids = input_ids[batch_index, valid]
        token_types = (ids == image_token_id).to(torch.int64)
        current_position = 0
        parts: list[torch.Tensor] = []
        for modality, group in itertools.groupby(
            enumerate(token_types.tolist()), lambda item: item[1]
        ):
            group_list = list(group)
            length = group_list[-1][0] - group_list[0][0] + 1
            if modality == 0:
                part = torch.arange(length, device=input_ids.device).expand(3, -1)
                parts.append(part + current_position)
                current_position += length
            else:
                grid = image_grid_thw[grid_index]
                grid_index += 1
                part = _vision_mrope_positions(
                    current_position, grid, spatial_merge_size
                )
                if part.shape[1] != length:
                    raise ValueError(
                        f"image placeholder length {length} does not match merged "
                        f"grid token count {part.shape[1]}."
                    )
                parts.append(part)
                current_position += max(
                    int(grid[0]),
                    int(grid[1]) // spatial_merge_size,
                    int(grid[2]) // spatial_merge_size,
                )
        if parts:
            position_ids[:, batch_index, valid] = torch.cat(parts, dim=1)
    if grid_index != int(image_grid_thw.shape[0]):
        raise ValueError(
            f"consumed {grid_index} image grids but received {image_grid_thw.shape[0]}."
        )
    return position_ids


class LingBotV2Model(nn.Module):
    """Flat LingBot-VLA 2.0 parameter container, following PI0's split."""

    VISION_PREFIX = "model.qwenvl_with_expert.qwenvl.model.visual"
    TEXT_PREFIX = "model.qwenvl_with_expert.qwenvl.model.language_model"

    def __init__(
        self,
        config: LingBotVLA2Config,
        *,
        params_dtype: torch.dtype | None = None,
        vision_params_dtype: torch.dtype | None = None,
        vision_patch_embed_backend: str = "conv3d",
        attn_backend: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        params_dtype = resolve_params_dtype(params_dtype)
        if device is None:
            device = get_engine_config().device.target
        vision_dtype = vision_params_dtype or params_dtype
        self.config = config
        self.params_dtype = params_dtype
        self.vision_params_dtype = vision_dtype
        self.vision = Qwen3VLVisionModel(
            config.vision,
            params_dtype=vision_dtype,
            attn_backend=attn_backend,
            patch_embed_backend=vision_patch_embed_backend,
            device=device,
            prefix=self.VISION_PREFIX,
        )
        self.text = Qwen3VLTextModel(
            config.text,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            device=device,
            prefix=self.TEXT_PREFIX,
            attn_causal=config.vlm_causal,
        )
        self.expert_stack = LingBotV2ExpertStack(
            config.expert,
            config.moe,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            device=device,
        )
        self.dual_query = LingBotV2DualQuery(
            config.dual_query,
            params_dtype=params_dtype,
            device=device,
        )
        self.heads = LingBotV2ActionTimeHeads(
            config.expert,
            config.flow_matching,
            params_dtype=params_dtype,
            device=device,
        )

    @property
    def mrope(self) -> InterleavedMRotaryEmbedding:
        """Expose the text tower's interleaved multimodal rotary embedding."""

        return self.text.rotary_emb

    def embed_language(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs with the language model's vocabulary table."""

        return self.text.embed_tokens(input_ids)

    def embed_special(self, token_id: int) -> torch.Tensor:
        """Embed one special token ID for prefix construction."""

        token = torch.tensor(
            [token_id], dtype=torch.int64, device=self.text.embed_tokens.weight.device
        )
        return self.embed_language(token)[0]


_DROP_WEIGHT_FRAGMENTS = (
    ".lm_head.",
    ".qwen_expert.model.embed_tokens.",
    "depth_align_head",
    "future_depth_align_head",
    "video_align_head",
    "future_video_cls_align_emb",
    "future_video_cls_head",
    "expert_visual",
)


def lingbot_v2_weight_remap(name: str) -> str | None:
    """Drop training-only tensors while keeping public V2 checkpoint keys intact."""

    while name.startswith("module."):
        name = name[len("module.") :]
    if any(fragment in name for fragment in _DROP_WEIGHT_FRAGMENTS):
        return None
    return name


__all__ = [
    "LingBotV2ActionTimeHeads",
    "LingBotV2AdaRMSNorm",
    "LingBotV2DualQuery",
    "LingBotV2ExpertLayer",
    "LingBotV2ExpertStack",
    "LingBotV2Model",
    "LingBotV2TokenMoE",
    "build_lingbot_v2_mrope_position_ids",
    "create_sinusoidal_pos_embedding",
    "lingbot_v2_weight_remap",
]
