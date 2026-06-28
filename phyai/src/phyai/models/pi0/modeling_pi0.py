"""pi0 inference model: vision + language + numeric-state action expert.

Module layout for the pi0 inference path. Configs live in
:mod:`configuration_pi0`; future runners and schedulers should own
packing, cache layout, Euler integration, and the pi0-specific
three-block attention mask. This file owns the ``nn.Module`` classes
only; every parameter declares its own ``hf_keys`` so
:func:`phyai.weights.load_pretrained` can fill the model without a
separate remap when checkpoint keys match the OpenPI / LeRobot layout.

Sections (top -> bottom):

1. **Engine defaults** -- :func:`_resolve_engine_defaults` and
   :func:`_engine_to_paged_backend`.
2. **Vision tower** -- SigLIP-So400m + ``multi_modal_projector``.
3. **PaliGemma language model** -- gemma_2b text side. Decoder layers
   use :class:`ARAttention` and run pre-norm GQA attention + gated MLP.
4. **Action expert** -- gemma_300m with plain :class:`GemmaRMSNorm`.
   The expert writes Q/K/V in the same joint attention space as the
   text tower, but keeps its own 1024-wide hidden stream via an
   asymmetric ``o_proj``.
5. **State / action / time heads** -- ``state_proj`` creates the
   numeric robot-state token; ``action_in_proj`` embeds noisy actions;
   sinusoidal timestep embeddings are fused into action tokens by
   ``action_time_mlp_in`` / ``action_time_mlp_out``.
6. **Top-level** -- :class:`PI0Model` is a flat container holding
   ``vision``, ``paligemma_lm``, ``expert_stack``, ``rope`` (shared
   instance), and ``heads``. No top-level forward lives here.

Default ``prefix`` strings follow the
``paligemma_with_expert.*`` namespace used by pi0 checkpoints. The
state/action/time projection heads live at the safetensors root.
``embed_tokens.weight`` reuses the tied LM-head tensor at
``paligemma_with_expert.paligemma.lm_head.weight``.

State-dict layout under
``{root}=PI0VisionTower.DEFAULT_PREFIX``::

    {root}.vision_tower.vision_model.embeddings.patch_embedding.{weight,bias}
    {root}.vision_tower.vision_model.embeddings.position_embedding.weight
    {root}.vision_tower.vision_model.encoder.layers.{i}.layer_norm{1,2}.{weight,bias}
    {root}.vision_tower.vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}
    {root}.vision_tower.vision_model.encoder.layers.{i}.mlp.{fc1,fc2}.{weight,bias}
    {root}.vision_tower.vision_model.post_layernorm.{weight,bias}
    {root}.multi_modal_projector.linear.{weight,bias}
    {root}.language_model.layers.{i}.input_layernorm.weight
    {root}.language_model.layers.{i}.post_attention_layernorm.weight
    {root}.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    {root}.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
    {root}.language_model.norm.weight
    paligemma_with_expert.paligemma.lm_head.weight             <- embed_tokens
    paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight
    paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight
    paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.{gate,up,down}_proj.weight
    paligemma_with_expert.gemma_expert.model.norm.weight
    state_proj.{weight,bias}
    action_in_proj.{weight,bias}
    action_out_proj.{weight,bias}
    action_time_mlp_in.{weight,bias}
    action_time_mlp_out.{weight,bias}

Inference-only -- every parameter is allocated with
``requires_grad=False``. Training belongs in a separate implementation.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.engine_config import get_engine_config
from phyai.layers.attention.ar import ARAttention, ARAttnCtx
from phyai.layers.attention.diffusion import DiffusionAttention, DiffusionAttnCtx
from phyai.layers.conv import Conv2d
from phyai.layers.layer_norm import GemmaRMSNorm, LayerNorm
from phyai.layers.linear.layers import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.layers.mlp.dense_mlp import DenseMLP
from phyai.layers.rotary_embedding import RotaryEmbedding
from phyai.layers.transformer_block import TransformerBlock
from phyai.layers.vocab_embedding.layers import VocabParallelEmbedding
from phyai.models.pi0.configuration_pi0 import (
    GemmaExpertConfig,
    PaliGemmaTextConfig,
    PI0Config,
    SiglipVisionConfig,
)
from phyai.weights.shards import replicated


def _resolve_engine_defaults(
    params_dtype: torch.dtype | None,
    attn_backend: str | None,
    norm_backend: str | None,
) -> tuple[torch.dtype, str, str]:
    """Fill in ``None`` overrides from the process EngineConfig."""

    if (
        params_dtype is not None
        and attn_backend is not None
        and norm_backend is not None
    ):
        return params_dtype, attn_backend, norm_backend
    ec = get_engine_config()
    return (
        ec.device.params_dtype if params_dtype is None else params_dtype,
        ec.backends.attn if attn_backend is None else attn_backend,
        ec.backends.norm if norm_backend is None else norm_backend,
    )


def _engine_to_paged_backend(attn_backend: str) -> str:
    """Map EngineConfig attention backend names to paged backend names."""

    canonical = attn_backend.lower().replace("_", "-")
    if canonical == "sdpa":
        return "eager"
    return canonical


def _vision_norm_backend(norm_backend: str, vision_dtype: torch.dtype) -> str:
    """Pick a norm backend that accepts the vision tower's compute dtype."""

    if norm_backend == "flashinfer" and vision_dtype != torch.bfloat16:
        return "phyai-kernel"
    return norm_backend


SIGLIP_NORM_HF_NAMES: dict[str, str] = {
    "input_layernorm": "layer_norm1",
    "post_attention_layernorm": "layer_norm2",
}


class PositionEmbedding(nn.Module):
    """Replicated, learned ``(N, D)`` position embedding."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if device is None:
            device = get_engine_config().device.target
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.prefix = prefix
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()

    def forward(self) -> torch.Tensor:
        return self.weight

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        )


class SiglipVisionEmbeddings(nn.Module):
    """Patch embed + learned position embed."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if params_dtype is None:
            params_dtype = get_engine_config().device.params_dtype
        self.config = config
        self.prefix = prefix
        self.patch_embedding = Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
            bias=True,
            dtype=params_dtype,
            prefix=f"{prefix}.patch_embedding" if prefix else "",
        )
        self.position_embedding = PositionEmbedding(
            num_embeddings=config.num_patches,
            embedding_dim=config.hidden_size,
            dtype=params_dtype,
            prefix=f"{prefix}.position_embedding" if prefix else "",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 4:
            raise ValueError(
                f"pixel_values must be 4-D (B, C, H, W); got shape "
                f"{tuple(pixel_values.shape)}."
            )
        _, c, h, w = pixel_values.shape
        if (
            c != self.config.num_channels
            or h != self.config.image_size
            or w != self.config.image_size
        ):
            raise ValueError(
                f"pixel_values shape {tuple(pixel_values.shape)} does not match "
                f"config: expected (B, {self.config.num_channels}, "
                f"{self.config.image_size}, {self.config.image_size})."
            )
        h_patch = self.patch_embedding(pixel_values)
        embeds = h_patch.flatten(2).transpose(1, 2)
        embeds = embeds + self.position_embedding()
        return embeds


class SiglipVisionEncoder(nn.Module):
    """Stack of SigLIP encoder layers."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = _resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        layer_prefix = f"{prefix}.layers" if prefix else ""
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    attn_causal=False,
                    attn_bias=True,
                    attn_out_bias=True,
                    rope=None,
                    mlp_gated=False,
                    mlp_activation="gelu_pytorch_tanh",
                    mlp_bias=True,
                    norm_type="layernorm",
                    norm_eps=config.layer_norm_eps,
                    norm_bias=True,
                    norm_hf_names=SIGLIP_NORM_HF_NAMES,
                    attn_out_hf_name="out_proj",
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    params_dtype=params_dtype,
                    prefix=f"{layer_prefix}.{i}" if layer_prefix else "",
                )
                for i in range(config.num_hidden_layers)
            ]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h)
        return h


class SiglipVisionModel(nn.Module):
    """SigLIP vision tower: embeddings -> encoder -> post_layernorm."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        prefix: str = "vision_model",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = _resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.config = config
        self.prefix = prefix
        self.embeddings = SiglipVisionEmbeddings(
            config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.embeddings" if prefix else "",
        )
        self.encoder = SiglipVisionEncoder(
            config,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            prefix=f"{prefix}.encoder" if prefix else "",
        )
        self.post_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            prefix=f"{prefix}.post_layernorm" if prefix else "",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self.embeddings(pixel_values)
        h = self.encoder(h)
        h = self.post_layernorm(h)
        return h


class MultiModalProjector(nn.Module):
    """PaliGemma multi_modal_projector: a single biased linear."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        prefix: str = "multi_modal_projector",
    ) -> None:
        super().__init__()
        if params_dtype is None:
            params_dtype = get_engine_config().device.params_dtype
        self.config = config
        self.prefix = prefix
        self.linear = ReplicatedLinear(
            in_features=config.hidden_size,
            out_features=config.projection_dim,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.linear" if prefix else "",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.linear(x)
        return out


class VisionTowerWrapper(nn.Module):
    """Wrap SigLIP so checkpoint keys include the ``vision_tower`` parent."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        *,
        params_dtype: torch.dtype | None,
        attn_backend: str,
        norm_backend: str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.vision_model = SiglipVisionModel(
            config,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            prefix=f"{prefix}.vision_model" if prefix else "vision_model",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vision_model(pixel_values)


class PI0VisionTower(nn.Module):
    """SigLIP + PaliGemma multi-modal projector for pi0."""

    DEFAULT_PREFIX: str = "paligemma_with_expert.paligemma.model"

    def __init__(
        self,
        config: SiglipVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        io_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = _resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.compute_dtype = params_dtype
        self.io_dtype = io_dtype if io_dtype is not None else params_dtype
        vision_norm_backend = _vision_norm_backend(norm_backend, params_dtype)
        self.config = config
        self.prefix = prefix
        self.vision_tower = VisionTowerWrapper(
            config,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=vision_norm_backend,
            prefix=f"{prefix}.vision_tower" if prefix else "vision_tower",
        )
        self.multi_modal_projector = MultiModalProjector(
            config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.multi_modal_projector"
            if prefix
            else "multi_modal_projector",
        )
        self.projection_scale = float(config.projection_dim) ** 0.5

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.to(self.compute_dtype)
        h = self.vision_tower(x)
        h = self.multi_modal_projector(h)
        # return h * self.projection_scale
        return h.to(self.io_dtype)


class PaliGemmaEmbedTokens(nn.Module):
    """Gemma vocab embedding using the tied PaliGemma LM-head key."""

    def __init__(
        self,
        config: PaliGemmaTextConfig,
        *,
        params_dtype: torch.dtype | None = None,
        prefix: str = "paligemma_with_expert.paligemma.lm_head",
    ) -> None:
        super().__init__()
        if params_dtype is None:
            params_dtype = get_engine_config().device.params_dtype
        self.config = config
        self.prefix = prefix
        self.embedding = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=params_dtype,
            # embed_scale=torch.tensor(config.hidden_size, dtype=torch.float32).sqrt().item(),
            embed_scale=float(config.hidden_size) ** 0.5,
            prefix=prefix,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class PaliGemmaDecoderLayer(nn.Module):
    """One PaliGemma decoder layer: RMSNorm, GQA attention, gated MLP."""

    def __init__(
        self,
        config: PaliGemmaTextConfig,
        layer_idx: int,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = _resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        attn_prefix = f"{prefix}.self_attn" if prefix else "self_attn"
        mlp_prefix = f"{prefix}.mlp" if prefix else "mlp"

        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.input_layernorm" if prefix else "",
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            bias=False,
            params_dtype=params_dtype,
            prefix=f"{attn_prefix}.qkv_proj",
        )
        self.q_heads_local = self.qkv_proj.num_heads
        self.kv_heads_local = max(
            1, self.qkv_proj.num_kv_heads * self.qkv_proj.num_kv_replicas
        )
        self.o_proj = RowParallelLinear(
            in_features=config.num_attention_heads * config.head_dim,
            out_features=config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            prefix=f"{attn_prefix}.o_proj",
        )
        self.attn = ARAttention(
            num_heads=self.q_heads_local,
            head_dim=config.head_dim,
            layer_id=layer_idx,
            num_kv_heads=self.kv_heads_local,
            causal=False,
            backend=_engine_to_paged_backend(attn_backend),
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.post_attention_layernorm" if prefix else "",
        )
        self.mlp = DenseMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="gelu_pytorch_tanh",
            gated=True,
            bias=False,
            params_dtype=params_dtype,
            prefix=mlp_prefix,
        )

    def _split_qkv(
        self,
        fused: torch.Tensor,
        leading: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_dim = self.q_heads_local * self.head_dim
        kv_dim = self.kv_heads_local * self.head_dim
        q, k, v = fused.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(*leading, self.q_heads_local, self.head_dim)
        k = k.reshape(*leading, self.kv_heads_local, self.head_dim)
        v = v.reshape(*leading, self.kv_heads_local, self.head_dim)
        return q, k, v

    def forward(
        self,
        h: torch.Tensor,
        position_ids: torch.Tensor,
        rope: RotaryEmbedding,
        attn_ctx: ARAttnCtx,
    ) -> torch.Tensor:
        residual = h
        n = self.input_layernorm(h)
        fused, _ = self.qkv_proj(n)
        q, k, v = self._split_qkv(fused, h.shape[:-1])
        q, k = rope(position_ids, q, k)
        attn_out = self.attn(q, k, v, attn_ctx)
        attn_flat = attn_out.reshape(*attn_out.shape[:-2], -1)
        out, _ = self.o_proj(attn_flat)
        h = residual + out
        residual = h
        m = self.post_attention_layernorm(h)
        m = self.mlp(m)
        return residual + m


class PaliGemmaLanguageModel(nn.Module):
    """PaliGemma language model: embeddings, decoder layers, final RMSNorm."""

    DEFAULT_PREFIX: str = "paligemma_with_expert.paligemma.model.language_model"

    def __init__(
        self,
        config: PaliGemmaTextConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = _resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.config = config
        self.prefix = prefix
        self.embed_tokens = PaliGemmaEmbedTokens(
            config,
            params_dtype=params_dtype,
            prefix="paligemma_with_expert.paligemma.lm_head",
        )
        layers_prefix = f"{prefix}.layers" if prefix else "layers"
        self.layers = nn.ModuleList(
            [
                PaliGemmaDecoderLayer(
                    config,
                    layer_idx=i,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    prefix=f"{layers_prefix}.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.norm" if prefix else "",
        )

    def embed_lang(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        rope: RotaryEmbedding,
        attn_ctx: ARAttnCtx,
    ) -> torch.Tensor:
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids, rope, attn_ctx)
        return self.norm(h)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    *,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    """Sin/cos timestep embedding for the flow-matching scheduler."""

    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.dim() != 1:
        raise ValueError(f"time must be 1-D (B,), got shape {tuple(time.shape)}.")
    device = time.device
    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=torch.float32, device=device
    )
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2.0 * math.pi
    sin_input = scaling[None, :] * time[:, None].to(torch.float32)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).to(time.dtype)


class PI0ExpertLayer(nn.Module):
    """One pi0 gemma_300m action-expert decoder layer.

    This mirrors the pi0.5 expert's attention geometry but uses plain
    :class:`GemmaRMSNorm`. ``q_proj`` writes into the shared joint
    attention space ``num_heads * head_dim`` (2048 by default), while
    ``o_proj`` reduces the attention output back to the expert width
    (1024 by default).
    """

    def __init__(
        self,
        config: GemmaExpertConfig,
        layer_idx: int,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if config.use_adarms:
            raise ValueError(
                "PI0ExpertLayer requires GemmaExpertConfig.use_adarms=False; "
                "AdaRMS conditioning belongs to pi0.5."
            )
        params_dtype, attn_backend, norm_backend = _resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        attn_prefix = f"{prefix}.self_attn" if prefix else "self_attn"
        mlp_prefix = f"{prefix}.mlp" if prefix else "mlp"

        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.input_layernorm" if prefix else "",
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            bias=False,
            params_dtype=params_dtype,
            prefix=f"{attn_prefix}.qkv_proj",
        )
        self.q_heads_local = self.qkv_proj.num_heads
        self.kv_heads_local = max(
            1, self.qkv_proj.num_kv_heads * self.qkv_proj.num_kv_replicas
        )
        self.o_proj = RowParallelLinear(
            in_features=config.num_attention_heads * config.head_dim,
            out_features=config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            prefix=f"{attn_prefix}.o_proj",
        )
        self.attn = DiffusionAttention(
            num_heads=self.q_heads_local,
            head_dim=config.head_dim,
            layer_id=layer_idx,
            num_kv_heads=self.kv_heads_local,
            causal=False,
            backend=_engine_to_paged_backend(attn_backend),
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.post_attention_layernorm" if prefix else "",
        )
        self.mlp = DenseMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="gelu_pytorch_tanh",
            gated=True,
            bias=False,
            params_dtype=params_dtype,
            prefix=mlp_prefix,
        )

    def _split_qkv(
        self,
        fused: torch.Tensor,
        leading: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_dim = self.q_heads_local * self.head_dim
        kv_dim = self.kv_heads_local * self.head_dim
        q, k, v = fused.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.reshape(*leading, self.q_heads_local, self.head_dim)
        k = k.reshape(*leading, self.kv_heads_local, self.head_dim)
        v = v.reshape(*leading, self.kv_heads_local, self.head_dim)
        return q, k, v

    def forward(
        self,
        h: torch.Tensor,
        position_ids: torch.Tensor,
        rope: RotaryEmbedding,
        attn_ctx: DiffusionAttnCtx,
    ) -> torch.Tensor:
        """Pre-norm self-attention + gated MLP, with KV cache scatter."""

        residual = h
        n = self.input_layernorm(h)
        fused, _ = self.qkv_proj(n)
        q, k, v = self._split_qkv(fused, h.shape[:-1])
        q, k = rope(position_ids, q, k)
        attn_out = self.attn(q, k, v, attn_ctx)
        attn_flat = attn_out.reshape(*attn_out.shape[:-2], -1)
        out, _ = self.o_proj(attn_flat)
        h = residual + out

        residual = h
        m = self.post_attention_layernorm(h)
        m = self.mlp(m)
        return residual + m


class PI0ExpertStack(nn.Module):
    """pi0 gemma_300m action expert: decoder layers + final RMSNorm."""

    DEFAULT_PREFIX: str = "paligemma_with_expert.gemma_expert.model"

    def __init__(
        self,
        config: GemmaExpertConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        super().__init__()
        if config.use_adarms:
            raise ValueError("PI0ExpertStack requires use_adarms=False.")
        params_dtype, attn_backend, norm_backend = _resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.config = config
        self.prefix = prefix
        layers_prefix = f"{prefix}.layers" if prefix else "layers"
        self.layers = nn.ModuleList(
            [
                PI0ExpertLayer(
                    config,
                    layer_idx=i,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    prefix=f"{layers_prefix}.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.norm" if prefix else "",
        )

    def forward(
        self,
        h: torch.Tensor,
        position_ids: torch.Tensor,
        rope: RotaryEmbedding,
        attn_ctx: DiffusionAttnCtx,
    ) -> torch.Tensor:
        """Run every expert layer + final RMSNorm."""

        for layer in self.layers:
            h = layer(h, position_ids, rope, attn_ctx)
        return self.norm(h)


class ActionTimeHeads(nn.Module):
    """pi0 state/action/timestep projection heads.

    Root-level checkpoint keys:

    * ``state_proj.{weight,bias}``
    * ``action_in_proj.{weight,bias}``
    * ``action_out_proj.{weight,bias}``
    * ``action_time_mlp_in.{weight,bias}``
    * ``action_time_mlp_out.{weight,bias}``

    ``embed_state`` produces the single numeric state token. Timestep is
    not a token; ``embed_action_time`` fuses it into every noisy-action
    token before the expert stack.
    """

    def __init__(
        self,
        config: PI0Config,
        *,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if params_dtype is None:
            params_dtype = get_engine_config().device.params_dtype
        self.config = config
        self.expert_hidden = config.expert.hidden_size
        self.max_state_dim = config.max_state_dim
        self.max_action_dim = config.max_action_dim
        self.min_period = config.min_period
        self.max_period = config.max_period
        self.state_proj = ReplicatedLinear(
            in_features=config.max_state_dim,
            out_features=config.expert.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix="state_proj",
        )
        self.action_in_proj = ReplicatedLinear(
            in_features=config.max_action_dim,
            out_features=config.expert.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix="action_in_proj",
        )
        self.action_out_proj = ReplicatedLinear(
            in_features=config.expert.hidden_size,
            out_features=config.max_action_dim,
            bias=True,
            params_dtype=params_dtype,
            prefix="action_out_proj",
        )
        self.action_time_mlp_in = ReplicatedLinear(
            in_features=2 * config.expert.hidden_size,
            out_features=config.expert.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix="action_time_mlp_in",
        )
        self.action_time_mlp_out = ReplicatedLinear(
            in_features=config.expert.hidden_size,
            out_features=config.expert.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix="action_time_mlp_out",
        )

    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        """``(B, max_state_dim) -> (B, expert_hidden)``."""

        state = state.to(self.state_proj.weight.dtype)
        out, _ = self.state_proj(state)
        return out

    def embed_action(self, x: torch.Tensor) -> torch.Tensor:
        """Raw action projection: ``(B, T, max_action_dim) -> (B, T, expert_hidden)``."""

        x = x.to(self.action_in_proj.weight.dtype)
        out, _ = self.action_in_proj(x)
        return out

    def embed_action_time(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse noisy actions and scalar flow time into expert action tokens."""

        action_emb = self.embed_action(x_t)
        time_emb = create_sinusoidal_pos_embedding(
            time,
            dimension=self.expert_hidden,
            min_period=self.min_period,
            max_period=self.max_period,
        )
        time_emb = time_emb.to(action_emb.dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)
        action_time_emb = action_time_emb.to(self.action_time_mlp_in.weight.dtype)
        h, _ = self.action_time_mlp_in(action_time_emb)
        h = F.silu(h)
        h, _ = self.action_time_mlp_out(h)
        return h

    def project_action(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, expert_hidden) -> (B, T, max_action_dim)``."""

        x = x.to(self.action_out_proj.weight.dtype)
        out, _ = self.action_out_proj(x)
        return out


class PI0Model(nn.Module):
    """Full pi0 inference model as a flat parameter container.

    Holds:

    * ``vision``: SigLIP + multi-modal projector.
    * ``paligemma_lm``: PaliGemma/Gemma text stack.
    * ``expert_stack``: plain-RMSNorm gemma_300m action expert.
    * ``rope``: shared RoPE instance for text and expert layers.
    * ``heads``: numeric state, action, and action-time projections.

    The pi0 attention pattern is a three-block mask:

    * image + language prefix attends within prefix only;
    * state token attends to prefix + state;
    * action tokens attend to prefix + state + the full action chunk.

    This module does not build that mask directly. The future runner and
    scheduler should realize it by staging the right AR/Diffusion
    attention contexts and cache reads.
    """

    def __init__(
        self,
        config: PI0Config,
        *,
        params_dtype: torch.dtype | None = None,
        vision_params_dtype: torch.dtype | None = torch.float32,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        rope_backend: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = _resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        vision_dtype = (
            vision_params_dtype if vision_params_dtype is not None else params_dtype
        )
        if rope_backend is None:
            rope_backend = "flashinfer" if attn_backend == "flashinfer" else "eager"
        self.config = config
        self.params_dtype = params_dtype
        self.vision_params_dtype = vision_dtype
        self.attn_backend = attn_backend

        vision_attn_backend = attn_backend
        if attn_backend == "flashinfer" and config.vision.head_dim not in (
            64,
            128,
            256,
        ):
            vision_attn_backend = "sdpa"
            warnings.warn(
                f"PI0Model: vision tower head_dim={config.vision.head_dim} "
                f"not in flashinfer's supported set {{64, 128, 256}}; "
                f"vision attention silently downgraded to 'sdpa'. The "
                f"language + expert joint attention path still uses "
                f"'flashinfer' as requested.",
                stacklevel=2,
            )

        self.vision = PI0VisionTower(
            config.vision,
            params_dtype=vision_dtype,
            io_dtype=params_dtype,
            attn_backend=vision_attn_backend,
            norm_backend=norm_backend,
        )
        self.paligemma_lm = PaliGemmaLanguageModel(
            config.text,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        self.expert_stack = PI0ExpertStack(
            config.expert,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        self.rope = RotaryEmbedding(
            head_dim=config.text.head_dim,
            max_position_embeddings=config.text.max_position_embeddings,
            rope_theta=config.text.rope_theta,
            backend=rope_backend,
            device=device,
        )
        self.heads = ActionTimeHeads(
            config,
            params_dtype=params_dtype,
        )


__all__ = [
    "ActionTimeHeads",
    "MultiModalProjector",
    "PaliGemmaDecoderLayer",
    "PaliGemmaEmbedTokens",
    "PaliGemmaLanguageModel",
    "PI0ExpertLayer",
    "PI0ExpertStack",
    "PI0Model",
    "PI0VisionTower",
    "PositionEmbedding",
    "SIGLIP_NORM_HF_NAMES",
    "SiglipVisionEmbeddings",
    "SiglipVisionEncoder",
    "SiglipVisionModel",
    "VisionTowerWrapper",
    "create_sinusoidal_pos_embedding",
]
