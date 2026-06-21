"""pi0.5

https://www.pi.website/blog/pi05
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.engine_config import get_engine_config, resolve_engine_defaults
from phyai.layers.attention.ar import ARAttention, ARAttnCtx
from phyai.layers.attention.diffusion import DiffusionAttention, DiffusionAttnCtx
from phyai.layers.conv import Conv2d
from phyai.layers.layer_norm import AdaRMSNorm, GemmaRMSNorm, LayerNorm
from phyai.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.layers.mlp.dense_mlp import DenseMLP
from phyai.layers.rotary_embedding import RotaryEmbedding
from phyai.layers.transformer_block import TransformerBlock
from phyai.layers.vocab_embedding import VocabParallelEmbedding
from phyai.models.pi05.configuration_pi05 import (
    GemmaExpertConfig,
    PaliGemmaTextConfig,
    PI05Config,
    SiglipVisionConfig,
)
from phyai.weights.shards import replicated


def _adarms_backend(norm_backend: str) -> str:
    """Map a generic norm backend to one :class:`AdaRMSNorm` accepts.

    flashinfer has no AdaRMS kernel; transparently fall back to
    ``phyai-kernel`` (Triton, CUDA) so callers can leave their
    :class:`EngineConfig` at the production default and still have
    pi0.5 construction succeed.
    """
    if norm_backend == "flashinfer":
        return "phyai-kernel"
    return norm_backend


def _vision_norm_backend(norm_backend: str, vision_dtype: torch.dtype) -> str:
    """Map a generic norm backend to one the vision tower can run at ``vision_dtype``.

    flashinfer's LayerNorm / RMSNorm CUDA kernels hard-require a **bf16**
    input tensor (``flashinfer.norm.layernorm``: "input ... Need to be
    bfloat16"). When the vision tower runs in fp32 (the openpi / lerobot
    parity path keeps SigLIP + projector + their norms in fp32), the
    flashinfer norm path cannot consume the fp32 activations, so we fall
    back to ``phyai-kernel`` (Triton, which accepts any floating dtype) —
    mirroring :func:`_adarms_backend`. When the tower stays at the bf16
    default, the backend is left untouched.
    """
    if norm_backend == "flashinfer" and vision_dtype != torch.bfloat16:
        return "phyai-kernel"
    return norm_backend


def _engine_to_paged_backend(attn_backend: str) -> str:
    """Map :class:`EngineConfig`'s ``attn_backend`` onto the AR / Diffusion
    paged backend name.

    The AR and Diffusion paged stacks are **flashinfer-only** (GPU):
    ``"flashinfer"`` is the only backend registered in either
    subpackage. ``"sdpa"`` / ``"eager"`` have no paged backend (SDPA
    cannot read paged KV; there is no CPU reference path), so any
    non-flashinfer name is rejected here rather than silently coerced.
    """
    canonical = attn_backend.lower().replace("_", "-")
    if canonical != "flashinfer":
        raise ValueError(
            f"AR / Diffusion paged stacks are flashinfer-only (GPU); got "
            f"attn_backend={attn_backend!r}. pi0.5 inference requires "
            f"backend='flashinfer'."
        )
    return canonical


# SigLIP encoder layers name their pre-norms ``layer_norm1`` /
# ``layer_norm2`` instead of HF's ``input_layernorm`` /
# ``post_attention_layernorm``. The override map is keyed by HF
# default names and points to the SigLIP-side names.
SIGLIP_NORM_HF_NAMES: dict[str, str] = {
    "input_layernorm": "layer_norm1",
    "post_attention_layernorm": "layer_norm2",
}


class PositionEmbedding(nn.Module):
    """Replicated, learned ``(N, D)`` position embedding.

    SigLIP's positional embedding is HF's ``nn.Embedding`` whose weight
    tensor is exactly ``(num_positions, embed_dim)``. The vision tower
    always reads positions ``arange(N)`` so we keep just the parameter
    and broadcast at forward time — no ``F.embedding`` call needed.
    """

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
    """Patch embed + learned position embed -> ``(B, num_patches, hidden_size)``.

    HF state dict layout::

        {prefix}.patch_embedding.{weight,bias}
        {prefix}.position_embedding.weight
    """

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
        h_patch = self.patch_embedding(pixel_values)  # (B, hidden, H/p, W/p)
        embeds = h_patch.flatten(2).transpose(1, 2)  # (B, num_patches, hidden)
        embeds = embeds + self.position_embedding()  # broadcast (N, D) over batch
        return embeds


class SiglipVisionEncoder(nn.Module):
    """Stack of ``num_hidden_layers`` SigLIP encoder layers.

    HF layout: ``{prefix}.layers.{i}.<encoder layer subkeys>``.

    SigLIP's encoder layer is pre-norm with LayerNorm (bias=True),
    bidirectional (causal=False), q/k/v/out_proj all biased, and a
    plain ``fc1 -> GELU(tanh) -> fc2`` MLP (bias=True). HF source names
    differ from the Llama / Gemma defaults: ``layer_norm1`` /
    ``layer_norm2`` for the norms and ``out_proj`` for the attention
    output projection.
    """

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
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
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
    """SigLIP-So400m vision tower: embeddings -> encoder -> post_layernorm.

    Output shape ``(B, num_patches, hidden_size)``.

    HF state dict layout (default ``prefix="vision_model"`` matches the
    HF ``SiglipVisionModel`` root)::

        {prefix}.embeddings.patch_embedding.{weight,bias}
        {prefix}.embeddings.position_embedding.weight
        {prefix}.encoder.layers.{i}.<layer subkeys>
        {prefix}.post_layernorm.{weight,bias}
    """

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
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
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
    """PaliGemma multi_modal_projector: a single ``Linear(vision -> text)``.

    HF state dict::

        {prefix}.linear.{weight,bias}

    PaliGemma's projector is a single biased Linear with no
    activation. The ``projector_hidden_act`` knob in HF's
    PaliGemmaConfig is consumed by SigLIP's classification head, not
    by this projector — so it doesn't appear here.
    """

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
    """Wraps :class:`SiglipVisionModel` so the HF prefix gains a
    ``vision_tower`` parent — pi0.5 checkpoints store the encoder under
    ``vision_tower.vision_model``, not just ``vision_model``.
    """

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


class PI05VisionTower(nn.Module):
    """SigLIP-So400m + PaliGemma multi_modal_projector, glued for pi0.5."""

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
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        # ``params_dtype`` is the vision *compute* dtype (may be fp32 for the
        # parity path); ``io_dtype`` is the surrounding model's dtype that the
        # output is cast back to (bf16). When the two match (bf16 default) the
        # boundary casts in forward are no-ops.
        self.compute_dtype = params_dtype
        self.io_dtype = io_dtype if io_dtype is not None else params_dtype
        # fp32 activations can't flow through flashinfer's bf16-only norm
        # kernels; route the vision norms to phyai-kernel when running fp32.
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

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Upcast the incoming pixels to the vision compute dtype (bf16 -> fp32
        # on the parity path; a no-op when the tower is bf16), run the tower +
        # projector in that dtype, then cast back to the model's io_dtype. The
        # casts are captured into the vision CUDA graph; its external interface
        # stays io_dtype.
        x = pixel_values.to(self.compute_dtype)
        h = self.vision_tower(x)  # (B, N, hidden)
        h = self.multi_modal_projector(h)  # (B, N, projection_dim)
        return h.to(self.io_dtype)


class PaliGemmaEmbedTokens(nn.Module):
    """Gemma vocab embedding with the sqrt(hidden) input scaling."""

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
            embed_scale=float(config.hidden_size) ** 0.5,
            prefix=prefix,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class PaliGemmaDecoderLayer(nn.Module):
    """One PaliGemma decoder layer — pre-norm GQA self-attention + gated MLP."""

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
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
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


class PaliGemmaLanguageModel(nn.Module):
    """PaliGemma language model: embed_tokens + 18 decoder layers + final norm."""

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
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
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
        """Look up ``input_ids`` and apply Gemma's sqrt(hidden) scaling.

        Returns ``(B, S, hidden_size)``. The vision embeddings produced
        by :class:`PI05VisionTower` already include the same
        ``sqrt(projection_dim)`` scaling, so downstream code can concat
        the two without further bookkeeping.
        """
        return self.embed_tokens(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        rope: RotaryEmbedding,
        attn_ctx: ARAttnCtx,
    ) -> torch.Tensor:
        """Run every decoder layer + final norm over ``inputs_embeds``."""
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids, rope, attn_ctx)
        return self.norm(h)


@dataclass(frozen=True)
class ExpertLayerModulation:
    """Precomputed AdaRMS modulation for one expert layer at one step."""

    input_ln: torch.Tensor
    post_attention_ln: torch.Tensor


@dataclass(frozen=True)
class ExpertStepModulation:
    """Precomputed AdaRMS modulation for the whole expert stack at one step."""

    layers: tuple[ExpertLayerModulation, ...]
    final: torch.Tensor


@dataclass(frozen=True)
class ExpertModulationTables:
    """Precomputed AdaRMS modulation for the whole stack across all steps."""

    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    final: torch.Tensor

    def step(self, i: int) -> ExpertStepModulation:
        """Select step ``i``'s modulation for every norm in the stack.

        Uses ``t[i : i + 1]`` row slices: each is a contiguous
        ``(1, 3 * hidden_size)`` view into the table at a *constant* offset,
        so this is safe to call inside a captured graph (the slice address is
        baked in and the underlying storage stays the table's).
        """
        return ExpertStepModulation(
            layers=tuple(
                ExpertLayerModulation(inp[i : i + 1], post[i : i + 1])
                for inp, post in self.layers
            ),
            final=self.final[i : i + 1],
        )


class PI05ExpertLayer(nn.Module):
    """One gemma_300m action-expert decoder layer with AdaRMS norms."""

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
        if not config.use_adarms:
            raise ValueError(
                "PI05ExpertLayer requires GemmaExpertConfig.use_adarms=True; "
                "non-AdaRMS expert is not part of pi0.5."
            )
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        adarms_backend = _adarms_backend(norm_backend)
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        attn_prefix = f"{prefix}.self_attn" if prefix else "self_attn"
        mlp_prefix = f"{prefix}.mlp" if prefix else "mlp"

        self.input_layernorm = AdaRMSNorm(
            hidden_size=config.hidden_size,
            cond_dim=config.adarms_cond_dim,
            eps=config.rms_norm_eps,
            backend=adarms_backend,
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
        # The expert's o_proj is asymmetric: the joint attention writes
        # ``num_heads * head_dim = 2048`` per token (same space as the
        # paligemma stream), and o_proj reduces back down to the
        # expert's ``hidden_size = 1024``.
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
        self.post_attention_layernorm = AdaRMSNorm(
            hidden_size=config.hidden_size,
            cond_dim=config.adarms_cond_dim,
            eps=config.rms_norm_eps,
            backend=adarms_backend,
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
        cond: torch.Tensor | None,
        rope: RotaryEmbedding,
        attn_ctx: DiffusionAttnCtx,
        *,
        modulation: ExpertLayerModulation | None = None,
    ) -> torch.Tensor:
        """Pre-AdaRMS self-attention + gated MLP, with KV cache scatter."""
        residual = h
        if modulation is None:
            n, gate_attn = self.input_layernorm(h, cond)
        else:
            n, gate_attn = self.input_layernorm(h, modulation=modulation.input_ln)
        fused, _ = self.qkv_proj(n)
        q, k, v = self._split_qkv(fused, h.shape[:-1])
        q, k = rope(position_ids, q, k)
        attn_out = self.attn(q, k, v, attn_ctx)
        attn_flat = attn_out.reshape(*attn_out.shape[:-2], -1)
        out, _ = self.o_proj(attn_flat)
        # ``residual + out * gate`` as one fused-multiply-add kernel (saves a
        # separate mul + add per gated residual; FMA rounds once instead of
        # twice, so it matches the old two-op form to bf16 ulp).
        h = torch.addcmul(residual, out, gate_attn)
        residual = h
        if modulation is None:
            m, gate_mlp = self.post_attention_layernorm(h, cond)
        else:
            m, gate_mlp = self.post_attention_layernorm(
                h, modulation=modulation.post_attention_ln
            )
        m = self.mlp(m)
        return torch.addcmul(residual, m, gate_mlp)


class PI05ExpertStack(nn.Module):
    """gemma_300m action-expert stack: 18 :class:`PI05ExpertLayer` + final AdaRMSNorm."""

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
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        adarms_backend = _adarms_backend(norm_backend)
        self.config = config
        self.prefix = prefix
        layers_prefix = f"{prefix}.layers" if prefix else "layers"
        self.layers = nn.ModuleList(
            [
                PI05ExpertLayer(
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
        self.norm = AdaRMSNorm(
            hidden_size=config.hidden_size,
            cond_dim=config.adarms_cond_dim,
            eps=config.rms_norm_eps,
            backend=adarms_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.norm" if prefix else "",
        )

    def build_modulation_tables(self, conds: torch.Tensor) -> ExpertModulationTables:
        """Project ``conds`` through every norm's ``dense`` once."""
        layers = tuple(
            (
                layer.input_layernorm.project_modulation(conds),
                layer.post_attention_layernorm.project_modulation(conds),
            )
            for layer in self.layers
        )
        return ExpertModulationTables(
            layers=layers, final=self.norm.project_modulation(conds)
        )

    def final_norm(
        self,
        h: torch.Tensor,
        cond: torch.Tensor | None,
        *,
        modulation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the trailing AdaRMSNorm and discard the unused gate.

        The trailing norm produces ``(out, gate)`` like every other
        AdaRMSNorm call, but no sublayer follows it so the gate has
        nothing to multiply — drop it here so callers don't have to.
        """
        if modulation is None:
            out, _ = self.norm(h, cond)
        else:
            out, _ = self.norm(h, modulation=modulation)
        return out

    def forward(
        self,
        h: torch.Tensor,
        position_ids: torch.Tensor,
        cond: torch.Tensor | None,
        rope: RotaryEmbedding,
        attn_ctx: DiffusionAttnCtx,
        *,
        modulation: ExpertStepModulation | None = None,
    ) -> torch.Tensor:
        """Run every expert layer + final AdaRMSNorm.

        ``cond`` is the per-token AdaRMS condition (already broadcast
        to ``(B * chunk_size, adarms_cond_dim)``); the same ``cond`` is
        threaded through every layer's gated norms and the trailing
        :meth:`final_norm`.

        When the condition is one step of a fixed, precomputed schedule,
        pass ``modulation`` (an :class:`ExpertStepModulation`) instead
        (``cond`` may be ``None``); each layer gets its precomputed row pair
        and the final norm its row, so no ``dense`` projection runs here.
        """
        for j, layer in enumerate(self.layers):
            layer_mod = None if modulation is None else modulation.layers[j]
            h = layer(h, position_ids, cond, rope, attn_ctx, modulation=layer_mod)
        final_mod = None if modulation is None else modulation.final
        return self.final_norm(h, cond, modulation=final_mod)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    *,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    """Sin cos timestep embedding for the flow-matching scheduler."""
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


class ActionTimeHeads(nn.Module):
    """Action and time embedding projections, all biased Linears."""

    def __init__(
        self,
        config: PI05Config,
        *,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if params_dtype is None:
            params_dtype = get_engine_config().device.params_dtype
        self.config = config
        self.expert_hidden = config.expert.hidden_size
        self.max_action_dim = config.max_action_dim
        self.min_period = config.min_period
        self.max_period = config.max_period
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
        self.time_mlp_in = ReplicatedLinear(
            in_features=config.expert.hidden_size,
            out_features=config.expert.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix="time_mlp_in",
        )
        self.time_mlp_out = ReplicatedLinear(
            in_features=config.expert.hidden_size,
            out_features=config.expert.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix="time_mlp_out",
        )

    def embed_action(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, max_action_dim) -> (B, T, expert_hidden)``."""
        out, _ = self.action_in_proj(x)
        return out

    def project_action(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, expert_hidden) -> (B, T, max_action_dim)``."""
        out, _ = self.action_out_proj(x)
        return out

    def embed_time(self, time: torch.Tensor) -> torch.Tensor:
        """``(B,) scalar time -> (B, expert_hidden)`` AdaRMS condition.

        Pipeline: sinusoidal pos embed (with the configured
        ``min_period`` / ``max_period``) -> Linear -> SiLU -> Linear ->
        SiLU.

        Casts the sinusoidal embedding to the time_mlp parameter dtype
        before the first Linear so callers can pass an fp32 ``time``
        tensor regardless of whether the heads themselves are bf16.
        """
        emb = create_sinusoidal_pos_embedding(
            time,
            dimension=self.expert_hidden,
            min_period=self.min_period,
            max_period=self.max_period,
        )
        emb = emb.to(self.time_mlp_in.weight.dtype)
        h, _ = self.time_mlp_in(emb)
        h = F.silu(h)
        h, _ = self.time_mlp_out(h)
        return F.silu(h)


class PI05Model(nn.Module):
    """Full pi0.5 inference model — flat composition of forward-able sub-modules.

    Holds every parameter the pi0.5 inference path needs as a flat set
    of attributes::

        vision         : PI05VisionTower                # SigLIP + projector
        paligemma_lm   : PaliGemmaLanguageModel         # text 18 layers + norm
        expert_stack   : PI05ExpertStack                # expert 18 layers + AdaRMS final
        rope           : RotaryEmbedding                # shared between paligemma + expert
        heads          : ActionTimeHeads                # action_in/out + time MLP

    Each decoder layer (paligemma + expert) owns its own paged
    attention instance bound to its layer index — paligemma uses
    :class:`ARAttention` and the expert uses :class:`DiffusionAttention`.
    """

    def __init__(
        self,
        config: PI05Config,
        *,
        params_dtype: torch.dtype | None = None,
        vision_params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        rope_backend: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        # The vision tower may run at a different (typically higher) precision
        # than the rest of the model — openpi / lerobot keep SigLIP + projector
        # in fp32 while the language + expert stacks are bf16. ``None`` keeps it
        # at the model dtype (the byte-identical single-dtype default).
        vision_dtype = (
            vision_params_dtype if vision_params_dtype is not None else params_dtype
        )
        # RoPE backend defaults follow the attention backend's preference:
        # both flashinfer paths exist and are fast; the eager path is the
        # fallback for non-flashinfer attention backends.
        if rope_backend is None:
            rope_backend = "flashinfer" if attn_backend == "flashinfer" else "eager"
        self.config = config
        self.params_dtype = params_dtype
        self.vision_params_dtype = vision_dtype
        self.attn_backend = attn_backend

        # flashinfer's prefill kernel hard-asserts ``head_dim ∈ {64, 128, 256}``.
        # SigLIP-So400m has head_dim=72 (= 1152 / 16) and would JIT-fail.
        # Auto-fall back to ``sdpa`` for the vision tower when the requested
        # backend is flashinfer; the joint attention path continues with the
        # user-requested backend (head_dim = 256 in pi0.5 always satisfies the
        # assert).
        vision_attn_backend = attn_backend
        if attn_backend == "flashinfer" and config.vision.head_dim not in (
            64,
            128,
            256,
        ):
            vision_attn_backend = "sdpa"
            # TODO: the sdpa vision path can be wrapped in torch.compile
            # to recover some of the throughput lost vs flashinfer.
            warnings.warn(
                f"PI05Model: vision tower head_dim={config.vision.head_dim} "
                f"not in flashinfer's supported set {{64, 128, 256}}; "
                f"vision attention silently downgraded to 'sdpa'. The "
                f"language + expert joint attention path still uses "
                f"'flashinfer' as requested. (Expected for SigLIP-So400m, "
                f"head_dim = 1152 / 16 = 72.)",
                stacklevel=2,
            )
        self.vision = PI05VisionTower(
            config.vision,
            params_dtype=vision_dtype,
            io_dtype=params_dtype,
            attn_backend=vision_attn_backend,
            norm_backend=norm_backend,
        )

        # Text and expert stacks. Each layer owns its own paged
        # attention bound to layer_id=i: paligemma uses ARAttention,
        # expert uses DiffusionAttention. The runners build the
        # right ctx type per stack and thread it through the
        # stack's forward(h, position_ids, [cond,] rope, ctx).
        #
        # The mathematical attention pattern of pi0.5 is a 2D
        # block-prefix-LM mask (the prefix and suffix embedders build
        # it as a block ``make_att_2d_masks``-style structure):
        #
        #   - Image + language tokens form one block; bidirectional
        #     within the block, **cannot see** action tokens.
        #   - Action tokens form a second block; cross-attend to
        #     image + language AND bidirectional within themselves.
        #
        # The runner pair realises this same mask by splitting the
        # forward into two phases: PI05LLMRunner runs the image+lang
        # block in isolation (its attention range is intrinsically
        # image+lang only because nothing else exists yet), and
        # PI05ExpertRunner runs joint attention over
        # ``[cached prefix K/V, fresh suffix K/V]``. Since image+lang's
        # hidden state at any layer never depends on action's K/V
        # (per the block mask), splitting the forward is
        # mathematically equivalent to the one-pass mask.
        #
        # Each layer's attention is therefore ``causal=False`` — the
        # per-phase attention range encodes the block-LM constraint,
        # not a per-token causal mask.
        self.paligemma_lm = PaliGemmaLanguageModel(
            config.text,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        self.expert_stack = PI05ExpertStack(
            config.expert,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )

        # Shared RoPE — held at the model level (not per-layer) so the
        # 8 MiB cos/sin cache is allocated once, not 18x. Layers take
        # rope as a forward argument; nothing about RoPE is
        # layer-specific in pi0.5 (same head_dim / theta / max_pos for
        # paligemma and expert). RotaryEmbedding's ``device`` kwarg
        # threads down to the cos/sin cache; sub-models read
        # ``engine_config.device`` directly for their own params.
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
    # Configuration re-exports come from ``configuration_pi05``; this
    # module owns every nn.Module the pi0.5 inference path needs.
    "ActionTimeHeads",
    "ExpertLayerModulation",
    "ExpertModulationTables",
    "ExpertStepModulation",
    "MultiModalProjector",
    "PaliGemmaDecoderLayer",
    "PaliGemmaEmbedTokens",
    "PaliGemmaLanguageModel",
    "PI05ExpertLayer",
    "PI05ExpertStack",
    "PI05Model",
    "PI05VisionTower",
    "PositionEmbedding",
    "SIGLIP_NORM_HF_NAMES",
    "SiglipVisionEmbeddings",
    "SiglipVisionEncoder",
    "SiglipVisionModel",
    "VisionTowerWrapper",
    "create_sinusoidal_pos_embedding",
]
