"""DreamZero DiT modeling helpers.

This module owns the stateless DreamZero DiT architecture. Runtime cache
lifetime belongs to ``model_runner_dreamzero.py``; forward methods here only
consume optional cache tensors and return updated tensors.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import phyai.parallel as P
from phyai.engine_config import get_engine_config
from phyai.layers import LayerNorm, SiLU
from phyai.layers.conv import Conv3d
from phyai.layers.mlp import DenseMLP
from phyai.layers.attention.nocache.layer import Attention
from phyai.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroConfig
from phyai.weights.shards import replicated, sharded


_DREAMZERO_DIT_PREFIX = "action_head.model."


def _validate_kv_cache(
    kv_cache: torch.Tensor,
    batch_size: int,
    num_heads: int,
    head_dim: int,
) -> None:
    if kv_cache.dim() != 5 or kv_cache.shape[0] != 2:
        raise ValueError(
            "Expected KV cache with shape (2, B, S, H, D), got "
            f"{tuple(kv_cache.shape)}."
        )
    if kv_cache.shape[1] != batch_size:
        raise ValueError(
            f"KV cache batch={kv_cache.shape[1]} does not match batch={batch_size}."
        )
    if kv_cache.shape[3] != num_heads or kv_cache.shape[4] != head_dim:
        raise ValueError(
            "KV cache head shape does not match local attention shape: "
            f"cache=({kv_cache.shape[3]}, {kv_cache.shape[4]}), "
            f"expected=({num_heads}, {head_dim})."
        )


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class DreamZeroTensorParallelRMSNorm(nn.Module):
    """RMSNorm over the full hidden dimension with TP-sharded activations.

    DreamZero/Wan normalizes Q/K before reshaping to attention heads, so the RMS
    statistic is over the full hidden dim. In TP, each rank owns only a hidden
    shard; the local squared sums are all-reduced before applying the local
    weight shard.
    """

    def __init__(
        self,
        hidden_size: int,
        local_size: int,
        *,
        eps: float,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size={hidden_size} must be positive.")
        if local_size <= 0:
            raise ValueError(f"local_size={local_size} must be positive.")
        self.hidden_size = hidden_size
        self.local_size = local_size
        self.eps = eps
        self.mesh = P.default_mesh()
        self.tp_size = self.mesh.group_size("dense_tp")
        dtype = params_dtype or torch.get_default_dtype()
        self.weight = nn.Parameter(
            torch.ones(local_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = sharded(dim=0, group="dense_tp", mesh=self.mesh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.local_size:
            raise ValueError(
                f"Expected local shard dim={self.local_size}, got {x.shape[-1]}."
            )
        local_sum = x.float().pow(2).sum(dim=-1, keepdim=True)
        if self.tp_size > 1:
            local_sum = P.all_reduce(local_sum, group="dense_tp", mesh=self.mesh)
        inv_rms = torch.rsqrt(local_sum / self.hidden_size + self.eps)
        return (x.float() * inv_rms).to(dtype=x.dtype) * self.weight


def dreamzero_sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim <= 0 or dim % 2:
        raise ValueError(f"dim={dim} must be a positive even int.")
    half = dim // 2
    position = position.to(dtype=torch.float64)
    freqs = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, dtype=position.dtype, device=position.device).div(half),
        ),
    )
    return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=1)


def dreamzero_rope_params(
    max_seq_len: int,
    dim: int,
    *,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if dim <= 0 or dim % 2:
        raise ValueError(f"dim={dim} must be a positive even int.")
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float64, device=device),
        1.0
        / torch.pow(
            theta,
            torch.arange(0, dim, 2, dtype=torch.float64, device=device).div(dim),
        ),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _dreamzero_causal_rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    *,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, _ = x.shape
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(batch_size, seq_len, num_heads, -1, 2)
    )
    if action_register_length is not None:
        if action_register_length != num_action_per_block + num_state_per_block:
            raise ValueError(
                "KV-cache causal RoPE expects exactly one action/state block; got "
                f"action_register_length={action_register_length}."
            )
        action_start = action_state_index * num_action_per_block
        state_start = action_state_index * num_state_per_block
        action_freqs = freqs_action[action_start : action_start + num_action_per_block]
        state_freqs = freqs_state[state_start : state_start + num_state_per_block]
        action_state_freqs = torch.cat([action_freqs, state_freqs], dim=0).view(
            action_register_length, 1, -1
        )
        freqs = torch.cat([freqs, action_state_freqs], dim=0)
    rotated = torch.view_as_real(x_complex * freqs.unsqueeze(0)).flatten(3)
    return rotated.to(dtype=x.dtype)


class DreamZeroSinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal timestep encoding used by the DreamZero action encoder."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim={embedding_dim} must be positive.")
        if embedding_dim % 2:
            raise ValueError(f"embedding_dim={embedding_dim} must be even.")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() != 2:
            raise ValueError(
                f"Expected timesteps with shape (B, T), got {tuple(timesteps.shape)}."
            )
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim,
            dtype=torch.float32,
            device=timesteps.device,
        ) * (math.log(10000.0) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class DreamZeroCategorySpecificLinear(nn.Module):
    """Category-specific linear projection with checkpoint-native weight layout.

    The reference DreamZero action/state encoder stores weights as
    ``[num_categories, in_dim, out_dim]`` and applies them with ``torch.bmm``.
    """

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        output_dim: int,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dtype = params_dtype or torch.get_default_dtype()
        self.num_categories = num_categories
        self.W = nn.Parameter(
            torch.empty(
                num_categories, input_dim, output_dim, dtype=dtype, device=device
            ),
            requires_grad=False,
        )
        self.b = nn.Parameter(
            torch.empty(num_categories, output_dim, dtype=dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            self.W.hf_keys = [(f"{prefix}.W", None)]
            self.W.weight_loader = replicated()
            self.b.hf_keys = [(f"{prefix}.b", None)]
            self.b.weight_loader = replicated()

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B, T, C), got {tuple(x.shape)}.")
        if cat_ids.dim() != 1 or cat_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "Expected cat_ids with shape (B,), got "
                f"{tuple(cat_ids.shape)} for batch={x.shape[0]}."
            )
        if self.num_categories == 1:
            selected_W = self.W.expand(x.shape[0], -1, -1)
            selected_b = self.b.expand(x.shape[0], -1)
        else:
            selected_W = self.W.index_select(0, cat_ids)
            selected_b = self.b.index_select(0, cat_ids)
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class DreamZeroCategorySpecificMLP(nn.Module):
    """Two-layer category-specific MLP used by state/action decoder paths."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer1 = DreamZeroCategorySpecificLinear(
            num_categories,
            input_dim,
            hidden_dim,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer1" if prefix else "",
        )
        self.layer2 = DreamZeroCategorySpecificLinear(
            num_categories,
            hidden_dim,
            output_dim,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.layer2" if prefix else "",
        )

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class DreamZeroActionEncoder(nn.Module):
    """Encode actions and timesteps for the selected embodiment."""

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        num_embodiments: int,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.W1 = DreamZeroCategorySpecificLinear(
            num_embodiments,
            action_dim,
            hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.W1" if prefix else "",
        )
        self.W2 = DreamZeroCategorySpecificLinear(
            num_embodiments,
            2 * hidden_size,
            hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.W2" if prefix else "",
        )
        self.W3 = DreamZeroCategorySpecificLinear(
            num_embodiments,
            hidden_size,
            hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.W3" if prefix else "",
        )
        self.pos_encoding = DreamZeroSinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        if actions.dim() != 3:
            raise ValueError(
                f"Expected actions with shape (B, T, C), got {tuple(actions.shape)}."
            )
        batch_size, horizon, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == batch_size:
            timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        elif timesteps.dim() == 2 and timesteps.shape == actions.shape[:2]:
            pass
        else:
            raise ValueError(
                "Expected timesteps with shape (B,) or (B, T), got "
                f"{tuple(timesteps.shape)} for actions={tuple(actions.shape)}."
            )

        action_embedding = self.W1(actions, cat_ids)
        timestep_embedding = self.pos_encoding(timesteps).to(
            dtype=action_embedding.dtype
        )
        x = torch.cat([action_embedding, timestep_embedding], dim=-1)
        x = _swish(self.W2(x, cat_ids))
        return self.W3(x, cat_ids)


class DreamZeroSelfAttention(nn.Module):
    """TP-sharded DreamZero causal self-attention.

    The module is stateless: callers may pass a KV cache and receive an updated
    cache, but the cache is never stored on ``self``. Persistent cache ownership
    belongs in ``model_runner_dreamzero.py``.
    """

    def __init__(
        self,
        config: DreamZeroConfig,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        attn_backend: str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dim = config.dit.dim
        heads = config.dit.num_heads
        head_dim = config.dit.head_dim
        self.q = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.q" if prefix else "",
        )
        self.k = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.k" if prefix else "",
        )
        self.v = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.v" if prefix else "",
        )
        self.o = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.o" if prefix else "",
        )
        local_dim = self.q.output_size_per_partition
        self.norm_q = DreamZeroTensorParallelRMSNorm(
            dim,
            local_dim,
            eps=config.dit.eps,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm_q" if prefix else "",
        )
        self.norm_k = DreamZeroTensorParallelRMSNorm(
            dim,
            local_dim,
            eps=config.dit.eps,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm_k" if prefix else "",
        )
        self.num_heads = heads
        self.num_local_heads = self.q.output_size_per_partition // head_dim
        self.head_dim = head_dim
        self.frame_seqlen = config.dit.frame_seqlen
        self.num_frame_per_block = config.dit.num_frame_per_block
        self.num_action_per_block = config.dit.num_action_per_block
        self.num_state_per_block = config.dit.num_state_per_block
        self.local_attn_size = (
            config.dit.max_chunk_size * config.dit.num_frame_per_block + 1
            if config.dit.max_chunk_size != -1
            else -1
        )
        self.max_attention_size = (
            21 * config.dit.frame_seqlen
            if self.local_attn_size == -1
            else self.local_attn_size * config.dit.frame_seqlen
        )
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_local_heads,
            causal=False,
            backend=attn_backend,
            kernel_role="dreamzero.self_attention",
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None = None,
        *,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q, _ = self.q(x)
        k, _ = self.k(x)
        v, _ = self.v(x)
        q = self.norm_q(q).reshape(
            batch_size, seq_len, self.num_local_heads, self.head_dim
        )
        k = self.norm_k(k).reshape(
            batch_size, seq_len, self.num_local_heads, self.head_dim
        )
        v = v.reshape(batch_size, seq_len, self.num_local_heads, self.head_dim)
        if kv_cache is not None:
            _validate_kv_cache(
                kv_cache, batch_size, self.num_local_heads, self.head_dim
            )
        action_state_index = max(
            0, (current_start_frame - 1) // self.num_frame_per_block
        )
        roped_query = _dreamzero_causal_rope_action_apply(
            q,
            freqs,
            freqs_action,
            freqs_state,
            action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
            action_state_index=action_state_index,
        )
        roped_key = _dreamzero_causal_rope_action_apply(
            k,
            freqs,
            freqs_action,
            freqs_state,
            action_register_length,
            num_action_per_block=self.num_action_per_block,
            num_state_per_block=self.num_state_per_block,
            action_state_index=action_state_index,
        )
        roped_action_query = None
        roped_action_key = None
        action_v = None
        if action_register_length is not None:
            roped_action_query = roped_query[:, -action_register_length:]
            roped_query = roped_query[:, :-action_register_length]
            roped_action_key = roped_key[:, -action_register_length:]
            roped_key = roped_key[:, :-action_register_length]
            action_v = v[:, -action_register_length:]
            v = v[:, :-action_register_length]

        if kv_cache is None:
            new_k, new_v = roped_key, v
        else:
            new_k = torch.cat([kv_cache[0], roped_key], dim=1)
            new_v = torch.cat([kv_cache[1], v], dim=1)
        new_k = new_k[:, -self.max_attention_size :]
        new_v = new_v[:, -self.max_attention_size :]
        if action_register_length is not None:
            if (
                roped_action_query is None
                or roped_action_key is None
                or action_v is None
            ):
                raise RuntimeError("Missing action/state tensors after split.")
            attn_query = torch.cat([roped_query, roped_action_query], dim=1)
            attn_key = torch.cat([new_k, roped_action_key], dim=1)
            attn_value = torch.cat([new_v, action_v], dim=1)
        else:
            attn_query = roped_query
            attn_key = new_k
            attn_value = new_v
        x_heads = self.attn(attn_query, attn_key, attn_value)
        updated_kv_cache = torch.stack([new_k, new_v], dim=0)
        merged = x_heads.reshape(batch_size, x_heads.shape[1], -1)
        out, _ = self.o(merged)
        return out, updated_kv_cache


class DreamZeroCrossAttention(nn.Module):
    """TP-sharded DreamZero image-to-video cross-attention.

    Context follows the Wan I2V convention: the first ``image_context_tokens``
    are CLIP/image tokens and the rest are text tokens. Text K/V may be supplied
    by the runner as a stateless cache input and returned as a new cache.
    """

    def __init__(
        self,
        config: DreamZeroConfig,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        attn_backend: str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dim = config.dit.dim
        head_dim = config.dit.head_dim
        for name in ("q", "k", "v", "k_img", "v_img"):
            setattr(
                self,
                name,
                ColumnParallelLinear(
                    dim,
                    dim,
                    bias=True,
                    gather_output=False,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.{name}" if prefix else "",
                ),
            )
        self.o = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.o" if prefix else "",
        )
        local_dim = self.q.output_size_per_partition
        self.norm_q = DreamZeroTensorParallelRMSNorm(
            dim,
            local_dim,
            eps=config.dit.eps,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm_q" if prefix else "",
        )
        self.norm_k = DreamZeroTensorParallelRMSNorm(
            dim,
            local_dim,
            eps=config.dit.eps,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm_k" if prefix else "",
        )
        self.norm_k_img = DreamZeroTensorParallelRMSNorm(
            dim,
            local_dim,
            eps=config.dit.eps,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm_k_img" if prefix else "",
        )
        self.num_heads = config.dit.num_heads
        self.num_local_heads = self.q.output_size_per_partition // head_dim
        self.head_dim = head_dim
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_local_heads,
            causal=False,
            backend=attn_backend,
            kernel_role="dreamzero.cross_attention",
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        image_context_tokens: int = 257,
        crossattn_cache: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if context.shape[1] <= image_context_tokens:
            raise ValueError(
                "DreamZeroCrossAttention requires image and text context tokens; "
                f"got context length={context.shape[1]} and "
                f"image_context_tokens={image_context_tokens}."
            )

        batch_size, seq_len, _ = x.shape
        context_img = context[:, :image_context_tokens]
        context_text = context[:, image_context_tokens:]

        q, _ = self.q(x)
        q = self.norm_q(q)
        q = q.reshape(batch_size, seq_len, self.num_local_heads, self.head_dim)

        if crossattn_cache is None:
            k, _ = self.k(context_text)
            v, _ = self.v(context_text)
            text_seq_len = context_text.shape[1]
            k = self.norm_k(k)
            k = k.reshape(batch_size, text_seq_len, self.num_local_heads, self.head_dim)
            v = v.reshape(batch_size, text_seq_len, self.num_local_heads, self.head_dim)
        else:
            _validate_kv_cache(
                crossattn_cache, batch_size, self.num_local_heads, self.head_dim
            )
            k = crossattn_cache[0]
            v = crossattn_cache[1]

        k_img, _ = self.k_img(context_img)
        v_img, _ = self.v_img(context_img)
        k_img = self.norm_k_img(k_img)
        k_img = k_img.reshape(
            batch_size, image_context_tokens, self.num_local_heads, self.head_dim
        )
        v_img = v_img.reshape(
            batch_size, image_context_tokens, self.num_local_heads, self.head_dim
        )

        text_out = self.attn(q, k, v)
        image_out = self.attn(q, k_img, v_img)
        summed_heads = text_out + image_out
        merged = summed_heads.reshape(batch_size, seq_len, -1)
        out, _ = self.o(merged)

        updated_crossattn_cache = None
        if use_cache or crossattn_cache is not None:
            updated_crossattn_cache = torch.stack([k, v], dim=0)
        return out, updated_crossattn_cache


class DreamZeroDiTBlock(nn.Module):
    """One DreamZero DiT block with TP-sharded attention/FFN parameters."""

    def __init__(
        self,
        config: DreamZeroConfig,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dim = config.dit.dim
        self.norm1 = LayerNorm(
            dim,
            eps=config.dit.eps,
            bias=False,
            elementwise_affine=False,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
        )
        self.self_attn = DreamZeroSelfAttention(
            config,
            params_dtype=params_dtype,
            device=device,
            attn_backend=attn_backend,
            prefix=f"{prefix}.self_attn" if prefix else "",
        )
        self.norm3 = LayerNorm(
            dim,
            eps=config.dit.eps,
            bias=True,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm3" if prefix else "",
        )
        self.cross_attn = DreamZeroCrossAttention(
            config,
            params_dtype=params_dtype,
            device=device,
            attn_backend=attn_backend,
            prefix=f"{prefix}.cross_attn" if prefix else "",
        )
        self.norm2 = LayerNorm(
            dim,
            eps=config.dit.eps,
            bias=False,
            elementwise_affine=False,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
        )
        self.ffn = DenseMLP(
            dim,
            config.dit.ffn_dim,
            activation="gelu_tanh",
            gated=False,
            bias=True,
            sequence_parallel=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.ffn" if prefix else "",
        )
        dtype = params_dtype or torch.get_default_dtype()
        self.modulation = nn.Parameter(
            torch.empty(1, 6, dim, dtype=dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            self.modulation.hf_keys = [(f"{prefix}.modulation", None)]
            self.modulation.weight_loader = replicated()

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        *,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        image_context_tokens: int = 257,
        use_crossattn_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        e0, e1, e2, e3, e4, e5 = (self.modulation.unsqueeze(1) + e).unbind(dim=2)

        self_attn_input = self.norm1(x) * (1 + e1) + e0
        y, updated_kv_cache = self.self_attn(
            self_attn_input,
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )
        x = x + y * e2

        cross_attn_input = self.norm3(x)
        y, updated_crossattn_cache = self.cross_attn(
            cross_attn_input,
            context,
            image_context_tokens=image_context_tokens,
            crossattn_cache=crossattn_cache,
            use_cache=use_crossattn_cache,
        )
        x = x + y
        ffn_input = self.norm2(x) * (1 + e4) + e3
        y = self.ffn(ffn_input)
        x = x + y * e5
        return x, updated_kv_cache, updated_crossattn_cache


class DreamZeroCausalHead(nn.Module):
    """Project video tokens to latent predictions."""

    def __init__(
        self,
        config: DreamZeroConfig,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        norm_backend: str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dim = config.dit.dim
        patch_size = (1, 2, 2)
        out_features = math.prod(patch_size) * config.dit.out_dim
        self.norm = LayerNorm(
            dim,
            eps=config.dit.eps,
            bias=False,
            elementwise_affine=False,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
        )
        self.head = ReplicatedLinear(
            dim,
            out_features,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.head" if prefix else "",
        )
        dtype = params_dtype or torch.get_default_dtype()
        self.modulation = nn.Parameter(
            torch.empty(1, 2, dim, dtype=dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            self.modulation.hf_keys = [(f"{prefix}.modulation", None)]
            self.modulation.weight_loader = replicated()

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.unsqueeze(1) + e).unbind(dim=2)
        return self.head(self.norm(x) * (1 + scale) + shift)[0]


class DreamZeroDiT(nn.Module):
    """DreamZero Causal Wan DiT with TP-sharded DiT blocks."""

    patch_size = (1, 2, 2)

    def __init__(
        self,
        config: DreamZeroConfig,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
    ) -> None:
        super().__init__()
        device = get_engine_config().device.target if device is None else device
        params_dtype = params_dtype or get_engine_config().device.params_dtype
        config.dit.validate_tp_size(P.default_mesh().group_size("dense_tp"))
        self.config = config
        self.patch_embedding = Conv3d(
            config.dit.in_dim,
            config.dit.dim,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            device=device,
            dtype=params_dtype,
            prefix="patch_embedding",
        )

        self.text_embedding = nn.ModuleList(
            [
                ReplicatedLinear(
                    4096,
                    config.dit.dim,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix="text_embedding.0",
                ),
                ReplicatedLinear(
                    config.dit.dim,
                    config.dit.dim,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix="text_embedding.2",
                ),
            ]
        )
        self.time_activation = SiLU()
        self.time_embedding = nn.ModuleList(
            [
                ReplicatedLinear(
                    config.dit.freq_dim,
                    config.dit.dim,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix="time_embedding.0",
                ),
                ReplicatedLinear(
                    config.dit.dim,
                    config.dit.dim,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix="time_embedding.2",
                ),
            ]
        )
        self.time_projection = ReplicatedLinear(
            config.dit.dim,
            config.dit.dim * 6,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix="time_projection.1",
        )
        self.img_emb = nn.ModuleDict(
            {
                "proj_0_norm": LayerNorm(
                    1280,
                    # MLPProj uses torch.nn.LayerNorm's default epsilon.
                    eps=1e-5,
                    bias=True,
                    backend=norm_backend,
                    dtype=params_dtype,
                    device=device,
                    prefix="img_emb.proj.0",
                ),
                "proj_1": ReplicatedLinear(
                    1280,
                    1280,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix="img_emb.proj.1",
                ),
                "proj_3": ReplicatedLinear(
                    1280,
                    config.dit.dim,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix="img_emb.proj.3",
                ),
                "proj_4_norm": LayerNorm(
                    config.dit.dim,
                    eps=1e-5,
                    bias=True,
                    backend=norm_backend,
                    dtype=params_dtype,
                    device=device,
                    prefix="img_emb.proj.4",
                ),
            }
        )
        self.state_encoder = DreamZeroCategorySpecificMLP(
            1,
            config.max_state_dim,
            config.hidden_size,
            config.dit.dim,
            params_dtype=params_dtype,
            device=device,
            prefix="state_encoder",
        )
        self.action_encoder = DreamZeroActionEncoder(
            config.action_dim,
            config.dit.dim,
            1,
            params_dtype=params_dtype,
            device=device,
            prefix="action_encoder",
        )
        self.action_decoder = DreamZeroCategorySpecificMLP(
            1,
            config.dit.dim,
            config.hidden_size,
            config.action_dim,
            params_dtype=params_dtype,
            device=device,
            prefix="action_decoder",
        )
        self.blocks = nn.ModuleList(
            [
                DreamZeroDiTBlock(
                    config,
                    params_dtype=params_dtype,
                    device=device,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    prefix=f"blocks.{idx}",
                )
                for idx in range(config.dit.num_layers)
            ]
        )
        self.head = DreamZeroCausalHead(
            config,
            params_dtype=params_dtype,
            device=device,
            norm_backend=norm_backend,
            prefix="head",
        )
        head_dim = config.dit.head_dim
        self.freqs_action = dreamzero_rope_params(1024 * 10, head_dim, device=device)
        self.freqs_state = dreamzero_rope_params(1024, head_dim, device=device)
        self.freqs = [
            dreamzero_rope_params(1024, head_dim - 4 * (head_dim // 6), device=device),
            dreamzero_rope_params(1024, 2 * (head_dim // 6), device=device),
            dreamzero_rope_params(1024, 2 * (head_dim // 6), device=device),
        ]

    def _text_embedding(self, context: torch.Tensor) -> torch.Tensor:
        context = self.text_embedding[0](context)[0]
        context = F.gelu(context, approximate="tanh")
        return self.text_embedding[1](context)[0]

    def _time_embedding(
        self, timestep: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        emb = dreamzero_sinusoidal_embedding_1d(
            self.config.dit.freq_dim, timestep.flatten()
        ).to(dtype=dtype, device=timestep.device)
        emb = self.time_embedding[0](emb)[0]
        emb = self.time_activation(emb)
        return self.time_embedding[1](emb)[0]

    def _img_embedding(self, clip_feature: torch.Tensor) -> torch.Tensor:
        x = self.img_emb["proj_0_norm"](clip_feature)
        x = self.img_emb["proj_1"](x)[0]
        x = F.gelu(x)
        x = self.img_emb["proj_3"](x)[0]
        return self.img_emb["proj_4_norm"](x)

    def _create_freqs(
        self, grid_size: tuple[int, int, int], start_frame: int
    ) -> torch.Tensor:
        freqs = self.freqs
        f, h, w = grid_size
        return torch.cat(
            [
                freqs[0][start_frame : start_frame + f]
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)

    def unpatchify(
        self, x: torch.Tensor, grid_size: tuple[int, int, int]
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        out_dim = self.config.dit.out_dim
        grid = grid_size
        if x.shape[1] != math.prod(grid):
            raise ValueError(
                f"Expected {math.prod(grid)} patch tokens, got {x.shape[1]}."
            )
        x = x.view(batch_size, *grid, *self.patch_size, out_dim)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(
            batch_size,
            out_dim,
            *[i * j for i, j in zip(grid, self.patch_size, strict=True)],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        kv_cache: list[torch.Tensor | None] | None = None,
        crossattn_cache: list[torch.Tensor | None] | None = None,
        current_start_frame: int = 0,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
        image_context_tokens: int = 257,
        use_crossattn_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list[torch.Tensor | None],
        list[torch.Tensor | None],
    ]:
        if y is not None:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)
        x = self.patch_embedding(x)
        grid_size = tuple(x.shape[2:])
        x_tokens = x.flatten(start_dim=2).transpose(1, 2)
        freqs = self._create_freqs(grid_size, current_start_frame)

        batch_size = x_tokens.shape[0]
        seq_len = x_tokens.shape[1]
        timestep_frames = timestep.shape[1]

        if action is not None:
            if embodiment_id is None:
                embodiment_id = torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=x_tokens.device,
                )
            else:
                embodiment_id = embodiment_id.to(
                    device=x_tokens.device, dtype=torch.long
                )
                if embodiment_id.ndim != 1 or embodiment_id.shape[0] != batch_size:
                    raise ValueError(
                        "embodiment_id must have shape "
                        f"({batch_size},), got {tuple(embodiment_id.shape)}."
                    )
            if timestep_action is None or state is None:
                raise ValueError("action forward requires timestep_action and state.")
            action_features = self.action_encoder(
                action, timestep_action, embodiment_id
            )
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x_tokens = torch.cat([x_tokens, action_register], dim=1)
        else:
            embodiment_id = None
            state_features = None
            action_length = 0
            action_register_length = None

        if timestep_frames <= seq_len:
            repeat = (seq_len + timestep_frames - 1) // timestep_frames
            timestep_video = timestep.repeat_interleave(repeat, dim=1)[:, :seq_len]
        else:
            indices = torch.linspace(
                0,
                timestep_frames - 1,
                seq_len,
                device=timestep.device,
                dtype=torch.long,
            )
            timestep_video = timestep[:, indices]
        if action is not None:
            if state_features is None or timestep_action is None:
                raise RuntimeError("Missing state/action tensors.")
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep_video = torch.cat(
                [timestep_video, timestep_action, timestep_state], dim=1
            )

        e = self._time_embedding(timestep_video, x_tokens.dtype)
        e = e.unflatten(dim=0, sizes=(batch_size, -1))
        e0 = self.time_projection(self.time_activation(e))[0].unflatten(
            dim=2, sizes=(6, self.config.dit.dim)
        )

        context = self._text_embedding(context)
        if clip_feature is not None:
            clip_embedding = self._img_embedding(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)
        if crossattn_cache is None:
            crossattn_cache = [None] * len(self.blocks)
        if len(kv_cache) != len(self.blocks) or len(crossattn_cache) != len(
            self.blocks
        ):
            raise ValueError("Cache list length must match number of DiT blocks.")

        updated_kv_caches: list[torch.Tensor | None] = []
        updated_crossattn_caches: list[torch.Tensor | None] = []
        for block_index, block in enumerate(self.blocks):
            x_tokens, updated_kv_cache, updated_crossattn_cache = block(
                x_tokens,
                e0,
                freqs,
                self.freqs_action,
                self.freqs_state,
                action_register_length,
                context,
                kv_cache=kv_cache[block_index],
                crossattn_cache=crossattn_cache[block_index],
                current_start_frame=current_start_frame,
                image_context_tokens=image_context_tokens,
                use_crossattn_cache=use_crossattn_cache,
            )
            updated_kv_caches.append(updated_kv_cache)
            updated_crossattn_caches.append(updated_crossattn_cache)

        if action is not None:
            if embodiment_id is None:
                raise RuntimeError("Missing embodiment ids for action decode.")
            action_noise_pred = x_tokens[:, seq_len : seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        x_video = x_tokens[:, :seq_len]
        e_video = e[:, :seq_len]
        x_video = self.head(x_video, e_video.unsqueeze(2))
        return (
            self.unpatchify(x_video, grid_size),
            action_noise_pred,
            updated_kv_caches,
            updated_crossattn_caches,
        )


def dreamzero_dit_weight_remap(key: str) -> str | None:
    """Load the ``action_head.model`` subtree into the DreamZero DiT."""
    if key.startswith(_DREAMZERO_DIT_PREFIX):
        return (
            key.removeprefix(_DREAMZERO_DIT_PREFIX)
            .replace(".ffn.0.", ".ffn.fc1.")
            .replace(".ffn.2.", ".ffn.fc2.")
        )
    return None


__all__ = [
    "DreamZeroActionEncoder",
    "DreamZeroCategorySpecificLinear",
    "DreamZeroCategorySpecificMLP",
    "DreamZeroCausalHead",
    "DreamZeroCrossAttention",
    "DreamZeroDiT",
    "DreamZeroDiTBlock",
    "DreamZeroSelfAttention",
    "DreamZeroSinusoidalPositionalEncoding",
    "dreamzero_dit_weight_remap",
]
