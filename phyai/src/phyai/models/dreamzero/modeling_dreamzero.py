"""DreamZero DiT modeling helpers.

This module owns the stateless DreamZero DiT architecture. Runtime cache
lifetime belongs to ``model_runner_dreamzero.py``; forward methods here only
consume optional cache tensors and return updated tensors.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers import LayerNorm
from phyai.layers.attention.nocache.layer import Attention
from phyai.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroConfig
from phyai.weights.shards import replicated, sharded


_DREAMZERO_DIT_PREFIX = "action_head.model."
_DREAMZERO_DROPPED_PREFIXES = (
    "action_head.text_encoder.",
    "action_head.image_encoder.",
    "action_head.vae.",
)


def _get_flash_attn_varlen_func() -> Callable[..., torch.Tensor]:
    try:
        flash_attn = importlib.import_module("flash_attn")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "DreamZero TE cross-attention requires FlashAttention 2."
        ) from exc
    return flash_attn.flash_attn_varlen_func


def _dreamzero_fa2_cross_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Match the official DreamZero FlashAttention 2 cross-attention call."""
    batch_size, query_length = query.shape[:2]
    key_length = key.shape[1]
    query_flat = query.flatten(0, 1).to(value.dtype)
    key_flat = key.flatten(0, 1).to(value.dtype)
    value_flat = value.flatten(0, 1)
    query_lengths = torch.full(
        (batch_size,), query_length, dtype=torch.int32, device=query.device
    )
    key_lengths = torch.full(
        (batch_size,), key_length, dtype=torch.int32, device=key.device
    )
    cu_seqlens_q = torch.cat([query_lengths.new_zeros(1), query_lengths]).cumsum(
        0, dtype=torch.int32
    )
    cu_seqlens_k = torch.cat([key_lengths.new_zeros(1), key_lengths]).cumsum(
        0, dtype=torch.int32
    )
    output = _get_flash_attn_varlen_func()(
        q=query_flat,
        k=key_flat,
        v=value_flat,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=query_length,
        max_seqlen_k=key_length,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
    )
    return output.unflatten(0, (batch_size, query_length)).type(query.dtype)


def _attach_replicated(param: nn.Parameter, key: str) -> None:
    param.hf_keys = [(key, None)]
    param.weight_loader = replicated()


def _attach_tp_sharded(param: nn.Parameter, key: str, dim: int = 0) -> None:
    param.hf_keys = [(key, None)]
    mesh = _current_mesh_or_none()
    if mesh is None:
        param.weight_loader = replicated()
    else:
        param.weight_loader = sharded(dim=dim, group="dense_tp", mesh=mesh)


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


def _linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = module(x)
    if isinstance(out, tuple):
        return out[0]
    return out


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
        dtype = params_dtype or torch.get_default_dtype()
        self.weight = nn.Parameter(
            torch.ones(local_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            _attach_tp_sharded(self.weight, f"{prefix}.weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.local_size:
            raise ValueError(
                f"Expected local shard dim={self.local_size}, got {x.shape[-1]}."
            )
        local_sum = x.float().pow(2).sum(dim=-1, keepdim=True)
        if _current_tp_size() > 1:
            import phyai.parallel as P

            local_sum = P.all_reduce(local_sum, group="dense_tp")
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


def _dreamzero_rope_action_apply(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    *,
    num_action_per_block: int,
    num_state_per_block: int,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, _ = x.shape
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(batch_size, seq_len, num_heads, -1, 2)
    )
    freqs = freqs.to(device=x.device)
    if action_register_length is not None:
        chunk_size = action_register_length // (
            num_action_per_block + num_state_per_block
        )
        action_freqs = freqs_action[: chunk_size * num_action_per_block].to(
            device=x.device
        )
        state_freqs = freqs_state[: chunk_size * num_state_per_block].to(
            device=x.device
        )
        action_state_freqs = torch.cat([action_freqs, state_freqs], dim=0).view(
            action_register_length, 1, -1
        )
        freqs = torch.cat([freqs, action_state_freqs], dim=0)
    rotated = torch.view_as_real(x_complex * freqs.unsqueeze(0)).flatten(3)
    return rotated.to(dtype=x.dtype)


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
    freqs = freqs.to(device=x.device)
    if action_register_length is not None:
        if action_register_length != num_action_per_block + num_state_per_block:
            raise ValueError(
                "KV-cache causal RoPE expects exactly one action/state block; got "
                f"action_register_length={action_register_length}."
            )
        action_start = action_state_index * num_action_per_block
        state_start = action_state_index * num_state_per_block
        action_freqs = freqs_action[
            action_start : action_start + num_action_per_block
        ].to(device=x.device)
        state_freqs = freqs_state[state_start : state_start + num_state_per_block].to(
            device=x.device
        )
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
    """Category-specific linear weight container.

    The reference DreamZero action/state encoder stores weights as
    ``[num_categories, in_dim, out_dim]`` and applies them with ``torch.bmm``.
    This phase only declares parameters and loaders; the forward path is added
    with the runner/KV-cache implementation.
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
        self.input_dim = input_dim
        self.output_dim = output_dim
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
            _attach_replicated(self.W, f"{prefix}.W")
            _attach_replicated(self.b, f"{prefix}.b")

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x with shape (B, T, C), got {tuple(x.shape)}.")
        if cat_ids.dim() != 1 or cat_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "Expected cat_ids with shape (B,), got "
                f"{tuple(cat_ids.shape)} for batch={x.shape[0]}."
            )
        if self.num_categories == 1:
            cat_ids = torch.zeros_like(cat_ids)
        elif bool(((cat_ids < 0) | (cat_ids >= self.num_categories)).any().item()):
            raise ValueError(
                "cat_ids must be in range "
                f"[0, {self.num_categories}), got {cat_ids.detach().cpu().tolist()}."
            )
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
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
    """Multi-embodiment action encoder parameter skeleton."""

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
        norm_backend: str | None = None,
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
            backend="sdpa" if attn_backend == "official" else attn_backend,
            backend_kwargs={"select_kernel": True}
            if attn_backend == "official"
            else None,
        )
        self.causal_attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_local_heads,
            causal=True,
            backend="sdpa" if attn_backend == "official" else attn_backend,
            backend_kwargs={"select_kernel": True}
            if attn_backend == "official"
            else None,
        )

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q, _ = self.q(x)
        k, _ = self.k(x)
        v, _ = self.v(x)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = q.reshape(batch_size, seq_len, self.num_local_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_local_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_local_heads, self.head_dim)
        return q, k, v

    def _simple_forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        q, k, v = self._project_qkv(x)

        k_all = k
        v_all = v
        if kv_cache is not None:
            _validate_kv_cache(
                kv_cache, batch_size, self.num_local_heads, self.head_dim
            )
            k_all = torch.cat([kv_cache[0], k], dim=1)
            v_all = torch.cat([kv_cache[1], v], dim=1)

        out = self.attn(q, k_all, v_all)
        out = out.reshape(batch_size, seq_len, -1)
        out, _ = self.o(out)

        updated_kv_cache = None
        if use_cache or kv_cache is not None:
            updated_kv_cache = torch.stack([k_all, v_all], dim=0)
        return out, updated_kv_cache

    def _blockwise_causal_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        action_horizon: int | None,
        state_horizon: int | None,
    ) -> torch.Tensor:
        total_len = q.shape[1]
        has_action_state = action_horizon is not None and state_horizon is not None
        if not has_action_state:
            num_frames = total_len // self.frame_seqlen
            block_size = self.frame_seqlen * self.num_frame_per_block
            num_blocks = (num_frames - 1) // self.num_frame_per_block
            if num_blocks <= 0:
                return self.attn(q, k, v)
            if self.local_attn_size == -1:
                return self.causal_attn(q, k, v)

            output = torch.empty_like(q)
            output[:, : self.frame_seqlen] = self.attn(
                q[:, : self.frame_seqlen],
                k[:, : self.frame_seqlen],
                v[:, : self.frame_seqlen],
            )
            for block_idx in range(num_blocks):
                block_start = self.frame_seqlen + block_idx * block_size
                block_end = min(block_start + block_size, total_len)
                kv_start = max(0, block_end - self.local_attn_size * self.frame_seqlen)
                output[:, block_start:block_end] = self.attn(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end],
                )
            return output

        if action_horizon is None or state_horizon is None:
            raise ValueError(
                "action_horizon and state_horizon must be provided together."
            )
        first_image_len = self.frame_seqlen
        image_blocks_len = total_len - first_image_len - action_horizon - state_horizon
        image_block_size = self.num_frame_per_block * self.frame_seqlen
        num_image_blocks = image_blocks_len // image_block_size
        num_action_blocks = action_horizon // self.num_action_per_block
        num_state_blocks = state_horizon // self.num_state_per_block
        if (
            num_image_blocks != num_action_blocks
            or num_image_blocks != num_state_blocks
        ):
            raise ValueError(
                "DreamZero block layout mismatch: "
                f"image_blocks={num_image_blocks}, action_blocks={num_action_blocks}, "
                f"state_blocks={num_state_blocks}."
            )

        image_start = first_image_len
        image_end = image_start + image_blocks_len
        action_start = image_end
        state_start = action_start + action_horizon

        output = torch.empty_like(q)
        output[:, :first_image_len] = self.attn(
            q[:, :first_image_len], k[:, :first_image_len], v[:, :first_image_len]
        )

        for block_idx in range(num_image_blocks):
            block_start = image_start + block_idx * image_block_size
            block_end = image_start + (block_idx + 1) * image_block_size
            image_kv_start = (
                max(image_start, block_end - self.local_attn_size * self.frame_seqlen)
                if self.local_attn_size != -1
                else image_start
            )
            action_block_start = action_start + block_idx * self.num_action_per_block
            action_block_end = action_block_start + self.num_action_per_block
            state_block_start = state_start + block_idx * self.num_state_per_block
            state_block_end = state_block_start + self.num_state_per_block
            k_context = torch.cat(
                [
                    k[:, :first_image_len],
                    k[:, image_kv_start:block_end],
                    k[:, action_block_start:action_block_end],
                    k[:, state_block_start:state_block_end],
                ],
                dim=1,
            )
            v_context = torch.cat(
                [
                    v[:, :first_image_len],
                    v[:, image_kv_start:block_end],
                    v[:, action_block_start:action_block_end],
                    v[:, state_block_start:state_block_end],
                ],
                dim=1,
            )
            output[:, block_start:block_end] = self.attn(
                q[:, block_start:block_end], k_context, v_context
            )

        for block_idx in range(num_action_blocks):
            action_block_start = action_start + block_idx * self.num_action_per_block
            action_block_end = action_block_start + self.num_action_per_block
            image_block_end = image_start + (block_idx + 1) * image_block_size
            image_kv_start = (
                max(
                    image_start,
                    image_block_end - self.local_attn_size * self.frame_seqlen,
                )
                if self.local_attn_size != -1
                else image_start
            )
            state_block_start = state_start + block_idx * self.num_state_per_block
            state_block_end = state_block_start + self.num_state_per_block
            k_context = torch.cat(
                [
                    k[:, :first_image_len],
                    k[:, image_kv_start:image_block_end],
                    k[:, action_block_start:action_block_end],
                    k[:, state_block_start:state_block_end],
                ],
                dim=1,
            )
            v_context = torch.cat(
                [
                    v[:, :first_image_len],
                    v[:, image_kv_start:image_block_end],
                    v[:, action_block_start:action_block_end],
                    v[:, state_block_start:state_block_end],
                ],
                dim=1,
            )
            output[:, action_block_start:action_block_end] = self.attn(
                q[:, action_block_start:action_block_end], k_context, v_context
            )

        for block_idx in range(num_state_blocks):
            state_block_start = state_start + block_idx * self.num_state_per_block
            state_block_end = state_block_start + self.num_state_per_block
            output[:, state_block_start:state_block_end] = self.attn(
                q[:, state_block_start:state_block_end],
                k[:, state_block_start:state_block_end],
                v[:, state_block_start:state_block_end],
            )
        return output

    def _process_clean_image_only(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        clean_frames: int,
    ) -> torch.Tensor:
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (clean_frames - 1) // self.num_frame_per_block
        if num_blocks == 0:
            return self.attn(
                q[:, : self.frame_seqlen],
                k[:, : self.frame_seqlen],
                v[:, : self.frame_seqlen],
            )
        output = torch.empty_like(q)
        output[:, : self.frame_seqlen] = self.attn(
            q[:, : self.frame_seqlen],
            k[:, : self.frame_seqlen],
            v[:, : self.frame_seqlen],
        )
        if self.local_attn_size == -1:
            output[:, self.frame_seqlen :] = self.causal_attn(
                q[:, self.frame_seqlen :], k, v
            )
            return output
        for block_idx in range(num_blocks):
            block_start = self.frame_seqlen + block_idx * block_size
            block_end = min(block_start + block_size, q.shape[1])
            image_kv_start = max(
                self.frame_seqlen,
                block_end - self.local_attn_size * self.frame_seqlen,
            )
            output[:, block_start:block_end] = self.attn(
                q[:, block_start:block_end],
                torch.cat(
                    [k[:, : self.frame_seqlen], k[:, image_kv_start:block_end]], dim=1
                ),
                torch.cat(
                    [v[:, : self.frame_seqlen], v[:, image_kv_start:block_end]], dim=1
                ),
            )
        return output

    def _process_state_blocks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state_horizon: int,
    ) -> torch.Tensor:
        num_blocks = state_horizon // self.num_state_per_block
        if num_blocks == 1:
            return self.attn(q, k, v)
        output = torch.empty_like(q)
        for block_idx in range(num_blocks):
            start = block_idx * self.num_state_per_block
            end = start + self.num_state_per_block
            output[:, start:end] = self.attn(
                q[:, start:end], k[:, start:end], v[:, start:end]
            )
        return output

    def _process_noisy_image_blocks(
        self,
        noisy_image_q: torch.Tensor,
        noisy_image_k: torch.Tensor,
        noisy_image_v: torch.Tensor,
        clean_image_k: torch.Tensor,
        clean_image_v: torch.Tensor,
        noisy_action_k: torch.Tensor,
        noisy_action_v: torch.Tensor,
        noisy_state_k: torch.Tensor,
        noisy_state_v: torch.Tensor,
        half_frames: int,
    ) -> torch.Tensor:
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        output = torch.empty_like(noisy_image_q)
        output[:, : self.frame_seqlen] = self.attn(
            noisy_image_q[:, : self.frame_seqlen],
            noisy_image_k[:, : self.frame_seqlen],
            noisy_image_v[:, : self.frame_seqlen],
        )
        for block_idx in range(num_blocks):
            noisy_start = self.frame_seqlen + block_idx * block_size
            noisy_end = noisy_start + block_size
            clean_end = self.frame_seqlen + block_idx * block_size
            action_start = block_idx * self.num_action_per_block
            action_end = action_start + self.num_action_per_block
            state_start = block_idx * self.num_state_per_block
            state_end = state_start + self.num_state_per_block
            output[:, noisy_start:noisy_end] = self.attn(
                noisy_image_q[:, noisy_start:noisy_end],
                torch.cat(
                    [
                        clean_image_k[:, :clean_end],
                        noisy_image_k[:, noisy_start:noisy_end],
                        noisy_action_k[:, action_start:action_end],
                        noisy_state_k[:, state_start:state_end],
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        clean_image_v[:, :clean_end],
                        noisy_image_v[:, noisy_start:noisy_end],
                        noisy_action_v[:, action_start:action_end],
                        noisy_state_v[:, state_start:state_end],
                    ],
                    dim=1,
                ),
            )
        return output

    def _process_noisy_action_blocks(
        self,
        noisy_action_q: torch.Tensor,
        noisy_action_k: torch.Tensor,
        noisy_action_v: torch.Tensor,
        clean_image_k: torch.Tensor,
        clean_image_v: torch.Tensor,
        noisy_image_k: torch.Tensor,
        noisy_image_v: torch.Tensor,
        noisy_state_k: torch.Tensor,
        noisy_state_v: torch.Tensor,
        half_frames: int,
    ) -> torch.Tensor:
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        output = torch.empty_like(noisy_action_q)
        for block_idx in range(num_blocks):
            action_start = block_idx * self.num_action_per_block
            action_end = action_start + self.num_action_per_block
            clean_end = (
                self.frame_seqlen
                + block_idx * self.frame_seqlen * self.num_frame_per_block
            )
            noisy_start = (
                self.frame_seqlen
                + block_idx * self.frame_seqlen * self.num_frame_per_block
            )
            noisy_end = noisy_start + self.frame_seqlen * self.num_frame_per_block
            state_start = block_idx * self.num_state_per_block
            state_end = state_start + self.num_state_per_block
            output[:, action_start:action_end] = self.attn(
                noisy_action_q[:, action_start:action_end],
                torch.cat(
                    [
                        clean_image_k[:, :clean_end],
                        noisy_image_k[:, noisy_start:noisy_end],
                        noisy_action_k[:, action_start:action_end],
                        noisy_state_k[:, state_start:state_end],
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        clean_image_v[:, :clean_end],
                        noisy_image_v[:, noisy_start:noisy_end],
                        noisy_action_v[:, action_start:action_end],
                        noisy_state_v[:, state_start:state_end],
                    ],
                    dim=1,
                ),
            )
        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor | None = None,
        freqs_action: torch.Tensor | None = None,
        freqs_state: torch.Tensor | None = None,
        action_register_length: int | None = None,
        *,
        kv_cache: torch.Tensor | None = None,
        use_cache: bool = False,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if freqs is None:
            return self._simple_forward(x, kv_cache=kv_cache, use_cache=use_cache)
        if freqs_action is None or freqs_state is None:
            raise ValueError("freqs_action and freqs_state are required with freqs.")

        batch_size, seq_len, _ = x.shape
        q, k, v = self._project_qkv(x)
        updated_kv_cache: torch.Tensor | None = None

        if kv_cache is None:
            if is_tf:
                register_len = action_register_length or 0
                half_seq_len = (seq_len - register_len) // 2
                q_context = q[:, :half_seq_len]
                k_context = k[:, :half_seq_len]
                q_noisy = q[:, half_seq_len:]
                k_noisy = k[:, half_seq_len:]
                rq_context = _dreamzero_rope_action_apply(
                    q_context,
                    freqs,
                    freqs_action,
                    freqs_state,
                    None,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                )
                rk_context = _dreamzero_rope_action_apply(
                    k_context,
                    freqs,
                    freqs_action,
                    freqs_state,
                    None,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                )
                rq_noisy = _dreamzero_rope_action_apply(
                    q_noisy,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                )
                rk_noisy = _dreamzero_rope_action_apply(
                    k_noisy,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                )
                roped_query = torch.cat([rq_context, rq_noisy], dim=1)
                roped_key = torch.cat([rk_context, rk_noisy], dim=1)

                if action_register_length is not None:
                    clean_image_seq_len = half_seq_len
                    noisy_image_seq_len = half_seq_len
                    noisy_frames = noisy_image_seq_len // self.frame_seqlen
                    num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
                    action_horizon = num_image_blocks * self.num_action_per_block
                    state_horizon = num_image_blocks * self.num_state_per_block
                    expected = (
                        half_seq_len
                        + noisy_image_seq_len
                        + action_horizon
                        + state_horizon
                    )
                    if roped_query.shape[1] != expected:
                        raise ValueError(
                            "Sequence length does not match DreamZero "
                            f"teacher-forcing block layout: got={roped_query.shape[1]}, "
                            f"expected={expected}."
                        )

                    clean_image_q = roped_query[:, :clean_image_seq_len]
                    clean_image_k = roped_key[:, :clean_image_seq_len]
                    clean_image_v = v[:, :clean_image_seq_len]
                    noisy_base = half_seq_len
                    action_base = noisy_base + noisy_image_seq_len
                    state_base = action_base + action_horizon
                    noisy_image_q = roped_query[:, noisy_base:action_base]
                    noisy_image_k = roped_key[:, noisy_base:action_base]
                    noisy_image_v = v[:, noisy_base:action_base]
                    noisy_action_q = roped_query[:, action_base:state_base]
                    noisy_action_k = roped_key[:, action_base:state_base]
                    noisy_action_v = v[:, action_base:state_base]
                    noisy_state_q = roped_query[:, state_base:]
                    noisy_state_k = roped_key[:, state_base:]
                    noisy_state_v = v[:, state_base:]

                    x_heads = torch.cat(
                        [
                            self._process_clean_image_only(
                                clean_image_q,
                                clean_image_k,
                                clean_image_v,
                                clean_image_seq_len // self.frame_seqlen,
                            ),
                            self._process_noisy_image_blocks(
                                noisy_image_q,
                                noisy_image_k,
                                noisy_image_v,
                                clean_image_k,
                                clean_image_v,
                                noisy_action_k,
                                noisy_action_v,
                                noisy_state_k,
                                noisy_state_v,
                                noisy_frames,
                            ),
                            self._process_noisy_action_blocks(
                                noisy_action_q,
                                noisy_action_k,
                                noisy_action_v,
                                clean_image_k,
                                clean_image_v,
                                noisy_image_k,
                                noisy_image_v,
                                noisy_state_k,
                                noisy_state_v,
                                noisy_frames,
                            ),
                            self._process_state_blocks(
                                noisy_state_q,
                                noisy_state_k,
                                noisy_state_v,
                                state_horizon,
                            ),
                        ],
                        dim=1,
                    )
                else:
                    clean_q = roped_query[:, :half_seq_len]
                    clean_k = roped_key[:, :half_seq_len]
                    clean_v = v[:, :half_seq_len]
                    noisy_q = roped_query[:, half_seq_len:]
                    noisy_k = roped_key[:, half_seq_len:]
                    noisy_v = v[:, half_seq_len:]
                    x_heads = torch.cat(
                        [
                            self._blockwise_causal_attn(
                                clean_q,
                                clean_k,
                                clean_v,
                                action_horizon=None,
                                state_horizon=None,
                            ),
                            self.attn(
                                noisy_q,
                                torch.cat([clean_k, noisy_k], dim=1),
                                torch.cat([clean_v, noisy_v], dim=1),
                            ),
                        ],
                        dim=1,
                    )
            else:
                roped_query = _dreamzero_rope_action_apply(
                    q,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                )
                roped_key = _dreamzero_rope_action_apply(
                    k,
                    freqs,
                    freqs_action,
                    freqs_state,
                    action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                )
                if action_register_length is not None:
                    chunk_size = action_register_length // (
                        self.num_action_per_block + self.num_state_per_block
                    )
                    action_horizon = chunk_size * self.num_action_per_block
                    state_horizon = chunk_size * self.num_state_per_block
                else:
                    action_horizon = None
                    state_horizon = None
                x_heads = self._blockwise_causal_attn(
                    roped_query,
                    roped_key,
                    v,
                    action_horizon=action_horizon,
                    state_horizon=state_horizon,
                )
        else:
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
        if use_cache and updated_kv_cache is None:
            updated_kv_cache = torch.stack([k, v], dim=0)
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
        norm_backend: str | None = None,
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
        self.use_fa2_cross_attention = attn_backend in ("te", "official")
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_local_heads,
            causal=False,
            backend="sdpa" if attn_backend == "official" else attn_backend,
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

        if self.use_fa2_cross_attention:
            text_out = _dreamzero_fa2_cross_attention(q, k, v)
        else:
            text_out = self.attn(q, k, v)
        if self.use_fa2_cross_attention:
            image_out = _dreamzero_fa2_cross_attention(q, k_img, v_img)
        else:
            image_out = self.attn(q, k_img, v_img)
        summed_heads = text_out + image_out
        merged = summed_heads.reshape(batch_size, seq_len, -1)
        out, _ = self.o(merged)

        updated_crossattn_cache = None
        if use_cache or crossattn_cache is not None:
            updated_crossattn_cache = torch.stack([k, v], dim=0)
        return out, updated_crossattn_cache


class DreamZeroMLP(nn.Module):
    """TP-sharded FFN skeleton matching ``ffn.0`` and ``ffn.2`` keys."""

    def __init__(
        self,
        config: DreamZeroConfig,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.dit.dim,
            config.dit.ffn_dim,
            bias=True,
            gather_output=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.0" if prefix else "",
        )
        self.fc2 = RowParallelLinear(
            config.dit.ffn_dim,
            config.dit.dim,
            bias=True,
            input_is_parallel=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.2" if prefix else "",
        )

    def forward(self, *args, **kwargs):
        if len(args) != 1 or kwargs:
            raise TypeError("DreamZeroMLP.forward expects exactly one tensor argument.")
        x = args[0]
        x, _ = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x, _ = self.fc2(x)
        return x


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
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
        )
        self.self_attn = DreamZeroSelfAttention(
            config,
            params_dtype=params_dtype,
            device=device,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
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
            norm_backend=norm_backend,
            prefix=f"{prefix}.cross_attn" if prefix else "",
        )
        self.norm2 = LayerNorm(
            dim,
            eps=config.dit.eps,
            bias=False,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
        )
        self.ffn = DreamZeroMLP(
            config,
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
            _attach_replicated(self.modulation, f"{prefix}.modulation")

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
        is_tf: bool = True,
        image_context_tokens: int = 257,
        use_crossattn_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        modulation = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        aligned = []
        seq_len = x.shape[1]
        for part in modulation:
            part_len = part.shape[1]
            if part_len == seq_len:
                aligned.append(part)
            elif part_len >= seq_len:
                aligned.append(part[:, :seq_len])
            else:
                repeat = (seq_len + part_len - 1) // part_len
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :seq_len])
        e0, e1, e2, e3, e4, e5 = [part.squeeze(2) for part in aligned]

        self_attn_input = self.norm1(x) * (1 + e1) + e0
        y, updated_kv_cache = self.self_attn(
            self_attn_input,
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
            is_tf=is_tf,
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
    """Output head skeleton matching ``head.head`` and ``head.modulation``."""

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
            _attach_replicated(self.modulation, f"{prefix}.modulation")

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        modulation = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        aligned = []
        seq_len = x.shape[1]
        for part in modulation:
            part_len = part.shape[1]
            if part_len == seq_len:
                aligned.append(part)
            elif part_len >= seq_len:
                aligned.append(part[:, :seq_len])
            else:
                repeat = (seq_len + part_len - 1) // part_len
                aligned.append(part.repeat_interleave(repeat, dim=1)[:, :seq_len])
        shift, scale = [part.squeeze(2) for part in aligned]
        return _linear(self.head, self.norm(x) * (1 + scale) + shift)


class DreamZeroDiT(nn.Module):
    """DreamZero Causal Wan DiT with TP-sharded DiT blocks."""

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
        config.dit.validate_tp_size(_current_tp_size())
        self.config = config
        self.patch_embedding = nn.Conv3d(
            config.dit.in_dim,
            config.dit.dim,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            device=device,
            dtype=params_dtype,
        )
        _attach_replicated(self.patch_embedding.weight, "patch_embedding.weight")
        if self.patch_embedding.bias is not None:
            _attach_replicated(self.patch_embedding.bias, "patch_embedding.bias")

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
        self.time_projection = nn.ModuleList(
            [
                ReplicatedLinear(
                    config.dit.dim,
                    config.dit.dim * 6,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix="time_projection.1",
                )
            ]
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

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return (1, 2, 2)

    def _text_embedding(self, context: torch.Tensor) -> torch.Tensor:
        context = _linear(self.text_embedding[0], context)
        context = F.gelu(context, approximate="tanh")
        return _linear(self.text_embedding[1], context)

    def _time_embedding(
        self, timestep: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        emb = dreamzero_sinusoidal_embedding_1d(
            self.config.dit.freq_dim, timestep.flatten()
        ).to(dtype=dtype, device=timestep.device)
        emb = _linear(self.time_embedding[0], emb)
        emb = F.silu(emb)
        return _linear(self.time_embedding[1], emb)

    def _time_projection(self, e: torch.Tensor) -> torch.Tensor:
        e = F.silu(e)
        return _linear(self.time_projection[0], e)

    def _img_embedding(self, clip_feature: torch.Tensor) -> torch.Tensor:
        x = self.img_emb["proj_0_norm"](clip_feature)
        x = _linear(self.img_emb["proj_1"], x)
        x = F.gelu(x)
        x = _linear(self.img_emb["proj_3"], x)
        return self.img_emb["proj_4_norm"](x)

    def _create_freqs(self, grid_size: torch.Tensor, start_frame: int) -> torch.Tensor:
        device = self.patch_embedding.weight.device
        freqs = [freq.to(device) for freq in self.freqs]
        self.freqs_action = self.freqs_action.to(device)
        self.freqs_state = self.freqs_state.to(device)
        f, h, w = grid_size.tolist()
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

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out_dim = self.config.dit.out_dim
        grid = grid_size.tolist()
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

    def _prepare_video_tokens(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        *,
        concat_first_frame_latent: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if y is not None and concat_first_frame_latent:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long, device=x.device)
        return x.flatten(start_dim=2).transpose(1, 2), grid_size

    def _forward_blocks(
        self,
        x_tokens: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        kv_cache: list[torch.Tensor | None] | None,
        crossattn_cache: list[torch.Tensor | None] | None,
        current_start_frame: int,
        *,
        clean_tokens: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        is_tf: bool = False,
        image_context_tokens: int = 257,
        use_crossattn_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list[torch.Tensor | None],
        list[torch.Tensor | None],
    ]:
        batch_size = x_tokens.shape[0]
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
        timestep_original = timestep_video
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
        e0 = self._time_projection(e).unflatten(dim=2, sizes=(6, self.config.dit.dim))

        if clean_tokens is not None:
            x_tokens = torch.cat([clean_tokens, x_tokens], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(timestep_original)
            e_clean = self._time_embedding(aug_t, x_tokens.dtype)
            e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
            e0_clean = self._time_projection(e_clean).unflatten(
                dim=2, sizes=(6, self.config.dit.dim)
            )
            e0 = torch.cat([e0_clean, e0], dim=1)
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
                is_tf=is_tf,
                image_context_tokens=image_context_tokens,
                use_crossattn_cache=use_crossattn_cache,
            )
            updated_kv_caches.append(updated_kv_cache)
            updated_crossattn_caches.append(updated_crossattn_cache)

        if clean_tokens is not None:
            x_tokens = x_tokens[:, clean_tokens.shape[1] :]

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
        return x_video, action_noise_pred, updated_kv_caches, updated_crossattn_caches

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        *,
        seq_len: int | None = None,
        kv_cache: list[torch.Tensor | None] | None = None,
        crossattn_cache: list[torch.Tensor | None] | None = None,
        current_start_frame: int = 0,
        y: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        timestep_action: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        concat_first_frame_latent: bool = False,
        image_context_tokens: int = 257,
        use_crossattn_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        list[torch.Tensor | None],
        list[torch.Tensor | None],
    ]:
        x_tokens, grid_size = self._prepare_video_tokens(
            x,
            y,
            concat_first_frame_latent=concat_first_frame_latent,
        )
        if seq_len is None:
            seq_len = x_tokens.shape[1]
        if x_tokens.shape[1] != seq_len:
            raise ValueError(
                f"seq_len={seq_len} does not match tokens={x_tokens.shape[1]}."
            )

        clean_tokens = None
        if clean_x is not None:
            clean_tokens, clean_grid_size = self._prepare_video_tokens(
                clean_x,
                y,
                concat_first_frame_latent=concat_first_frame_latent,
            )
            if not torch.equal(clean_grid_size, grid_size):
                raise ValueError("clean_x grid size must match x grid size.")

        freqs = self._create_freqs(grid_size, current_start_frame)
        video_tokens, action_noise_pred, updated_kv_caches, updated_crossattn_caches = (
            self._forward_blocks(
                x_tokens,
                seq_len,
                freqs,
                timestep,
                context,
                clip_feature,
                action,
                timestep_action,
                state,
                embodiment_id,
                kv_cache,
                crossattn_cache,
                current_start_frame,
                clean_tokens=clean_tokens,
                aug_t=aug_t,
                is_tf=clean_tokens is not None,
                image_context_tokens=image_context_tokens,
                use_crossattn_cache=use_crossattn_cache,
            )
        )
        return (
            self.unpatchify(video_tokens, grid_size),
            action_noise_pred,
            updated_kv_caches,
            updated_crossattn_caches,
        )


def _current_tp_size() -> int:
    import phyai.parallel as P

    return P.default_mesh().group_size("dense_tp")


def _current_mesh_or_none():
    import phyai.parallel as P

    return P.default_mesh()


def dreamzero_dit_weight_remap(key: str) -> str | None:
    """Map a DreamZero checkpoint key to the future DiT module namespace.

    The DreamZero-DROID checkpoint stores every component under ``action_head``.
    The DiT implementation in PhyAI will own only the ``action_head.model``
    subtree, so this helper strips that prefix and drops text/image/VAE keys.
    Unknown action-head leaves are dropped until their target module exists.
    """
    if key.startswith(_DREAMZERO_DIT_PREFIX):
        return key[len(_DREAMZERO_DIT_PREFIX) :]
    for prefix in _DREAMZERO_DROPPED_PREFIXES:
        if key.startswith(prefix):
            return None
    return None


def dreamzero_component_from_key(key: str) -> str | None:
    """Classify a raw DreamZero checkpoint key by high-level component."""
    if key.startswith(_DREAMZERO_DIT_PREFIX):
        return "dit"
    if key.startswith("action_head.text_encoder."):
        return "text_encoder"
    if key.startswith("action_head.image_encoder."):
        return "image_encoder"
    if key.startswith("action_head.vae."):
        return "vae"
    if key.startswith("action_head."):
        return "action_head_other"
    return None


__all__ = [
    "DreamZeroActionEncoder",
    "DreamZeroCategorySpecificLinear",
    "DreamZeroCategorySpecificMLP",
    "DreamZeroCausalHead",
    "DreamZeroCrossAttention",
    "DreamZeroDiT",
    "DreamZeroDiTBlock",
    "DreamZeroMLP",
    "DreamZeroSelfAttention",
    "DreamZeroSinusoidalPositionalEncoding",
    "dreamzero_component_from_key",
    "dreamzero_dit_weight_remap",
]
