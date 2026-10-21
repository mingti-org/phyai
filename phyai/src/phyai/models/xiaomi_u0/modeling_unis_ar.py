"""Single-rank UNIS backbone with runner-supplied optional paged context."""

from __future__ import annotations

import torch
from torch import nn

from phyai.kernel.call import CallSite, backend_preference

from phyai.parallel.state import default_mesh
from phyai.layers.attention.paged.base import PagedAttnCtx
from phyai.layers.linear import ReplicatedLinear
from phyai.layers.vocab_embedding import VocabParallelEmbedding
from phyai.models.xiaomi_u0.configuration_unis import UNISConfig
from phyai.models.xiaomi_u0.modeling_unis import UNISDecoderLayer, UNISRMSNorm


class UNISModel(nn.Module):
    """Embedding, decoder stack, and final normalization."""

    def __init__(
        self,
        config: UNISConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        attention_backend: str | None = None,
        rope_backend: str | None = None,
    ) -> None:
        """Build the single-rank token embedding, decoder stack and final RMS norm."""
        super().__init__()
        if default_mesh().replica_world_size != 1:
            raise ValueError("The initial UNIS implementation requires one rank.")
        self.max_position_embeddings = config.max_position_embeddings
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=dtype,
            padding_multiple=1,
            device=device,
            prefix="model.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                UNISDecoderLayer(
                    config,
                    layer_idx=index,
                    dtype=dtype,
                    device=device,
                    attention_backend=attention_backend,
                    rope_backend=rope_backend,
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = UNISRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix="model.norm",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        ctx: PagedAttnCtx | None = None,
    ) -> torch.Tensor:
        """Return normalized hidden states for a nonempty batch of integer token IDs.

        Without a paged context, positions default to a sequence starting at
        zero. Paged execution requires one sequence and explicit absolute
        positions matching the current cache writes.
        """
        if self.training:
            raise RuntimeError("UNISModel currently supports eval mode only.")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty (batch, sequence) shape.")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs.")
        batch, length = input_ids.shape
        if ctx is not None:
            if batch != 1:
                raise ValueError("Initial paged UNIS supports one sequence only.")
            if position_ids is None:
                raise ValueError("Paged UNIS requires explicit absolute positions.")
            if ctx.write_indices.numel() != length:
                raise ValueError("Cache write slots must match input token count.")
        if position_ids is None:
            if length > self.max_position_embeddings:
                raise ValueError("Sequence exceeds the configured RoPE cache.")
            position_ids = torch.arange(
                length, device=input_ids.device, dtype=torch.long
            ).unsqueeze(0)
        if position_ids.shape not in ((1, length), (batch, length)):
            raise ValueError("position_ids must have shape (1, S) or (B, S).")
        if position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("position_ids must contain integer positions.")
        if position_ids.device != input_ids.device:
            raise ValueError("position_ids and input_ids must share a device.")

        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, ctx)
        return self.norm(hidden_states)


class UNISForCausalLM(nn.Module):
    """UNIS forward returning FP32 vocabulary logits."""

    def __init__(
        self,
        config: UNISConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        attention_backend: str | None = None,
        rope_backend: str | None = None,
        lm_head_backend: str | None = "torch",
    ) -> None:
        """Build the UNIS backbone and a separate vocabulary head with a local GEMM preference."""
        super().__init__()
        self.model = UNISModel(
            config,
            dtype=dtype,
            device=device,
            attention_backend=attention_backend,
            rope_backend=rope_backend,
        )
        self.lm_head = ReplicatedLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            params_dtype=dtype,
            device=device,
            prefix="lm_head",
        )
        # Preserve the reference LM head's BF16 linear and FP32 output cast.
        # The preference is local to this layer, not a global GEMM policy.
        self.lm_head.gemm_call = CallSite(
            "gemm",
            role=self.lm_head.kernel_role,
            prefer=backend_preference("gemm", lm_head_backend),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        ctx: PagedAttnCtx | None = None,
    ) -> torch.Tensor:
        """Project backbone states to vocabulary logits and return them in FP32."""
        hidden_states = self.model(input_ids, position_ids, ctx)
        logits, _ = self.lm_head(hidden_states)
        return logits.float()
