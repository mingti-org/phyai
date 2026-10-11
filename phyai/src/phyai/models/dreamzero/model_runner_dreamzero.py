"""DreamZero DiT runner.

The runner owns per-request cache tensors and calls the stateless DiT model for
one forward pass. Scheduler code decides when to reset or reuse this runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.models.dreamzero.modeling_dreamzero import DreamZeroDiT
from phyai.runtime.model_runner import ModelRunner


@dataclass
class DreamZeroDiTForwardBatch:
    x: torch.Tensor
    timestep: torch.Tensor
    context: torch.Tensor
    seq_len: int | None = None
    current_start_frame: int = 0
    y: torch.Tensor | None = None
    clip_feature: torch.Tensor | None = None
    action: torch.Tensor | None = None
    timestep_action: torch.Tensor | None = None
    state: torch.Tensor | None = None
    embodiment_id: torch.Tensor | None = None
    clean_x: torch.Tensor | None = None
    aug_t: torch.Tensor | None = None
    concat_first_frame_latent: bool = False
    image_context_tokens: int = 257
    use_kv_cache: bool = True
    update_kv_cache: bool = True
    use_crossattn_cache: bool = False
    update_crossattn_cache: bool = True


@dataclass
class DreamZeroDiTForwardOutput:
    video: torch.Tensor
    action: torch.Tensor | None
    kv_cache: list[torch.Tensor | None]
    crossattn_cache: list[torch.Tensor | None]


class DreamZeroDiTRunner(ModelRunner):
    """Owns DreamZero DiT KV caches for one scheduler branch."""

    def __init__(
        self,
        model: DreamZeroDiT,
        *,
        device: torch.device | str | None = None,
        max_kv_cache_tokens: int | None = None,
        max_crossattn_cache_tokens: int | None = None,
    ) -> None:
        self.model = model
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.max_kv_cache_tokens = max_kv_cache_tokens
        self.max_crossattn_cache_tokens = max_crossattn_cache_tokens
        self._kv_cache: list[torch.Tensor | None] = []
        self._crossattn_cache: list[torch.Tensor | None] = []

    def setup(self) -> None:
        return None

    def reset(self) -> None:
        self._kv_cache = []
        self._crossattn_cache = []

    @property
    def kv_cache(self) -> list[torch.Tensor | None]:
        return self._kv_cache

    @property
    def crossattn_cache(self) -> list[torch.Tensor | None]:
        return self._crossattn_cache

    def create_kv_caches(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        seq_len: int = 0,
    ) -> list[torch.Tensor]:
        if device is None:
            device = self.device
        first_attn = self.model.blocks[0].self_attn
        shape = (
            2,
            batch_size,
            seq_len,
            first_attn.num_local_heads,
            first_attn.head_dim,
        )
        return [
            torch.empty(shape, dtype=dtype, device=device)
            for _ in range(len(self.model.blocks))
        ]

    def _input_kv_cache(
        self, batch: DreamZeroDiTForwardBatch
    ) -> list[torch.Tensor | None] | None:
        if not batch.use_kv_cache:
            return None
        if not self._kv_cache:
            return self.create_kv_caches(
                batch_size=batch.x.shape[0],
                dtype=batch.x.dtype,
                device=batch.x.device,
            )
        return self._kv_cache

    def _input_crossattn_cache(
        self, batch: DreamZeroDiTForwardBatch
    ) -> list[torch.Tensor | None] | None:
        del batch
        return self._crossattn_cache or None

    def _store_cache(
        self,
        caches: list[torch.Tensor | None],
        *,
        max_tokens: int | None,
        name: str,
    ) -> list[torch.Tensor | None]:
        if len(caches) != len(self.model.blocks):
            raise ValueError(
                f"{name} list length {len(caches)} does not match "
                f"num_layers={len(self.model.blocks)}."
            )
        stored: list[torch.Tensor | None] = []
        seq_len: int | None = None
        for layer_id, cache in enumerate(caches):
            if cache is None:
                raise ValueError(f"updated {name} for layer {layer_id} is None.")
            layer_seq_len = int(cache.shape[2])
            if seq_len is None:
                seq_len = layer_seq_len
            elif layer_seq_len != seq_len:
                raise ValueError(
                    f"all {name} tensors must have the same sequence length."
                )
            if max_tokens is not None and layer_seq_len > max_tokens:
                raise ValueError(
                    f"updated {name} seq_len={layer_seq_len} exceeds configured "
                    f"max_tokens={max_tokens}."
                )
            stored.append(cache)
        return stored

    @torch.no_grad()
    def forward(self, batch: DreamZeroDiTForwardBatch) -> DreamZeroDiTForwardOutput:
        kv_cache = self._input_kv_cache(batch)
        crossattn_cache = self._input_crossattn_cache(batch)
        video, action, updated_kv_cache, updated_crossattn_cache = self.model(
            batch.x,
            batch.timestep,
            batch.context,
            seq_len=batch.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=batch.current_start_frame,
            y=batch.y,
            clip_feature=batch.clip_feature,
            action=batch.action,
            timestep_action=batch.timestep_action,
            state=batch.state,
            embodiment_id=batch.embodiment_id,
            clean_x=batch.clean_x,
            aug_t=batch.aug_t,
            concat_first_frame_latent=batch.concat_first_frame_latent,
            image_context_tokens=batch.image_context_tokens,
            use_crossattn_cache=batch.use_crossattn_cache,
        )

        if batch.update_kv_cache:
            if not any(cache is not None for cache in updated_kv_cache):
                raise RuntimeError("model did not return KV cache to update.")
            self._kv_cache = self._store_cache(
                updated_kv_cache,
                max_tokens=self.max_kv_cache_tokens,
                name="KV cache",
            )
        if batch.update_crossattn_cache:
            if any(cache is not None for cache in updated_crossattn_cache):
                self._crossattn_cache = self._store_cache(
                    updated_crossattn_cache,
                    max_tokens=self.max_crossattn_cache_tokens,
                    name="cross-attention cache",
                )
        return DreamZeroDiTForwardOutput(
            video=video,
            action=action,
            kv_cache=self.kv_cache if batch.update_kv_cache else updated_kv_cache,
            crossattn_cache=(
                self.crossattn_cache
                if batch.update_crossattn_cache
                else updated_crossattn_cache
            ),
        )


__all__ = [
    "DreamZeroDiTForwardBatch",
    "DreamZeroDiTForwardOutput",
    "DreamZeroDiTRunner",
]
