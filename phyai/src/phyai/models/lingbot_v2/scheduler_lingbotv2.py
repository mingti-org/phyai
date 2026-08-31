"""Single-card LingBot-VLA 2.0 inference scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from phyai.layers.attention import (
    ARAttnMetadata,
    AttnLayout,
    AttnMode,
    DiffusionAttnMetadata,
)
from phyai.models.lingbot_v2.configuration_lingbotv2 import LingBotVLA2Config
from phyai.models.lingbot_v2.model_runner_lingbotv2 import (
    LingBotV2ExpertRunner,
    LingBotV2PrefixForwardBatch,
    LingBotV2PrefixRunner,
    LingBotV2VisionForwardBatch,
    LingBotV2VisionRunner,
)
from phyai.runtime.schedule import Scheduler
from phyai.utils.profile import event_scope

if TYPE_CHECKING:
    from phyai.models.lingbot_v2.modeling_lingbotv2 import LingBotV2Model


def build_prefix_padded_write_indices(
    real_lens: torch.Tensor,
    *,
    n_per_sample: int,
    prefix_slot_base: int,
    sentinel_slot: int = 0,
) -> torch.Tensor:
    """Map fixed-stride prefix rows to compact real-token KV slots."""

    batch_size = int(real_lens.shape[0])
    real64 = real_lens.to(torch.int64)
    prefix_starts = torch.zeros(
        batch_size + 1, dtype=torch.int64, device=real_lens.device
    )
    prefix_starts[1:] = torch.cumsum(real64, dim=0)
    row = torch.arange(
        n_per_sample, dtype=torch.int64, device=real_lens.device
    ).unsqueeze(0)
    real_slot = prefix_slot_base + prefix_starts[:-1, None] + row
    return torch.where(
        row < real64[:, None],
        real_slot,
        torch.full_like(real_slot, sentinel_slot),
    ).flatten()


def build_expert_visible_prefix_lens(
    full_prefix_lens: torch.Tensor,
    *,
    future_query_tokens: int,
    block_future_depth_to_action: bool,
    block_suffix_to_future_video: bool,
) -> torch.Tensor:
    """Return the prefix length visible to state and action expert queries."""

    if future_query_tokens < 0:
        raise ValueError("future_query_tokens must be non-negative.")
    if block_future_depth_to_action or block_suffix_to_future_video:
        if bool((full_prefix_lens < future_query_tokens).any()):
            raise ValueError("future query tokens exceed the full prefix length.")
        return full_prefix_lens - future_query_tokens
    return full_prefix_lens


def _prefix_and_suffix_indices(
    full_prefix_lens: torch.Tensor,
    visible_prefix_lens: torch.Tensor,
    *,
    suffix_tokens: int,
    suffix_len: int,
    prefix_slot_base: int,
    suffix_slot_base: int,
) -> torch.Tensor:
    """Build sample-major KV lists while skipping hidden future-query slots."""

    device = full_prefix_lens.device
    pieces: list[torch.Tensor] = []
    prefix_start = 0
    for batch_index in range(int(full_prefix_lens.shape[0])):
        visible = int(visible_prefix_lens[batch_index])
        if visible:
            pieces.append(
                torch.arange(
                    prefix_slot_base + prefix_start,
                    prefix_slot_base + prefix_start + visible,
                    dtype=torch.int32,
                    device=device,
                )
            )
        pieces.append(
            torch.arange(
                suffix_slot_base + batch_index * suffix_len,
                suffix_slot_base + batch_index * suffix_len + suffix_tokens,
                dtype=torch.int32,
                device=device,
            )
        )
        prefix_start += int(full_prefix_lens[batch_index])
    return torch.cat(pieces)


def build_state_paged_kv_indices(
    full_prefix_lens: torch.Tensor,
    visible_prefix_lens: torch.Tensor,
    *,
    suffix_len: int,
    prefix_slot_base: int,
    suffix_slot_base: int,
) -> torch.Tensor:
    """KV slots visible to each state query: visible prefix plus state."""

    return _prefix_and_suffix_indices(
        full_prefix_lens,
        visible_prefix_lens,
        suffix_tokens=1,
        suffix_len=suffix_len,
        prefix_slot_base=prefix_slot_base,
        suffix_slot_base=suffix_slot_base,
    )


def build_action_paged_kv_indices(
    full_prefix_lens: torch.Tensor,
    visible_prefix_lens: torch.Tensor,
    *,
    suffix_len: int,
    prefix_slot_base: int,
    suffix_slot_base: int,
) -> torch.Tensor:
    """KV slots visible to action queries: visible prefix plus full suffix."""

    return _prefix_and_suffix_indices(
        full_prefix_lens,
        visible_prefix_lens,
        suffix_tokens=suffix_len,
        suffix_len=suffix_len,
        prefix_slot_base=prefix_slot_base,
        suffix_slot_base=suffix_slot_base,
    )


def build_suffix_mrope_position_ids(
    prefix_position_ids: torch.Tensor,
    prefix_mask: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return state and action MRoPE ids after each sample's full prefix."""

    valid = prefix_position_ids.masked_fill(~prefix_mask.unsqueeze(0), 0)
    offsets = valid.amax(dim=(0, 2)).to(torch.int64) + 1
    state = offsets.view(1, -1, 1).expand(3, -1, 1)
    action_steps = torch.arange(
        1,
        chunk_size + 1,
        dtype=torch.int64,
        device=prefix_position_ids.device,
    )
    action = offsets[:, None] + action_steps[None, :]
    return state, action.unsqueeze(0).expand(3, -1, -1)


@dataclass
class LingBotV2Request:
    """Preprocessed LingBot request.

    pixel_values has shape (B, N, P, patch_vector_dim). Only the leading
    product(image_grid_thw) patch rows of an active image are consumed.
    """

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    image_masks: torch.Tensor
    input_ids: torch.Tensor
    lang_lens: torch.Tensor
    state: torch.Tensor
    noise: torch.Tensor | None = None


class LingBotV2WS1Scheduler(Scheduler):
    """PI0-style single-card orchestration for the LingBot V2 graph."""

    def __init__(
        self,
        model: LingBotV2Model,
        *,
        max_batch_size: int = 1,
        num_images: int = 3,
        max_vision_tokens_per_image: int | None = None,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        self.cfg: LingBotVLA2Config = model.config
        self.max_batch_size = int(max_batch_size)
        self.num_images = int(num_images)
        if self.max_batch_size <= 0 or self.num_images <= 0:
            raise ValueError("max_batch_size and num_images must be positive.")
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.params_dtype = model.params_dtype
        self.max_vision_tokens_per_image = int(
            max_vision_tokens_per_image
            if max_vision_tokens_per_image is not None
            else self.cfg.dual_query.image_token_size**2
        )
        boundary_tokens = 2 if self.cfg.use_vision_boundaries else 0
        self.max_image_prefix_tokens = self.num_images * (
            self.max_vision_tokens_per_image + boundary_tokens
        )
        self.n_per_sample = (
            self.max_image_prefix_tokens
            + self.cfg.tokenizer_max_length
            + self.cfg.dual_query.prefix_query_token_count
        )
        self.suffix_len = self.cfg.suffix_len

        prefix_capacity = self.max_batch_size * self.n_per_sample
        suffix_capacity = self.max_batch_size * self.suffix_len
        self.vision_runner = LingBotV2VisionRunner(
            model.vision,
            params_dtype=model.vision_params_dtype,
            device=self.device,
        )
        self.prefix_runner = LingBotV2PrefixRunner(
            model,
            max_batch_size=self.max_batch_size,
            prefix_capacity=prefix_capacity,
            suffix_capacity=suffix_capacity,
            params_dtype=self.params_dtype,
            device=self.device,
        )
        self.expert_runner = LingBotV2ExpertRunner(
            model.expert_stack,
            model.heads,
            model.mrope,
            self.prefix_runner.kv_pool,
            max_batch_size=self.max_batch_size,
            chunk_size=self.cfg.chunk_size,
            max_action_dim=self.cfg.max_action_dim,
            num_inference_steps=self.cfg.num_inference_steps,
            max_num_tokens=self.max_batch_size * self.cfg.chunk_size,
            max_paged_kv_indices=self.max_batch_size
            * (self.n_per_sample + self.suffix_len),
            params_dtype=self.params_dtype,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
        )

    def setup(self) -> None:
        """Initialize the vision, prefix, and expert runners in order."""

        self.vision_runner.setup()
        self.prefix_runner.setup()
        self.expert_runner.setup()

    def reset(self) -> None:
        """Reset all request-local runtime state before a new inference."""

        self.vision_runner.reset()
        self.prefix_runner.reset()
        self.expert_runner.reset()

    def close(self) -> None:
        """Release runner-owned attention plans, backends, and condition caches."""

        self.expert_runner.close()
        self.prefix_runner.close()
        self.vision_runner.close()

    @torch.no_grad()
    def step(self, request: LingBotV2Request) -> torch.Tensor:
        """Run one request and return (B, chunk_size, action_dim)."""

        self._validate(request)
        self.reset()
        batch_size = int(request.pixel_values.shape[0])
        with event_scope("lingbot_v2.vision"):
            merged, deepstack, active_grids, merged_counts = self._run_vision(request)
        with event_scope("lingbot_v2.prefix_pack"):
            (
                packed,
                fake_ids,
                prefix_mask,
                visual_mask,
                real_lens,
                visible_lens,
            ) = self._pack_prefix(
                request,
                merged,
                active_grids,
                merged_counts,
            )
            position_ids = self.prefix_runner.build_position_ids(
                fake_ids,
                prefix_mask,
                active_grids,
            )
            ragged_position_ids = position_ids[:, prefix_mask].unsqueeze(1)

        with event_scope("lingbot_v2.prefix_plan"):
            n_real_total = int(real_lens.sum())
            cache_allocation = self.prefix_runner.allocate_request_cache(
                prefix_tokens=n_real_total,
                suffix_tokens=batch_size * self.suffix_len,
            )
            write_indices = cache_allocation.prefix_slots
            prefix_indptr = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=self.device
            )
            prefix_indptr[1:] = torch.cumsum(real_lens, dim=0)
            prefix_metadata = ARAttnMetadata(
                mode=AttnMode.PREFILL,
                layout=AttnLayout.RAGGED_3D,
                batch_size=batch_size,
                num_query_tokens=n_real_total,
                cu_seqlens_q=prefix_indptr,
                paged_kv_indptr=prefix_indptr,
                paged_kv_indices=write_indices.to(torch.int32),
                paged_kv_last_page_len=torch.ones(
                    batch_size, dtype=torch.int32, device=self.device
                ),
                write_indices=write_indices,
                position_ids=ragged_position_ids,
            )
            self.prefix_runner.plan_inference(prefix_metadata)

        with event_scope("lingbot_v2.prefix_forward"):
            self.prefix_runner.forward(
                LingBotV2PrefixForwardBatch(
                    hidden_states=packed,
                    position_ids=ragged_position_ids,
                    write_indices=write_indices,
                    visual_pos_masks=visual_mask,
                    deepstack_visual_embeds=deepstack,
                )
            )

        with event_scope("lingbot_v2.expert_plan"):
            self._plan_expert(
                real_lens,
                visible_lens,
                position_ids,
                prefix_mask,
                cache_allocation.suffix_slots,
            )
        with event_scope("lingbot_v2.euler"):
            state = request.state.to(device=self.device, dtype=self.params_dtype)
            if request.noise is None:
                x_t = torch.randn(
                    batch_size,
                    self.cfg.chunk_size,
                    self.cfg.max_action_dim,
                    dtype=self.params_dtype,
                    device=self.device,
                )
            else:
                x_t = request.noise.to(device=self.device, dtype=self.params_dtype)
            x_t = self.expert_runner.forward_euler(state, x_t)
        # A captured graph owns static output storage that is overwritten on
        # its next replay. Return an independent action tensor to the caller.
        return x_t[..., : self.cfg.action_dim].clone()

    def _run_vision(
        self, request: LingBotV2Request
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[int]]:
        patch_parts: list[torch.Tensor] = []
        grid_parts: list[torch.Tensor] = []
        for batch_index in range(request.pixel_values.shape[0]):
            for image_index in range(self.num_images):
                if not bool(request.image_masks[batch_index, image_index]):
                    continue
                grid = request.image_grid_thw[batch_index, image_index]
                patch_count = int(grid.prod())
                patch_parts.append(
                    request.pixel_values[batch_index, image_index, :patch_count]
                )
                grid_parts.append(grid)
        if not patch_parts:
            raise ValueError("at least one active image is required.")
        pixel_values = torch.cat(patch_parts, dim=0)
        active_grids = torch.stack(grid_parts).to(device=self.device, dtype=torch.int64)
        merged, deepstack = self.vision_runner.forward(
            LingBotV2VisionForwardBatch(
                pixel_values=pixel_values,
                image_grid_thw=active_grids,
            )
        )
        merged_counts = [
            int(grid.prod()) // self.cfg.vision.spatial_merge_unit
            for grid in active_grids
        ]
        if any(count > self.max_vision_tokens_per_image for count in merged_counts):
            raise ValueError(
                f"merged image tokens {merged_counts} exceed "
                f"max_vision_tokens_per_image={self.max_vision_tokens_per_image}."
            )
        return merged, deepstack, active_grids, merged_counts

    def _pack_prefix(
        self,
        request: LingBotV2Request,
        merged: torch.Tensor,
        active_grids: torch.Tensor,
        merged_counts: list[int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = int(request.pixel_values.shape[0])
        hidden_size = self.cfg.text.hidden_size
        packed = torch.zeros(
            batch_size,
            self.n_per_sample,
            hidden_size,
            dtype=self.params_dtype,
            device=self.device,
        )
        fake_ids = torch.full(
            (batch_size, self.n_per_sample),
            self.cfg.text.eos_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        prefix_mask = torch.zeros(
            batch_size,
            self.n_per_sample,
            dtype=torch.bool,
            device=self.device,
        )
        visual_mask = torch.zeros_like(prefix_mask)
        embeddings = self.prefix_runner.prepare_prefix_embeddings(
            request.input_ids,
            batch_size=batch_size,
        )
        language = embeddings.language
        current_query = embeddings.current_query
        future_query = embeddings.future_query
        start_embedding = embeddings.vision_start
        end_embedding = embeddings.vision_end
        merged_offset = 0
        image_counter = 0
        real_lens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        for batch_index in range(batch_size):
            cursor = 0
            for image_index in range(self.num_images):
                if not bool(request.image_masks[batch_index, image_index]):
                    continue
                count = merged_counts[image_counter]
                if start_embedding is not None:
                    packed[batch_index, cursor] = start_embedding
                    fake_ids[batch_index, cursor] = self.cfg.vision_start_token_id
                    cursor += 1
                packed[batch_index, cursor : cursor + count] = merged[
                    merged_offset : merged_offset + count
                ].to(self.params_dtype)
                fake_ids[batch_index, cursor : cursor + count] = self.cfg.image_token_id
                visual_mask[batch_index, cursor : cursor + count] = True
                cursor += count
                merged_offset += count
                image_counter += 1
                if end_embedding is not None:
                    packed[batch_index, cursor] = end_embedding
                    fake_ids[batch_index, cursor] = self.cfg.vision_end_token_id
                    cursor += 1
            language_length = int(request.lang_lens[batch_index])
            packed[batch_index, cursor : cursor + language_length] = language[
                batch_index, :language_length
            ]
            fake_ids[batch_index, cursor : cursor + language_length] = (
                request.input_ids[
                    batch_index, :language_length
                ].to(device=self.device, dtype=torch.int64)
            )
            cursor += language_length
            packed[
                batch_index,
                cursor : cursor + current_query.shape[1],
            ] = current_query[batch_index]
            cursor += current_query.shape[1]
            packed[
                batch_index,
                cursor : cursor + future_query.shape[1],
            ] = future_query[batch_index]
            cursor += future_query.shape[1]
            prefix_mask[batch_index, :cursor] = True
            real_lens[batch_index] = cursor
        if merged_offset != merged.shape[0] or image_counter != active_grids.shape[0]:
            raise RuntimeError("vision split accounting mismatch.")
        visible_lens = build_expert_visible_prefix_lens(
            real_lens,
            future_query_tokens=self.cfg.dual_query.future_query_token_count,
            block_future_depth_to_action=(
                self.cfg.dual_query.block_future_depth_to_action
            ),
            block_suffix_to_future_video=(
                self.cfg.dual_query.block_suffix_to_future_video
            ),
        )
        return (
            packed[prefix_mask],
            fake_ids,
            prefix_mask,
            visual_mask[prefix_mask],
            real_lens,
            visible_lens,
        )

    def _plan_expert(
        self,
        real_lens: torch.Tensor,
        visible_lens: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        suffix_slots: torch.Tensor,
    ) -> None:
        batch_size = int(real_lens.shape[0])
        state_position_ids, action_position_ids = build_suffix_mrope_position_ids(
            prefix_position_ids,
            prefix_mask,
            self.cfg.chunk_size,
        )
        suffix_slots = suffix_slots.view(batch_size, self.suffix_len)
        state_write = suffix_slots[:, 0]
        action_write = suffix_slots[:, 1:].flatten()

        state_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=self.device
        )
        state_indptr[1:] = torch.cumsum(visible_lens + 1, dim=0)
        action_indptr = torch.zeros_like(state_indptr)
        action_indptr[1:] = torch.cumsum(visible_lens + self.suffix_len, dim=0)
        state_metadata = DiffusionAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=batch_size,
            num_query_tokens=batch_size,
            cu_seqlens_q=torch.arange(
                batch_size + 1,
                dtype=torch.int32,
                device=self.device,
            ),
            paged_kv_indptr=state_indptr,
            paged_kv_indices=build_state_paged_kv_indices(
                real_lens,
                visible_lens,
                suffix_len=self.suffix_len,
                prefix_slot_base=self.prefix_runner.prefix_base,
                suffix_slot_base=self.prefix_runner.suffix_base,
            ),
            paged_kv_last_page_len=torch.ones(
                batch_size, dtype=torch.int32, device=self.device
            ),
            write_indices=state_write,
            position_ids=state_position_ids,
        )
        action_metadata = DiffusionAttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=batch_size,
            num_query_tokens=batch_size * self.cfg.chunk_size,
            cu_seqlens_q=torch.arange(
                0,
                (batch_size + 1) * self.cfg.chunk_size,
                self.cfg.chunk_size,
                dtype=torch.int32,
                device=self.device,
            ),
            paged_kv_indptr=action_indptr,
            paged_kv_indices=build_action_paged_kv_indices(
                real_lens,
                visible_lens,
                suffix_len=self.suffix_len,
                prefix_slot_base=self.prefix_runner.prefix_base,
                suffix_slot_base=self.prefix_runner.suffix_base,
            ),
            paged_kv_last_page_len=torch.ones(
                batch_size, dtype=torch.int32, device=self.device
            ),
            write_indices=action_write,
            position_ids=action_position_ids,
        )
        self.expert_runner.plan_inference(state_metadata, action_metadata)

    def _validate(self, request: LingBotV2Request) -> None:
        if request.pixel_values.dim() != 4:
            raise ValueError("pixel_values must be (B, N, P, patch_vector_dim).")
        batch_size, num_images, max_patches, patch_dim = request.pixel_values.shape
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(
                f"batch size {batch_size} outside [1, {self.max_batch_size}]."
            )
        if num_images != self.num_images:
            raise ValueError(
                f"request has {num_images} images, expected {self.num_images}."
            )
        if patch_dim != self.cfg.vision.patch_vector_dim:
            raise ValueError(
                f"patch vector dim {patch_dim} != {self.cfg.vision.patch_vector_dim}."
            )
        if request.image_grid_thw.shape != (batch_size, num_images, 3):
            raise ValueError("image_grid_thw must be (B, N, 3).")
        if request.image_masks.shape != (batch_size, num_images):
            raise ValueError("image_masks must be (B, N).")
        active_patch_counts = request.image_grid_thw.prod(dim=-1)
        if bool((active_patch_counts[request.image_masks.bool()] > max_patches).any()):
            raise ValueError("pixel_values P is smaller than an active grid.")
        if request.input_ids.shape != (
            batch_size,
            self.cfg.tokenizer_max_length,
        ):
            raise ValueError("input_ids has the wrong fixed tokenizer length.")
        if request.lang_lens.shape != (batch_size,):
            raise ValueError("lang_lens must be (B,).")
        if bool(
            (
                (request.lang_lens < 0)
                | (request.lang_lens > self.cfg.tokenizer_max_length)
            ).any()
        ):
            raise ValueError("lang_lens contains an out-of-range value.")
        if request.state.shape != (batch_size, self.cfg.max_state_dim):
            raise ValueError("state must be (B, max_state_dim).")
        if request.noise is not None and request.noise.shape != (
            batch_size,
            self.cfg.chunk_size,
            self.cfg.max_action_dim,
        ):
            raise ValueError("noise must be (B, chunk_size, max_action_dim).")


__all__ = [
    "LingBotV2Request",
    "LingBotV2WS1Scheduler",
    "build_action_paged_kv_indices",
    "build_prefix_padded_write_indices",
    "build_state_paged_kv_indices",
    "build_suffix_mrope_position_ids",
]
