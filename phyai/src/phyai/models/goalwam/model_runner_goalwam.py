"""GoalWAM runtime state and graph ownership."""

import hashlib
from collections import OrderedDict
from dataclasses import fields, replace, dataclass
from collections.abc import Callable, Hashable

import torch

from phyai.parallel import default_mesh
from phyai.kernel.types import KernelMode
from phyai.layers.attention.enums import AttnLayout
from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Condition
from phyai.models.goalwam.modeling_goalwam import GoalWAMLayout, GoalWAMTransformer
from phyai.models.goalwam.requests_goalwam import GoalWAMDenoiseRequest
from phyai.models.cosmos3.model_runner_staged_cosmos3 import (
    Cosmos3DiTRunner,
    Cosmos3VLMRunner,
)
from phyai.models.cosmos3.model_runner_vae_optimized_cosmos3 import (
    Cosmos3OptimizedVAERunner,
)


@dataclass(frozen=True)
class GoalWAMPlan:
    shapes: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]
    patch_counts: tuple[int, ...]
    layout: GoalWAMLayout


class GoalWAMVAERunner(Cosmos3OptimizedVAERunner):
    """Cache independent CPU-supplied frames by their complete pixel contents."""

    def __init__(self, *args, max_cached_frames: int = 8, **kwargs):
        if max_cached_frames < 0:
            raise ValueError("The frame cache bound must be nonnegative")
        self.max_cached_frames = max_cached_frames
        self.frames: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        super().__init__(*args, **kwargs)

    def invalidate_cache(self) -> None:
        super().invalidate_cache()
        self.frames.clear()

    def stage_encoder(self, device: torch.device | str) -> None:
        # Encoder movement can invalidate cached latents as well as graphs.
        self.frames.clear()
        super().stage_encoder(device)

    def encode(self, pixels, *, sample=False, generator=None):
        if (
            not self.max_cached_frames
            or pixels.device.type != "cpu"
            or pixels.shape[:3] != (1, 3, 1)
            or sample
        ):
            return super().encode(pixels, sample=sample, generator=generator)
        source = pixels.detach().contiguous()
        key = (
            source.shape,
            source.dtype,
            hashlib.sha256(source.view(torch.uint8).numpy()).digest(),
        )
        latent = self.frames.get(key)
        if latent is None:
            latent = super().encode(source)
            if len(self.frames) >= self.max_cached_frames:
                self.frames.popitem(last=False)
            self.frames[key] = latent
        self.frames.move_to_end(key)
        return latent.clone()

    def _stage_normalization(self, device: torch.device) -> None:
        super()._stage_normalization(device)
        if self.vae.latents_inv_std.device != device:
            self.invalidate_cache()
            self.vae.latents_inv_std = self.vae.latents_inv_std.to(device)


class GoalWAMTextRunner(Cosmos3VLMRunner):
    """Prefill the full text tower, retaining only this Generator rank's KV."""

    @torch.no_grad()
    def forward(
        self, text_ids: torch.Tensor, text_mask: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if text_ids.ndim != 2 or text_ids.shape[0] != 1:
            raise ValueError("GoalWAM text prefill requires a single prompt")
        cached = super().forward(text_ids, text_mask)
        mesh = default_mesh()
        size, rank = mesh.group_size("attention_tp"), mesh.group_rank("attention_tp")
        if size > 1:
            self.cached_kv = [
                (
                    key.chunk(size, dim=2)[rank].contiguous(),
                    value.chunk(size, dim=2)[rank].contiguous(),
                )
                for key, value in cached
            ]
        return self.cached_kv


class GoalWAMRunner(Cosmos3DiTRunner):
    """Reuse Cosmos3 stage residency, attention plans, and graph lifecycle."""

    def __init__(
        self,
        transformer: GoalWAMTransformer,
        *,
        device: torch.device | str,
        use_cuda_graph: bool = True,
    ) -> None:
        super().__init__(transformer, device=device, use_cuda_graph=use_cuda_graph)
        self.model = transformer
        self.graph.use_cuda_graph = use_cuda_graph
        self.time_signature: tuple | None = None
        self.prepared_times: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
        self.selected_context = None
        self.selected_context_signature = None

    def condition_for_plan(self, plan: GoalWAMPlan) -> Cosmos3Condition:
        condition = self.rollout_condition() if self.use_cuda_graph else self.condition
        if condition is None:
            raise RuntimeError("Prepare Generator conditioning before denoising")
        if not self.use_cuda_graph:
            return condition
        key = condition.cached_kv[-1][0]
        count = (
            plan.layout.video_noisy_rows.numel() + plan.layout.action_noisy_rows.numel()
        )
        signature = (count, self._contexts.revision, key.shape, key.dtype, key.device)
        if signature != self.selected_context_signature:
            attention = self.transformer.gen_layers[-1].cross_attention
            full = sum(plan.patch_counts) + self.transformer.config.horizon + 1
            query = torch.empty(
                (1, count, attention.num_local_heads, attention.head_dim),
                device=key.device,
                dtype=key.dtype,
            )
            kv = torch.empty(
                (
                    1,
                    full + key.shape[1],
                    attention.num_local_kv_heads,
                    attention.head_dim,
                ),
                device=key.device,
                dtype=key.dtype,
            )
            self.selected_context = attention.attn._build_default_ctx(
                query, kv, kv, AttnLayout.PADDED_4D, None, None, mode=KernelMode.CAPTURE
            )
            self.selected_context_signature = signature
        return replace(
            condition,
            attention_contexts=(
                *condition.attention_contexts[:-1],
                self.selected_context,
            ),
            attention_preparation=None,
        )

    def plan(self, request: GoalWAMDenoiseRequest) -> GoalWAMPlan:
        config = self.transformer.config
        if not request.vision or not (
            len(request.vision)
            == len(request.video_noisy_masks)
            == len(request.video_roles)
        ):
            raise ValueError("Each visual stream requires a noisy mask and role")
        metadata = (
            *request.video_noisy_masks,
            request.action_noisy_mask,
            request.action_valid_mask,
            request.action_domains,
        )
        if any(value.device.type != "cpu" for value in metadata):
            raise ValueError("Keep layout masks and domain IDs on CPU")
        goals, noisy, counts = [], [], []
        patch = config.latent_patch_size
        for value, mask, role in zip(
            request.vision,
            request.video_noisy_masks,
            request.video_roles,
            strict=True,
        ):
            if value.ndim != 5 or value.shape[:2] != (1, config.latent_channel):
                raise ValueError("Visual streams must be [1,latent_channels,T,H,W]")
            t, h, w = value.shape[2:]
            if min(t, h, w) < 1:
                raise ValueError("Visual dimensions must be positive")
            if mask.shape != (1, t) or mask.dtype != torch.bool:
                raise ValueError("Visual noisy masks must be bool [1,T]")
            if role not in (0, 1) or (role == 1 and mask.any()):
                raise ValueError("Visual roles must be default (0) or clean goal (1)")
            spatial = ((h + patch - 1) // patch) * ((w + patch - 1) // patch)
            counts.append(t * spatial)
            goals.append(torch.full((t * spatial,), role == 1, dtype=torch.bool))
            noisy.append(mask.flatten().repeat_interleave(spatial))
        rows = config.horizon + 1
        if request.action.shape != (1, rows, config.action_dim):
            raise ValueError("Actions must contain one state row and the full horizon")
        for mask, shape in (
            (request.action_noisy_mask, (1, rows)),
            (request.action_valid_mask, request.action.shape),
        ):
            if mask.shape != shape or mask.dtype != torch.bool:
                raise ValueError(f"Action mask must be bool {shape}")
        action_noisy = request.action_noisy_mask[0]
        if action_noisy[0] or not action_noisy.any():
            raise ValueError("The measured state is clean and future actions are noisy")
        if (action_noisy[:-1] & ~action_noisy[1:]).any():
            raise ValueError("Clean actions must form a contiguous RTC prefix")
        domains = request.action_domains
        if domains.shape != (rows,) or domains.dtype != torch.long:
            raise ValueError("Action domains must be int64 [horizon+1]")
        valid = request.action_valid_mask[0]
        joint_active = valid[:, : config.joint_dim].any(dim=-1)
        if (
            (domains[joint_active] < 0)
            | (domains[joint_active] >= config.num_embodiment_domains)
        ).any():
            raise ValueError("Active joint rows require a valid embodiment domain")
        if valid[:, config.joint_dim + config.shared_dim :].any():
            raise ValueError("Padded action channels must be invalid")
        if request.gen_position_ids.shape != (3, 1, sum(counts) + rows):
            raise ValueError(
                "GEN positions must match the packed visual and action rows"
            )
        action_rows = action_noisy.nonzero().flatten()
        layout = GoalWAMLayout(
            goal_rows=torch.cat(goals),
            video_noisy_rows=torch.cat(noisy).nonzero().flatten(),
            action_noisy_rows=action_rows,
            action_valid=valid,
            domains=domains,
            input_joint_rows=joint_active.nonzero().flatten(),
            output_joint_rows=joint_active[action_rows].nonzero().flatten(),
        )
        shapes = tuple(
            tuple(value.shape) for value in (*request.vision, request.action)
        )
        return GoalWAMPlan(
            shapes=shapes,
            sizes=tuple(value.numel() for value in (*request.vision, request.action)),
            patch_counts=tuple(counts),
            layout=layout,
        )

    def layout_inputs(self, plan: GoalWAMPlan) -> dict[str, torch.Tensor]:
        return {
            field.name: getattr(plan.layout, field.name).to(self.device)
            for field in fields(GoalWAMLayout)
        }

    def time_embeddings(
        self, timesteps: torch.Tensor, plan: GoalWAMPlan
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        counts = (
            plan.layout.video_noisy_rows.numel(),
            plan.layout.action_noisy_rows.numel(),
        )
        signature = (id(timesteps), counts, self.transformer.proj_in.weight.device)
        if signature != self.time_signature:
            dtype = self.transformer.proj_in.weight.dtype
            # Keep separate visual/action GEMM shapes, even for repeated times.
            self.prepared_times = tuple(
                tuple(
                    self.transformer.time_embedder(
                        t.float().reshape(1).expand(count)
                        * self.transformer.timestep_scale
                    ).to(dtype)
                    for count in counts
                )
                for t in timesteps.unbind()
            )
            self.time_signature = signature
        return self.prepared_times

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: Cosmos3Condition,
        plan: GoalWAMPlan,
        layout: GoalWAMLayout,
        *,
        time_embedding: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        values = tuple(
            value.reshape(shape)
            for value, shape in zip(
                latent.split(plan.sizes, dim=1), plan.shapes, strict=True
            )
        )
        patches = torch.cat(
            [
                self.transformer.patchify(value, *value.shape[2:])
                for value in values[:-1]
            ],
            dim=1,
        ).squeeze(0)
        video_velocity, action_velocity = self.model(
            patches,
            values[-1].squeeze(0),
            timestep,
            condition,
            layout,
            time_embedding=time_embedding,
        )
        packed_video = video_velocity.new_zeros(patches.shape).index_copy(
            0, layout.video_noisy_rows, video_velocity
        )
        action = action_velocity.new_zeros(values[-1].shape[1:]).index_copy(
            0, layout.action_noisy_rows, action_velocity
        )
        vision = tuple(
            self.transformer.unpatchify(value.unsqueeze(0), *shape[2:])
            for value, shape in zip(
                packed_video.split(plan.patch_counts), plan.shapes[:-1], strict=True
            )
        )
        # Keep the prediction dtype: UniPC's first scalar product rounds in
        # BF16 before it is subtracted from the FP32 sampler state.
        return torch.cat([value.reshape(1, -1) for value in (*vision, action)], dim=1)

    def execute_rollout(
        self,
        function: Callable[..., torch.Tensor],
        inputs: dict[str, torch.Tensor],
        *,
        graph_key: Hashable,
    ) -> torch.Tensor:
        """Refresh every request buffer, including history positions and RTC values."""
        return self.graph.forward(function, inputs, graph_key=graph_key).clone()

    def close(self) -> None:
        super().close()
        self.time_signature = None
        self.prepared_times = ()
        self.selected_context = None
        self.selected_context_signature = None
