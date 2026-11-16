"""Text conditioning, visual encoding, and ten-step HumanWAM-Goal sampling."""

import math
from collections import OrderedDict
from dataclasses import fields, replace

import numpy as np
import torch
from phyai_utils_tools.models.humanwam_goal.positions_humanwam_goal import (
    humanwam_goal_position_ids,
)

from phyai.runtime.schedule import Scheduler
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE
from phyai.models.humanwam_goal.sampler_humanwam_goal import (
    build_humanwam_goal_schedule,
)
from phyai.models.humanwam_goal.modeling_humanwam_goal import (
    HumanWAMGoalLayout,
    HumanWAMGoalTransformer,
)
from phyai.models.humanwam_goal.requests_humanwam_goal import (
    HumanWAMGoalTextRequest,
    HumanWAMGoalStageRequest,
    HumanWAMGoalEncodeRequest,
    HumanWAMGoalDenoiseRequest,
    HumanWAMGoalObservationRequest,
)
from phyai.models.humanwam_goal.model_runner_humanwam_goal import (
    HumanWAMGoalRunner,
    HumanWAMGoalVAERunner,
    HumanWAMGoalTextRunner,
)
from phyai.models.humanwam_goal.configuration_humanwam_goal import (
    HumanWAMGoalSamplingConfig,
)
from phyai.models.cosmos3.model_runner_rollout_cosmos3 import (
    Cosmos3UniPCState,
)


class HumanWAMGoalScheduler(Scheduler):
    def __init__(
        self,
        transformer: HumanWAMGoalTransformer,
        *,
        device: torch.device | str,
        sampling: HumanWAMGoalSamplingConfig,
        vae: Cosmos3WanVAE | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.sampling = sampling
        self.vlm = HumanWAMGoalTextRunner(transformer, device=device)
        self.dit = HumanWAMGoalRunner(
            transformer,
            device=device,
            use_cuda_graph=use_cuda_graph,
        )
        self.vae = (
            None
            if vae is None
            else HumanWAMGoalVAERunner(
                vae,
                device=device,
                dtype=transformer.proj_in.weight.dtype,
                use_cuda_graph=use_cuda_graph,
            )
        )
        self.schedule = None
        self.noise_cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    def initial_noise(self, shape: tuple[int, ...], seed: int) -> torch.Tensor:
        """Retain bounded CPU draws; each modality still converts to its own dtype."""
        key = (seed, shape)
        cacheable = type(seed) is int
        noise = self.noise_cache.get(key) if cacheable else None
        if noise is None:
            noise = torch.from_numpy(
                np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
            )
            if cacheable:
                self.noise_cache[key] = noise
                if len(self.noise_cache) > 8:
                    self.noise_cache.popitem(last=False)
        if cacheable:
            self.noise_cache.move_to_end(key)
        return noise

    def setup(self) -> None:
        self.vlm.setup()
        self.dit.setup()
        self.dit.stage(self.device)
        if self.vae is not None:
            self.vae.setup()
            self.vae.stage_encoder(self.device)
        self.schedule = build_humanwam_goal_schedule(self.sampling, self.device)

    @torch.inference_mode()
    def step(self, request: HumanWAMGoalStageRequest):
        if isinstance(request, HumanWAMGoalTextRequest):
            self.vlm.stage(self.device)
            result = self.vlm.forward(request.text_ids, request.text_mask)
            if request.offload_after:
                self.vlm.offload()
            return {"text_tokens": result[0][0].shape[1]}
        if isinstance(request, HumanWAMGoalEncodeRequest):
            if self.vae is None:
                raise RuntimeError("Load the VAE before encoding observations")
            return tuple(self.vae.encode(pixels) for pixels in request.pixels)
        if isinstance(request, HumanWAMGoalDenoiseRequest):
            return self.denoise(request)
        if isinstance(request, HumanWAMGoalObservationRequest):
            return self.observation(request)
        raise TypeError(f"Unsupported HumanWAM-Goal request: {type(request).__name__}")

    def observation(self, request: HumanWAMGoalObservationRequest) -> dict[str, object]:
        if self.vae is None or not self.vlm.cached_kv:
            raise RuntimeError("Load the VAE and prefill text before observations")
        if len(request.pixels) != len(request.history_ages_seconds) + 2:
            raise ValueError(
                "Provide one goal, histories, then the current camera frame"
            )
        config = self.dit.transformer.config
        if any(
            not math.isfinite(value) or value <= 0
            for value in (request.video_fps, request.action_fps)
        ):
            raise ValueError("Frame rates must be finite and positive")
        future_frames = config.horizon * request.video_fps / request.action_fps
        future_latents = future_frames / config.temporal_compression_factor
        if not float(future_latents).is_integer() or future_latents < 1:
            raise ValueError(
                "Frame rates and horizon must form complete temporal VAE groups"
            )
        latents = self.vae.encode_frames(request.pixels, self.dit.collective)
        current = latents[-1]
        rollout = torch.cat(
            (
                current,
                current.new_zeros(
                    (1, config.latent_channel, int(future_latents), *current.shape[-2:])
                ),
            ),
            dim=2,
        )
        vision = (*latents[:-1], rollout)
        masks = tuple(
            torch.zeros((1, value.shape[2]), dtype=torch.bool) for value in vision
        )
        masks[-1][:, 1:] = True
        patch = config.latent_patch_size
        grids = tuple(
            (
                value.shape[2],
                (value.shape[3] + patch - 1) // patch,
                (value.shape[4] + patch - 1) // patch,
            )
            for value in vision
        )
        positions = humanwam_goal_position_ids(
            text_tokens=self.vlm.cached_kv[0][0].shape[1],
            grids=grids,
            history_ages_seconds=request.history_ages_seconds,
            action_rows=config.horizon + 1,
            video_fps=request.video_fps,
            action_fps=request.action_fps,
            base_fps=config.base_fps,
            temporal_compression_factor=config.temporal_compression_factor,
            temporal_modality_margin=config.temporal_modality_margin,
        )
        return self.denoise(
            HumanWAMGoalDenoiseRequest(
                vision=vision,
                video_noisy_masks=masks,
                video_roles=(1,) + (0,) * (len(vision) - 1),
                action=request.action,
                action_noisy_mask=request.action_noisy_mask,
                action_valid_mask=request.action_valid_mask,
                action_domains=request.action_domains,
                gen_position_ids=positions,
                seed=request.seed,
            )
        )

    def denoise(self, request: HumanWAMGoalDenoiseRequest) -> dict[str, object]:
        if not self.vlm.cached_kv or self.schedule is None:
            raise RuntimeError("Set up the scheduler and prefill text before denoising")
        plan = self.dit.plan(request)
        prefix = request.action.shape[1] - 1 - plan.layout.action_noisy_rows.numel()
        if prefix > self.sampling.prefix_length:
            raise ValueError(
                "RTC prefix exceeds the checkpoint's inference prefix length"
            )
        vision = tuple(value.to(self.device, torch.float32) for value in request.vision)
        valid = request.action_valid_mask.to(self.device)
        action = request.action.to(
            self.device, self.dit.transformer.proj_in.weight.dtype
        ).masked_fill(~valid, 0)
        clean = (*vision, action)
        masks = tuple(
            mask.to(self.device)[:, None, :, None, None].expand_as(value)
            for value, mask in zip(vision, request.video_noisy_masks, strict=True)
        ) + (request.action_noisy_mask.to(self.device).unsqueeze(-1) & valid,)
        noisy = torch.cat([mask.reshape(1, -1) for mask in masks], dim=1)
        # Each modality uses the same seed; clean rows still consume noise.
        initial_parts = []
        for value, mask in zip(clean, masks, strict=True):
            noise = self.initial_noise(tuple(value.shape[1:]), request.seed).to(
                device=self.device, dtype=value.dtype
            )
            mask = mask.to(value.dtype)
            # Action noise is rounded to the model dtype before joining the
            # FP32 visual state; the solver then accumulates both in FP32.
            initial_parts.append(((1.0 - mask) * value + mask * noise).reshape(1, -1))
        initial = torch.cat(initial_parts, dim=1).float()
        frequencies = self.vlm.frequencies(request.gen_position_ids)
        self.dit.prepare(
            self.vlm.cached_kv,
            frequencies,
            tuple(shape[2:] for shape in plan.shapes[:-1]),
            action.shape[1],
        )
        condition = self.dit.condition_for_plan(plan)
        schedule = self.schedule
        embeddings = self.dit.time_embeddings(schedule.timesteps, plan)
        inputs = {
            "initial": initial,
            "noisy": noisy,
            "cos": frequencies[0],
            "sin": frequencies[1],
            **self.dit.layout_inputs(plan),
        }
        for index, (key, value) in enumerate(condition.cached_kv):
            inputs[f"key_{index}"] = key
            inputs[f"value_{index}"] = value

        def rollout(**buffers: torch.Tensor) -> torch.Tensor:
            local_condition = replace(
                condition,
                cached_kv=[
                    (buffers[f"key_{index}"], buffers[f"value_{index}"])
                    for index in range(len(condition.cached_kv))
                ],
                freqs_gen=(buffers["cos"], buffers["sin"]),
            )
            layout = HumanWAMGoalLayout(
                **{
                    field.name: buffers[field.name]
                    for field in fields(HumanWAMGoalLayout)
                }
            )
            state = Cosmos3UniPCState(schedule)
            current = buffers["initial"]
            for timestep, embedding in zip(
                schedule.timesteps.unbind(), embeddings, strict=True
            ):
                velocity = self.dit.forward(
                    current,
                    timestep,
                    local_condition,
                    plan,
                    layout,
                    time_embedding=embedding,
                )
                current = state.step(velocity * buffers["noisy"], current)
            return current

        final = self.dit.execute_rollout(
            rollout,
            inputs,
            graph_key=(
                plan.shapes,
                plan.patch_counts,
                self.sampling,
                self.dit._contexts.revision,
                condition.attention_preparation,
            ),
        )
        outputs = tuple(
            value.reshape(shape)
            for value, shape in zip(
                final.split(plan.sizes, dim=1), plan.shapes, strict=True
            )
        )
        return {"vision": outputs[:-1], "action": outputs[-1]}

    def close(self) -> None:
        self.dit.close()
        self.vlm.close()
        if self.vae is not None:
            self.vae.close()
        self.schedule = None
        self.noise_cache.clear()
