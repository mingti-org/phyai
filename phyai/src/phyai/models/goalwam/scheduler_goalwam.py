"""Text conditioning, visual encoding, and ten-step GoalWAM sampling."""

import math
from dataclasses import fields, replace

import numpy as np
import torch
from phyai_utils_tools.models.goalwam.positions_goalwam import goalwam_position_ids

from phyai.runtime.schedule import Scheduler
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE
from phyai.models.goalwam.sampler_goalwam import build_goalwam_schedule
from phyai.models.goalwam.modeling_goalwam import GoalWAMLayout, GoalWAMTransformer
from phyai.models.goalwam.requests_goalwam import (
    GoalWAMTextRequest,
    GoalWAMStageRequest,
    GoalWAMEncodeRequest,
    GoalWAMDenoiseRequest,
    GoalWAMObservationRequest,
)
from phyai.models.goalwam.model_runner_goalwam import (
    GoalWAMRunner,
    GoalWAMVAERunner,
    GoalWAMTextRunner,
)
from phyai.models.goalwam.configuration_goalwam import GoalWAMSamplingConfig
from phyai.models.cosmos3.model_runner_rollout_cosmos3 import (
    Cosmos3UniPCState,
)


class GoalWAMScheduler(Scheduler):
    def __init__(
        self,
        transformer: GoalWAMTransformer,
        *,
        device: torch.device | str,
        sampling: GoalWAMSamplingConfig,
        vae: Cosmos3WanVAE | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.sampling = sampling
        self.vlm = GoalWAMTextRunner(transformer, device=device)
        self.dit = GoalWAMRunner(
            transformer,
            device=device,
            use_cuda_graph=use_cuda_graph,
        )
        self.vae = (
            None
            if vae is None
            else GoalWAMVAERunner(
                vae,
                device=device,
                dtype=transformer.proj_in.weight.dtype,
                use_cuda_graph=use_cuda_graph,
            )
        )
        self.schedule = None

    def setup(self) -> None:
        self.vlm.setup()
        self.dit.setup()
        self.dit.stage(self.device)
        if self.vae is not None:
            self.vae.setup()
            self.vae.stage_encoder(self.device)
        self.schedule = build_goalwam_schedule(self.sampling, self.device)

    @torch.inference_mode()
    def step(self, request: GoalWAMStageRequest):
        if isinstance(request, GoalWAMTextRequest):
            self.vlm.stage(self.device)
            result = self.vlm.forward(request.text_ids, request.text_mask)
            if request.offload_after:
                self.vlm.offload()
            return {"text_tokens": result[0][0].shape[1]}
        if isinstance(request, GoalWAMEncodeRequest):
            if self.vae is None:
                raise RuntimeError("Load the VAE before encoding observations")
            return tuple(self.vae.encode(pixels) for pixels in request.pixels)
        if isinstance(request, GoalWAMDenoiseRequest):
            return self.denoise(request)
        if isinstance(request, GoalWAMObservationRequest):
            return self.observation(request)
        raise TypeError(f"Unsupported GoalWAM request: {type(request).__name__}")

    def observation(self, request: GoalWAMObservationRequest) -> dict[str, object]:
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
        latents = tuple(self.vae.encode(frame) for frame in request.pixels)
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
        positions = goalwam_position_ids(
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
            GoalWAMDenoiseRequest(
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

    def denoise(self, request: GoalWAMDenoiseRequest) -> dict[str, object]:
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
        # Each modality restarts the CPU RNG; clean rows still consume noise.
        initial_parts = []
        for value, mask in zip(clean, masks, strict=True):
            noise = torch.from_numpy(
                np.random.RandomState(request.seed)
                .standard_normal(tuple(value.shape[1:]))
                .astype(np.float32)
            ).to(device=self.device, dtype=value.dtype)
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
            layout = GoalWAMLayout(
                **{field.name: buffers[field.name] for field in fields(GoalWAMLayout)}
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
