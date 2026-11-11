"""Cached-video action inference and CFG video sampling for VPP2."""

import math

import torch

import phyai.parallel as P
from phyai.models.vpp2.model_runner_vpp2 import (
    ActionForward,
    VPP2ForwardBatch,
    VPP2Runner,
)
from phyai.models.vpp2.requests_vpp2 import (
    VPP2ActionRequest,
    VPP2StageRequest,
    VPP2VideoRequest,
)
from phyai.models.vpp2.sampler_vpp2 import (
    VPP2DiscreteFlowSampler,
    VPP2EulerSampler,
)
from phyai.runtime.schedule import Scheduler


class VPP2Scheduler(Scheduler):
    def __init__(self, runner: VPP2Runner) -> None:
        self.runner = runner
        self.config = runner.config
        self.ready = False
        self.cfg_rank = 0
        self.cfg_size = 1
        self.noise_groups: tuple[str, ...] = ()
        if P.registered_meshes():
            mesh = P.default_mesh()
            self.cfg_rank = mesh.group_rank("cfg")
            self.cfg_size = mesh.group_size("cfg")
            seen: set[tuple[int, ...]] = set()
            groups = []
            for name in ("attention_tp", "dense_tp", "cfg"):
                if name not in mesh.group_names or mesh.group_size(name) == 1:
                    continue
                members = mesh.group_members(name)
                if members not in seen:
                    seen.add(members)
                    groups.append(name)
            self.noise_groups = tuple(groups)
        if self.cfg_size not in (1, 2):
            raise ValueError("VPP2 video CFG supports cfg_size=1 or cfg_size=2.")
        self.action_sampler = VPP2EulerSampler(
            self.config.action_num_train_timesteps, self.config.action_infer_shift
        )
        self.video_sampler = VPP2EulerSampler(
            self.config.video_num_train_timesteps, self.config.video_infer_shift
        )
        self.discrete_video_sampler = VPP2DiscreteFlowSampler(
            self.config.video_num_train_timesteps, self.config.video_infer_shift
        )

    def setup(self) -> None:
        self.runner.setup()
        self.ready = True

    def noise(
        self,
        shape: tuple[int, ...],
        *,
        seed: int | None,
        rand_device: str | torch.device,
        dtype: torch.dtype,
        initial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if initial is None:
            generator = (
                None
                if seed is None
                else torch.Generator(device=rand_device).manual_seed(seed)
            )
            value = torch.randn(
                shape,
                device=rand_device,
                dtype=torch.float32,
                generator=generator,
            )
        else:
            if tuple(initial.shape) != shape:
                raise ValueError(
                    f"Initial noise must have shape {shape}, got {tuple(initial.shape)}."
                )
            value = initial
        value = value.to(device=self.runner.device, dtype=dtype, non_blocking=True)
        for group in self.noise_groups:
            value = P.broadcast(value, group=group, src=0)
        return value

    def video_shape(
        self, condition: torch.Tensor, num_video_frames: int
    ) -> tuple[int, ...]:
        if condition.ndim != 5 or condition.shape[0] != 1 or condition.shape[2] < 1:
            raise ValueError("condition_latents must have shape [1,C,T>=1,H,W].")
        factor = self.config.vae_temporal_downsample_factor
        if num_video_frames <= 0 or (num_video_frames - 1) % factor:
            raise ValueError(
                f"num_video_frames must be positive and equal to {factor}n+1."
            )
        frames = (num_video_frames - 1) // factor + 1
        if condition.shape[2] > frames:
            raise ValueError(
                "Condition has more latent frames than the generated video."
            )
        return (1, condition.shape[1], frames, condition.shape[3], condition.shape[4])

    @torch.no_grad()
    def step(self, request: VPP2StageRequest) -> dict[str, object]:
        if not self.ready:
            raise RuntimeError("Call VPP2Scheduler.setup() before inference.")
        self.runner.reset()
        try:
            if isinstance(request, VPP2ActionRequest):
                return self.infer_action(request)
            if isinstance(request, VPP2VideoRequest):
                return self.infer_video(request)
            raise TypeError(f"Unsupported VPP2 request: {type(request).__name__}.")
        finally:
            self.runner.reset()

    def infer_action(self, request: VPP2ActionRequest) -> dict[str, object]:
        if not self.runner.supports_action:
            raise ValueError("Action inference requires an action checkpoint.")
        if request.text_cfg_scale != 1.0:
            raise ValueError(
                "The released VPP2 action policy supports text_cfg_scale=1 only."
            )
        chunk_size = self.config.action_chunk_size
        if request.action_horizon <= 0 or request.action_horizon % chunk_size:
            raise ValueError(
                "action_horizon must be positive and divisible by action_chunk_size."
            )
        if request.num_video_frames <= 1:
            raise ValueError("Action inference requires more than one video frame.")
        dtype = self.runner.params_dtype
        device = self.runner.device
        condition = request.condition_latents.to(device=device, dtype=dtype)
        video_shape = self.video_shape(condition, request.num_video_frames)
        if video_shape[2] < self.config.action_visible_video_frames:
            raise ValueError(
                "Video is shorter than the configured action-visible frames."
            )
        action_tokens = request.action_horizon // chunk_size
        action_shape = (1, action_tokens, self.config.action.action_dim)
        action = self.noise(
            action_shape,
            seed=request.seed,
            rand_device=request.rand_device,
            dtype=dtype,
            initial=request.initial_action,
        )
        video = self.noise(
            video_shape,
            seed=request.seed if request.video_seed is None else request.video_seed,
            rand_device=request.rand_device,
            dtype=dtype,
            initial=request.initial_video,
        )
        action_schedule = self.action_sampler.schedule(
            request.num_inference_steps,
            device=device,
            dtype=dtype,
            shift=request.sigma_shift,
        )
        full_steps = (
            request.num_inference_steps
            if request.full_video_num_inference_steps is None
            else request.full_video_num_inference_steps
        )
        full_schedule = (
            self.video_sampler.schedule(
                full_steps, device=device, dtype=dtype, shift=request.sigma_shift
            )
            if request.return_full_video
            else None
        )
        capture_video = request.return_one_step_video or request.return_full_video
        video_timestep = torch.full(
            (1,), self.config.video_num_train_timesteps, device=device, dtype=dtype
        )
        video_prediction = self.runner.prepare_policy(
            request.context,
            request.context_mask,
            request.clip_fea,
            video,
            video_timestep,
            condition,
            proprio=request.proprio,
            action_seq_len=action_tokens,
            return_prediction=capture_video,
        )
        action = self.runner.execute_action_rollout(
            self.action_rollout,
            action,
            action_schedule.timesteps,
            action_schedule.deltas,
        )
        result: dict[str, object] = {
            "action": action.reshape(request.action_horizon, -1).to(
                device=request.output_device, dtype=torch.float32
            )
        }
        # The large video K/V tensors are unnecessary for video continuation.
        self.runner.release_video_cache()
        if not capture_video:
            return result
        if video_prediction is None:
            raise RuntimeError("Video prefill did not return the requested prediction.")
        one_step = self.video_sampler.schedule(
            1, device=device, dtype=dtype, shift=request.sigma_shift
        )
        one_step_video = self.video_sampler.step(
            video_prediction, one_step.deltas[0], video
        )
        result["video_one_step_latents"] = one_step_video
        result["video_latents"] = one_step_video
        result["video_inference_steps"] = 1
        if full_schedule is not None:
            trajectory = self.video_sampler.step(
                video_prediction, full_schedule.deltas[0], video
            )
            for timestep, delta in zip(
                full_schedule.timesteps[1:].unbind(),
                full_schedule.deltas[1:].unbind(),
                strict=True,
            ):
                prediction = self.runner.forward(
                    VPP2ForwardBatch(
                        sample=trajectory,
                        timestep=timestep.reshape(1),
                        mode="video",
                        condition_latents=condition,
                    )
                )
                trajectory = self.video_sampler.step(prediction, delta, trajectory)
            result["video_full_latents"] = trajectory
            result["video_full_inference_steps"] = full_steps
        return result

    def action_rollout(
        self,
        action: torch.Tensor,
        timesteps: torch.Tensor,
        deltas: torch.Tensor,
        forward: ActionForward,
    ) -> torch.Tensor:
        for timestep, delta in zip(timesteps.unbind(), deltas.unbind(), strict=True):
            action = self.action_sampler.step(
                forward(action, timestep.reshape(1)), delta, action
            )
        return action

    def video_prediction(
        self,
        video: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        *,
        guidance_scale: float,
        fp32_prediction: bool,
    ) -> torch.Tensor:
        use_cfg = not math.isclose(guidance_scale, 1.0)
        branch = "uncond" if use_cfg and self.cfg_rank == 1 else "cond"
        prediction = self.runner.forward(
            VPP2ForwardBatch(
                sample=video.to(self.runner.params_dtype),
                timestep=timestep.reshape(1),
                mode="video",
                branch=branch,
                condition_latents=condition,
            )
        )
        if fp32_prediction:
            prediction = prediction.float()
        if not use_cfg:
            return prediction
        if self.cfg_size == 2:
            branches = P.all_gather(prediction.unsqueeze(0), group="cfg", dim=0)
            positive, negative = branches.unbind(0)
        else:
            positive = prediction
            negative = self.runner.forward(
                VPP2ForwardBatch(
                    sample=video.to(self.runner.params_dtype),
                    timestep=timestep.reshape(1),
                    mode="video",
                    branch="uncond",
                    condition_latents=condition,
                )
            )
            if fp32_prediction:
                negative = negative.float()
        return negative + (positive - negative) * guidance_scale

    def infer_video(self, request: VPP2VideoRequest) -> dict[str, object]:
        if not math.isfinite(request.text_cfg_scale) or request.text_cfg_scale < 0:
            raise ValueError("text_cfg_scale must be finite and non-negative.")
        if request.scheduler_mode not in ("discrete", "continuous"):
            raise ValueError("scheduler_mode must be 'discrete' or 'continuous'.")
        use_cfg = not math.isclose(request.text_cfg_scale, 1.0)
        if use_cfg and request.negative_context is None:
            raise ValueError("Video CFG requires negative_context.")
        dtype = self.runner.params_dtype
        device = self.runner.device
        condition = request.condition_latents.to(device=device, dtype=dtype)
        shape = self.video_shape(condition, request.num_video_frames)
        discrete = request.scheduler_mode == "discrete"
        video = self.noise(
            shape,
            seed=request.seed,
            rand_device=device if request.rand_device is None else request.rand_device,
            dtype=torch.float32 if discrete else dtype,
            initial=request.initial_video,
        )
        if self.cfg_size == 1 or not use_cfg or self.cfg_rank == 0:
            self.runner.prepare_condition(
                request.context, request.context_mask, request.clip_fea
            )
        if use_cfg and (self.cfg_size == 1 or self.cfg_rank == 1):
            assert request.negative_context is not None
            self.runner.prepare_condition(
                request.negative_context,
                request.negative_context_mask,
                request.clip_fea,
                branch="uncond",
            )
        if discrete:
            timesteps = self.discrete_video_sampler.schedule(
                request.num_inference_steps, device=device, shift=request.sigma_shift
            )
            for index, timestep in enumerate(timesteps.unbind()):
                prediction = self.video_prediction(
                    video,
                    timestep,
                    condition,
                    guidance_scale=request.text_cfg_scale,
                    fp32_prediction=True,
                )
                following = timesteps[index + 1] if index + 1 < len(timesteps) else 0
                video = self.discrete_video_sampler.step(
                    prediction, timestep, following, video
                )
        else:
            schedule = self.video_sampler.schedule(
                request.num_inference_steps,
                device=device,
                dtype=dtype,
                shift=request.sigma_shift,
            )
            for timestep, delta in zip(
                schedule.timesteps.unbind(), schedule.deltas.unbind(), strict=True
            ):
                prediction = self.video_prediction(
                    video,
                    timestep,
                    condition,
                    guidance_scale=request.text_cfg_scale,
                    fp32_prediction=False,
                )
                video = self.video_sampler.step(prediction, delta, video)
        return {
            "video_latents": video,
            "video_inference_steps": request.num_inference_steps,
        }

    def close(self) -> None:
        self.runner.close()
        self.ready = False
