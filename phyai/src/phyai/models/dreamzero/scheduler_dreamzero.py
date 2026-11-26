"""DreamZero latent-level scheduler.

This scheduler starts at the model-ready tensor boundary: text/image/VAE
encoding is handled by the pipeline. The scheduler owns DreamZero DiT runners,
resets them when an attention window ends, performs optional clean
context KV prefill, and runs the flow denoise loop through the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

import phyai.parallel as P
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroConfig
from phyai.models.dreamzero.model_runner_dreamzero import (
    DreamZeroDiTForwardBatch,
    DreamZeroDiTRunner,
)
from phyai.models.dreamzero.modeling_dreamzero import DreamZeroDiT
from phyai.runtime.schedule import Scheduler


@dataclass
class DreamZeroRequest:
    """One model-ready DreamZero DiT inference request.

    Tensors are already encoded and placed on the desired device. Video latents
    use DreamZero DiT's `(B, C, T, H, W)` layout. `context` is the conditional
    text/image-context tensor before the DiT's internal context projections.
    """

    video: torch.Tensor
    action: torch.Tensor
    state: torch.Tensor
    context: torch.Tensor
    embodiment_id: torch.Tensor | None = None
    clip_feature: torch.Tensor | None = None
    y: torch.Tensor | None = None
    clean_video: torch.Tensor | None = None
    reference_video: torch.Tensor | None = None
    uncond_context: torch.Tensor | None = None
    uncond_clip_feature: torch.Tensor | None = None
    current_start_frame: int | None = None
    image_context_tokens: int = 257
    num_inference_steps: int | None = None
    dynamic_dit: bool = False
    dynamic_dit_scheduler_steps: int = 16
    guidance_scale: float | None = None
    sigma_shift: float | None = None
    decouple_inference_noise: bool | None = None
    video_inference_final_noise: float | None = None
    update_kv_cache: bool = False
    prefill_clean_cache: bool | None = None
    reset_kv_cache: bool | None = None


@dataclass
class DreamZeroSchedulerOutput:
    video: torch.Tensor
    action: torch.Tensor
    last_video_pred: torch.Tensor | None
    last_action_pred: torch.Tensor | None
    cond_kv_cache: list[torch.Tensor | None]
    uncond_kv_cache: list[torch.Tensor | None] | None
    current_start_frame: int = 0
    dit_compute_steps: int = 0
    scheduler_steps: int = 0


class DreamZeroFlowStepper:
    """DreamZero Flow UniPC multistep scheduler.

    Flow prediction uses x0 conversion, BH2 updates, and lower-order final
    steps. The default solver order is two. Video and action trajectories can
    use different final sigmas while sharing the same update equations.
    """

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        solver_order: int = 2,
        compile_updates: bool = False,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.solver_order = int(solver_order)
        if self.num_train_timesteps <= 0 or self.shift <= 0:
            raise ValueError("num_train_timesteps and shift must be positive.")
        if self.solver_order <= 0:
            raise ValueError(f"solver_order must be positive, got {solver_order}.")
        alphas = torch.linspace(
            1.0 / self.num_train_timesteps,
            1.0,
            self.num_train_timesteps,
            dtype=torch.float32,
        )
        train_sigmas = 1.0 - alphas
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.sigma_min = train_sigmas[-1].item()
        self.sigma_max = train_sigmas[0].item()
        self.model_outputs: list[torch.Tensor | None] = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample: torch.Tensor | None = None
        self.this_order = 1
        if compile_updates:
            # Official UniPC fuses these updates. In BF16, eager intermediate
            # rounding changes subsequent denoising inputs and predictions.
            self._multistep_uni_p_bh_update = torch.compile(
                self._multistep_uni_p_bh_update, fullgraph=True, dynamic=False
            )
            self._multistep_uni_c_bh_update = torch.compile(
                self._multistep_uni_c_bh_update, fullgraph=True, dynamic=False
            )

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str,
        final_sigma: float = 0.0,
        shift: float | None = None,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {num_inference_steps}."
            )
        sigmas = np.linspace(
            self.sigma_max,
            self.sigma_min,
            num_inference_steps + 1,
        ).copy()[:-1]
        sigma_shift = self.shift if shift is None else float(shift)
        sigmas = sigma_shift * sigmas / (1 + (sigma_shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        sigmas_tensor = torch.from_numpy(sigmas).to(device=device)
        if final_sigma:
            sigma_max = sigmas_tensor[0]
            sigmas_tensor = (
                sigmas_tensor * (sigma_max - final_sigma) / sigma_max + final_sigma
            )
        self.sigmas = sigmas_tensor
        if final_sigma:
            self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).to(
                torch.int64
            )
        else:
            self.timesteps = torch.from_numpy(timesteps).to(
                device=device, dtype=torch.int64
            )
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 1

    @staticmethod
    def _sigma_to_alpha_sigma_t(
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma

    def _multistep_uni_p_bh_update(
        self,
        *,
        sample: torch.Tensor,
        order: int,
        sigma_t: torch.Tensor,
        sigma_s0: torch.Tensor,
        sigma_history: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        if m0 is None:
            raise RuntimeError("missing current model output for UniPC update.")
        x = sample

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(sigma_history[i - 1])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(torch.ones_like(sigma_s0))
        rks = torch.stack(rks, dim=0)

        R = []
        b = []

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        B_h = torch.expm1(hh)

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R, dim=0)
        b = torch.stack(b, dim=0)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.full((1,), 0.5, dtype=x.dtype, device=x.device)
            else:
                rhos_p = torch.linalg.solve_ex(R[:-1, :-1], b[:-1])[0].to(x.dtype)
        else:
            D1s = None
            rhos_p = None

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
        else:
            pred_res = 0
        x_t = x_t_ - alpha_t * B_h * pred_res

        return x_t.to(x.dtype)

    def _multistep_uni_c_bh_update(
        self,
        *,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
        sigma_t: torch.Tensor,
        sigma_s0: torch.Tensor,
        sigma_history: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        if m0 is None:
            raise RuntimeError("missing previous model output for UniPC correction.")
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(sigma_history[i - 1])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)
        rks.append(torch.ones_like(sigma_s0))
        rks = torch.stack(rks, dim=0)

        R = []
        b = []

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        B_h = torch.expm1(hh)

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R, dim=0)
        b = torch.stack(b, dim=0)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = torch.full((1,), 0.5, dtype=x.dtype, device=x.device)
        else:
            rhos_c = torch.linalg.solve_ex(R, b)[0].to(x.dtype)

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0
        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t.to(x.dtype)

    def step(
        self,
        *,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        if self.sigmas is None or self.timesteps is None:
            raise RuntimeError("call set_timesteps() before step().")
        use_corrector = step_index > 0 and self.last_sample is not None

        converted = sample - self.sigmas[step_index] * model_output
        if use_corrector:
            # Index outside the compiled updates so each step reuses the same
            # graphs. The corrector's history precedes the previous step.
            sample = self._multistep_uni_c_bh_update(
                this_model_output=converted,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                sigma_t=self.sigmas[step_index],
                sigma_s0=self.sigmas[step_index - 1],
                sigma_history=tuple(
                    self.sigmas[step_index - i - 1] for i in range(1, self.this_order)
                ),
            ).clone()

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = converted

        this_order = min(self.solver_order, len(self.timesteps) - step_index)
        self.this_order = min(this_order, self.lower_order_nums + 1)
        if self.this_order <= 0:
            raise RuntimeError("UniPC order must be positive.")

        self.last_sample = sample
        prev_sample = self._multistep_uni_p_bh_update(
            sample=sample,
            order=self.this_order,
            sigma_t=self.sigmas[step_index + 1],
            sigma_s0=self.sigmas[step_index],
            sigma_history=tuple(
                self.sigmas[step_index - i] for i in range(1, self.this_order)
            ),
        ).clone()
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        return prev_sample


class DreamZeroScheduler(Scheduler):
    """DreamZero DiT denoise scheduler with TP and optional CFG parallelism."""

    def __init__(
        self,
        model: DreamZeroDiT,
        *,
        device: torch.device | str | None = None,
        use_cfg_runner: bool = True,
        compile_updates: bool = False,
    ) -> None:
        self.model = model
        self.compile_updates = compile_updates
        self.cfg: DreamZeroConfig = model.config
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        mesh = P.default_mesh()
        self.cfg_rank = mesh.group_rank("cfg")
        self.cfg_size = mesh.group_size("cfg")
        max_attention_size = model.blocks[0].self_attn.max_attention_size
        max_kv_cache_tokens = max_attention_size if max_attention_size > 0 else None
        self.cond_runner = DreamZeroDiTRunner(
            model,
            max_kv_cache_tokens=max_kv_cache_tokens,
        )
        self.uncond_runner = (
            DreamZeroDiTRunner(
                model,
                max_kv_cache_tokens=max_kv_cache_tokens,
            )
            if use_cfg_runner or self.cfg_size > 1
            else None
        )
        self.current_start_frame = 0
        self._ready = False

    def setup(self) -> None:
        self.cond_runner.setup()
        if self.uncond_runner is not None:
            self.uncond_runner.setup()
        self._ready = True

    def reset_sequence(self) -> None:
        self.cond_runner.reset()
        if self.uncond_runner is not None:
            self.uncond_runner.reset()
        self.current_start_frame = 0

    def close(self) -> None:
        self.cond_runner.close()
        if self.uncond_runner is not None:
            self.uncond_runner.close()
        self.current_start_frame = 0
        self._ready = False

    @property
    def local_attn_size(self) -> int:
        return self.model.blocks[0].self_attn.local_attn_size

    @staticmethod
    def _should_run_dynamic_dit(
        previous_video_predictions: list[torch.Tensor],
        skip_countdown: int,
    ) -> tuple[bool, int]:
        """Match DreamZero's cosine-similarity dynamic DiT schedule."""
        if len(previous_video_predictions) < 2:
            return True, skip_countdown
        if skip_countdown > 1:
            return False, skip_countdown - 1
        if skip_countdown == 1:
            return True, 0

        last = previous_video_predictions[-1].flatten(1).float()
        previous = previous_video_predictions[-2].flatten(1).float()
        similarity = torch.nn.functional.cosine_similarity(last, previous, dim=1).mean()
        for threshold, countdown in ((0.95, 4), (0.93, 2)):
            if similarity > threshold:
                return False, countdown
        return True, 0

    @staticmethod
    def _slice_temporal_condition(
        condition: torch.Tensor | None,
        *,
        start: int,
        length: int,
    ) -> torch.Tensor | None:
        if condition is None:
            return None
        if length <= 0:
            raise ValueError(f"condition length must be positive, got {length}.")
        total = int(condition.shape[2])
        if total == length:
            return condition
        if total <= 0:
            raise ValueError("condition tensor must have a non-empty temporal axis.")
        if start + length <= total:
            return condition[:, :, start : start + length]
        return condition[:, :, max(0, total - length) :]

    def _cfg_parallel_branch(
        self,
        *,
        request: DreamZeroRequest,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        use_cfg: bool,
    ) -> tuple[DreamZeroDiTRunner, torch.Tensor, torch.Tensor | None]:
        if self.cfg_size == 1 or not use_cfg:
            return self.cond_runner, context, clip_feature
        if self.cfg_size != 2:
            raise ValueError(
                "DreamZero CFG parallelism currently supports cfg_size=2; "
                f"got cfg_size={self.cfg_size}."
            )
        if request.uncond_context is None:
            raise ValueError("CFG parallelism requires uncond_context.")
        if self.cfg_rank == 0:
            return self.cond_runner, context, clip_feature
        if self.uncond_runner is None:
            raise RuntimeError("CFG requested but scheduler has no uncond runner.")
        return self.uncond_runner, request.uncond_context, request.uncond_clip_feature

    def _combine_cfg_parallel(
        self,
        *,
        video_local: torch.Tensor,
        action_local: torch.Tensor,
        guidance_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg_size != 2:
            raise ValueError(
                "DreamZero CFG parallel combine requires cfg_size=2; "
                f"got cfg_size={self.cfg_size}."
            )
        # all_gather over cfg is rank ordered: rank 0 is cond, rank 1 is uncond.
        video_pair = P.all_gather(video_local.unsqueeze(0), group="cfg", dim=0)
        action_pair = P.all_gather(action_local.unsqueeze(0), group="cfg", dim=0)
        video_pred = video_pair[1] + guidance_scale * (video_pair[0] - video_pair[1])
        action_pred = action_pair[0]
        return video_pred, action_pred

    def _prefill_cache(
        self,
        runner: DreamZeroDiTRunner,
        *,
        video: torch.Tensor,
        request: DreamZeroRequest,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        start_frame: int,
        use_crossattn_cache: bool,
    ) -> None:
        runner.forward(
            DreamZeroDiTForwardBatch(
                x=video,
                timestep=torch.zeros(
                    video.shape[0],
                    video.shape[2],
                    dtype=torch.int64,
                    device=video.device,
                ),
                context=context,
                current_start_frame=start_frame,
                y=self._slice_temporal_condition(
                    request.y,
                    start=start_frame,
                    length=video.shape[2],
                ),
                clip_feature=clip_feature,
                image_context_tokens=request.image_context_tokens,
                update_kv_cache=True,
                use_crossattn_cache=use_crossattn_cache,
                update_crossattn_cache=not use_crossattn_cache,
            )
        )

    @torch.no_grad()
    def step(self, request: DreamZeroRequest) -> DreamZeroSchedulerOutput:
        if not self._ready:
            raise RuntimeError("call setup() before step().")

        requested_start_frame = (
            self.current_start_frame
            if request.current_start_frame is None
            else int(request.current_start_frame)
        )
        local_attn_size = self.local_attn_size
        reset_kv_cache = (
            requested_start_frame == 0
            if request.reset_kv_cache is None
            else bool(request.reset_kv_cache)
        )
        if local_attn_size != -1 and requested_start_frame >= local_attn_size:
            reset_kv_cache = True
            requested_start_frame = 0
        if reset_kv_cache:
            self.reset_sequence()
            requested_start_frame = 0
        else:
            self.current_start_frame = requested_start_frame

        def on_device(value: torch.Tensor | None) -> torch.Tensor | None:
            return value.to(self.device) if value is not None else None

        request = replace(
            request,
            video=request.video.to(self.device),
            action=request.action.to(self.device),
            state=request.state.to(self.device),
            context=request.context.to(self.device),
            embodiment_id=on_device(request.embodiment_id),
            clip_feature=on_device(request.clip_feature),
            y=on_device(request.y),
            clean_video=on_device(request.clean_video),
            reference_video=on_device(request.reference_video),
            uncond_context=on_device(request.uncond_context),
            uncond_clip_feature=on_device(request.uncond_clip_feature),
            current_start_frame=requested_start_frame,
        )
        video = request.video
        action = request.action
        state = request.state
        embodiment_id = request.embodiment_id
        context = request.context
        clip_feature = request.clip_feature
        y = request.y
        reference_video = request.reference_video

        guidance_scale = (
            self.cfg.cfg_scale
            if request.guidance_scale is None
            else float(request.guidance_scale)
        )
        use_cfg = request.uncond_context is not None and guidance_scale != 1.0
        if use_cfg and self.uncond_runner is None:
            raise RuntimeError("CFG requested but scheduler has no uncond runner.")
        cfg_parallel = self.cfg_size > 1
        branch_runner, branch_context, branch_clip = self._cfg_parallel_branch(
            request=request,
            context=context,
            clip_feature=clip_feature,
            use_cfg=use_cfg,
        )

        prefill_clean_cache = (
            requested_start_frame == 0 and request.clean_video is not None
            if request.prefill_clean_cache is None
            else bool(request.prefill_clean_cache)
        )
        if prefill_clean_cache:
            if request.clean_video is None:
                raise ValueError("prefill_clean_cache requires clean_video.")
            self._prefill_cache(
                branch_runner,
                video=request.clean_video,
                request=request,
                context=branch_context,
                clip_feature=branch_clip,
                start_frame=0,
                use_crossattn_cache=False,
            )
            if use_cfg and not cfg_parallel:
                assert self.uncond_runner is not None
                self._prefill_cache(
                    self.uncond_runner,
                    video=request.clean_video,
                    request=request,
                    context=request.uncond_context,
                    clip_feature=request.uncond_clip_feature,
                    start_frame=0,
                    use_crossattn_cache=False,
                )
            if requested_start_frame == 0:
                requested_start_frame = int(request.clean_video.shape[2])
                self.current_start_frame = requested_start_frame

        if reference_video is not None and requested_start_frame != 1:
            reference_start_frame = max(
                0, requested_start_frame - reference_video.shape[2]
            )
            self._prefill_cache(
                branch_runner,
                video=reference_video,
                request=request,
                context=branch_context,
                clip_feature=branch_clip,
                start_frame=reference_start_frame,
                use_crossattn_cache=True,
            )
            if use_cfg and not cfg_parallel:
                assert self.uncond_runner is not None
                self._prefill_cache(
                    self.uncond_runner,
                    video=reference_video,
                    request=request,
                    context=request.uncond_context,
                    clip_feature=request.uncond_clip_feature,
                    start_frame=reference_start_frame,
                    use_crossattn_cache=True,
                )

        denoise_y = self._slice_temporal_condition(
            y,
            start=requested_start_frame,
            length=video.shape[2],
        )

        steps = (
            self.cfg.num_inference_timesteps
            if request.num_inference_steps is None
            else request.num_inference_steps
        )
        if request.dynamic_dit:
            steps = request.dynamic_dit_scheduler_steps
        if request.dynamic_dit and steps < 2:
            raise ValueError(
                f"dynamic_dit_scheduler_steps must be at least 2, got {steps}."
            )
        sigma_shift = (
            self.cfg.sigma_shift if request.sigma_shift is None else request.sigma_shift
        )
        video_stepper = DreamZeroFlowStepper(
            num_train_timesteps=self.cfg.num_timestep_buckets,
            shift=sigma_shift,
            compile_updates=self.compile_updates,
        )
        action_stepper = DreamZeroFlowStepper(
            num_train_timesteps=self.cfg.num_timestep_buckets,
            shift=sigma_shift,
            compile_updates=self.compile_updates,
        )
        final_sigma = 0.0
        decouple = (
            self.cfg.decouple_inference_noise
            if request.decouple_inference_noise is None
            else bool(request.decouple_inference_noise)
        )
        if decouple:
            final_sigma = (
                self.cfg.video_inference_final_noise
                if request.video_inference_final_noise is None
                else float(request.video_inference_final_noise)
            )
        video_stepper.set_timesteps(
            steps,
            device=self.device,
            final_sigma=final_sigma,
        )
        action_stepper.set_timesteps(steps, device=self.device)
        assert video_stepper.timesteps is not None
        assert action_stepper.timesteps is not None

        last_video_pred: torch.Tensor | None = None
        last_action_pred: torch.Tensor | None = None
        previous_video_predictions: list[torch.Tensor] = []
        skip_countdown = 0
        dit_compute_steps = 0
        for step_index, video_timestep in enumerate(video_stepper.timesteps):
            action_timestep = action_stepper.timesteps[step_index]
            timestep = video_timestep.to(torch.int64).expand(
                video.shape[0], video.shape[2]
            )
            timestep_action = action_timestep.to(torch.int64).expand(
                action.shape[0], action.shape[1]
            )
            should_run_dit = True
            if request.dynamic_dit:
                should_run_dit, skip_countdown = self._should_run_dynamic_dit(
                    previous_video_predictions,
                    skip_countdown,
                )
            if not should_run_dit:
                if last_video_pred is None or last_action_pred is None:
                    raise RuntimeError(
                        "dynamic DiT skipped before a prediction was available."
                    )
                video_pred = last_video_pred
                action_pred = last_action_pred
            else:
                local = branch_runner.forward(
                    DreamZeroDiTForwardBatch(
                        x=video,
                        timestep=timestep,
                        context=branch_context,
                        current_start_frame=requested_start_frame,
                        y=denoise_y,
                        clip_feature=branch_clip,
                        action=action,
                        timestep_action=timestep_action,
                        state=state,
                        embodiment_id=embodiment_id,
                        image_context_tokens=request.image_context_tokens,
                        update_kv_cache=request.update_kv_cache,
                        use_crossattn_cache=True,
                        update_crossattn_cache=step_index == 0,
                    )
                )
                if local.action is None:
                    raise RuntimeError(
                        "DreamZero local CFG forward returned no action."
                    )
                if use_cfg and cfg_parallel:
                    video_pred, action_pred = self._combine_cfg_parallel(
                        video_local=local.video,
                        action_local=local.action,
                        guidance_scale=guidance_scale,
                    )
                else:
                    video_pred = local.video
                    action_pred = local.action
            if should_run_dit and use_cfg and not cfg_parallel:
                assert self.uncond_runner is not None
                uncond = self.uncond_runner.forward(
                    DreamZeroDiTForwardBatch(
                        x=video,
                        timestep=timestep,
                        context=request.uncond_context,
                        current_start_frame=requested_start_frame,
                        y=denoise_y,
                        clip_feature=request.uncond_clip_feature,
                        action=action,
                        timestep_action=timestep_action,
                        state=state,
                        embodiment_id=embodiment_id,
                        image_context_tokens=request.image_context_tokens,
                        update_kv_cache=request.update_kv_cache,
                        use_crossattn_cache=True,
                        update_crossattn_cache=step_index == 0,
                    )
                )
                if uncond.action is None:
                    raise RuntimeError("DreamZero uncond forward returned no action.")
                video_pred = uncond.video + guidance_scale * (
                    local.video - uncond.video
                )
                action_pred = local.action
            if should_run_dit:
                dit_compute_steps += 1
                if request.dynamic_dit:
                    previous_video_predictions.append(video_pred)
                    if len(previous_video_predictions) > 2:
                        previous_video_predictions.pop(0)
            if video_pred.shape != video.shape:
                raise ValueError(
                    "DreamZero video prediction shape must match the denoise "
                    f"sample shape; got pred={tuple(video_pred.shape)} and "
                    f"sample={tuple(video.shape)}."
                )
            video = video_stepper.step(
                model_output=video_pred,
                sample=video,
                step_index=step_index,
            )
            action = action_stepper.step(
                model_output=action_pred,
                sample=action,
                step_index=step_index,
            )
            last_video_pred = video_pred
            last_action_pred = action_pred

        self.current_start_frame = requested_start_frame + int(video.shape[2])

        return DreamZeroSchedulerOutput(
            video=video,
            action=action,
            last_video_pred=last_video_pred,
            last_action_pred=last_action_pred,
            cond_kv_cache=self.cond_runner.kv_cache,
            uncond_kv_cache=self.uncond_runner.kv_cache if use_cfg else None,
            current_start_frame=self.current_start_frame,
            dit_compute_steps=dit_compute_steps,
            scheduler_steps=steps,
        )


__all__ = [
    "DreamZeroFlowStepper",
    "DreamZeroRequest",
    "DreamZeroSchedulerOutput",
    "DreamZeroScheduler",
]
