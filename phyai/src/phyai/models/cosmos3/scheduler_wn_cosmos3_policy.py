"""Cosmos3 world-size-N (tensor-parallel) action / policy orchestrator."""

from __future__ import annotations

import logging

import numpy as np
import torch

import phyai.parallel as P
from phyai.models.cosmos3.model_runner_policy_cosmos3 import Cosmos3ActionRunner
from phyai.models.cosmos3.model_runner_vae_cosmos3 import Cosmos3VAERunner
from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Transformer
from phyai.models.cosmos3.sampler_unipc import UniPCMultistepSampler
from phyai.models.cosmos3.scheduler_ws1_cosmos3_policy import (
    _ACTION_MODES,
    Cosmos3ActionRequest,
)
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE
from phyai.runtime.schedule import Scheduler
from phyai.utils import this_rank_log
from phyai.utils.profile import event_scope


logger = logging.getLogger(__name__)


def _tp_rank_size() -> tuple[int, int]:
    """Current ``(tp_rank, tp_size)`` (``(0, 1)`` when no mesh / not initialised)."""
    try:
        mesh = P.default_mesh()
        return mesh.axis_local_rank("tp"), mesh.axis_size("tp")
    except Exception:
        return 0, 1


def _cfg_rank_size() -> tuple[int, int]:
    """Current ``(cfg_rank, cfg_size)`` on the CFG-parallel axis.

    ``(0, 1)`` when there is no ``cfg`` axis / no mesh. With ``cfg_size == 2`` the
    cond branch runs on ``cfg_rank == 0`` and the uncond branch on ``cfg_rank == 1``.
    """
    try:
        mesh = P.default_mesh()
        return mesh.axis_local_rank("cfg"), mesh.axis_size("cfg")
    except Exception:
        return 0, 1


class Cosmos3PolicyWNScheduler(Scheduler):
    """Tensor-parallel Cosmos3 action/policy solver.

    Mirrors :class:`~phyai.models.cosmos3.scheduler_ws1_cosmos3_policy.Cosmos3PolicyScheduler`
    but every rank holds a TP shard of the transformer; the deterministic joint
    video+action denoise loop is identical across ranks. Video decode (when
    requested) is split across ranks via the TP-aware VAE runner.
    """

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        vae: Cosmos3WanVAE | None = None,
        device: torch.device | str | None = None,
        flow_shift: float = 10.0,
        use_karras_sigmas: bool = True,
        use_cuda_graph: bool = False,
    ) -> None:
        self.transformer = transformer
        self.vae = vae
        if device is None:
            device = next(transformer.parameters()).device
        self.device = torch.device(device)
        self.dtype = next(transformer.parameters()).dtype
        self.latent_channel = transformer.latent_channel_size
        self._flow_shift = flow_shift
        self._use_karras_sigmas = bool(use_karras_sigmas)
        self.tp_rank, self.tp_size = _tp_rank_size()
        self.cfg_rank, self.cfg_size = _cfg_rank_size()
        self.runner = Cosmos3ActionRunner(
            transformer, device=self.device, use_cuda_graph=use_cuda_graph
        )
        self.vae_runner = (
            Cosmos3VAERunner(vae, device=self.device, dtype=self.dtype)
            if vae is not None
            else None
        )
        self._ready = False

    def setup(self) -> None:
        """Warm the runner."""
        self.runner.setup()
        if self.vae_runner is not None:
            self.vae_runner.setup()
        self._ready = True
        this_rank_log(
            logger,
            logging.INFO,
            "Cosmos3 policy scheduler ready (UniPC, tp=%d, cfg=%d).",
            self.tp_size,
            self.cfg_size,
        )

    @torch.no_grad()
    def step(
        self, request: Cosmos3ActionRequest, *, decode_video: bool = False
    ) -> dict[str, torch.Tensor]:
        """Joint video+action denoising for the three action modes.

        Identical loop to the single-card policy scheduler — every rank seeds the
        same ``np.random.RandomState(seed)`` noise and steps the same deterministic
        UniPC solvers over the TP-sharded transformer, so the action / video are
        bit-identical across ranks. Returns ``{"action", "video"[, "pixels"]}`` on
        every rank.
        """
        if not self._ready:
            raise RuntimeError("call setup() before step().")
        if request.mode not in _ACTION_MODES:
            raise ValueError(
                f"mode must be one of {_ACTION_MODES}, got {request.mode!r}."
            )

        if request.cond_video_latents is None and request.cond_video_pixels is not None:
            if self.vae_runner is None:
                raise RuntimeError(
                    "Cosmos3PolicyWNScheduler requires a VAE to encode "
                    "cond_video_pixels, but no VAE was provided at construction."
                )
            import dataclasses

            encoded = self.vae_runner.encode(request.cond_video_pixels)
            request = dataclasses.replace(request, cond_video_latents=encoded)

        dev, dt = self.device, self.dtype
        t_lat, h_lat, w_lat = request.video_shape
        chunk, ad, raw = (
            request.action_chunk,
            request.action_dim,
            request.raw_action_dim,
        )
        batch = request.text_ids.shape[0]
        domain = torch.full((batch,), request.domain_id, device=dev, dtype=torch.long)

        if request.cond_frame_indexes is not None:
            video_clean = list(request.cond_frame_indexes)
        elif request.mode == "inverse_dynamics":
            video_clean = list(range(t_lat))
        else:
            video_clean = [0]
        action_clean = request.mode == "forward_dynamics"

        seed = int(request.seed)
        video = torch.from_numpy(
            np.random.RandomState(seed)
            .standard_normal((batch, self.latent_channel, t_lat, h_lat, w_lat))
            .astype("float32")
        ).to(dev, dt)
        action = torch.from_numpy(
            np.random.RandomState(seed)
            .standard_normal((batch, chunk, ad))
            .astype("float32")
        ).to(dev, dt)
        action[:, :, raw:] = 0.0

        cond_video = (
            request.cond_video_latents.to(dev, dt)
            if request.cond_video_latents is not None
            else None
        )
        if cond_video is not None:
            video[:, :, video_clean] = cond_video[:, :, video_clean]
        cond_action = (
            request.cond_action.to(dev, dt) if request.cond_action is not None else None
        )
        if action_clean and cond_action is not None:
            action = cond_action.clone()
            action[:, :, raw:] = 0.0

        video_mask = torch.ones(batch, t_lat, dtype=torch.bool, device=dev)
        video_mask[:, video_clean] = False
        action_mask = (
            torch.zeros(batch, chunk, dtype=torch.bool, device=dev)
            if action_clean
            else torch.ones(batch, chunk, dtype=torch.bool, device=dev)
        )

        text_ids, text_mask = request.text_ids.to(dev), request.text_mask.to(dev)
        neg_ids, neg_mask = request.neg_text_ids.to(dev), request.neg_text_mask.to(dev)
        do_cfg = request.guidance_scale > 1.0

        uni_v = UniPCMultistepSampler(
            flow_shift=self._flow_shift, use_karras_sigmas=self._use_karras_sigmas
        )
        uni_v.set_timesteps(request.num_inference_steps, device=dev)
        uni_a = UniPCMultistepSampler(
            flow_shift=self._flow_shift, use_karras_sigmas=self._use_karras_sigmas
        )
        uni_a.set_timesteps(request.num_inference_steps, device=dev)

        self.runner.reset()

        # CFG-parallel: cond / uncond branches run concurrently on the two halves
        # of the ``cfg`` axis; this rank computes one branch, an all-gather over
        # ``cfg`` brings both the video and action velocities to every rank, which
        # all compute the identical combine + UniPC steps. Gated on cfg_size>1;
        # otherwise the sequential cond+uncond loop below runs unchanged.
        if self.cfg_size > 1:
            if not do_cfg:
                this_rank_log(
                    logger,
                    logging.WARNING,
                    "cfg_size=%d but guidance_scale=%.3f<=1 (CFG off): the uncond "
                    "branch is redundant — run with cfg_size=1 to save the GPUs.",
                    self.cfg_size,
                    request.guidance_scale,
                )
            if self.cfg_rank == 0:
                br_ids, br_mask, branch = text_ids, text_mask, "cond"
            else:
                br_ids, br_mask, branch = neg_ids, neg_mask, "uncond"
            with event_scope("cosmos3.policy_denoise_loop"):
                for timestep in uni_v.timesteps:
                    tval = timestep.to(dev).reshape(1).to(dt)
                    v_local, a_local = self.runner.forward(
                        branch,
                        video,
                        tval,
                        text_ids=br_ids,
                        text_mask=br_mask,
                        video_shape=request.video_shape,
                        fps=request.fps,
                        noisy_frame_mask=video_mask,
                        action_latents=action,
                        action_domain_id=domain,
                        action_noisy_mask=action_mask,
                    )
                    # all_gather over cfg is rank-ordered -> [cond, uncond] at dim 0.
                    v_pair = P.all_gather(v_local.unsqueeze(0), axis="cfg", dim=0)
                    a_pair = P.all_gather(a_local.unsqueeze(0), axis="cfg", dim=0)
                    v_vel = v_pair[1] + request.guidance_scale * (v_pair[0] - v_pair[1])
                    a_vel = a_pair[1] + request.guidance_scale * (a_pair[0] - a_pair[1])
                    a_vel[:, :, raw:] = 0.0
                    video = uni_v.step(v_vel, timestep, video)
                    action = uni_a.step(a_vel, timestep, action)
                    if cond_video is not None:
                        video[:, :, video_clean] = cond_video[:, :, video_clean]
                    if action_clean and cond_action is not None:
                        action = cond_action.to(action.dtype).clone()
                        action[:, :, raw:] = 0.0
        else:
            with event_scope("cosmos3.policy_denoise_loop"):
                for timestep in uni_v.timesteps:
                    tval = timestep.to(dev).reshape(1).to(dt)
                    v_vel, a_vel = self.runner.forward(
                        "cond",
                        video,
                        tval,
                        text_ids=text_ids,
                        text_mask=text_mask,
                        video_shape=request.video_shape,
                        fps=request.fps,
                        noisy_frame_mask=video_mask,
                        action_latents=action,
                        action_domain_id=domain,
                        action_noisy_mask=action_mask,
                    )
                    if do_cfg:
                        vu, au = self.runner.forward(
                            "uncond",
                            video,
                            tval,
                            text_ids=neg_ids,
                            text_mask=neg_mask,
                            video_shape=request.video_shape,
                            fps=request.fps,
                            noisy_frame_mask=video_mask,
                            action_latents=action,
                            action_domain_id=domain,
                            action_noisy_mask=action_mask,
                        )
                        v_vel = vu + request.guidance_scale * (v_vel - vu)
                        a_vel = au + request.guidance_scale * (a_vel - au)
                    a_vel[:, :, raw:] = 0.0
                    video = uni_v.step(v_vel, timestep, video)
                    action = uni_a.step(a_vel, timestep, action)
                    if cond_video is not None:
                        video[:, :, video_clean] = cond_video[:, :, video_clean]
                    if action_clean and cond_action is not None:
                        action = cond_action.to(action.dtype).clone()
                        action[:, :, raw:] = 0.0

        out = {"video": video, "action": action[:, :, :raw]}
        if decode_video:
            if self.vae_runner is None:
                raise RuntimeError(
                    "decode_video=True but the scheduler was built without a VAE."
                )
            pixels = self.vae_runner.decode(video)
            out["pixels"] = ((pixels.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        return out


__all__ = ["Cosmos3PolicyWNScheduler"]
