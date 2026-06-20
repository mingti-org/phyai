"""Cosmos3 single-card action/policy orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from phyai.models.cosmos3.model_runner_policy_cosmos3 import Cosmos3ActionRunner
from phyai.models.cosmos3.model_runner_vae_cosmos3 import Cosmos3VAERunner
from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Transformer
from phyai.models.cosmos3.sampler_unipc import UniPCMultistepSampler
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE
from phyai.runtime.schedule import Scheduler
from phyai.utils import this_rank_log
from phyai.utils.profile import event_scope


logger = logging.getLogger(__name__)


@dataclass
class Cosmos3ActionRequest:
    """One Cosmos3 action request — policy / forward_dynamics / inverse_dynamics.

    ``mode`` selects what is clean (conditioned) vs noised (generated):

    * ``policy`` — observation frame 0 clean, rest of the video noised; action all
      noised. Produces the action trajectory (+ a rollout video).
    * ``forward_dynamics`` — frame 0 clean + the action all clean (given); video
      noised. Produces the rollout video.
    * ``inverse_dynamics`` — the whole video clean (given); action all noised.
      Recovers the action trajectory.

    ``cond_video_latents`` are VAE-encoded observation latents ``[1, C, t, h, w]``
    (the clean frames are read from it); ``cond_action`` ``[1, chunk, action_dim]``
    is the clean action (forward_dynamics). ``raw_action_dim`` is the embodiment's
    true action width (the tail up to ``action_dim`` is zero-padded / sliced off).
    """

    text_ids: torch.Tensor
    text_mask: torch.Tensor
    neg_text_ids: torch.Tensor
    neg_text_mask: torch.Tensor
    video_shape: tuple[int, int, int]
    mode: str
    domain_id: int
    action_chunk: int
    raw_action_dim: int
    action_dim: int = 64
    cond_video_latents: torch.Tensor | None = None
    cond_video_pixels: torch.Tensor | None = None
    cond_action: torch.Tensor | None = None
    # Clean (conditioned) video latent-frame indices. ``None`` uses the per-mode
    # default (inverse_dynamics = all frames; policy/forward_dynamics = ``[0]``).
    # For a multi-frame VIDEO observation conditioned on its first two latent frames,
    # pass ``(0, 1)``.
    cond_frame_indexes: tuple[int, ...] | None = None
    fps: float = 24.0
    num_inference_steps: int = 30
    guidance_scale: float = 1.0
    seed: int = 42


_ACTION_MODES = ("policy", "forward_dynamics", "inverse_dynamics")


class Cosmos3PolicyScheduler(Scheduler):
    """Cosmos3 action/policy solver"""

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        vae: Cosmos3WanVAE | None = None,
        device: torch.device | str | None = None,
        flow_shift: float = 10.0,
        use_karras_sigmas: bool = True,
        use_cuda_graph: bool = True,
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
        """Warm the runner

        TODO(wch): add cuda graph later
        """
        self.runner.setup()
        if self.vae_runner is not None:
            self.vae_runner.setup()
        self._ready = True
        this_rank_log(
            logger,
            logging.INFO,
            "Cosmos3 policy scheduler ready (UniPC, ws=1, cuda_graph=%s).",
            self.runner.use_cuda_graph,
        )

    @torch.no_grad()
    def step(
        self, request: Cosmos3ActionRequest, *, decode_video: bool = False
    ) -> dict[str, torch.Tensor]:
        """Joint video+action denoising for the three action modes.

        Returns ``{"action": [1, chunk, raw_action_dim], "video": [1, C, t, h, w]}``
        (video = rollout latents). When ``decode_video`` and a VAE is present, also adds
        ``"pixels": [1, 3, T, H, W]`` in ``[0, 1]``. Video and action share the timestep;
        each is stepped by its own UniPC solver and its clean frames are re-imposed every
        step.
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
                    "Cosmos3PolicyScheduler requires a VAE to encode cond_video_pixels, "
                    "but no VAE was provided at construction."
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
        domain = torch.tensor([request.domain_id], device=dev, dtype=torch.long)

        # Clean (conditioned) vs noised (generated) per mode. ``cond_frame_indexes``
        # overrides the per-mode default (e.g. [0,1] for a video observation).
        if request.cond_frame_indexes is not None:
            video_clean = list(request.cond_frame_indexes)
        elif request.mode == "inverse_dynamics":
            video_clean = list(range(t_lat))
        else:
            video_clean = [0]
        action_clean = request.mode == "forward_dynamics"

        # Initial noise uses a fresh ``np.random.RandomState(seed)`` per modality
        # (video and action each reseeded with the same seed). The leading batch-1
        # axis is row-major over the per-sample draw. Kept identical to
        # ``Cosmos3T2VScheduler.step_action`` so the two paths stay comparable.
        seed = int(request.seed)
        video = torch.from_numpy(
            np.random.RandomState(seed)
            .standard_normal((1, self.latent_channel, t_lat, h_lat, w_lat))
            .astype("float32")
        ).to(dev, dt)
        action = torch.from_numpy(
            np.random.RandomState(seed)
            .standard_normal((1, chunk, ad))
            .astype("float32")
        ).to(dev, dt)
        action[:, :, raw:] = 0.0  # zero the pad tail beyond the embodiment's dim

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

        video_mask = torch.ones(1, t_lat, dtype=torch.bool, device=dev)
        video_mask[:, video_clean] = False
        action_mask = (
            torch.zeros(1, chunk, dtype=torch.bool, device=dev)
            if action_clean
            else torch.ones(1, chunk, dtype=torch.bool, device=dev)
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


__all__ = ["Cosmos3ActionRequest", "Cosmos3PolicyScheduler"]
