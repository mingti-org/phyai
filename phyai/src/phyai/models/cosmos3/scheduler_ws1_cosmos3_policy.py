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
    supplies rows selected by ``cond_action_indexes`` (or all rows for forward
    dynamics). ``action_start_frame_offset`` aligns action token zero with the
    corresponding video frame. ``raw_action_dim`` is the embodiment's true action
    width (the tail up to ``action_dim`` is zero-padded / sliced off).
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
    cond_action_indexes: tuple[int, ...] | None = None
    action_start_frame_offset: int = 1
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


def _prepare_condition_latents(
    latents: torch.Tensor,
    *,
    video_shape: tuple[int, int, int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Quantize like the VAE output, promote to FP32, and remove latent padding."""
    if latents.ndim != 5:
        raise ValueError(
            f"cond_video_latents must have shape [B,C,T,H,W], got {tuple(latents.shape)}."
        )
    t_lat, h_lat, w_lat = video_shape
    if latents.shape[-2] < h_lat or latents.shape[-1] < w_lat:
        raise ValueError(
            "cond_video_latents spatial shape is smaller than video_shape: "
            f"{tuple(latents.shape[-2:])} vs {(h_lat, w_lat)}."
        )
    return latents.to(device=device, dtype=model_dtype).float()[
        :, :, :t_lat, :h_lat, :w_lat
    ]


class Cosmos3PolicyScheduler(Scheduler):
    """Cosmos3 action/policy solver"""

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        vae: Cosmos3WanVAE | None = None,
        device: torch.device | str | None = None,
        flow_shift: float = 5.0,
        use_karras_sigmas: bool = False,
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

        Returns ``{"action": [B, chunk, raw_action_dim], "video": [B, C, t, h, w]}``
        (video = rollout latents). When ``decode_video`` and a VAE is present, also adds
        ``"pixels": [B, 3, T, H, W]`` in ``[0, 1]``. Video and action share the timestep;
        each is stepped by its own UniPC solver and its clean frames are re-imposed every
        step. Sampler state stays FP32; only transformer inputs are cast to model dtype.

        Batch ``B`` is read from ``request.text_ids.shape[0]`` and applies to every
        per-sample tensor (``cond_video_latents`` / ``cond_action`` / the text pairs);
        the prompt, observation grid, embodiment ``domain_id`` and ``action_chunk`` are
        shared across the batch (``encode_condition`` requires identical real text
        lengths within a batch). ``B=1`` is bit-identical to the single-request path.
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
        # Batch size from the (required) text tensor; every per-sample tensor shares
        # it. B=1 reproduces the single-request path exactly.
        batch = request.text_ids.shape[0]
        domain = torch.full((batch,), request.domain_id, device=dev, dtype=torch.long)

        # Clean (conditioned) vs noised (generated) per mode. ``cond_frame_indexes``
        # overrides the per-mode default (e.g. [0,1] for a video observation).
        if request.cond_frame_indexes is not None:
            video_clean = list(request.cond_frame_indexes)
        elif request.mode == "inverse_dynamics":
            video_clean = list(range(t_lat))
        else:
            video_clean = [0]
        if request.cond_action_indexes is not None:
            action_clean = list(request.cond_action_indexes)
        elif request.mode == "forward_dynamics":
            action_clean = list(range(chunk))
        else:
            action_clean = []
        if any(index < 0 or index >= t_lat for index in video_clean):
            raise ValueError(
                f"condition frame indexes {video_clean} are outside latent T={t_lat}."
            )
        if any(index < 0 or index >= chunk for index in action_clean):
            raise ValueError(
                f"condition action indexes {action_clean} are outside action T={chunk}."
            )

        # Initial noise uses a fresh ``np.random.RandomState(seed)`` per modality
        # (video and action each reseeded with the same seed). The leading batch
        # axis is row-major over the per-sample draw, so each sample gets distinct
        # noise. Kept identical to ``Cosmos3T2VScheduler.step_action`` so the two
        # paths stay comparable.
        seed = int(request.seed)
        video = torch.from_numpy(
            np.random.RandomState(seed)
            .standard_normal((batch, self.latent_channel, t_lat, h_lat, w_lat))
            .astype("float32")
        ).to(dev)
        action = (
            torch.from_numpy(
                np.random.RandomState(seed)
                .standard_normal((batch, chunk, ad))
                .astype("float32")
            )
            .to(dev, dt)
            .float()
        )
        action[:, :, raw:] = 0.0  # zero the pad tail beyond the embodiment's dim

        cond_video = (
            _prepare_condition_latents(
                request.cond_video_latents,
                video_shape=request.video_shape,
                device=dev,
                model_dtype=dt,
            )
            if request.cond_video_latents is not None
            else None
        )
        if cond_video is not None:
            if any(index >= cond_video.shape[2] for index in video_clean):
                raise ValueError(
                    f"condition frame indexes {video_clean} exceed encoded latent "
                    f"T={cond_video.shape[2]}."
                )
            video[:, :, video_clean] = cond_video[:, :, video_clean]
        cond_action = (
            request.cond_action.to(dev, dt).float().clone()
            if request.cond_action is not None
            else None
        )
        if action_clean:
            if cond_action is None:
                raise ValueError(
                    "cond_action is required when action conditioning indexes are set."
                )
            if cond_action.shape != action.shape:
                raise ValueError(
                    f"cond_action shape {tuple(cond_action.shape)} must equal "
                    f"{tuple(action.shape)}."
                )
            cond_action[:, :, raw:] = 0.0
            action[:, action_clean] = cond_action[:, action_clean]

        video_mask = torch.ones(batch, t_lat, dtype=torch.bool, device=dev)
        video_mask[:, video_clean] = False
        action_mask = torch.ones(batch, chunk, dtype=torch.bool, device=dev)
        action_mask[:, action_clean] = False

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
                tval = timestep.to(dev).reshape(1).float()
                model_video = video.to(dt)
                model_action = action.to(dt)
                v_vel, a_vel = self.runner.forward(
                    "cond",
                    model_video,
                    tval,
                    text_ids=text_ids,
                    text_mask=text_mask,
                    video_shape=request.video_shape,
                    fps=request.fps,
                    noisy_frame_mask=video_mask,
                    action_latents=model_action,
                    action_domain_id=domain,
                    action_noisy_mask=action_mask,
                    action_start_frame_offset=request.action_start_frame_offset,
                )
                if do_cfg:
                    vu, au = self.runner.forward(
                        "uncond",
                        model_video,
                        tval,
                        text_ids=neg_ids,
                        text_mask=neg_mask,
                        video_shape=request.video_shape,
                        fps=request.fps,
                        noisy_frame_mask=video_mask,
                        action_latents=model_action,
                        action_domain_id=domain,
                        action_noisy_mask=action_mask,
                        action_start_frame_offset=request.action_start_frame_offset,
                    )
                    v_vel = vu + request.guidance_scale * (v_vel - vu)
                    a_vel = au + request.guidance_scale * (a_vel - au)
                v_vel = v_vel * video_mask[:, None, :, None, None].to(v_vel.dtype)
                a_vel = a_vel * action_mask.unsqueeze(-1).to(a_vel.dtype)
                a_vel[:, :, raw:] = 0.0
                video = uni_v.step(v_vel, timestep, video)
                action = uni_a.step(a_vel, timestep, action)
                if cond_video is not None:
                    video[:, :, video_clean] = cond_video[:, :, video_clean]
                if action_clean:
                    action[:, action_clean] = cond_action[:, action_clean]

        out = {"video": video, "action": action[:, :, :raw]}
        if decode_video:
            if self.vae_runner is None:
                raise RuntimeError(
                    "decode_video=True but the scheduler was built without a VAE."
                )
            pixels = self.vae_runner.decode(video.to(dt))
            out["pixels"] = ((pixels.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        return out


__all__ = ["Cosmos3ActionRequest", "Cosmos3PolicyScheduler"]
