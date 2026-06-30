"""Cosmos3 world-size-N (tensor-parallel) T2V / T2AV denoise orchestrator."""

from __future__ import annotations

import logging

import numpy as np
import torch

import phyai.parallel as P
from phyai.models.cosmos3.avae_sound import Cosmos3AVAESoundDecoder
from phyai.models.cosmos3.model_runner_cosmos3 import Cosmos3T2VRunner
from phyai.models.cosmos3.model_runner_vae_cosmos3 import (
    Cosmos3SoundVAERunner,
    Cosmos3VAERunner,
)
from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Transformer
from phyai.models.cosmos3.sampler_unipc import UniPCMultistepSampler
from phyai.models.cosmos3.scheduler_ws1_cosmos3 import (
    Cosmos3T2VRequest,
    pixel_to_latent_shape,
)
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE
from phyai.runtime.schedule import Scheduler
from phyai.utils import report_progress, this_rank_log
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


class Cosmos3T2VWNScheduler(Scheduler):
    """Tensor-parallel Cosmos3 video [+ sound] denoising orchestrator (UniPC + CFG).

    Mirrors :class:`~phyai.models.cosmos3.scheduler_ws1_cosmos3.Cosmos3T2VScheduler`
    but every rank holds a TP shard of the transformer and the VAE decode is split
    across ranks. The denoise loop is deterministic and identical across ranks.
    """

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        vae: Cosmos3WanVAE | None = None,
        avae: Cosmos3AVAESoundDecoder | None = None,
        device: torch.device | str | None = None,
        flow_shift: float = 10.0,
        use_karras_sigmas: bool = False,
        torch_compile: bool = False,
        compile_kwargs: dict | None = None,
    ) -> None:
        self.transformer = transformer
        self.vae = vae
        self.avae = avae
        if device is None:
            device = next(transformer.parameters()).device
        self.device = torch.device(device)
        self.dtype = next(transformer.parameters()).dtype
        self.latent_channel = transformer.latent_channel_size
        self._flow_shift = flow_shift
        self._use_karras_sigmas = bool(use_karras_sigmas)
        self.tp_rank, self.tp_size = _tp_rank_size()
        self.cfg_rank, self.cfg_size = _cfg_rank_size()
        self.runner = Cosmos3T2VRunner(
            transformer,
            device=self.device,
            torch_compile=torch_compile,
            compile_kwargs=compile_kwargs,
        )
        # The VAE runners auto-route to the TP spatial-tile decode when tp_size > 1.
        self.vae_runner = (
            Cosmos3VAERunner(vae, device=self.device, dtype=self.dtype)
            if vae is not None
            else None
        )
        self.sound_runner = (
            Cosmos3SoundVAERunner(avae, device=self.device, dtype=self.dtype)
            if avae is not None
            else None
        )
        self.unipc: UniPCMultistepSampler | None = None

    def setup(self) -> None:
        """Build the UniPC sampler (no graph capture; plain Python loop)."""
        self.runner.setup()
        if self.vae_runner is not None:
            self.vae_runner.setup()
        if self.sound_runner is not None:
            self.sound_runner.setup()
        self.unipc = UniPCMultistepSampler(
            flow_shift=self._flow_shift, use_karras_sigmas=self._use_karras_sigmas
        )
        this_rank_log(
            logger,
            logging.INFO,
            "Cosmos3 video scheduler ready (UniPC, tp=%d, cfg=%d).",
            self.tp_size,
            self.cfg_size,
        )

    @torch.no_grad()
    def step(
        self, request: Cosmos3T2VRequest
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the full denoise loop (T2V/I2V, or T2AV/I2AV when audio is requested).

        Identical to the single-card scheduler's loop: every rank seeds the same
        ``np.random.RandomState(seed)`` noise and steps the same deterministic
        UniPC solver over the TP-sharded transformer, so the latent is bit-identical
        across ranks. Returns the video latents ``[1, C, t, h, w]`` (or the
        ``{"video", "sound"}`` dict for T2AV) on every rank.
        """
        if self.unipc is None:
            raise RuntimeError("call setup() before step().")
        dev, dt = self.device, self.dtype
        t_lat, h_lat, w_lat = request.video_shape
        with_sound = request.sound_frames is not None

        seed = int(request.seed)
        if request.noise is not None:
            video = request.noise.to(dev, dt)
        else:
            video = torch.from_numpy(
                np.random.RandomState(seed)
                .standard_normal((1, self.latent_channel, t_lat, h_lat, w_lat))
                .astype("float32")
            ).to(dev, dt)
        sound = None
        if with_sound:
            sound = (
                torch.from_numpy(
                    np.random.RandomState(seed)
                    .standard_normal((request.sound_dim, request.sound_frames))
                    .astype("float32")
                )
                .to(dev, dt)
                .transpose(0, 1)
                .unsqueeze(0)
                .contiguous()
            )

        cond_idx = list(request.cond_frame_indexes)
        cond_latents = None
        noisy_frame_mask = None
        if request.cond_latents is not None and cond_idx:
            cond_latents = request.cond_latents.to(dev, dt)
            video[:, :, cond_idx] = cond_latents[:, :, cond_idx]
            noisy_frame_mask = torch.ones(1, t_lat, dtype=torch.bool, device=dev)
            noisy_frame_mask[:, cond_idx] = False

        text_ids = request.text_ids.to(dev)
        text_mask = request.text_mask.to(dev)
        neg_ids = request.neg_text_ids.to(dev)
        neg_mask = request.neg_text_mask.to(dev)
        do_cfg = request.guidance_scale > 1.0
        sound_fps = request.sound_latent_fps if with_sound else None

        self.unipc.set_timesteps(request.num_inference_steps, device=dev)
        uni_s = None
        if with_sound:
            uni_s = UniPCMultistepSampler(
                flow_shift=self._flow_shift, use_karras_sigmas=self._use_karras_sigmas
            )
            uni_s.set_timesteps(request.num_inference_steps, device=dev)

        self.runner.reset()
        scope = "cosmos3.t2av_denoise_loop" if with_sound else "cosmos3.denoise_loop"

        # CFG-parallel: the cond / uncond branches run concurrently on the two
        # halves of the ``cfg`` axis (each a full ``tp`` group). This rank computes
        # only its branch, then a single all-gather over ``cfg`` brings both
        # velocities to every rank, which all compute the identical combine + step.
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
            # This rank's branch: cfg_rank 0 -> cond (real text), 1 -> uncond (neg).
            if self.cfg_rank == 0:
                br_ids, br_mask, branch = text_ids, text_mask, "cond"
            else:
                br_ids, br_mask, branch = neg_ids, neg_mask, "uncond"
            with event_scope(scope):
                total_steps = len(self.unipc.timesteps)
                for i, timestep in enumerate(self.unipc.timesteps):
                    tval = timestep.to(dev).reshape(1).to(dt)
                    out = self.runner.forward(
                        branch,
                        video,
                        tval,
                        text_ids=br_ids,
                        text_mask=br_mask,
                        video_shape=request.video_shape,
                        fps=request.fps,
                        noisy_frame_mask=noisy_frame_mask,
                        sound_latents=sound,
                        sound_fps=sound_fps,
                    )
                    v_local, s_local = out if with_sound else (out, None)
                    # all_gather over cfg is rank-ordered -> [cond, uncond] at dim 0.
                    v_pair = P.all_gather(v_local.unsqueeze(0), axis="cfg", dim=0)
                    v_vel = v_pair[1] + request.guidance_scale * (v_pair[0] - v_pair[1])
                    if with_sound:
                        s_pair = P.all_gather(s_local.unsqueeze(0), axis="cfg", dim=0)
                        s_vel = s_pair[1] + request.guidance_scale * (
                            s_pair[0] - s_pair[1]
                        )
                    video = self.unipc.step(v_vel, timestep, video)
                    if with_sound:
                        sound = uni_s.step(s_vel, timestep, sound)
                    if cond_latents is not None:
                        video[:, :, cond_idx] = cond_latents[:, :, cond_idx]
                    report_progress(i + 1, total_steps, phase="denoise")
            if with_sound:
                return {"video": video, "sound": sound.transpose(1, 2).contiguous()}
            return video

        # No cfg parallel below:
        with event_scope(scope):
            total_steps = len(self.unipc.timesteps)
            for i, timestep in enumerate(self.unipc.timesteps):
                tval = timestep.to(dev).reshape(1).to(dt)
                out_c = self.runner.forward(
                    "cond",
                    video,
                    tval,
                    text_ids=text_ids,
                    text_mask=text_mask,
                    video_shape=request.video_shape,
                    fps=request.fps,
                    noisy_frame_mask=noisy_frame_mask,
                    sound_latents=sound,
                    sound_fps=sound_fps,
                )
                v_cond, s_cond = out_c if with_sound else (out_c, None)
                if do_cfg:
                    out_u = self.runner.forward(
                        "uncond",
                        video,
                        tval,
                        text_ids=neg_ids,
                        text_mask=neg_mask,
                        video_shape=request.video_shape,
                        fps=request.fps,
                        noisy_frame_mask=noisy_frame_mask,
                        sound_latents=sound,
                        sound_fps=sound_fps,
                    )
                    v_unc, s_unc = out_u if with_sound else (out_u, None)
                    v_vel = v_unc + request.guidance_scale * (v_cond - v_unc)
                    if with_sound:
                        s_vel = s_unc + request.guidance_scale * (s_cond - s_unc)
                else:
                    v_vel = v_cond
                    s_vel = s_cond
                video = self.unipc.step(v_vel, timestep, video)
                if with_sound:
                    sound = uni_s.step(s_vel, timestep, sound)
                if cond_latents is not None:
                    video[:, :, cond_idx] = cond_latents[:, :, cond_idx]
                report_progress(i + 1, total_steps, phase="denoise")

        if with_sound:
            return {"video": video, "sound": sound.transpose(1, 2).contiguous()}
        return video

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents -> pixels ``[B, 3, T, H, W]`` in ``[0, 1]`` (TP-split decode)."""
        if self.vae_runner is None:
            raise RuntimeError("Cosmos3T2VWNScheduler was constructed without a VAE.")
        pixels = self.vae_runner.decode(latents)
        return ((pixels.float() + 1.0) / 2.0).clamp(0.0, 1.0)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels ``[B, 3, T, H, W]`` in ``[-1, 1]`` -> normalized latent (needs a VAE)."""
        if self.vae_runner is None:
            raise RuntimeError("Cosmos3T2VWNScheduler was constructed without a VAE.")
        return self.vae_runner.encode(pixels)

    @torch.no_grad()
    def decode_sound(self, sound_latent: torch.Tensor) -> torch.Tensor:
        """Sound latent ``[B, latent_ch, T]`` -> waveform in ``[-1, 1]`` (needs an AVAE)."""
        if self.sound_runner is None:
            raise RuntimeError("Cosmos3T2VWNScheduler was constructed without an AVAE.")
        return self.sound_runner.decode(sound_latent)

    @property
    def sound_sample_rate(self) -> int:
        """Output waveform sample rate (Hz) of the wrapped AVAE (needs an AVAE)."""
        if self.sound_runner is None:
            raise RuntimeError("Cosmos3T2VWNScheduler was constructed without an AVAE.")
        return self.sound_runner.sample_rate


__all__ = ["Cosmos3T2VWNScheduler"]
