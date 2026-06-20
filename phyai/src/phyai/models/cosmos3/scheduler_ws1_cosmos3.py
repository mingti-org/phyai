"""Cosmos3 single-card (ws=1) T2V / T2AV denoise orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from phyai.models.cosmos3.avae_sound import Cosmos3AVAESoundDecoder
from phyai.models.cosmos3.model_runner_cosmos3 import Cosmos3T2VRunner
from phyai.models.cosmos3.model_runner_vae_cosmos3 import (
    Cosmos3SoundVAERunner,
    Cosmos3VAERunner,
)
from phyai.models.cosmos3.modeling_cosmos3 import Cosmos3Transformer
from phyai.models.cosmos3.sampler_unipc import UniPCMultistepSampler
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE
from phyai.runtime.schedule import Scheduler
from phyai.utils import this_rank_log
from phyai.utils.profile import event_scope


logger = logging.getLogger(__name__)


@dataclass
class Cosmos3T2VRequest:
    """One Cosmos3 text/image-to-video [+ sound] request — already-tokenized tensors.

    Tokenization (Qwen chat template + eos/vision_start append) happens in the
    caller's processor (``phyai_utils_tools.models.cosmos3``), not here. Both the
    conditional prompt and the unconditional (negative / empty) prompt are
    pre-tokenized; right-padding is allowed but all real lengths in a batch must
    match (the transformer requires it).

    ``video_shape`` is the **latent** grid ``(t_lat, h_lat, w_lat)``; helper
    :func:`pixel_to_latent_shape` converts pixel dims. ``noise`` is optional
    (sampled from ``seed`` when ``None``).

    Optional audio (T2AV / I2AV): set ``sound_frames`` to the sound-latent length
    ``T_sound`` to jointly denoise a sound stream alongside the video (sharing the
    timestep, each stepped by its own UniPC solver). The audio stream is always
    fully noised (generated); the audio requires a ``sound_gen=True`` checkpoint.
    As a rule of thumb the audio runs at ``sound_latent_fps`` (25) and the video at
    ``fps``, so ``T_sound ≈ ceil(num_video_frames / fps * sound_latent_fps)``; the
    caller passes the resolved ``sound_frames``. Leave ``sound_frames=None`` for
    plain video.

    I2V/V2V (and, with sound, I2AV) conditioning seeds VAE-encoded clean latents
    ``[1, C, t_lat, h_lat, w_lat]`` at ``cond_frame_indexes`` and holds them fixed
    across denoising. Empty ``cond_frame_indexes`` / ``None`` cond_latents = plain
    text-conditioned generation.
    """

    text_ids: torch.Tensor
    text_mask: torch.Tensor
    neg_text_ids: torch.Tensor
    neg_text_mask: torch.Tensor
    video_shape: tuple[int, int, int]
    fps: float = 24.0
    num_inference_steps: int = 35
    guidance_scale: float = 6.0
    noise: torch.Tensor | None = None
    seed: int = 42
    # I2V/V2V conditioning: VAE-encoded clean latents ``[1, C, t_lat, h_lat, w_lat]``
    # and the latent-frame indices that are clean (kept fixed across denoising).
    # Empty ``cond_frame_indexes`` / ``None`` cond_latents = plain T2V.
    cond_latents: torch.Tensor | None = None
    cond_frame_indexes: tuple[int, ...] = ()
    # Optional audio (T2AV / I2AV): ``sound_frames`` = T_sound enables a jointly
    # denoised sound stream; ``None`` = video only. The sound latent is generated
    # token-major ``[1, T_sound, sound_dim]`` and returned channel-major.
    sound_frames: int | None = None
    sound_dim: int = 64
    sound_latent_fps: float = 25.0


def pixel_to_latent_shape(
    num_frames: int, height: int, width: int, *, temporal: int = 4, spatial: int = 16
) -> tuple[int, int, int]:
    """Pixel ``(T, H, W)`` -> latent ``(t_lat, h_lat, w_lat)`` via VAE compression."""
    return (num_frames - 1) // temporal + 1, height // spatial, width // spatial


class Cosmos3T2VScheduler(Scheduler):
    """Single-card Cosmos3 video [+ sound] denoising orchestrator (UniPC + CFG).

    Drives the denoise loop over a
    :class:`~phyai.models.cosmos3.model_runner_cosmos3.Cosmos3T2VRunner`, which owns
    the per-CFG-branch UND condition (encoded once, reused across all steps).
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
        # Owns the dense per-branch UND condition cache (no KVCachePool); see
        # Cosmos3T2VRunner. Keeps the modeling stateless and avoids recomputing the
        # timestep-independent UND tower every denoise step. ``torch_compile`` opts
        # into regional torch.compile of the decoder blocks (applied in setup()).
        self.runner = Cosmos3T2VRunner(
            transformer,
            device=self.device,
            torch_compile=torch_compile,
            compile_kwargs=compile_kwargs,
        )
        # VAEs are wrapped in their own runners so every VAE inference call goes
        # through a runner (parity with the transformer). ``None`` when the module
        # was not supplied — the stub-transformer tests construct without VAEs.
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
            logger, logging.INFO, "Cosmos3 video scheduler ready (UniPC, ws=1)."
        )

    @torch.no_grad()
    def step(
        self, request: Cosmos3T2VRequest
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the full denoise loop (T2V/I2V, or T2AV/I2AV when audio is requested).

        Returns the video latents ``[1, C, t, h, w]`` for a video-only request, or
        ``{"video": [1, C, t, h, w], "sound": [1, sound_dim, T_sound]}`` (the sound
        latent channel-major, the AVAE decode input) when ``request.sound_frames`` is
        set. Video and sound share the timestep; each is stepped by its own UniPC
        solver. I2V/V2V condition frames are re-imposed after every step; the audio
        stream is always fully noised (generated).
        """
        if self.unipc is None:
            raise RuntimeError("call setup() before step().")
        dev, dt = self.device, self.dtype
        t_lat, h_lat, w_lat = request.video_shape
        with_sound = request.sound_frames is not None

        # Initial latents: cosmos-framework native draws each modality's noise with a
        # fresh ``np.random.RandomState(seed)`` (``arch_invariant_rand``), so phyai
        # matches it modality-by-modality (both reseeded with the same seed). Video is
        # channel-major ``[1, C, t, h, w]`` (native draws ``[C,T,H,W]``; the batch-1
        # axis is trivial). The sound latent is token-major ``[1, T_sound, sound_dim]``
        # here, but native draws it channel-major ``[sound_dim, T_sound]`` — transpose.
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

        # I2V/V2V: seed clean condition frames into the initial latent and build a
        # per-frame mask so the transformer skips the timestep on those frames. The
        # clean latents are re-imposed after every UniPC step (the solver rescales
        # the whole sample, so a velocity/timestep mask alone is not enough).
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

        # The runner encodes the UND condition once per branch and reuses it across
        # every step (it is timestep-independent); reset clears it for this request.
        self.runner.reset()
        scope = "cosmos3.t2av_denoise_loop" if with_sound else "cosmos3.denoise_loop"
        with event_scope(scope):
            for timestep in self.unipc.timesteps:
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

        if with_sound:
            return {"video": video, "sound": sound.transpose(1, 2).contiguous()}
        return video

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents -> pixels ``[B, 3, T, H, W]`` in ``[0, 1]`` (needs a VAE)."""
        if self.vae_runner is None:
            raise RuntimeError("Cosmos3T2VScheduler was constructed without a VAE.")
        pixels = self.vae_runner.decode(latents)
        return ((pixels.float() + 1.0) / 2.0).clamp(0.0, 1.0)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels ``[B, 3, T, H, W]`` in ``[-1, 1]`` -> normalized latent (needs a VAE).

        Used for I2V / I2VS conditioning (VAE-encode the condition image, then hold
        the resulting latent frame fixed across denoising).
        """
        if self.vae_runner is None:
            raise RuntimeError("Cosmos3T2VScheduler was constructed without a VAE.")
        return self.vae_runner.encode(pixels)

    @torch.no_grad()
    def decode_sound(self, sound_latent: torch.Tensor) -> torch.Tensor:
        """Sound latent ``[B, latent_ch, T]`` -> waveform in ``[-1, 1]`` (needs an AVAE)."""
        if self.sound_runner is None:
            raise RuntimeError("Cosmos3T2VScheduler was constructed without an AVAE.")
        return self.sound_runner.decode(sound_latent)

    @property
    def sound_sample_rate(self) -> int:
        """Output waveform sample rate (Hz) of the wrapped AVAE (needs an AVAE)."""
        if self.sound_runner is None:
            raise RuntimeError("Cosmos3T2VScheduler was constructed without an AVAE.")
        return self.sound_runner.sample_rate


__all__ = [
    "Cosmos3T2VScheduler",
    "Cosmos3T2VRequest",
    "pixel_to_latent_shape",
]
