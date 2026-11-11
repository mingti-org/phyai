"""VPP2 inference requests at the encoded-condition boundary."""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class VPP2ActionRequest:
    """Cached-video policy inference with normalized proprioception.

    Conditions use [1,C,T,H,W] VAE latents, [1,L,D] text embeddings,
    [1,L] text masks, and [1,257,1280] CLIP features. Optional initial
    action noise is packed as [1,horizon/chunk_size,action_dim].
    """

    condition_latents: torch.Tensor
    clip_fea: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None = None
    proprio: torch.Tensor | None = None
    action_horizon: int = 32
    num_video_frames: int = 17
    num_inference_steps: int = 10
    sigma_shift: float | None = None
    seed: int | None = 0
    video_seed: int | None = None
    rand_device: str | torch.device = "cpu"
    initial_action: torch.Tensor | None = None
    initial_video: torch.Tensor | None = None
    text_cfg_scale: float = 1.0
    return_one_step_video: bool = False
    return_full_video: bool = False
    full_video_num_inference_steps: int | None = None
    output_device: str | torch.device = "cpu"


@dataclass
class VPP2VideoRequest:
    """Video generation; discrete sampling matches the standalone command.

    The discrete sampler retains FP32 latents and uses integer timesteps.
    Continuous sampling matches the policy's optional video trajectory.
    CFG uses the same image conditions for both text branches.
    """

    condition_latents: torch.Tensor
    clip_fea: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None = None
    negative_context: torch.Tensor | None = None
    negative_context_mask: torch.Tensor | None = None
    num_video_frames: int = 49
    num_inference_steps: int = 30
    sigma_shift: float | None = None
    text_cfg_scale: float = 4.0
    seed: int | None = 3402
    rand_device: str | torch.device | None = None
    initial_video: torch.Tensor | None = None
    scheduler_mode: Literal["discrete", "continuous"] = "discrete"


VPP2Request = VPP2ActionRequest
VPP2StageRequest = VPP2ActionRequest | VPP2VideoRequest
