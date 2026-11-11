"""Compose VPP2 observation encoding, sampling, and optional video decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from phyai.models.vpp2.model_runner_encoders import VPP2EncoderInputs, VPP2EncoderRunner
from phyai.models.vpp2.requests_vpp2 import (
    VPP2ActionRequest,
    VPP2StageRequest,
    VPP2VideoRequest,
)
from phyai.models.vpp2.scheduler_vpp2 import VPP2Scheduler
from phyai.runtime.schedule import Scheduler


@dataclass
class VPP2ObservationRequest:
    """Preprocessed observations; inference options override the release defaults."""

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    clip_pixel_values: torch.Tensor | None = None
    condition_video: torch.Tensor | None = None
    condition_observation_video: torch.Tensor | None = None
    negative_input_ids: torch.Tensor | None = None
    negative_attention_mask: torch.Tensor | None = None
    proprio: torch.Tensor | None = None
    mode: Literal["action", "video"] = "action"
    inference: dict[str, Any] = field(default_factory=dict)
    decode_video: bool = True
    tiled: bool = False


class VPP2Pipeline(Scheduler):
    def __init__(
        self,
        scheduler: VPP2Scheduler,
        encoders: VPP2EncoderRunner | None = None,
        *,
        defaults: dict[str, dict[str, Any]] | None = None,
        video_seed_offset: int = 1_000_003,
    ) -> None:
        self.scheduler = scheduler
        self.encoders = encoders
        self.defaults = defaults or {}
        self.video_seed_offset = video_seed_offset

    def setup(self) -> None:
        self.scheduler.setup()
        if self.encoders is not None:
            self.encoders.setup()

    @torch.inference_mode()
    def step(
        self, request: VPP2ObservationRequest | VPP2StageRequest
    ) -> dict[str, Any]:
        if not isinstance(request, VPP2ObservationRequest):
            return self.scheduler.step(request)
        if request.mode not in {"action", "video"}:
            raise ValueError("mode must be 'action' or 'video'.")
        if self.encoders is None:
            raise RuntimeError("Raw observation requests require load_encoders=True.")
        conditions = self.encoders.forward(
            VPP2EncoderInputs(
                pixel_values=request.pixel_values,
                input_ids=request.input_ids,
                attention_mask=request.attention_mask,
                clip_pixel_values=request.clip_pixel_values,
                condition_video=request.condition_video,
                condition_observation_video=request.condition_observation_video,
                negative_input_ids=request.negative_input_ids
                if request.mode == "video"
                else None,
                negative_attention_mask=request.negative_attention_mask,
                policy=request.mode == "action",
                tiled=request.tiled,
            )
        )
        options = {**self.defaults.get(request.mode, {}), **request.inference}
        if conditions.keys() & options.keys():
            raise ValueError("inference options cannot replace encoded observations.")
        if request.mode == "action":
            seed = options.get("seed", 0)
            if seed is not None:
                options.setdefault("video_seed", seed + self.video_seed_offset)
            stage = VPP2ActionRequest(**conditions, proprio=request.proprio, **options)
        else:
            if request.proprio is not None:
                raise ValueError(
                    "Standalone video requests do not consume proprioception."
                )
            stage = VPP2VideoRequest(**conditions, **options)
        output = self.scheduler.step(stage)
        if request.decode_video:
            decoded: dict[int, torch.Tensor] = {}
            for key in (
                "video_latents",
                "video_one_step_latents",
                "video_full_latents",
            ):
                if key in output:
                    identity = id(output[key])
                    if identity not in decoded:
                        decoded[identity] = self.encoders.decode(
                            output[key], tiled=request.tiled
                        )
                    output[key.removesuffix("_latents")] = decoded[identity]
        return output

    def close(self) -> None:
        self.scheduler.close()
        if self.encoders is not None:
            self.encoders.close()
