"""Prompt preparation, CFG, skip-layer guidance, and Euler denoising."""

from __future__ import annotations

import torch

from phyai_utils_tools.models.moss_image import MOSSImageProcessor

from phyai.models.moss_image.model_runner_moss_image import MOSSImageRunner
from phyai.models.moss_image.modeling_moss_image import validate_skip_layers
from phyai.models.moss_image.requests import MOSSImageOutput, MOSSImageRequest
from phyai.models.moss_image.sampler_moss_image import (
    MOSSImageEulerSampler,
    MOSSImageSamplerConfig,
)
from phyai.runtime.schedule import Scheduler


class MOSSImageScheduler(Scheduler):
    def __init__(
        self,
        runner: MOSSImageRunner,
        *,
        sampler_config: MOSSImageSamplerConfig | None = None,
        processor: MOSSImageProcessor | None = None,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        latent_channels: int = 128,
    ) -> None:
        self.runner = runner
        self.sampler = MOSSImageEulerSampler(sampler_config)
        self.processor = processor or MOSSImageProcessor()
        self.device = torch.device(device)
        self.dtype = dtype
        self.latent_channels = latent_channels
        self.ready = False

    def setup(self) -> None:
        self.runner.setup()
        self.ready = True

    def prepare_embeddings(
        self,
        request: MOSSImageRequest,
        *,
        negative: bool = False,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = (
            request.negative_prompt_embeds if negative else request.prompt_embeds
        )
        mask = (
            request.negative_prompt_attention_mask
            if negative
            else request.prompt_attention_mask
        )
        if embeddings is None:
            prompt = request.negative_prompt if negative else request.prompt
            if prompt is None:
                raise ValueError("Provide prompt strings or prompt embeddings")
            if negative and isinstance(prompt, str):
                prompt = [prompt] * (batch_size or 1)
            tokens = self.processor.tokenize(prompt, request.max_sequence_length)
            embeddings, mask = self.runner.encode(tokens)
        if embeddings.ndim != 3 or any(size < 1 for size in embeddings.shape):
            raise ValueError(
                "Prompt embeddings must have nonempty shape (batch, sequence, channels)"
            )
        if batch_size is not None and embeddings.shape[0] != batch_size:
            raise ValueError("Negative and positive prompt batches must match")
        if mask is None:
            mask = torch.ones(
                embeddings.shape[:2], device=embeddings.device, dtype=torch.bool
            )
        if mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "Prompt attention masks must match the batch and sequence axes"
            )
        mask = mask.to(self.device, torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Each prompt must contain at least one unmasked token")
        return (
            embeddings.to(self.device, self.dtype).repeat_interleave(
                request.num_images_per_prompt, dim=0
            ),
            mask.repeat_interleave(request.num_images_per_prompt, dim=0),
        )

    def prepare_latents(
        self, request: MOSSImageRequest, batch_size: int
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.latent_channels,
            request.height // 16,
            request.width // 16,
        )
        if request.latents is not None:
            if tuple(request.latents.shape) != shape:
                raise ValueError(f"latents must have shape {shape}")
            return request.latents.to(device=self.device, dtype=torch.float32).clone()
        generator = request.generator
        if generator is None:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)

        def sample(sample_shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
            if gen.device.type != "cpu" and gen.device.type != self.device.type:
                raise ValueError(
                    "A generator must use CPU or the inference device type"
                )
            return torch.randn(
                sample_shape, generator=gen, device=gen.device, dtype=torch.float32
            ).to(self.device)

        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError("Supply one generator per generated image")
            return torch.cat([sample((1, *shape[1:]), gen) for gen in generator], dim=0)
        return sample(shape, generator)

    @torch.no_grad()
    def step(self, request: MOSSImageRequest) -> MOSSImageOutput:
        if not self.ready:
            raise RuntimeError("call setup() before step()")
        uncond_skip = tuple(
            validate_skip_layers(request.uncond_skip_layers, self.runner.depth)
        )
        slg_layers = tuple(validate_skip_layers(request.slg_layers, self.runner.depth))
        self.runner.reset()
        try:
            positive, positive_mask = self.prepare_embeddings(request)
            batch = positive.shape[0]
            do_cfg = request.guidance_scale != 1.0
            negative, negative_mask = None, None
            if do_cfg:
                negative, negative_mask = self.prepare_embeddings(
                    request,
                    negative=True,
                    batch_size=batch // request.num_images_per_prompt,
                )
            height, width = request.height // 16, request.width // 16
            self.runner.prepare_condition(
                "positive", positive, positive_mask, height, width
            )
            if negative is not None and negative_mask is not None:
                self.runner.prepare_condition(
                    "negative", negative, negative_mask, height, width
                )
            latents = self.prepare_latents(request, batch)
            timesteps, sigmas = self.sampler.timesteps(
                request.num_inference_steps, device=self.device
            )
            for index, timestep in enumerate(timesteps):
                skip_active = (
                    request.slg_start <= index / len(timesteps) < request.slg_stop
                )
                conditional = self.runner.forward(
                    latents, timestep, condition="positive"
                ).float()
                prediction = conditional
                if do_cfg:
                    unconditional = self.runner.forward(
                        latents,
                        timestep,
                        condition="negative",
                        skip_layers=uncond_skip if skip_active else (),
                    ).float()
                    prediction = unconditional + request.guidance_scale * (
                        conditional - unconditional
                    )
                if slg_layers and request.slg_scale > 0 and skip_active:
                    skipped = self.runner.forward(
                        latents, timestep, condition="positive", skip_layers=slg_layers
                    ).float()
                    if not do_cfg:
                        prediction = skipped
                    prediction = prediction + request.slg_scale * (
                        conditional - skipped
                    )
                latents = self.sampler.step(
                    prediction, latents, sigmas[index], sigmas[index + 1]
                )
            if request.output_type == "latent":
                images = latents
            else:
                pixels = self.runner.decode(latents)
                images = self.processor.format_images(pixels, request.output_type)
            return MOSSImageOutput(images=images, latents=latents)
        finally:
            self.runner.reset()

    def close(self) -> None:
        self.runner.close()
        self.ready = False


__all__ = ["MOSSImageScheduler"]
