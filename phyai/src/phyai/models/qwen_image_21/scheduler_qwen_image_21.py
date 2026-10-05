"""Qwen-Image 2.1 text/image conditioning and denoising orchestration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from phyai_utils_tools.models.qwen_image_21 import (
    QwenImage21ImageInputs,
    QwenImage21Processor,
)

import phyai.parallel as P
from phyai.models.qwen_image_21.model_runner_condition import (
    QwenImage21Condition,
    QwenImage21ConditionRunner,
)
from phyai.models.qwen_image_21.model_runner_vae import QwenImage21VAERunner
from phyai.models.qwen_image_21.requests import QwenImage21Output, QwenImage21Request
from phyai.models.qwen_image_21.sampler_flow_match import (
    FlowMatchEulerConfig,
    FlowMatchEulerSampler,
)
from phyai.runtime.schedule import Scheduler
from phyai.utils import get_logger

if TYPE_CHECKING:
    from phyai.models.qwen_image_21.model_runner_qwen_image_21 import QwenImage21Runner


logger = get_logger(__name__)


class QwenImage21Scheduler(Scheduler):
    def __init__(
        self,
        runner: QwenImage21Runner,
        *,
        sampler_config: FlowMatchEulerConfig | None = None,
        condition_runner: QwenImage21ConditionRunner | None = None,
        vae_runner: QwenImage21VAERunner | None = None,
        processor: QwenImage21Processor | None = None,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        latent_channels: int = 64,
        cfg_rank: int = 0,
        cfg_size: int = 1,
    ) -> None:
        if cfg_size not in (1, 2) or not 0 <= cfg_rank < cfg_size:
            raise ValueError("Qwen-Image 2.1 supports cfg_size=1 or 2.")
        self.runner = runner
        self.condition_runner = condition_runner
        self.vae_runner = vae_runner
        self.processor = processor
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.latent_channels = latent_channels
        self.cfg_rank = cfg_rank
        self.cfg_size = cfg_size
        self.sampler = FlowMatchEulerSampler(sampler_config)
        self.ready = False

    def setup(self) -> None:
        self.runner.setup()
        if self.condition_runner is not None:
            self.condition_runner.setup()
        if self.vae_runner is not None:
            self.vae_runner.setup()
        self.ready = True

    def prepare_condition(
        self,
        request: QwenImage21Request,
        images: QwenImage21ImageInputs,
        *,
        negative: bool = False,
        batch_size: int | None = None,
    ) -> QwenImage21Condition:
        embeds = request.negative_prompt_embeds if negative else request.prompt_embeds
        mask = (
            request.negative_prompt_embeds_mask
            if negative
            else request.prompt_embeds_mask
        )
        image_mask = (
            request.negative_image_pad_mask if negative else request.image_pad_mask
        )
        if embeds is None:
            if self.condition_runner is None:
                raise ValueError(
                    "Text conditioning requires a condition runner or prompt_embeds."
                )
            processed = request.negative_processed if negative else request.processed
            if processed is None:
                prompt = request.negative_prompt if negative else request.prompt
                if prompt is None or self.processor is None:
                    raise ValueError(
                        "Supply a prompt with a processor, processed tokens, or prompt_embeds."
                    )
                if negative and isinstance(prompt, str) and batch_size is not None:
                    prompt = [prompt] * batch_size
                processed = self.processor.tokenize(prompt, images.images)
            condition = self.condition_runner.forward(
                processed, num_images_per_prompt=request.num_images_per_prompt
            )
            embeds, mask, image_mask = (
                condition.prompt_embeds,
                condition.prompt_embeds_mask,
                condition.image_pad_mask,
            )
        else:
            if embeds.ndim != 3:
                raise ValueError(
                    "prompt_embeds must have shape (batch, sequence, hidden)."
                )
            if mask is None:
                mask = torch.ones(
                    embeds.shape[:2], device=embeds.device, dtype=torch.bool
                )
            if image_mask is None:
                if images.images or request.condition_latents is not None:
                    raise ValueError(
                        "Image-conditioned prompt_embeds require image_pad_mask."
                    )
                image_mask = torch.zeros(
                    embeds.shape[:2], device=embeds.device, dtype=torch.bool
                )
            if mask.shape != embeds.shape[:2] or image_mask.shape != embeds.shape[:2]:
                raise ValueError(
                    "Prompt masks must match prompt_embeds' batch and sequence axes."
                )
            embeds = embeds.repeat_interleave(request.num_images_per_prompt, dim=0)
            mask = mask.repeat_interleave(request.num_images_per_prompt, dim=0)
            image_mask = image_mask.repeat_interleave(
                request.num_images_per_prompt, dim=0
            )
        if mask is not None:
            mask = mask.to(self.device, torch.bool)
            if bool(mask.all()):
                mask = None
        return QwenImage21Condition(
            embeds.to(self.device, self.dtype),
            mask,
            image_mask.to(self.device, torch.bool),
        )

    def prepare_images(self, request: QwenImage21Request) -> QwenImage21ImageInputs:
        if request.processed is not None:
            if request.image is not None:
                raise ValueError("Pass either processed images or image, not both.")
            return request.processed.images
        if request.image is None:
            return QwenImage21ImageInputs()
        if self.processor is None:
            raise ValueError("Raw condition images require a processor.")
        return self.processor.prepare_images(
            request.image, output_resolution=request.output_resolution
        )

    def prepare_latents(
        self,
        request: QwenImage21Request,
        images: QwenImage21ImageInputs,
        batch_size: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[int, int, int]]]:
        condition_latents = request.condition_latents
        condition_shapes = list(request.condition_image_shapes)
        if images.vae_images:
            if condition_latents is not None:
                raise ValueError(
                    "Pass either raw condition images or condition_latents."
                )
            if self.vae_runner is None:
                raise ValueError("Encoding condition images requires a VAE runner.")
            packed = []
            for pixels in images.vae_images:
                encoded = self.vae_runner.encode(pixels.to(self.device, self.dtype))
                condition_shapes.append(tuple(encoded.shape[-3:]))
                packed.append(encoded.flatten(2).transpose(1, 2))
            condition_latents = torch.cat(packed, dim=1)
        if condition_latents is not None:
            condition_latents = condition_latents.to(self.device, self.dtype)
            if (
                condition_latents.ndim != 3
                or condition_latents.shape[-1] != self.latent_channels
            ):
                raise ValueError(
                    "condition_latents must be packed (batch, tokens, channels)."
                )
            expected_tokens = sum(math.prod(shape) for shape in condition_shapes)
            if expected_tokens != condition_latents.shape[1]:
                raise ValueError(
                    "condition_image_shapes must cover all condition latent tokens."
                )
            if condition_latents.shape[0] == 1:
                condition_latents = condition_latents.expand(batch_size, -1, -1)
            elif (
                condition_latents.shape[0]
                == batch_size // request.num_images_per_prompt
            ):
                condition_latents = condition_latents.repeat_interleave(
                    request.num_images_per_prompt, dim=0
                )
            elif condition_latents.shape[0] != batch_size:
                raise ValueError(
                    "Condition latent batch does not match the prompt batch."
                )
        latent_height, latent_width = height // 16, width // 16
        shape = (batch_size, 1, self.latent_channels, latent_height, latent_width)
        if request.latents is not None:
            latents = request.latents.to(self.device, self.dtype)
            if latents.shape != (
                batch_size,
                latent_height * latent_width,
                self.latent_channels,
            ):
                raise ValueError(
                    "latents must be packed (batch, height/16 * width/16, channels)."
                )
            latents = latents.clone()
        else:
            generator = request.generator
            if generator is None:
                generator = torch.Generator(device=self.device).manual_seed(
                    request.seed
                )
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError("Supply one generator per generated image.")
                parts = [self.randn(shape[1:], gen) for gen in generator]
                noise = torch.stack(parts)
            else:
                noise = self.randn(shape, generator)
            latents = noise.view(batch_size, self.latent_channels, -1).transpose(1, 2)
        return (
            latents,
            condition_latents,
            [*condition_shapes, (1, latent_height, latent_width)],
        )

    def randn(self, shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
        generator_device = torch.device(generator.device)
        if generator_device.type != "cpu" and generator_device != self.device:
            raise ValueError("A CUDA generator must use the model's device.")
        return torch.randn(
            shape, generator=generator, device=generator_device, dtype=self.dtype
        ).to(self.device)

    @staticmethod
    def append_target_slots(
        condition: QwenImage21Condition, target_tokens: int
    ) -> QwenImage21Condition:
        mask = condition.image_pad_mask
        image_mask = torch.cat(
            [mask, mask.new_ones(mask.shape[0], target_tokens // 4)], dim=1
        )
        return QwenImage21Condition(
            condition.prompt_embeds, condition.prompt_embeds_mask, image_mask
        )

    def predict(
        self,
        latent_input: torch.Tensor,
        timestep: torch.Tensor,
        condition: QwenImage21Condition,
        img_shapes: list[list[tuple[int, int, int]]],
        branch: str,
    ) -> torch.Tensor:
        return self.runner.forward(
            latent_input,
            timestep,
            encoder_hidden_states=condition.prompt_embeds,
            encoder_hidden_states_mask=condition.prompt_embeds_mask,
            img_shapes=img_shapes,
            img_mask=condition.image_pad_mask,
            branch=branch,
        )

    @torch.inference_mode()
    def step(self, request: QwenImage21Request) -> QwenImage21Output:
        if not self.ready:
            raise RuntimeError("Call setup() before step().")
        if request.output_type != "latent" and self.vae_runner is None:
            raise ValueError(
                "Decoded output requires a VAE runner; use output_type='latent'."
            )
        self.runner.reset()
        images = self.prepare_images(request)
        default_width, default_height = (
            images.image_sizes[-1]
            if images.image_sizes
            else (request.output_resolution, request.output_resolution)
        )
        height = (request.height or default_height) // 32 * 32
        width = (request.width or default_width) // 32 * 32
        condition = self.prepare_condition(request, images)
        batch_size = condition.prompt_embeds.shape[0]
        has_negative = any(
            value is not None
            for value in (
                request.negative_prompt,
                request.negative_prompt_embeds,
                request.negative_processed,
            )
        )
        do_cfg = request.true_cfg_scale > 1 and has_negative
        negative = None
        if do_cfg:
            negative = self.prepare_condition(
                request,
                images,
                negative=True,
                batch_size=batch_size // request.num_images_per_prompt,
            )
            if negative.prompt_embeds.shape[0] != batch_size:
                raise ValueError(
                    "Positive and negative prompts must have the same batch size."
                )
        elif request.true_cfg_scale > 1:
            logger.warning_once(
                "CFG is disabled because no negative conditioning was supplied."
            )
        latents, condition_latents, shapes = self.prepare_latents(
            request, images, batch_size, height, width
        )
        condition = self.append_target_slots(condition, latents.shape[1])
        if negative is not None:
            negative = self.append_target_slots(negative, latents.shape[1])
        img_shapes = [shapes] * batch_size
        schedule = self.sampler.build_schedule(
            request.num_inference_steps,
            latents.shape[1],
            device=self.device,
            sigmas=request.sigmas,
        )
        for index, timestep in enumerate(schedule.timesteps):
            model_input = (
                latents
                if condition_latents is None
                else torch.cat([condition_latents, latents], dim=1)
            )
            t = (
                timestep.expand(batch_size).to(self.dtype)
                / self.sampler.config.num_train_timesteps
            )
            if self.cfg_size == 2 and negative is not None:
                branch = "cond" if self.cfg_rank == 0 else "uncond"
                local_condition = condition if self.cfg_rank == 0 else negative
                local = self.predict(
                    model_input, t, local_condition, img_shapes, branch
                )
                pair = P.all_gather(local.unsqueeze(0), group="cfg", dim=0)
                velocity = pair[1] + request.true_cfg_scale * (pair[0] - pair[1])
            else:
                velocity = self.predict(model_input, t, condition, img_shapes, "cond")
                if negative is not None:
                    uncond = self.predict(
                        model_input, t, negative, img_shapes, "uncond"
                    )
                    velocity = uncond + request.true_cfg_scale * (velocity - uncond)
            latents = self.sampler.step(velocity, latents, schedule, index)
        if request.output_type == "latent":
            return QwenImage21Output(latents, latents)
        if self.vae_runner is None:
            raise ValueError(
                "Decoded output requires a VAE runner; use output_type='latent'."
            )
        unpacked = latents.transpose(1, 2).reshape(
            batch_size, self.latent_channels, 1, height // 16, width // 16
        )
        pixels = self.vae_runner.decode(unpacked)[:, :, 0]
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        if request.output_type == "pt":
            output = pixels
        else:
            array = pixels.float().permute(0, 2, 3, 1).cpu().numpy()
            if request.output_type == "np":
                output = array
            else:
                array = (array * 255).round().astype(np.uint8)
                output = [Image.fromarray(sample) for sample in array]
        return QwenImage21Output(output, latents)

    def close(self) -> None:
        self.runner.close()
        if self.condition_runner is not None:
            self.condition_runner.close()
        if self.vae_runner is not None:
            self.vae_runner.close()
        self.ready = False


__all__ = ["QwenImage21Scheduler"]
