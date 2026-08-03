"""LingBot-VLA 2.0-specific processor steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import (
    ACTION,
    IMAGES,
    INPUT_IDS,
    LANG_LENS,
    PIXEL_VALUES,
    PROMPT,
    STATE,
    TASK,
    Transition,
)

IMAGE_GRID_THW = "image_grid_thw"
IMAGE_MASKS = "image_masks"
NOISE = "noise"

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def resolve_float_dtype(
    float_dtype: str | torch.dtype | None,
) -> torch.dtype | None:
    """Resolve a serialized dtype name to ``torch.dtype``."""

    if float_dtype is None or isinstance(float_dtype, torch.dtype):
        return float_dtype
    key = str(float_dtype).replace("torch.", "")
    if key not in _DTYPE_BY_NAME:
        raise ValueError(
            f"unknown float dtype {float_dtype!r}; "
            f"expected one of {sorted(_DTYPE_BY_NAME)} or None."
        )
    return _DTYPE_BY_NAME[key]


def camera_batches(
    images: Any,
    *,
    num_images: int,
    num_channels: int,
) -> list[torch.Tensor]:
    """Convert supported image containers to per-camera ``(B,C,H,W)`` tensors."""

    if isinstance(images, torch.Tensor):
        if images.ndim != 5:
            raise ValueError(
                "stacked images must be (B,N,C,H,W) or (B,N,H,W,C), "
                f"got {tuple(images.shape)}."
            )
        if images.shape[1] != num_images:
            raise ValueError(
                f"stacked images contain {images.shape[1]} cameras, "
                f"expected {num_images}."
            )
        if images.shape[2] == num_channels:
            stacked = images
        elif images.shape[-1] == num_channels:
            stacked = images.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(
                f"stacked images do not have a {num_channels}-channel axis."
            )
        return [stacked[:, index] for index in range(num_images)]

    cameras = list(images)
    if len(cameras) != num_images:
        raise ValueError(f"expected {num_images} camera tensors, got {len(cameras)}.")

    batches: list[torch.Tensor] = []
    batch_size: int | None = None
    for index, camera in enumerate(cameras):
        if not isinstance(camera, torch.Tensor):
            camera = torch.as_tensor(camera)
        if camera.ndim == 3:
            camera = camera.unsqueeze(0)
        if camera.ndim != 4:
            raise ValueError(
                f"camera {index} must be (B,C,H,W) or (B,H,W,C), "
                f"got {tuple(camera.shape)}."
            )
        if camera.shape[1] == num_channels:
            camera_batch = camera
        elif camera.shape[-1] == num_channels:
            camera_batch = camera.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"camera {index} does not have {num_channels} channels.")
        if batch_size is None:
            batch_size = int(camera_batch.shape[0])
        elif camera_batch.shape[0] != batch_size:
            raise ValueError(
                f"camera {index} batch size {camera_batch.shape[0]} "
                f"does not match {batch_size}."
            )
        batches.append(camera_batch)
    return batches


def prepare_raw_image(
    image: torch.Tensor,
    *,
    convert_unit_float_to_uint8: bool,
) -> torch.Tensor:
    """Match LingBot deployment's float ``[0,1]`` to uint8 conversion."""

    if not convert_unit_float_to_uint8 or not image.is_floating_point():
        return image
    try:
        max_value = float(image.max())
    except RuntimeError:
        max_value = 0.0
    if max_value <= 2.0:
        return (image * 255.0).round().clamp(0, 255).to(torch.uint8)
    return image


@ProcessorStepRegistry.register("lingbot_v2_qwen3vl_image_pack_step")
@dataclass
class Qwen3VLImagePackStep(ProcessorStep):
    """Run Qwen3-VL image preprocessing and pad patches per image.

    The Qwen processor returns one concatenated patch matrix and one ``(t,h,w)``
    row per active image. This step splits the matrix back into fixed-stride
    ``(B,N,P,D)`` storage required by ``LingBotV2Request``.
    """

    image_processor: Any = field(repr=False, default=None)
    processor_name: str | None = None
    num_images: int = 3
    num_channels: int = 3
    patch_vector_dim: int = 1536
    max_patches_per_image: int | None = None
    convert_unit_float_to_uint8: bool = True

    def __post_init__(self) -> None:
        if self.num_images <= 0:
            raise ValueError("num_images must be positive.")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive.")
        if self.patch_vector_dim <= 0:
            raise ValueError("patch_vector_dim must be positive.")
        if self.max_patches_per_image is not None and self.max_patches_per_image <= 0:
            raise ValueError("max_patches_per_image must be positive.")

    def __call__(self, transition: Transition) -> Transition:
        if self.image_processor is None:
            raise ValueError(
                "Qwen3VLImagePackStep requires an `image_processor` object."
            )
        if IMAGES not in transition:
            raise ValueError("Qwen3VLImagePackStep requires an IMAGES entry.")

        cameras = camera_batches(
            transition[IMAGES],
            num_images=self.num_images,
            num_channels=self.num_channels,
        )
        batch_size = int(cameras[0].shape[0])
        image_masks = transition.get(IMAGE_MASKS)
        if image_masks is None:
            masks = torch.ones(
                batch_size,
                self.num_images,
                dtype=torch.bool,
            )
        else:
            masks = torch.as_tensor(image_masks, dtype=torch.bool)
            if masks.shape != (batch_size, self.num_images):
                raise ValueError(
                    "image_masks must have shape "
                    f"({batch_size}, {self.num_images}), got {tuple(masks.shape)}."
                )
            masks = masks.cpu()

        active_images: list[torch.Tensor] = []
        active_indices: list[tuple[int, int]] = []
        for batch_index in range(batch_size):
            for image_index in range(self.num_images):
                if not bool(masks[batch_index, image_index]):
                    continue
                active_images.append(
                    prepare_raw_image(
                        cameras[image_index][batch_index],
                        convert_unit_float_to_uint8=(self.convert_unit_float_to_uint8),
                    )
                )
                active_indices.append((batch_index, image_index))
        if not active_images:
            raise ValueError("at least one image must be active.")

        processed = self.image_processor(
            images=active_images,
            return_tensors="pt",
        )
        if PIXEL_VALUES not in processed or IMAGE_GRID_THW not in processed:
            raise ValueError(
                "Qwen3-VL image processor must return pixel_values and "
                "image_grid_thw."
            )
        flat_pixels = torch.as_tensor(processed[PIXEL_VALUES])
        if flat_pixels.shape[-1] != self.patch_vector_dim:
            raise ValueError(
                f"Qwen3-VL patch width {flat_pixels.shape[-1]} does not match "
                f"patch_vector_dim={self.patch_vector_dim}."
            )
        flat_pixels = flat_pixels.reshape(-1, self.patch_vector_dim)
        active_grids = torch.as_tensor(
            processed[IMAGE_GRID_THW],
            dtype=torch.int64,
        ).reshape(-1, 3)
        if active_grids.shape[0] != len(active_indices):
            raise ValueError(
                f"image processor returned {active_grids.shape[0]} grid rows "
                f"for {len(active_indices)} active images."
            )

        patch_counts = active_grids.prod(dim=-1)
        total_patches = int(patch_counts.sum())
        if flat_pixels.shape[0] != total_patches:
            raise ValueError(
                f"image processor returned {flat_pixels.shape[0]} patch rows, "
                f"but image_grid_thw describes {total_patches}."
            )
        required_patches = int(patch_counts.max())
        max_patches = (
            required_patches
            if self.max_patches_per_image is None
            else self.max_patches_per_image
        )
        if required_patches > max_patches:
            raise ValueError(
                f"an image requires {required_patches} patches, exceeding "
                f"max_patches_per_image={max_patches}."
            )

        packed_pixels = flat_pixels.new_zeros(
            (
                batch_size,
                self.num_images,
                max_patches,
                self.patch_vector_dim,
            )
        )
        grids = torch.zeros(
            batch_size,
            self.num_images,
            3,
            dtype=torch.int64,
            device=active_grids.device,
        )
        offset = 0
        for active_index, (batch_index, image_index) in enumerate(active_indices):
            count = int(patch_counts[active_index])
            packed_pixels[batch_index, image_index, :count] = flat_pixels[
                offset : offset + count
            ]
            grids[batch_index, image_index] = active_grids[active_index]
            offset += count

        out = transition.copy()
        out[PIXEL_VALUES] = packed_pixels
        out[IMAGE_GRID_THW] = grids
        out[IMAGE_MASKS] = masks
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_name": self.processor_name,
            "num_images": self.num_images,
            "num_channels": self.num_channels,
            "patch_vector_dim": self.patch_vector_dim,
            "max_patches_per_image": self.max_patches_per_image,
            "convert_unit_float_to_uint8": self.convert_unit_float_to_uint8,
        }


@ProcessorStepRegistry.register("lingbot_v2_pad_state_step")
@dataclass
class LingBotV2PadStateStep(ProcessorStep):
    """Pad canonical LingBot state vectors to ``max_state_dim``."""

    max_state_dim: int = 55

    def __call__(self, transition: Transition) -> Transition:
        state = transition.get(STATE)
        if state is None:
            raise ValueError("LingBotV2PadStateStep requires a STATE entry.")
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(f"state must be (B,state_dim), got {tuple(state.shape)}.")
        state_dim = int(state.shape[-1])
        if state_dim > self.max_state_dim:
            raise ValueError(
                f"state_dim={state_dim} exceeds max_state_dim={self.max_state_dim}."
            )
        out = transition.copy()
        out[STATE] = (
            state
            if state_dim == self.max_state_dim
            else F.pad(state, (0, self.max_state_dim - state_dim))
        )
        return out

    def get_config(self) -> dict[str, Any]:
        return {"max_state_dim": self.max_state_dim}


@ProcessorStepRegistry.register("lingbot_v2_prompt_prepare_step")
@dataclass
class LingBotV2PromptPrepareStep(ProcessorStep):
    """Apply the Qwen3 chat template or the legacy BOS/newline format."""

    tokenizer: Any = field(repr=False, default=None)
    use_chat_template: bool = True
    bos_token: str = "<bos>"

    def __call__(self, transition: Transition) -> Transition:
        tasks = transition.get(TASK)
        if tasks is None:
            raise ValueError("LingBotV2PromptPrepareStep requires a TASK entry.")
        if isinstance(tasks, str):
            tasks = [tasks]
        prompts = list(tasks)
        if self.use_chat_template:
            if self.tokenizer is None:
                raise ValueError("use_chat_template=True requires a tokenizer object.")
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for prompt in prompts
            ]
        else:
            prompts = [
                prompt if prompt.startswith(self.bos_token) else self.bos_token + prompt
                for prompt in prompts
            ]
            prompts = [
                prompt if prompt.endswith("\n") else prompt + "\n" for prompt in prompts
            ]
        out = transition.copy()
        out[PROMPT] = prompts
        return out

    def get_config(self) -> dict[str, Any]:
        return {
            "use_chat_template": self.use_chat_template,
            "bos_token": self.bos_token,
        }


@ProcessorStepRegistry.register("lingbot_v2_device_step")
@dataclass
class LingBotV2DeviceStep(ProcessorStep):
    """Move all LingBot request fields to their runtime device and dtype."""

    device: torch.device | str = "cpu"
    float_dtype: str | torch.dtype | None = None

    def __post_init__(self) -> None:
        self.resolved_float_dtype = resolve_float_dtype(self.float_dtype)

    def __call__(self, transition: Transition) -> Transition:
        out = transition.copy()
        float_fields = (PIXEL_VALUES, STATE, ACTION, NOISE)
        int_fields = (INPUT_IDS, LANG_LENS, IMAGE_GRID_THW)
        for name in float_fields:
            value = out.get(name)
            if not isinstance(value, torch.Tensor):
                continue
            value = value.to(device=self.device)
            if self.resolved_float_dtype is not None and value.is_floating_point():
                value = value.to(dtype=self.resolved_float_dtype)
            out[name] = value
        for name in int_fields:
            value = out.get(name)
            if isinstance(value, torch.Tensor):
                out[name] = value.to(device=self.device, dtype=torch.int64)
        masks = out.get(IMAGE_MASKS)
        if isinstance(masks, torch.Tensor):
            out[IMAGE_MASKS] = masks.to(device=self.device, dtype=torch.bool)
        return out

    def get_config(self) -> dict[str, Any]:
        dtype = self.resolved_float_dtype
        return {
            "device": str(self.device).replace("torch.device", "").strip("()'\""),
            "float_dtype": (
                str(dtype).replace("torch.", "") if dtype is not None else None
            ),
        }


__all__ = [
    "IMAGE_GRID_THW",
    "IMAGE_MASKS",
    "NOISE",
    "LingBotV2DeviceStep",
    "LingBotV2PadStateStep",
    "LingBotV2PromptPrepareStep",
    "Qwen3VLImagePackStep",
]
