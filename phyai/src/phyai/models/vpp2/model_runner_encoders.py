"""VPP2 conditioning encoders and VAE runtime."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from phyai.models.vpp2.image_encoder_wan import VPP2WanImageEncoder
from phyai.models.vpp2.text_encoder_wan import VPP2WanTextEncoder
from phyai.models.vpp2.vae_wan import VPP2WanVAE
from phyai.parallel.state import graph_capture
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner


@dataclass
class VPP2EncoderInputs:
    """RGB tensors are in [-1, 1]; histories use [B, 3, T, H, W]."""

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    clip_pixel_values: torch.Tensor | None = None
    condition_video: torch.Tensor | None = None
    condition_observation_video: torch.Tensor | None = None
    negative_input_ids: torch.Tensor | None = None
    negative_attention_mask: torch.Tensor | None = None
    policy: bool = True
    tiled: bool = False


class VPP2EncoderRunner(ModelRunner):
    def __init__(
        self,
        text_encoder: VPP2WanTextEncoder,
        image_encoder: VPP2WanImageEncoder,
        vae: VPP2WanVAE,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        use_cuda_graph: bool = False,
        max_cached_graphs: int = 1,
    ) -> None:
        if max_cached_graphs < 1:
            raise ValueError("max_cached_graphs must be positive.")
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_cuda_graph = use_cuda_graph
        self.max_cached_graphs = max_cached_graphs
        self.graphs: dict[tuple[object, ...], CudaGraph] = {}
        self.graph_capture_count = 0
        self.graph_replay_count = 0

    def setup(self) -> None:
        self.text_encoder.eval()
        self.image_encoder.eval()
        self.vae.eval()

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = input_ids.to(device=self.device, dtype=torch.long)
        mask = attention_mask.to(device=self.device, dtype=torch.bool)
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError(
                "Text IDs and attention mask must have matching [B, L] shapes."
            )
        context = self.text_encoder(ids, mask)
        # Wan keeps padded tokens as zero embeddings and attends to those positions.
        context = context.masked_fill(~mask[..., None], 0)
        return context.to(self.dtype), torch.ones_like(mask)

    def encode_video(self, pixels: torch.Tensor, *, tiled: bool) -> torch.Tensor:
        if pixels.ndim != 5 or pixels.shape[1] != 3:
            raise ValueError("VAE input must have shape [B, 3, T, H, W].")
        if pixels.shape[2] < 1 or pixels.shape[2] % 4 != 1:
            raise ValueError("VAE frame count must be positive and of the form 4n+1.")
        with torch.autocast(self.device.type, enabled=False):
            return self.vae.encode(
                pixels.to(device=self.device, dtype=torch.float32),
                tiled=tiled,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            ).to(self.dtype)

    @torch.inference_mode()
    def forward(self, batch: VPP2EncoderInputs) -> dict[str, torch.Tensor]:
        if not self.use_cuda_graph or self.device.type != "cuda":
            return self.forward_eager(batch)
        inputs = {
            field.name: value.to(device=self.device)
            for field in fields(batch)
            if isinstance(value := getattr(batch, field.name), torch.Tensor)
        }
        autocast_dtype = (
            torch.get_autocast_dtype(self.device.type)
            if torch.is_autocast_enabled(self.device.type)
            else None
        )
        graph_key = (
            batch.policy,
            batch.tiled,
            autocast_dtype,
            tuple(
                (
                    name,
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.dtype,
                    value.device,
                )
                for name, value in inputs.items()
            ),
        )
        graph = self.graphs.get(graph_key)
        if graph is None:
            if len(self.graphs) >= self.max_cached_graphs:
                self.graphs.pop(next(iter(self.graphs))).reset()
            policy, tiled = batch.policy, batch.tiled

            def graph_forward(**buffers: torch.Tensor) -> dict[str, torch.Tensor]:
                return self.forward_eager(
                    VPP2EncoderInputs(**buffers, policy=policy, tiled=tiled)
                )

            # Resolve capture-compatible kernels before recording the graph.
            with graph_capture():
                graph_forward(**inputs)
            graph = CudaGraph(num_warmup_iters=1)
            graph.capture(graph_forward, inputs)
            self.graphs[graph_key] = graph
            self.graph_capture_count += 1
        result = graph.replay(inputs)
        self.graph_replay_count += 1
        # Returned conditions may outlive the next request's graph replay.
        return {name: value.clone() for name, value in result.items()}

    def clear_cuda_graphs(self) -> None:
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()

    def close(self) -> None:
        self.clear_cuda_graphs()

    @torch.inference_mode()
    def forward_eager(self, batch: VPP2EncoderInputs) -> dict[str, torch.Tensor]:
        pixels = batch.pixel_values
        if pixels.ndim != 4 or pixels.shape[:2] != (1, 3):
            raise ValueError("pixel_values must have shape [1, 3, H, W].")
        if pixels.shape[-2] % 16 or pixels.shape[-1] % 16:
            raise ValueError("Image height and width must be divisible by 16.")
        input_dtype = self.dtype if batch.policy else torch.float32
        pixels = pixels.to(device=self.device, dtype=input_dtype)
        history = batch.condition_video
        observation = batch.condition_observation_video
        if history is None:
            if observation is not None:
                raise ValueError(
                    "condition_observation_video requires condition_video."
                )
            condition = self.encode_video(pixels.unsqueeze(2), tiled=batch.tiled)
            clip_pixels = batch.clip_pixel_values
            if clip_pixels is None:
                clip_pixels = pixels
        else:
            history = history.to(device=self.device, dtype=input_dtype)
            if history.shape[:2] != (1, 3) or history.shape[-2:] != pixels.shape[-2:]:
                raise ValueError(
                    "Condition history must match the image batch and spatial size."
                )
            condition = self.encode_video(history, tiled=batch.tiled)
            clip_pixels = history[:, :, 0]
            if observation is not None:
                if observation.shape != (*pixels.shape[:2], 1, *pixels.shape[-2:]):
                    raise ValueError(
                        "Current observation must have shape [1, 3, 1, H, W]."
                    )
                observation = observation.to(device=self.device, dtype=input_dtype)
                condition = torch.cat(
                    [condition, self.encode_video(observation, tiled=batch.tiled)],
                    dim=2,
                )
        clip_pixels = clip_pixels.to(device=self.device, dtype=input_dtype).to(
            torch.float16
        )
        # Match the reference's FP32 resize/normalization and FP16 patch projection.
        clip_pixels = self.image_encoder.preprocess(clip_pixels.float()).to(
            torch.float16
        )
        clip = self.image_encoder(clip_pixels)
        context, context_mask = self.encode_text(batch.input_ids, batch.attention_mask)
        result = {
            "condition_latents": condition,
            "clip_fea": clip.to(self.dtype),
            "context": context,
            "context_mask": context_mask,
        }
        if batch.negative_input_ids is not None:
            if batch.negative_attention_mask is None:
                raise ValueError("Negative text IDs require a negative attention mask.")
            negative, negative_mask = self.encode_text(
                batch.negative_input_ids, batch.negative_attention_mask
            )
            result.update(
                negative_context=negative, negative_context_mask=negative_mask
            )
        return result

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor, *, tiled: bool = False) -> torch.Tensor:
        with torch.autocast(self.device.type, enabled=False):
            return self.vae.decode(
                latents.to(device=self.device, dtype=torch.float32),
                tiled=tiled,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )
