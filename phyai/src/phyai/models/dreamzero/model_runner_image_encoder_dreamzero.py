"""DreamZero image encoder runner."""

from __future__ import annotations

import torch

from phyai.models.dreamzero.image_encoder_wan import DreamZeroWanImageEncoder
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner


class DreamZeroImageEncoderRunner(ModelRunner):
    """Wraps the DreamZero Wan2.1 CLIP image encoder."""

    def __init__(
        self,
        image_encoder: DreamZeroWanImageEncoder,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        use_cuda_graph: bool = False,
        graph_batch_size: int = 1,
    ) -> None:
        self.image_encoder = image_encoder
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_cuda_graph = bool(use_cuda_graph)
        self.graph_batch_size = int(graph_batch_size)
        if self.graph_batch_size <= 0:
            raise ValueError(
                f"graph_batch_size={self.graph_batch_size} must be positive."
            )
        self.graph: CudaGraph | None = None

    def setup(self) -> None:
        if not self.use_cuda_graph or self.device.type != "cuda":
            return
        example = {
            "pixel_values": torch.zeros(
                self.graph_batch_size,
                self.image_encoder.config.num_channels,
                self.image_encoder.config.image_size,
                self.image_encoder.config.image_size,
                dtype=self.dtype,
                device=self.device,
            ),
        }
        self.graph = CudaGraph()
        self.graph.capture(self._forward_preprocessed, example)

    def _forward_preprocessed(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(pixel_values)

    @torch.no_grad()
    def encode_image(
        self,
        videos: torch.Tensor | list[torch.Tensor],
        *,
        preprocessed: bool = False,
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda",
        ):
            if preprocessed:
                pixel_values = videos
                if not isinstance(pixel_values, torch.Tensor):
                    raise TypeError("preprocessed videos must be a Tensor.")
                pixel_values = pixel_values.to(self.device, self.dtype)
            else:
                pixel_values = self.image_encoder.preprocess(videos).to(
                    self.device, self.dtype
                )
            if self.graph is not None:
                output = self.graph.replay({"pixel_values": pixel_values}).clone()
            else:
                output = self.image_encoder.encode_image(
                    pixel_values, preprocessed=True
                )
            return output

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode_image(pixel_values, preprocessed=True)


__all__ = ["DreamZeroImageEncoderRunner"]
