"""DreamZero image encoder runner."""

from __future__ import annotations

import torch

from phyai.layers.attention import AttnCtx, AttnLayout
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
        self._attn_contexts: tuple[AttnCtx, ...] = ()

    def setup(self) -> None:
        self.close()
        if not self.use_cuda_graph or self.device.type != "cuda":
            return
        config = self.image_encoder.config
        tokens = config.num_patches + int(config.vision_pool in {"token", "token_fc"})
        q = torch.empty(
            self.graph_batch_size,
            tokens,
            self.image_encoder.visual.transformer[0].attn.num_heads,
            config.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        # FlashInfer plans outside capture. Each selected block owns its
        # backend; the runner retains the plans for this graph's lifetime.
        self._attn_contexts = tuple(
            block.attn.attention._build_default_ctx(
                q, q, q, AttnLayout.PADDED_4D, None, None, mode="capture"
            )
            for block in self.image_encoder.visual.transformer
        )
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
        with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
            self.graph.capture(self._forward_preprocessed, example)

    def close(self) -> None:
        if self.graph is not None:
            self.graph.reset()
            self.graph = None
        self._attn_contexts = ()

    def _forward_preprocessed(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(pixel_values, attn_contexts=self._attn_contexts)

    @torch.no_grad()
    def forward(
        self,
        videos: torch.Tensor,
        *,
        preprocessed: bool = True,
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda",
        ):
            pixel_values = videos.to(self.device)
            if not preprocessed:
                pixel_values = self.image_encoder.preprocess(pixel_values)
            pixel_values = pixel_values.to(self.dtype)
            if self.graph is not None:
                output = self.graph.replay({"pixel_values": pixel_values}).clone()
            else:
                output = self.image_encoder(pixel_values)
            return output


__all__ = ["DreamZeroImageEncoderRunner"]
