"""VAE runtime wrapper with latent normalization and optional spatial tiling."""

from __future__ import annotations

import torch

from phyai.runtime.model_runner import ModelRunner

from phyai.models.qwen_image_21.vae import QwenImage21VAE


class QwenImage21VAERunner(ModelRunner):
    def __init__(
        self,
        vae: QwenImage21VAE,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        use_tiling: bool = False,
        tile_shape: tuple[int, int] = (256, 256),
        tile_stride: tuple[int, int] = (192, 192),
        use_slicing: bool = False,
    ) -> None:
        self.vae = vae
        self.config = vae.config
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_tiling = use_tiling
        self.use_slicing = use_slicing
        self.tile_shape = tile_shape
        self.tile_stride = tile_stride
        scale = self.config.scale_factor_spatial
        if any(size <= 0 or size % scale for size in tile_shape + tile_stride):
            raise ValueError(
                "VAE tile sizes and strides must be positive multiples of the spatial compression ratio"
            )
        if any(stride > size for size, stride in zip(tile_shape, tile_stride)):
            raise ValueError("VAE tile strides must not exceed tile sizes")
        self.latents_mean = torch.tensor(
            self.config.latents_mean or (0.0,) * self.config.z_dim,
            device=self.device,
            dtype=dtype,
        ).reshape(1, -1, 1, 1, 1)
        self.latents_std = torch.tensor(
            self.config.latents_std or (1.0,) * self.config.z_dim,
            device=self.device,
            dtype=dtype,
        ).reshape(1, -1, 1, 1, 1)

    def setup(self) -> None:
        self.vae.eval()

    def reset(self) -> None:
        return None

    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        normalize: bool = True,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        pixels = pixels.to(device=self.device, dtype=self.dtype)
        if self.use_slicing and pixels.shape[0] > 1:
            moments = torch.cat(
                [self.encode_moments(value) for value in pixels.split(1)], dim=0
            )
        else:
            moments = self.encode_moments(pixels)
        mean, logvar = moments.chunk(2, dim=1)
        if sample:
            noise = torch.randn(
                mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
            )
            latent = mean + (0.5 * logvar.clamp(-30, 20)).exp() * noise
        else:
            latent = mean
        return (latent - self.latents_mean) / self.latents_std if normalize else latent

    def encode_moments(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.use_tiling and any(
            length > tile for length, tile in zip(pixels.shape[-2:], self.tile_shape)
        ):
            return self.tiled_forward(pixels, encode=True)
        return self.vae.encode_moments(pixels)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.dtype)
        if normalize:
            latents = latents * self.latents_std + self.latents_mean
        if self.use_slicing and latents.shape[0] > 1:
            return torch.cat(
                [self.decode_latents(value) for value in latents.split(1)], dim=0
            )
        return self.decode_latents(latents)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        scale = self.config.scale_factor_spatial
        if self.use_tiling and any(
            length * scale > tile
            for length, tile in zip(latents.shape[-2:], self.tile_shape)
        ):
            return self.tiled_forward(latents, encode=False).clamp(-1, 1)
        return self.vae.decode(latents)

    def tiled_forward(self, value: torch.Tensor, *, encode: bool) -> torch.Tensor:
        scale = self.config.scale_factor_spatial
        input_divisor, output_divisor = (1, scale) if encode else (scale, 1)
        tile_h, tile_w = (size // input_divisor for size in self.tile_shape)
        stride_h, stride_w = (size // input_divisor for size in self.tile_stride)
        out_stride_h, out_stride_w = (
            size // output_divisor for size in self.tile_stride
        )
        blend_h, blend_w = (
            (size - stride) // output_divisor
            for size, stride in zip(self.tile_shape, self.tile_stride)
        )
        height, width = value.shape[-2:]
        rows = []
        for top in range(0, height, stride_h):
            row = []
            for left in range(0, width, stride_w):
                tile = value[..., top : top + tile_h, left : left + tile_w]
                row.append(
                    self.vae.encode_moments(tile)
                    if encode
                    else self.vae.decode(tile, clamp_output=False)
                )
            rows.append(row)
        stitched_rows = []
        for row_index, row in enumerate(rows):
            stitched_tiles = []
            for column_index, tile in enumerate(row):
                if row_index:
                    self.blend_vertical(
                        rows[row_index - 1][column_index], tile, blend_h
                    )
                if column_index:
                    self.blend_horizontal(row[column_index - 1], tile, blend_w)
                stitched_tiles.append(tile[..., :out_stride_h, :out_stride_w])
            stitched_rows.append(torch.cat(stitched_tiles, dim=-1))
        output = torch.cat(stitched_rows, dim=-2)
        output_height = height // scale if encode else height * scale
        output_width = width // scale if encode else width * scale
        return output[..., :output_height, :output_width]

    @staticmethod
    def blend_vertical(
        previous: torch.Tensor, current: torch.Tensor, extent: int
    ) -> None:
        extent = min(previous.shape[-2], current.shape[-2], extent)
        for offset in range(extent):
            current[..., offset, :] = previous[..., -extent + offset, :] * (
                1 - offset / extent
            ) + current[..., offset, :] * (offset / extent)

    @staticmethod
    def blend_horizontal(
        previous: torch.Tensor, current: torch.Tensor, extent: int
    ) -> None:
        extent = min(previous.shape[-1], current.shape[-1], extent)
        for offset in range(extent):
            current[..., offset] = previous[..., -extent + offset] * (
                1 - offset / extent
            ) + current[..., offset] * (offset / extent)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)


__all__ = ["QwenImage21VAERunner"]
