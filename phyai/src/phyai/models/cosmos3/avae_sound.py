"""AVAE sound decoder for Cosmos3"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from phyai.layers.activation import Snake1d
from phyai.layers.conv import Conv1d, ConvTranspose1d
from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3AVAESoundConfig


class OobleckResidualUnit(nn.Module):
    def __init__(self, dimension: int, dilation: int = 1, prefix: str = "") -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension, prefix=f"{prefix}.snake1" if prefix else "")
        self.conv1 = Conv1d(
            dimension,
            dimension,
            kernel_size=7,
            dilation=dilation,
            padding=pad,
            weight_norm=True,
            prefix=f"{prefix}.conv1" if prefix else "",
        )
        self.snake2 = Snake1d(dimension, prefix=f"{prefix}.snake2" if prefix else "")
        self.conv2 = Conv1d(
            dimension,
            dimension,
            kernel_size=1,
            weight_norm=True,
            prefix=f"{prefix}.conv2" if prefix else "",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.snake1(x))
        out = self.conv2(self.snake2(out))
        pad = (x.shape[-1] - out.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + out


class OobleckDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        output_padding: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.snake1 = Snake1d(input_dim, prefix=f"{prefix}.snake1" if prefix else "")
        self.conv_t1 = ConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=output_padding,
            weight_norm=True,
            prefix=f"{prefix}.conv_t1" if prefix else "",
        )
        self.res_unit1 = OobleckResidualUnit(
            output_dim, dilation=1, prefix=f"{prefix}.res_unit1" if prefix else ""
        )
        self.res_unit2 = OobleckResidualUnit(
            output_dim, dilation=3, prefix=f"{prefix}.res_unit2" if prefix else ""
        )
        self.res_unit3 = OobleckResidualUnit(
            output_dim, dilation=9, prefix=f"{prefix}.res_unit3" if prefix else ""
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        return self.res_unit3(x)


class OobleckDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: list[int],
        channel_multiples: list[int],
        prefix: str = "",
    ) -> None:
        super().__init__()
        strides = upsampling_ratios
        channel_multiples = [1] + list(channel_multiples)
        self.conv1 = Conv1d(
            input_channels,
            channels * channel_multiples[-1],
            kernel_size=7,
            padding=3,
            weight_norm=True,
            prefix=f"{prefix}.conv1" if prefix else "",
        )
        block = []
        for stride_index, stride in enumerate(strides):
            block.append(
                OobleckDecoderBlock(
                    input_dim=channels * channel_multiples[len(strides) - stride_index],
                    output_dim=channels
                    * channel_multiples[len(strides) - stride_index - 1],
                    stride=stride,
                    output_padding=stride % 2,
                    prefix=f"{prefix}.block.{stride_index}" if prefix else "",
                )
            )
        self.block = nn.ModuleList(block)
        self.snake1 = Snake1d(channels, prefix=f"{prefix}.snake1" if prefix else "")
        self.conv2 = Conv1d(
            channels,
            audio_channels,
            kernel_size=7,
            padding=3,
            bias=False,
            weight_norm=True,
            prefix=f"{prefix}.conv2" if prefix else "",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for layer in self.block:
            x = layer(x)
        x = self.snake1(x)
        return self.conv2(x)


class Cosmos3AVAESoundDecoder(nn.Module):
    """sound latent ``[B, latent_ch, T]`` to waveform."""

    def __init__(
        self,
        config: Cosmos3AVAESoundConfig,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(config, Cosmos3AVAESoundConfig):
            raise TypeError(
                f"Expected Cosmos3AVAESoundConfig, got {type(config).__name__}"
            )
        super().__init__()
        self.config = config
        self.latent_ch = config.latent_ch
        self.audio_channels = config.audio_channels
        self.sample_rate = config.sample_rate
        self.hop_size = math.prod(config.dec_strides)
        self.decoder = OobleckDecoder(
            channels=config.dec_dim,
            input_channels=config.latent_ch,
            audio_channels=config.audio_channels,
            upsampling_ratios=list(reversed(config.dec_strides)),
            channel_multiples=list(config.dec_c_mults),
            prefix="decoder",
        )
        if device is not None:
            self.to(device=device)
        if dtype is not None:
            self.to(dtype=dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``[B, latent_ch, T]`` -> ``[B, audio_channels, T * hop_size]`` in [-1, 1]."""
        return self.decoder(latent).clamp(-1.0, 1.0)


def cosmos3_avae_weight_remap(key: str) -> str | None:
    """Map a diffusers ``sound_tokenizer/`` checkpoint key to a phyai AVAE param name.

    Fed to :func:`phyai.weights.load_pretrained` as its ``remap``. Keeps the decode
    path (``decoder.*``) as identity — the phyai param paths and the legacy
    ``weight_norm`` leaf names (``weight_g`` / ``weight_v``, folded at load by the
    conv layers' ``weight_norm=True``) already match the checkpoint — and drops
    everything else (the encoder, which this decode-only module does not build).
    """
    return key if key.startswith("decoder.") else None


__all__ = ["Cosmos3AVAESoundDecoder", "OobleckDecoder", "cosmos3_avae_weight_remap"]
