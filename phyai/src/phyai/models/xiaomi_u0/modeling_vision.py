"""Native image-only Emu3.5 encoding and decoding.

Architecture and checkpoint names follow BAAI/Emu3.5-VisionTokenizer.
The encoder and decoder load separately.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from phyai.layers.attention.nocache.layer import Attention
from phyai.layers.conv import Conv2d
from phyai.models.xiaomi_u0.configuration_vision import U0VisionConfig
from phyai.weights.shards import replicated


def vision_norm(
    channels: int, *, prefix: str, device: torch.device | str
) -> nn.GroupNorm:
    """Build FP32 GroupNorm and register its affine checkpoint keys."""
    norm = nn.GroupNorm(32, channels, eps=1e-6, device=device, dtype=torch.float32)
    for name, param in norm.named_parameters():
        param.requires_grad_(False)
        param.hf_keys = [(f"{prefix}.{name}", None)]
        param.weight_loader = replicated()
    return norm


def vision_conv(
    in_channels: int,
    out_channels: int,
    kernel: int,
    *,
    prefix: str,
    device: torch.device | str,
) -> Conv2d:
    """Create a checkpoint-backed FP32 convolution without changing spatial size."""
    return Conv2d(
        in_channels,
        out_channels,
        kernel,
        padding=kernel // 2,
        dtype=torch.float32,
        device=device,
        prefix=prefix,
    )


class U0VisionResidual(nn.Module):
    """Two normalized convolutions with the checkpoint's optional 1x1 shortcut."""

    def __init__(
        self, inputs: int, outputs: int, *, prefix: str, device: torch.device | str
    ) -> None:
        """Construct FP32 residual projections using their original checkpoint names."""
        super().__init__()
        self.norm1 = vision_norm(inputs, prefix=f"{prefix}.norm1", device=device)
        self.norm2 = vision_norm(outputs, prefix=f"{prefix}.norm2", device=device)
        self.conv1 = vision_conv(
            inputs, outputs, 3, prefix=f"{prefix}.conv1", device=device
        )
        self.conv2 = vision_conv(
            outputs, outputs, 3, prefix=f"{prefix}.conv2", device=device
        )
        if inputs != outputs:
            self.nin_shortcut = vision_conv(
                inputs, outputs, 1, prefix=f"{prefix}.nin_shortcut", device=device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize, apply Swish and add the projected or identity residual."""
        h = self.norm1(x)
        h = self.conv1(h * torch.sigmoid(h))
        h = self.norm2(h)
        h = self.conv2(h * torch.sigmoid(h))
        return (self.nin_shortcut(x) if hasattr(self, "nin_shortcut") else x) + h


class U0VisionAttention(nn.Module):
    """One-head spatial attention using PhyAI's noncausal SDPA backend."""

    def __init__(
        self, channels: int, *, prefix: str, device: torch.device | str
    ) -> None:
        """Build the checkpoint's convolutional Q/K/V and output projections."""
        super().__init__()
        self.norm = vision_norm(channels, prefix=f"{prefix}.norm", device=device)
        for name in ("q", "k", "v", "proj_out"):
            setattr(
                self,
                name,
                vision_conv(
                    channels, channels, 1, prefix=f"{prefix}.{name}", device=device
                ),
            )
        # The released decoder has one wide FP32 head, outside FlashInfer's
        # paged AR contract. SDPA handles it through the shared layer API.
        self.attention = Attention(
            1,
            channels,
            scale=1.0,
            causal=False,
            backend="sdpa",
            backend_kwargs={"compile": False, "select_kernel": False},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over all spatial cells and add the projected result to the input."""
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q, k, v = (
            projection(h).flatten(2).transpose(1, 2).unsqueeze(2)
            for projection in (self.q, self.k, self.v)
        )
        # Match the released FP32 arithmetic without SDPA splitting the scale.
        h = self.attention(q * (channels**-0.5), k, v).squeeze(2).transpose(1, 2)
        return x + self.proj_out(h.reshape(batch, channels, height, width))


class U0VisionUpsample(nn.Module):
    """Nearest-neighbor spatial expansion followed by the learned convolution."""

    def __init__(
        self, channels: int, *, prefix: str, device: torch.device | str
    ) -> None:
        """Create the upsampling convolution with the checkpoint's nested key."""
        super().__init__()
        self.conv = vision_conv(
            channels, channels, 3, prefix=f"{prefix}.conv", device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double each spatial dimension before applying the convolution."""
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class U0VisionStage(nn.Module):
    """Residual blocks at one decoder resolution, with optional attention and upsampling."""

    def __init__(
        self,
        config: U0VisionConfig,
        level: int,
        inputs: int,
        resolution: int,
        *,
        device: torch.device | str,
    ) -> None:
        """Keep block and attention indices aligned with decoder.up checkpoint keys."""
        super().__init__()
        prefix = f"decoder.up.{level}"
        outputs = config.ch * config.ch_mult[level]
        self.block = nn.ModuleList(
            [
                U0VisionResidual(
                    inputs if index == 0 else outputs,
                    outputs,
                    prefix=f"{prefix}.block.{index}",
                    device=device,
                )
                for index in range(config.num_res_blocks + 1)
            ]
        )
        self.attn = nn.ModuleList(
            [
                U0VisionAttention(
                    outputs, prefix=f"{prefix}.attn.{index}", device=device
                )
                for index in range(config.num_res_blocks + 1)
            ]
            if resolution in config.attn_resolutions
            else []
        )
        if level:
            self.upsample = U0VisionUpsample(
                outputs, prefix=f"{prefix}.upsample", device=device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run this resolution's blocks and expand spatially when another stage follows."""
        for index, block in enumerate(self.block):
            x = block(x)
            if self.attn:
                x = self.attn[index](x)
        return self.upsample(x) if hasattr(self, "upsample") else x


class U0VisionMiddle(nn.Module):
    """Bottleneck residual-attention-residual sequence."""

    def __init__(
        self, channels: int, *, prefix: str = "decoder.mid", device: torch.device | str
    ) -> None:
        """Construct the three checkpoint-backed bottleneck modules."""
        super().__init__()
        self.block_1 = U0VisionResidual(
            channels, channels, prefix=f"{prefix}.block_1", device=device
        )
        self.attn_1 = U0VisionAttention(
            channels, prefix=f"{prefix}.attn_1", device=device
        )
        self.block_2 = U0VisionResidual(
            channels, channels, prefix=f"{prefix}.block_2", device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both residual blocks around the spatial attention layer."""
        return self.block_2(self.attn_1(self.block_1(x)))


class U0VisionDownsample(nn.Module):
    """Halve the spatial dimensions with right and bottom padding."""

    def __init__(
        self, channels: int, *, prefix: str, device: torch.device | str
    ) -> None:
        """Build the stride-two convolution under its checkpoint key."""
        super().__init__()
        self.conv = Conv2d(
            channels,
            channels,
            3,
            stride=2,
            dtype=torch.float32,
            device=device,
            prefix=f"{prefix}.conv",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the right and bottom edges before applying the convolution."""
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class U0VisionEncoderStage(nn.Module):
    """One encoder resolution with checkpoint-indexed residual and attention blocks."""

    def __init__(
        self,
        config: U0VisionConfig,
        level: int,
        inputs: int,
        *,
        device: torch.device | str,
    ) -> None:
        """Build the encoder blocks and downsample between resolutions."""
        super().__init__()
        prefix = f"encoder.down.{level}"
        outputs = config.ch * config.ch_mult[level]
        self.block = nn.ModuleList(
            U0VisionResidual(
                inputs if index == 0 else outputs,
                outputs,
                prefix=f"{prefix}.block.{index}",
                device=device,
            )
            for index in range(config.num_res_blocks)
        )
        self.attn = nn.ModuleList()
        if (config.resolution >> level) in config.attn_resolutions:
            self.attn.extend(
                U0VisionAttention(
                    outputs, prefix=f"{prefix}.attn.{index}", device=device
                )
                for index in range(config.num_res_blocks)
            )
        if level < len(config.ch_mult) - 1:
            self.downsample = U0VisionDownsample(
                outputs, prefix=f"{prefix}.downsample", device=device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks in order and downsample if another stage follows."""
        for index, block in enumerate(self.block):
            x = block(x)
            if self.attn:
                x = self.attn[index](x)
        return self.downsample(x) if hasattr(self, "downsample") else x


class U0VisionEncoder(nn.Module):
    """Map normalized RGB pixels to a latent grid for IBQ quantization."""

    def __init__(self, config: U0VisionConfig, *, device: torch.device | str) -> None:
        """Build the checkpoint's fine-to-coarse encoder."""
        super().__init__()
        channels = config.ch
        self.conv_in = vision_conv(
            3, channels, 3, prefix="encoder.conv_in", device=device
        )
        self.down = nn.ModuleList()
        for level, multiplier in enumerate(config.ch_mult):
            self.down.append(
                U0VisionEncoderStage(config, level, channels, device=device)
            )
            channels = config.ch * multiplier
        self.mid = U0VisionMiddle(channels, prefix="encoder.mid", device=device)
        self.norm_out = vision_norm(channels, prefix="encoder.norm_out", device=device)
        self.conv_out = vision_conv(
            channels, config.z_channels, 3, prefix="encoder.conv_out", device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent grid before the quantization projection."""
        x = self.conv_in(x)
        for stage in self.down:
            x = stage(x)
        x = self.norm_out(self.mid(x))
        return self.conv_out(x * torch.sigmoid(x))


class U0VisionDecoder(nn.Module):
    """Convert channel-first quantized features to unconstrained RGB values."""

    def __init__(self, config: U0VisionConfig, *, device: torch.device | str) -> None:
        """Build the released decoder topology without allocating its unused encoder."""
        super().__init__()
        channels = config.ch * config.ch_mult[-1]
        self.conv_in = vision_conv(
            config.z_channels, channels, 3, prefix="decoder.conv_in", device=device
        )
        self.mid = U0VisionMiddle(channels, device=device)
        resolution = config.resolution // config.spatial_factor
        stages = []
        for level in reversed(range(len(config.ch_mult))):
            stages.append(
                U0VisionStage(config, level, channels, resolution, device=device)
            )
            channels = config.ch * config.ch_mult[level]
            if level:
                resolution *= 2
        self.up = nn.ModuleList(reversed(stages))
        self.norm_out = vision_norm(channels, prefix="decoder.norm_out", device=device)
        self.conv_out = vision_conv(
            channels, config.out_ch, 3, prefix="decoder.conv_out", device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode one latent grid using the checkpoint's coarse-to-fine stage order."""
        x = self.mid(self.conv_in(x))
        for stage in reversed(self.up):
            x = stage(x)
        x = self.norm_out(x)
        return self.conv_out(x * torch.sigmoid(x))


def vision_codebook(config: U0VisionConfig, device: torch.device | str) -> nn.Module:
    """Create the shared checkpoint codebook with FP32 loader metadata."""
    quantize = nn.Module()
    quantize.embedding = nn.Embedding(
        config.codebook_size, config.embed_dim, dtype=torch.float32, device=device
    )
    weight = quantize.embedding.weight
    weight.requires_grad_(False)
    weight.hf_keys = [("quantize.embedding.weight", None)]
    weight.weight_loader = replicated()
    return quantize


class U0VisionImageEncoder(nn.Module):
    """Encode normalized RGB pixels into Emu3.5 codebook IDs."""

    def __init__(self, config: U0VisionConfig, *, device: torch.device | str) -> None:
        """Build the encoder, pre-quantization projection and IBQ codebook."""
        super().__init__()
        self.config = config
        self.encoder = U0VisionEncoder(config, device=device)
        self.quant_conv = vision_conv(
            config.z_channels, config.embed_dim, 1, prefix="quant_conv", device=device
        )
        self.quantize = vision_codebook(config, device)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Return one code ID per latent cell using the largest codebook dot product."""
        latent = self.quant_conv(self.encoder(pixels))
        scores = torch.einsum(
            "b d h w, n d -> b n h w", latent, self.quantize.embedding.weight
        )
        return scores.argmax(dim=1)


class U0VisionTokenizer(nn.Module):
    """Stateless image decoding from Emu3.5 codebook IDs, using FP32 parameters."""

    def __init__(self, config: U0VisionConfig, *, device: torch.device | str) -> None:
        """Allocate codebook lookup, post-quantization projection and the RGB decoder."""
        super().__init__()
        self.config = config
        self.quantize = vision_codebook(config, device)
        self.post_quant_conv = vision_conv(
            config.embed_dim,
            config.z_channels,
            1,
            prefix="post_quant_conv",
            device=device,
        )
        self.decoder = U0VisionDecoder(config, device=device)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode a batch of integer code grids shaped (batch, height, width)."""
        if self.training:
            raise RuntimeError("The U0 VisionTokenizer requires eval mode.")
        if (
            codes.ndim != 3
            or codes.dtype not in (torch.int32, torch.int64)
            or not codes.numel()
        ):
            raise ValueError(
                "Code grids must be nonempty integer (batch, height, width) tensors."
            )
        latent = self.quantize.embedding(codes).permute(0, 3, 1, 2).contiguous()
        return self.decoder(self.post_quant_conv(latent))


def vision_decode_weight_remap(name: str) -> str | None:
    """Skip only the encoder and encoder-side quantization projection during loading."""
    if name.startswith(("encoder.", "quant_conv.")):
        return None
    return name


def vision_encode_weight_remap(name: str) -> str | None:
    """Skip only the decoder and its post-quantization projection when encoding."""
    if name.startswith(("decoder.", "post_quant_conv.")):
        return None
    return name
