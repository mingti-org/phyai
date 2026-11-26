"""Wan auxiliary encoder configurations used by VPP2."""

from dataclasses import dataclass

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class VPP2TextEncoderConfig(PretrainedConfig):
    """VPP2 Wan/T5 text encoder checkpoint config."""

    vocab: int = 256384
    dim: int = 4096
    dim_attn: int = 4096
    dim_ffn: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32
    shared_pos: bool = False
    max_length: int = 512

    def __post_init__(self) -> None:
        if self.vocab <= 0:
            raise ValueError(f"vocab={self.vocab} must be positive.")
        if self.dim <= 0:
            raise ValueError(f"dim={self.dim} must be positive.")
        if self.dim_attn <= 0:
            raise ValueError(f"dim_attn={self.dim_attn} must be positive.")
        if self.dim_ffn <= 0:
            raise ValueError(f"dim_ffn={self.dim_ffn} must be positive.")
        if self.num_heads <= 0 or self.dim_attn % self.num_heads != 0:
            raise ValueError(
                f"dim_attn={self.dim_attn} must be divisible by "
                f"num_heads={self.num_heads}."
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers={self.num_layers} must be positive.")
        if self.num_buckets <= 0:
            raise ValueError(f"num_buckets={self.num_buckets} must be positive.")
        if self.max_length <= 0:
            raise ValueError(f"max_length={self.max_length} must be positive.")

    @property
    def head_dim(self) -> int:
        return self.dim_attn // self.num_heads


@dataclass(frozen=True)
class VPP2ImageEncoderConfig(PretrainedConfig):
    """VPP2 CLIP/XLM-R image encoder checkpoint config."""

    image_size: int = 224
    patch_size: int = 14
    vision_dim: int = 1280
    vision_mlp_ratio: int = 4
    vision_heads: int = 16
    vision_layers: int = 32
    vision_pool: str = "token"
    vision_pre_norm: bool = True
    vision_post_norm: bool = False
    activation: str = "gelu"
    norm_eps: float = 1e-5
    num_channels: int = 3
    feature_layer: int = 31

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size={self.image_size} must be positive.")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size={self.patch_size} must be positive.")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by "
                f"patch_size={self.patch_size}."
            )
        if self.vision_dim <= 0:
            raise ValueError(f"vision_dim={self.vision_dim} must be positive.")
        if self.vision_heads <= 0 or self.vision_dim % self.vision_heads != 0:
            raise ValueError(
                f"vision_dim={self.vision_dim} must be divisible by "
                f"vision_heads={self.vision_heads}."
            )
        if self.vision_layers <= 0:
            raise ValueError(f"vision_layers={self.vision_layers} must be positive.")
        if not 0 < self.feature_layer <= self.vision_layers:
            raise ValueError(
                f"feature_layer={self.feature_layer} must be in "
                f"[1, {self.vision_layers}]."
            )
        if self.vision_mlp_ratio <= 0:
            raise ValueError(
                f"vision_mlp_ratio={self.vision_mlp_ratio} must be positive."
            )
        if self.vision_pool not in {"token", "token_fc", "attn_pool"}:
            raise ValueError(f"unsupported vision_pool={self.vision_pool!r}.")
        if self.activation not in {"quick_gelu", "gelu", "swi_glu"}:
            raise ValueError(f"unsupported activation={self.activation!r}.")
        if self.num_channels != 3:
            raise ValueError("VPP2 CLIP requires three RGB channels.")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.vision_dim // self.vision_heads


@dataclass(frozen=True)
class VPP2VAEConfig(PretrainedConfig):
    """VPP2 Wan2.1 VAE checkpoint config."""

    nested_sources = {"temporal_downsample": "temperal_downsample"}

    z_dim: int = 16
    base_dim: int = 96
    decoder_base_dim: int = 96
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temporal_downsample: tuple[bool, ...] = (False, True, True)
    out_channels: int = 3
    patch_size: int = 1
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.dim_mult, tuple):
            object.__setattr__(self, "dim_mult", tuple(self.dim_mult))
        if not isinstance(self.temporal_downsample, tuple):
            object.__setattr__(
                self, "temporal_downsample", tuple(self.temporal_downsample)
            )
        if self.z_dim <= 0:
            raise ValueError(f"z_dim={self.z_dim} must be positive.")
        if self.base_dim <= 0:
            raise ValueError(f"base_dim={self.base_dim} must be positive.")
        if self.decoder_base_dim <= 0:
            raise ValueError(
                f"decoder_base_dim={self.decoder_base_dim} must be positive."
            )
        if self.out_channels <= 0:
            raise ValueError(f"out_channels={self.out_channels} must be positive.")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size={self.patch_size} must be positive.")
        if self.scale_factor_temporal <= 0:
            raise ValueError(
                f"scale_factor_temporal={self.scale_factor_temporal} must be positive."
            )
        if self.scale_factor_spatial <= 0:
            raise ValueError(
                f"scale_factor_spatial={self.scale_factor_spatial} must be positive."
            )
        if self.z_dim != 16 or self.out_channels != 3 or self.patch_size != 1:
            raise ValueError(
                "Wan2.1 requires z_dim=16, out_channels=3, and patch_size=1."
            )
        if self.decoder_base_dim != self.base_dim:
            raise ValueError("Wan2.1 encoder and decoder must use the same base_dim.")
        if len(self.dim_mult) < 2 or any(value <= 0 for value in self.dim_mult):
            raise ValueError("dim_mult must contain at least two positive multipliers.")
        if self.num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive.")
        if len(self.temporal_downsample) != len(self.dim_mult) - 1:
            raise ValueError(
                "temporal_downsample must have one entry per downsampling stage."
            )
        if self.scale_factor_spatial != 2 ** (len(self.dim_mult) - 1):
            raise ValueError(
                "scale_factor_spatial must match the VAE downsampling stages."
            )
        if self.scale_factor_temporal != 2 ** sum(self.temporal_downsample):
            raise ValueError("scale_factor_temporal must match temporal_downsample.")
        if self.scale_factor_temporal != 4:
            raise ValueError("VPP2 requires temporal downsampling by a factor of four.")
