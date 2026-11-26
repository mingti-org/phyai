"""Configuration for the Emu3.5 image decoder used by Xiaomi U0."""

from dataclasses import dataclass

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class U0VisionConfig(PretrainedConfig):
    """Decoder dimensions from the local VisionTokenizer configuration."""

    model_type: str = "Emu3p5VisionVQ"
    ch: int = 256
    ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4)
    num_res_blocks: int = 4
    attn_resolutions: tuple[int, ...] = (16,)
    resolution: int = 256
    z_channels: int = 256
    embed_dim: int = 256
    codebook_size: int = 131072
    out_ch: int = 3
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Reject configurations that cannot use the released image decoder layout."""
        if self.model_type != "Emu3p5VisionVQ":
            raise ValueError("Expected an Emu3p5VisionVQ checkpoint.")
        object.__setattr__(self, "ch_mult", tuple(self.ch_mult))
        object.__setattr__(self, "attn_resolutions", tuple(self.attn_resolutions))
        for name in (
            "ch",
            "num_res_blocks",
            "resolution",
            "z_channels",
            "embed_dim",
            "codebook_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not self.ch_mult or any(type(m) is not int or m <= 0 for m in self.ch_mult):
            raise ValueError("ch_mult must contain positive integer multipliers.")
        if self.ch % 32:
            raise ValueError("Decoder channels must be divisible by 32 for GroupNorm.")
        if self.resolution % self.spatial_factor:
            raise ValueError(
                "resolution must be divisible by the spatial scale factor."
            )
        if any(type(r) is not int or r <= 0 for r in self.attn_resolutions):
            raise ValueError("Attention resolutions must be positive integers.")
        if self.out_ch != 3 or self.dropout != 0:
            raise ValueError("Only RGB inference with dropout=0 is supported.")

    @property
    def spatial_factor(self) -> int:
        """Return the pixel width or height produced by one codebook cell."""
        return 2 ** (len(self.ch_mult) - 1)
