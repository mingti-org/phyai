"""Config classes for the Cosmos3 model family.

:class:`Cosmos3Config` mirrors ``transformer/config.json``,
:class:`Cosmos3WanVAEConfig` mirrors ``vae/config.json``, and
:class:`Cosmos3AVAESoundConfig` mirrors ``sound_tokenizer/config.json``
of the public ``Cosmos3-Nano`` checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class Cosmos3Config(PretrainedConfig):
    """Cosmos3 MoT transformer config (``Cosmos3OmniTransformer``).

    The UND and GEN pathways share ``num_hidden_layers`` (one GEN cross-attn
    layer per UND self-attn layer). ``mrope_section`` sums to ``head_dim // 2``;
    ``rope_theta`` is the long-context 5e6 used by the Cosmos3/Qwen3-VL text
    backbone. ``patch_latent_dim`` must equal ``latent_patch_size**2 *
    latent_channel`` (the per-patch flattened latent width fed to ``proj_in``).

    The checkpoint nests ``mrope_section`` inside ``rope_scaling``;
    :attr:`nested_sources` lifts it so ``Cosmos3Config.from_dict`` reads the real
    ``transformer/config.json`` directly (``latent_channel`` and the other knobs
    are already flat). Unrelated ``_class_name`` / ``_diffusers_version`` / dtype
    keys are dropped by the base filter.
    """

    nested_sources = {"mrope_section": "rope_scaling.mrope_section"}

    # Transformer core (Qwen3-VL text backbone dims).
    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    hidden_act: str = "silu"
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False

    # RoPE
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)
    max_position_embeddings: int = 262144

    # Latent
    latent_channel: int = 48
    latent_patch_size: int = 2
    patch_latent_dim: int = 192

    # Diffusion
    timestep_scale: float = 0.001
    base_fps: float = 24.0
    enable_fps_modulation: bool = True
    temporal_compression_factor: int = 4
    temporal_modality_margin: int = 15000

    # QK Norm
    qk_norm_for_text: bool = True
    qk_norm_for_diffusion: bool = True

    # Optional modalities (omitted from the T2V build when False).
    action_gen: bool = False
    action_dim: int = 64
    num_embodiment_domains: int = 32
    sound_gen: bool = False
    sound_dim: int = 64
    sound_latent_fps: float = 25.0
    temporal_compression_factor_sound: int = 1

    def __post_init__(self) -> None:
        # Lifted from JSON, mrope_section arrives as a list; coerce so the frozen
        # config stays hashable and equality with the tuple default holds.
        if not isinstance(self.mrope_section, tuple):
            object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"num_key_value_heads={self.num_key_value_heads}."
            )
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be a positive even int.")
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(self.mrope_section)} must equal "
                f"head_dim//2={self.head_dim // 2}; the temporal/height/width axes "
                f"partition the half-rotary frequency slots."
            )
        expected_patch = self.latent_patch_size**2 * self.latent_channel
        if self.patch_latent_dim != expected_patch:
            raise ValueError(
                f"patch_latent_dim={self.patch_latent_dim} must equal "
                f"latent_patch_size**2 * latent_channel={expected_patch}."
            )

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass(frozen=True)
class Cosmos3WanVAEConfig(PretrainedConfig):
    """WAN VAE config (``vae/config.json``)."""

    z_dim: int = 48
    decoder_base_dim: int = 256
    base_dim: int = 160
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    out_channels: int = 12
    patch_size: int = 2
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 16
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dim_mult, tuple):
            object.__setattr__(self, "dim_mult", tuple(self.dim_mult))
        if not isinstance(self.temperal_downsample, tuple):
            object.__setattr__(
                self, "temperal_downsample", tuple(self.temperal_downsample)
            )
        if self.latents_mean is not None and not isinstance(self.latents_mean, tuple):
            object.__setattr__(self, "latents_mean", tuple(self.latents_mean))
        if self.latents_std is not None and not isinstance(self.latents_std, tuple):
            object.__setattr__(self, "latents_std", tuple(self.latents_std))


@dataclass(frozen=True)
class Cosmos3AVAESoundConfig(PretrainedConfig):
    """AVAE sound decoder config (``sound_tokenizer/config.json``)."""

    nested_sources = {
        "latent_ch": "vocoder_input_dim",
        "audio_channels": "dec_out_channels",
        "sample_rate": "sampling_rate",
    }

    dec_dim: int = 320
    latent_ch: int = 64
    audio_channels: int = 2
    dec_strides: tuple[int, ...] = (2, 4, 5, 6, 8)
    dec_c_mults: tuple[int, ...] = (1, 2, 4, 8, 16)
    sample_rate: int = 48000

    def __post_init__(self) -> None:
        if not isinstance(self.dec_strides, tuple):
            object.__setattr__(self, "dec_strides", tuple(self.dec_strides))
        if not isinstance(self.dec_c_mults, tuple):
            object.__setattr__(self, "dec_c_mults", tuple(self.dec_c_mults))


__all__ = ["Cosmos3Config", "Cosmos3WanVAEConfig", "Cosmos3AVAESoundConfig"]
