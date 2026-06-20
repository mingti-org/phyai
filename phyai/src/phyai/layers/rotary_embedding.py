"""RoPE with a selectable kernel backend.

Most modern transformer decoders rotate Q and K with the rotate-half
(NeoX-style) variant of RoPE; :class:`Attention` is by design a
pure attention op and expects rotation to happen before it. This module
fills that gap.

A :class:`RotaryEmbedding` holds an fp32 ``cos_sin_cache`` of shape
``(max_position_embeddings, rotary_dim)`` — first half cos, second half
sin — which is exactly what flashinfer's
:func:`flashinfer.rope.apply_rope_with_cos_sin_cache` consumes. The eager
backend reads the same cache in PyTorch with a ``rotate_half`` (or, when
``interleave=True``, an even/odd-split) op.

Backends
--------
* ``"flashinfer"`` (default) — one fused kernel for Q and K.
* ``"eager"`` — pure PyTorch reference, also handy on CPU.

Supported rope_types
--------------------
``"default"`` (no scaling), ``"linear"`` (PI), ``"llama3"``. ``"yarn"``,
``"dynamic"``, ``"longrope"`` are reserved and raise ``NotImplementedError``
to keep the dispatch table closed.

Two usage shapes
----------------
The default call does the position -> cos/sin lookup and the rotation
in one fused launch per layer; signature passes
(``positions`` first, then ``q`` and ``k``):

    q, k = rope(positions, q, k)           # one fused kernel per layer

Alternatively the cos/sin gather can be hoisted out of the layer loop
and the rotation applied per layer with the cached tensors:

    cos, sin = rope.get_cos_sin(positions)   # once at top of stack
    for layer in layers:
        q, k = rope.apply(q, k, cos, sin)   # rotation only

Trades one extra method on the rope object for amortising the cache
gather across N layers. Mostly useful when the same ``positions`` apply
to every layer in the stack and you care about kernel-launch count
(cuda-graph capture, deep stacks).

Shape contract
--------------
Both Q and K must have the same number of tokens (``S`` for padded,
``nnz`` for ragged) — append-prefill (Q shorter than K) is the caller's
responsibility, who can call this layer twice with different
``positions``.

* 4-D padded: ``q (B, S, H_q, D)``, ``k (B, S, H_k, D)``,
  ``positions (B, S)`` or ``(S,)``.
* 3-D ragged: ``q (nnz, H_q, D)``, ``k (nnz, H_k, D)``,
  ``positions (nnz,)``.

Limitations
-----------
* No fused FP8 RoPE (flashinfer's ``rope_quantize_fp8``); not needed for
  prefill-only VLA today.
* No in-place variant — output buffers are always freshly allocated.
* The precomputed-cos/sin path uses the eager apply; flashinfer has no
  kernel that takes pre-gathered cos/sin tensors today.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
import torch.nn as nn

from phyai.engine_config import get_engine_config


_VALID_BACKENDS: tuple[str, ...] = ("flashinfer", "eager")


def _resolve_backend(name: str) -> str:
    canonical = name.replace("_", "-").lower()
    if canonical not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown RoPE backend {name!r}; expected one of {_VALID_BACKENDS!r}."
        )
    return canonical


# ---------------------------------------------------------------------------
# inv_freq computation per rope_type. Each helper returns
# ``(inv_freq, attention_scaling)`` so that the cos/sin cache is built
# uniformly: ``cache = cat([cos, sin], -1) * attention_scaling``.
# ---------------------------------------------------------------------------


def _default_inv_freq(
    rotary_dim: int,
    theta: float,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, float]:
    """Original RoPE: ``inv_freq[i] = theta ** (-2i / rotary_dim)``."""
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float32
            )
            / rotary_dim
        )
    )
    return inv_freq, 1.0


def _linear_inv_freq(
    rotary_dim: int,
    theta: float,
    *,
    factor: float,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, float]:
    """PI / linear scaling: divide every inv_freq by ``factor``."""
    inv_freq, _ = _default_inv_freq(rotary_dim, theta, device=device)
    return inv_freq / factor, 1.0


def _llama3_inv_freq(
    rotary_dim: int,
    theta: float,
    *,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, float]:
    """Llama 3.1 piecewise-smooth scaling.

    Implements the piecewise-smooth long-context scaling described in the
    Llama 3 technical report.
    """
    inv_freq, _ = _default_inv_freq(rotary_dim, theta, device=device)
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(
        wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
    )
    smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium, smoothed, inv_freq_llama)
    return inv_freq_llama, 1.0


def _unsupported_rope_type(name: str) -> Callable[..., tuple[torch.Tensor, float]]:
    def _raise(*_args, **_kwargs):
        raise NotImplementedError(
            f"rope_type={name!r} is reserved but not implemented yet; "
            f"supported types: {tuple(ROPE_INV_FREQ_FNS)!r}."
        )

    return _raise


ROPE_INV_FREQ_FNS: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {
    "default": _default_inv_freq,
    "linear": _linear_inv_freq,
    "llama3": _llama3_inv_freq,
    "yarn": _unsupported_rope_type("yarn"),
    "dynamic": _unsupported_rope_type("dynamic"),
    "longrope": _unsupported_rope_type("longrope"),
}


# ---------------------------------------------------------------------------
# Standalone ops (work on caller-supplied cos/sin).
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[a, b] -> [-b, a]`` along the last dim — the NeoX-style RoPE rotation."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotate-half RoPE with caller-supplied ``cos`` / ``sin``.

    NeoX-style rotate-half convention. ``cos`` and ``sin`` must already be
    sized to the full ``head_dim`` (i.e. each half duplicated).
    ``unsqueeze_dim`` controls which axis becomes the head axis after
    broadcasting:

    * ``unsqueeze_dim=1`` for ``q (B, H, S, D)`` (HF default),
    * ``unsqueeze_dim=2`` for ``q (B, S, H, D)``,
    * ``unsqueeze_dim=-2`` for ragged ``q (nnz, H, D)`` or any "head is
      second-to-last" layout.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


def _rotate_interleave(x: torch.Tensor) -> torch.Tensor:
    """GPT-J / GPT-NeoX-old rotation: interleaved even/odd dims.

    ``[a0, a1, a2, a3, ...] -> [-a1, a0, -a3, a2, ...]``.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1)
    return rotated.flatten(-2)


def _apply_interleaved(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = -2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interleaved-style RoPE on caller-supplied cos/sin.

    ``cos`` / ``sin`` here have layout
    ``[c0, c0, c1, c1, ...]`` / ``[s0, s0, s1, s1, ...]`` so that the
    same index multiplies its even and odd partners.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_out = (q * cos) + (_rotate_interleave(q) * sin)
    k_out = (k * cos) + (_rotate_interleave(k) * sin)
    return q_out, k_out


def compute_cos_sin_from_inv_freq(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    attention_scaling: float = 1.0,
    interleave: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-D RoPE cos/sin from ``inv_freq``, fp32"""
    freqs = positions.unsqueeze(-1).float() * inv_freq.to(
        device=positions.device, dtype=torch.float32
    )
    cos_h = freqs.cos() * attention_scaling
    sin_h = freqs.sin() * attention_scaling
    if interleave:
        cos = cos_h.repeat_interleave(2, dim=-1)
        sin = sin_h.repeat_interleave(2, dim=-1)
    else:
        cos = torch.cat([cos_h, cos_h], dim=-1)
        sin = torch.cat([sin_h, sin_h], dim=-1)
    return cos, sin


def _weave_qwen3vl_mrope(
    freqs: torch.Tensor, mrope_section: tuple[int, ...] | list[int]
) -> torch.Tensor:
    """Weave the 3 axes into ``rotary_dim/2`` slots at stride 3.

    ``freqs`` is ``(3, B, S, rotary_dim/2)``; returns ``(B, S, rotary_dim/2)``.
    Temporal fills every slot first, then height (offset 1) and width
    (offset 2) overwrite their stride-3 positions up to their section span.
    """
    freqs_t = freqs[0]  # temporal occupies all slots initially
    for dim, offset in enumerate((1, 2), start=1):  # height, width
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def compute_qwen3vl_mrope_cos_sin_from_inv_freq(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: tuple[int, ...] | list[int],
    *,
    attention_scaling: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interleaved 3-D mRoPE cos/sin from ``inv_freq`` (Qwen2/3-VL "M-RoPE").

    Recomputes freqs directly from ``inv_freq`` so arbitrary position values
    are supported with no cache-size limit, then weaves the temporal / height
    / width axes into the ``rotary_dim/2`` slots at stride 3
    (see :func:`_weave_qwen3vl_mrope`). Backs
    :meth:`InterleavedMRotaryEmbedding.get_cos_sin`.

    Parameters
    ----------
    position_ids:
        ``(3, B, S)`` temporal/height/width indices, or ``(B, S)`` which is
        broadcast across the three axes (degenerates to plain 1-D RoPE).
    inv_freq:
        ``(rotary_dim / 2,)`` inverse frequencies.
    mrope_section:
        Per-axis ``(t, h, w)`` frequency-slot budget, summing to
        ``rotary_dim / 2``.
    attention_scaling:
        Scalar multiplied into cos/sin.

    Returns
    -------
    cos, sin : torch.Tensor
        Shape ``(B, S, rotary_dim)``, fp32, half-duplicated (rotate-half layout).
    """
    if position_ids.ndim == 2:
        position_ids = position_ids[None].expand(3, *position_ids.shape)
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(
            f"mRoPE position_ids must be (3, B, S) or (B, S), got "
            f"{tuple(position_ids.shape)}."
        )
    inv_freq = inv_freq.to(device=position_ids.device, dtype=torch.float32)
    # (3, B, rotary_dim/2, 1) @ (3, B, 1, S) -> (3, B, rotary_dim/2, S)
    inv_freq_expanded = inv_freq[None, None, :, None].expand(
        3, position_ids.shape[1], -1, 1
    )
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
    freqs = _weave_qwen3vl_mrope(freqs, mrope_section)  # (B, S, rotary_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (B, S, rotary_dim)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return cos, sin


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary position embedding with a selectable kernel backend.

    Parameters
    ----------
    head_dim:
        Per-head channel width of Q / K. Cache rotates the leading
        ``rotary_dim = int(head_dim * partial_rotary_factor)`` channels
        and the trailing ``head_dim - rotary_dim`` are passed through
        unchanged.
    max_position_embeddings:
        Cache size along the sequence axis. ``positions`` must be
        ``< max_position_embeddings``.
    rope_theta:
        RoPE base wavelength. Common values: ``1e4`` for short-context
        configs, ``5e5`` (or larger) for long-context configs.
    rope_type:
        ``"default"``, ``"linear"``, or ``"llama3"``. ``"yarn"``,
        ``"dynamic"``, ``"longrope"`` raise NotImplementedError.
    rope_scaling:
        Extra params for the chosen ``rope_type``. ``"linear"`` requires
        ``{"factor": <float>}``; ``"llama3"`` requires
        ``{"factor", "low_freq_factor", "high_freq_factor",
        "original_max_position_embeddings"}``.
    partial_rotary_factor:
        Fraction of ``head_dim`` that gets rotated. Default ``1.0``.
    interleave:
        ``False`` (default): NeoX-style rotate-half geometry (the
        common case for modern transformer decoders). ``True``:
        GPT-J / NeoX-old even/odd geometry. flashinfer maps
        ``is_neox = not interleave``.
    backend:
        ``"flashinfer"`` (default) or ``"eager"``.
    device:
        Where to allocate the cache. Defaults to
        :attr:`phyai.engine_config.EngineConfig.device` (typically
        ``"cuda"``); pass an explicit ``"cpu"`` / ``"cuda:N"`` to
        override. ``.to(device)`` later still works.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 8192,
        *,
        rope_theta: float = 10000.0,
        rope_type: str = "default",
        rope_scaling: dict | None = None,
        partial_rotary_factor: float = 1.0,
        interleave: bool = False,
        backend: str = "flashinfer",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError(f"head_dim must be a positive even int, got {head_dim}.")
        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError(
                f"partial_rotary_factor must be in (0, 1], got {partial_rotary_factor}."
            )
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError(
                f"rotary_dim={rotary_dim} (=int(head_dim={head_dim} * "
                f"partial_rotary_factor={partial_rotary_factor})) must be a "
                f"positive even int."
            )
        if max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got "
                f"{max_position_embeddings}."
            )
        if rope_type not in ROPE_INV_FREQ_FNS:
            raise ValueError(
                f"Unknown rope_type {rope_type!r}; expected one of "
                f"{tuple(ROPE_INV_FREQ_FNS)!r}."
            )

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.rope_scaling = dict(rope_scaling or {})
        self.partial_rotary_factor = partial_rotary_factor
        self.interleave = interleave
        self.backend = _resolve_backend(backend)
        if device is None:
            device = get_engine_config().device.target

        if self.backend == "flashinfer":
            # Fail fast at construction rather than at first forward.
            try:
                import flashinfer.rope  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "backend='flashinfer' but flashinfer is not installed; "
                    "either install flashinfer-python or pass backend='eager'."
                ) from e

        inv_freq, attention_scaling = ROPE_INV_FREQ_FNS[rope_type](
            rotary_dim, rope_theta, device=device, **self.rope_scaling
        )
        self.attention_scaling = float(attention_scaling)

        # Build the cos/sin cache. flashinfer expects fp32 with first half
        # cos and second half sin; the eager path reads the same buffer.
        t = torch.arange(
            max_position_embeddings, dtype=torch.float32, device=inv_freq.device
        )
        freqs = torch.outer(t, inv_freq.float())  # (max_pos, rotary_dim/2)
        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling
        cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    # ------------------------------------------------------------------ #
    # Forward — positions first, then q, k                                #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-layer fused gather-and-rotate.

        Argument order is ``(positions, query, key)`` so the kernel
        sees the position lookup before the Q/K rotation in one fused
        call.
        """
        self._check_shapes(positions, q, k)
        if self.backend == "flashinfer":
            return self._forward_flashinfer(positions, q, k)
        return self._forward_eager(positions, q, k)

    def _check_shapes(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> None:
        if q.dim() not in (3, 4):
            raise ValueError(
                f"q must be 3-D (ragged) or 4-D (padded), got shape {tuple(q.shape)}."
            )
        if q.dim() != k.dim():
            raise ValueError(
                f"q and k must have the same number of dims, got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}."
            )
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(
                f"q/k last dim must equal head_dim={self.head_dim}, got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}."
            )
        # Q and K must have the same number of tokens. For 4-D this is
        # the (B, S) pair; for 3-D this is the leading dim.
        if q.dim() == 4:
            if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1]:
                raise ValueError(
                    f"q/k must share leading (B, S); got q={tuple(q.shape)}, "
                    f"k={tuple(k.shape)}. Append-prefill (Q shorter than K) "
                    f"is not supported here — call RotaryEmbedding twice "
                    f"with different positions."
                )
        else:  # 3-D
            if q.shape[0] != k.shape[0]:
                raise ValueError(
                    f"q/k ragged token counts differ: q={tuple(q.shape)}, "
                    f"k={tuple(k.shape)}."
                )

    # ---------------------------- flashinfer ---------------------------- #

    def _forward_flashinfer(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from flashinfer.rope import apply_rope_with_cos_sin_cache

        if self.cos_sin_cache.device != q.device:
            raise RuntimeError(
                f"cos_sin_cache is on {self.cos_sin_cache.device} but q is on "
                f"{q.device}; call ``rotary_emb.to(q.device)`` once before "
                f"forward."
            )

        orig_q_shape = q.shape
        orig_k_shape = k.shape
        if q.dim() == 4:
            B, S = q.shape[0], q.shape[1]
            H_q = q.shape[2]
            H_k = k.shape[2]
            nnz = B * S
            flat_q = q.reshape(nnz, H_q * self.head_dim)
            flat_k = k.reshape(nnz, H_k * self.head_dim)
            pos = positions
            if pos.dim() == 1:
                if pos.shape[0] != S:
                    raise ValueError(
                        f"1-D positions length {pos.shape[0]} does not "
                        f"match S={S} for 4-D q."
                    )
                pos = pos.unsqueeze(0).expand(B, S)
            elif pos.shape != (B, S):
                raise ValueError(
                    f"positions shape {tuple(pos.shape)} != (B, S)=({B}, {S})."
                )
            flat_pos = pos.reshape(nnz)
        else:  # 3-D
            nnz = q.shape[0]
            H_q = q.shape[1]
            H_k = k.shape[1]
            flat_q = q.reshape(nnz, H_q * self.head_dim)
            flat_k = k.reshape(nnz, H_k * self.head_dim)
            if positions.dim() != 1 or positions.shape[0] != nnz:
                raise ValueError(
                    f"positions shape {tuple(positions.shape)} does not "
                    f"match (nnz,)=({nnz},) for 3-D q."
                )
            flat_pos = positions
        flat_pos = flat_pos.contiguous().to(torch.int32)

        q_out, k_out = apply_rope_with_cos_sin_cache(
            positions=flat_pos,
            query=flat_q.contiguous(),
            key=flat_k.contiguous(),
            head_size=self.head_dim,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=not self.interleave,
        )
        return q_out.view(orig_q_shape), k_out.view(orig_k_shape)

    # ------------------------------ eager ------------------------------ #

    def _forward_eager(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Eager path: gather cos/sin from the cache and apply via
        # :meth:`apply`. Sharing the apply path with the
        # precomputed-cos/sin route keeps the two numerically identical.
        cos, sin = self.get_cos_sin(positions)
        return self.apply(q, k, cos, sin)

    # ------------------------------------------------------------------ #
    # Precomputed cos/sin                                                #
    # ------------------------------------------------------------------ #

    def get_cos_sin(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather cos/sin from the cache for the given positions.

        Hoist this call to the top of the layer loop when the same
        ``positions`` apply to every layer, then thread the resulting
        ``(cos, sin)`` through every layer's :meth:`apply` (or the
        equivalent ``cos`` / ``sin`` forward kwargs on
        :class:`~phyai.layers.transformer_block.TransformerBlock`).

        Returns
        -------
        cos, sin : torch.Tensor
            Shape ``(*positions.shape, rotary_dim)``, fp32 (cache dtype),
            laid out for ``self.interleave``:

            * ``interleave=False`` (rotate-half / NeoX): each cos/sin
              half is duplicated along the last dim
              (``[c0, c1, ..., c0, c1, ...]``).
            * ``interleave=True`` (GPT-J / NeoX-old): each entry is
              repeated twice in place (``[c0, c0, c1, c1, ...]``).

            The caller broadcasts to the head axis by passing through
            :meth:`apply` (which uses ``unsqueeze_dim=-2``) or with their
            own unsqueeze.
        """
        pos = positions.to(self.cos_sin_cache.device).long()
        cs = self.cos_sin_cache[pos]  # (*pos.shape, rotary_dim)
        cos_h, sin_h = cs.chunk(2, dim=-1)  # each (*pos.shape, rotary_dim/2)
        if self.interleave:
            cos = cos_h.repeat_interleave(2, dim=-1)
            sin = sin_h.repeat_interleave(2, dim=-1)
        else:
            cos = torch.cat([cos_h, cos_h], dim=-1)
            sin = torch.cat([sin_h, sin_h], dim=-1)
        return cos, sin

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE rotation with caller-supplied cos/sin.

        Picks rotate-half vs interleaved based on ``self.interleave`` and
        respects ``self.partial_rotary_factor`` (only the leading
        ``rotary_dim`` channels are rotated; the trailing channels pass
        through). The head axis is auto-broadcast via ``unsqueeze_dim=-2``,
        matching phyai's ``(..., H, D)`` layout for both padded and
        ragged inputs.
        """
        apply_fn = _apply_interleaved if self.interleave else apply_rotary_pos_emb
        cos = cos.to(dtype=q.dtype, device=q.device)
        sin = sin.to(dtype=q.dtype, device=q.device)
        if self.rotary_dim < self.head_dim:
            q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
            k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]
            q_rot, k_rot = apply_fn(q_rot, k_rot, cos, sin, unsqueeze_dim=-2)
            return (
                torch.cat([q_rot, q_pass], dim=-1),
                torch.cat([k_rot, k_pass], dim=-1),
            )
        return apply_fn(q, k, cos, sin, unsqueeze_dim=-2)

    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        s = (
            f"head_dim={self.head_dim}, rotary_dim={self.rotary_dim}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"rope_theta={self.rope_theta}, rope_type={self.rope_type!r}, "
            f"backend={self.backend!r}"
        )
        if self.partial_rotary_factor != 1.0:
            s += f", partial_rotary_factor={self.partial_rotary_factor}"
        if self.interleave:
            s += ", interleave=True"
        return s


class InterleavedMRotaryEmbedding(RotaryEmbedding):
    """Related Papers:
    https://arxiv.org/pdf/2409.12191 Qwen2-VL Technical Report
    https://arxiv.org/abs/2511.21631 Qwen3-VL Technical Report

    Interleaved multimodal 3-D RoPE (Qwen2-VL / Qwen3-VL "M-RoPE").

    Extends :class:`RotaryEmbedding` for the multimodal case where every token
    carries *three* position indices — temporal ``t``, height ``h``, width
    ``w`` — instead of one. The three axes are woven into the
    ``rotary_dim // 2`` frequency slots at stride 3: slot ``i`` takes the
    temporal axis when ``i % 3 == 0``, height when ``i % 3 == 1``, width when
    ``i % 3 == 2``, with each non-temporal axis bounded to its
    ``mrope_section`` span and the tail falling back to temporal. This is the
    *interleaved* layout (Qwen3-VL); it preserves frequency continuity, in
    contrast to the original chunked ``[T..T H..H W..W]`` mRoPE.

    Only cos/sin *computation* differs from the 1-D parent. The rotation itself
    (rotate-half, via the inherited :meth:`apply`) and ``inv_freq``
    are reused unchanged, so the parent's 1-D fused / eager paths stay
    byte-identical.

    Usage — Pattern B (precompute), since a token's three axes cannot be
    expressed as the single ``positions`` index the fused flashinfer kernel
    expects. Compute ``(cos, sin)`` once at the top of the decoder loop and
    thread them through every layer::

        rope = InterleavedMRotaryEmbedding(head_dim=128, mrope_section=[24, 20, 20],
                                           rope_theta=5e5)
        cos, sin = rope.get_cos_sin(position_ids)   # position_ids: (3, B, S)
        for layer in layers:                        # TransformerBlock(precompute_rope=True)
            h = layer(h, cos=cos, sin=sin, attn_ctx=ctx)

    The fused :meth:`forward` is only valid for 1-D / ``(B, S)`` positions and
    is therefore not used for mRoPE.

    Parameters
    ----------
    mrope_section:
        Per-axis ``(t, h, w)`` frequency-slot budget. Must sum to
        ``rotary_dim // 2`` (e.g. ``[24, 20, 20]`` for ``head_dim=128``).
    backend:
        Defaults to ``"eager"`` — the flashinfer fused kernel has no mRoPE
        path, and :meth:`get_cos_sin` is backend-independent regardless.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 8192,
        *,
        mrope_section: list[int] | tuple[int, ...],
        rope_theta: float = 10000.0,
        rope_type: str = "default",
        rope_scaling: dict | None = None,
        backend: str = "eager",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            head_dim,
            max_position_embeddings,
            rope_theta=rope_theta,
            rope_type=rope_type,
            rope_scaling=rope_scaling,
            interleave=False,
            backend=backend,
            device=device,
        )
        section = tuple(int(s) for s in mrope_section)
        if any(s <= 0 for s in section):
            raise ValueError(
                f"mrope_section entries must be positive ints, got {section}."
            )
        if sum(section) != self.rotary_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(section)} must equal rotary_dim//2="
                f"{self.rotary_dim // 2} (rotary_dim={self.rotary_dim}); the three "
                f"axes partition the half-rotary frequency slots."
            )
        self.mrope_section = section
        # The parent built a seq-indexed ``cos_sin_cache`` for its fused /
        # gather paths, but mRoPE recomputes from ``inv_freq`` (see
        # :meth:`get_cos_sin`) and never reads the cache — drop it so each
        # instance doesn't carry a dead ``(max_pos, rotary_dim)`` fp32 buffer.
        del self.cos_sin_cache

    def get_cos_sin(
        self, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build interleaved-mRoPE cos/sin from 3-D ``position_ids``.

        Unlike the parent (which gathers from the seq-indexed ``cos_sin_cache``),
        this recomputes freqs directly from ``inv_freq`` so arbitrary position
        values are supported with no cache-size limit — matching the HF
        reference. Delegates to
        :func:`compute_qwen3vl_mrope_cos_sin_from_inv_freq`.

        Parameters
        ----------
        position_ids:
            ``(3, B, S)`` temporal/height/width indices, or ``(B, S)`` which is
            broadcast across the three axes (degenerates to plain 1-D RoPE).

        Returns
        -------
        cos, sin : torch.Tensor
            Shape ``(B, S, rotary_dim)``, fp32, half-duplicated (rotate-half
            layout) — ready for the inherited :meth:`apply`.
        """
        return compute_qwen3vl_mrope_cos_sin_from_inv_freq(
            position_ids,
            self.inv_freq,
            self.mrope_section,
            attention_scaling=self.attention_scaling,
        )


__all__ = [
    "RotaryEmbedding",
    "InterleavedMRotaryEmbedding",
    "apply_rotary_pos_emb",
    "compute_cos_sin_from_inv_freq",
    "compute_qwen3vl_mrope_cos_sin_from_inv_freq",
    "rotate_half",
    "ROPE_INV_FREQ_FNS",
]
