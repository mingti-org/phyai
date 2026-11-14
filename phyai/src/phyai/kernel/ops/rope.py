"""Define rotary position embedding implementations."""

from __future__ import annotations

from phyai.kernel.facts import lib, attrs, dtype, shape, device
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import all_of

ROPE = OpSpec(
    name="rope",
    dims=("tokens", "head_dim", "rotary_dim", "heads_q", "heads_k"),
    dtypes=("input",),
    attributes=("rope_type", "interleave"),
    optional_attributes=("rank",),
    # ``cos_sin_cache`` is the layer's ``(max_pos, rotary_dim)`` fp32
    # buffer: first half cos, second half sin.
    signature="(positions, q, k, cos_sin_cache) -> tuple[Tensor, Tensor]",
)


FLASHINFER_ROPE = all_of(
    lib.has("flashinfer"),
    device.vendor == "nvidia",
    dtype.input.in_({"bf16", "fp16"}),
    shape.rotary_dim == shape.head_dim,
)


AITER_ROPE = all_of(
    lib.has("aiter"),
    device.vendor == "amd",
    dtype.input.in_({"bf16", "fp16"}),
    shape.rotary_dim == shape.head_dim,
)


def _flatten_tokens(positions, q, k):
    """Normalize padded/ragged Q, K, positions to flat ``(nnz, H*D)`` / ``(nnz,)``."""
    head_dim = q.shape[-1]
    if q.dim() == 4:
        B, S, H_q = q.shape[0], q.shape[1], q.shape[2]
        H_k = k.shape[2]
        nnz = B * S
        flat_q = q.reshape(nnz, H_q * head_dim)
        flat_k = k.reshape(nnz, H_k * head_dim)
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
        nnz, H_q = q.shape[0], q.shape[1]
        H_k = k.shape[1]
        flat_q = q.reshape(nnz, H_q * head_dim)
        flat_k = k.reshape(nnz, H_k * head_dim)
        if positions.dim() != 1 or positions.shape[0] != nnz:
            raise ValueError(
                f"positions shape {tuple(positions.shape)} does not "
                f"match (nnz,)=({nnz},) for 3-D q."
            )
        flat_pos = positions
    return flat_pos, flat_q, flat_k


def _flashinfer(facts, params):
    # Import during preparation so selection can fall back on failure.
    import torch
    from flashinfer.rope import apply_rope_with_cos_sin_cache

    is_neox = not bool(facts.lookup("attrs.interleave"))

    def rope(positions, q, k, cos_sin_cache):
        flat_pos, flat_q, flat_k = _flatten_tokens(positions, q, k)
        q_out, k_out = apply_rope_with_cos_sin_cache(
            positions=flat_pos.contiguous().to(torch.int32),
            query=flat_q.contiguous(),
            key=flat_k.contiguous(),
            head_size=q.shape[-1],
            cos_sin_cache=cos_sin_cache,
            is_neox=is_neox,
        )
        return q_out.view(q.shape), k_out.view(k.shape)

    return rope


def _aiter(facts, params):
    from aiter import rope_cached_2c_fwd

    # aiter rotate_style: 0 rotates halves (NeoX), 1 rotates pairs (GPT-J).
    rotate_style = 1 if facts.lookup("attrs.interleave") else 0

    def rope(positions, q, k, cos_sin_cache):
        flat_pos, flat_q, flat_k = _flatten_tokens(positions, q, k)
        head_dim = q.shape[-1]
        tokens = flat_pos.shape[0]
        # aiter reads cos and sin as contiguous ``(tokens, 1, 1, rotary_dim // 2)``
        # and silently misreads strided views of the packed cache.
        cos, sin = (
            cos_sin_cache.index_select(0, flat_pos)
            .view(tokens, 1, 1, -1)
            .chunk(2, dim=-1)
        )
        q_out, k_out = rope_cached_2c_fwd(
            flat_q.contiguous().view(tokens, 1, -1, head_dim),
            flat_k.contiguous().view(tokens, 1, -1, head_dim),
            cos.contiguous(),
            sin.contiguous(),
            rotate_style,
            True,
            False,
        )
        return q_out.view(q.shape), k_out.view(k.shape)

    return rope


def _eager(facts, params):
    # Import during preparation. The rotation helpers live with the layer's
    # other public RoPE math; reusing them here keeps the eager kernel and
    # the precomputed-cos/sin path numerically identical by construction.
    from phyai.layers.rotary_embedding import apply_rope, gather_cos_sin

    interleave = bool(facts.lookup("attrs.interleave"))

    def rope(positions, q, k, cos_sin_cache):
        cos, sin = gather_cos_sin(cos_sin_cache, positions, interleave=interleave)
        return apply_rope(
            q,
            k,
            cos,
            sin,
            interleave=interleave,
            rotary_dim=cos_sin_cache.shape[-1],
        )

    return rope


def register(catalog: Catalog) -> None:
    catalog.register_op(ROPE)
    catalog.register_many(
        (
            Impl(
                kernel_id="flashinfer.rope",
                op="rope",
                priority=Priority.OPTIMIZED + 2,
                when=FLASHINFER_ROPE,
                prepare=_flashinfer,
                metadata={"package": "flashinfer"},
            ),
            Impl(
                kernel_id="aiter.rope",
                op="rope",
                priority=Priority.OPTIMIZED + 2,
                when=AITER_ROPE,
                prepare=_aiter,
                metadata={"package": "aiter"},
            ),
            Impl(
                kernel_id="eager.rope",
                op="rope",
                priority=Priority.REFERENCE,
                reference=True,
                when=attrs.rope_type.is_set(),
                prepare=_eager,
                metadata={"package": "torch", "note": "eager reference"},
            ),
        )
    )


__all__ = ["ROPE", "register"]
