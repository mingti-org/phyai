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

ROPE_STRIDED = OpSpec(
    name="rope_strided",
    dims=ROPE.dims,
    dtypes=ROPE.dtypes,
    attributes=ROPE.attributes,
    optional_attributes=ROPE.optional_attributes,
    signature=ROPE.signature,
    doc="Full NeoX RoPE accepting aligned QKV row views without input materialization.",
)


FLASHINFER_ROPE = all_of(
    lib.has("flashinfer"),
    device.vendor == "nvidia",
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


def strided_input_supported(tensor, heads: int) -> bool:
    """Check the vector-load alignment and complete storage span, without a sync."""
    import torch

    if tensor.dtype != torch.bfloat16 or tensor.ndim != 3:
        return False
    n, h, d = tensor.shape
    pitch, head_stride, element_stride = tensor.stride()
    return (
        1 <= n <= 4096
        and (h, d) == (heads, 256)
        and (head_stride, element_stride) == (256, 1)
        and heads * 256 <= pitch <= 2**24
        and pitch % 8 == 0
        and tensor.data_ptr() % 16 == 0
        and 2 * (tensor.storage_offset() + (n - 1) * pitch + heads * 256)
        <= tensor.untyped_storage().nbytes()
    )


def _flashinfer_strided(facts, params):
    import torch
    from flashinfer.rope import apply_rope_with_cos_sin_cache

    def rope(positions, q, k, cos_sin_cache):
        if not (
            strided_input_supported(q, 8)
            and strided_input_supported(k, 1)
            and q.is_cuda
            and k.device == q.device == positions.device == cos_sin_cache.device
            and q.shape[0] == k.shape[0]
            and positions.shape == (q.shape[0],)
            and positions.dtype == torch.int32
            and positions.is_contiguous()
            and cos_sin_cache.ndim == 2
            and cos_sin_cache.shape[1] == 256
            and cos_sin_cache.dtype == torch.float32
            and cos_sin_cache.is_contiguous()
            and cos_sin_cache.data_ptr() % 16 == 0
        ):
            raise ValueError(
                "strided RoPE requires aligned canonical CUDA Q/K and metadata"
            )
        q_out, k_out = apply_rope_with_cos_sin_cache(
            positions=positions,
            query=q.view(q.shape[0], 2048),
            key=k.view(k.shape[0], 256),
            head_size=256,
            cos_sin_cache=cos_sin_cache,
            is_neox=True,
        )
        return q_out.view(q.shape), k_out.view(k.shape)

    return rope


def register(catalog: Catalog) -> None:
    catalog.register_op(ROPE)
    catalog.register_op(ROPE_STRIDED)
    catalog.register_many(
        (
            Impl(
                kernel_id="flashinfer.rope_strided.sm110",
                op="rope_strided",
                priority=Priority.OPTIMIZED,
                when=all_of(
                    FLASHINFER_ROPE,
                    device.arch == "sm110",
                    dtype.input == "bf16",
                    shape.head_dim == 256,
                    shape.heads_q == 8,
                    shape.heads_k == 1,
                    shape.tokens >= 1,
                    shape.tokens <= 4096,
                    attrs.interleave == False,  # noqa: E712 -- symbolic predicate
                    attrs.rank == 3,
                ),
                prepare=_flashinfer_strided,
                metadata={
                    "package": "flashinfer",
                    "note": "aligned strided input adapter",
                },
            ),
            Impl(
                kernel_id="eager.rope_strided",
                op="rope_strided",
                priority=Priority.REFERENCE,
                reference=True,
                when=attrs.rope_type.is_set(),
                prepare=_eager,
            ),
            Impl(
                kernel_id="flashinfer.rope",
                op="rope",
                priority=Priority.OPTIMIZED + 2,
                when=FLASHINFER_ROPE,
                prepare=_flashinfer,
                metadata={"package": "flashinfer"},
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


__all__ = ["ROPE", "ROPE_STRIDED", "strided_input_supported", "register"]
