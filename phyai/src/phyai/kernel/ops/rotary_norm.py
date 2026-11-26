"""Projection-wide Q/K normalization followed by precomputed rotary embedding."""

from __future__ import annotations

from phyai.kernel.facts import device, dtype, lib, shape
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.predicate import all_of
from phyai.kernel.registry import Catalog


QK_RMSNORM_ROPE = OpSpec(
    name="qk_rmsnorm_rope",
    dims=("batch", "tokens", "hidden", "head_dim", "local_heads"),
    dtypes=("input", "weight", "cos", "sin"),
    attributes=("interleave", "head_start"),
    signature="(q, k, q_weight, k_weight, cos, sin, eps, head_dim, interleave, head_start, local_heads) -> (Tensor, Tensor)",
    doc="Paired full-width RMSNorm, rounded affine and rotary for a local head slice.",
)


def _triton(facts, params):
    from functools import partial
    from phyai_kernel import qk_rmsnorm_rope

    num_warps = params.get("num_warps", 4)
    if type(num_warps) is not int or num_warps not in (4, 8, 16, 32):
        raise ValueError("num_warps must be 4, 8, 16, or 32")
    return partial(qk_rmsnorm_rope, num_warps=num_warps)


def _torch(facts, params):
    import torch
    from phyai.layers.rotary_embedding import apply_rope

    def qk_rmsnorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        eps,
        head_dim,
        interleave=True,
        head_start=0,
        local_heads=None,
    ):
        heads = q.shape[-1] // head_dim
        if local_heads is None:
            local_heads = heads - head_start

        def normalize(x, weight):
            value = x.float()
            normalized = value * torch.rsqrt(
                value.square().mean(-1, keepdim=True) + eps
            )
            affine = (normalized.to(x.dtype) * weight).to(x.dtype)
            return affine.unflatten(-1, (heads, head_dim)).narrow(
                -2, head_start, local_heads
            )

        q_rotated, k_rotated = apply_rope(
            normalize(q, q_weight).float(),
            normalize(k, k_weight).float(),
            cos.float(),
            sin.float(),
            interleave=interleave,
        )
        return q_rotated.to(q.dtype).contiguous(), k_rotated.to(k.dtype).contiguous()

    return qk_rmsnorm_rope


def register(catalog: Catalog) -> None:
    catalog.register_op(QK_RMSNORM_ROPE)
    catalog.register_many(
        (
            Impl(
                kernel_id="phyai_kernel.qk_rmsnorm_rope",
                op="qk_rmsnorm_rope",
                priority=Priority.OPTIMIZED,
                when=all_of(
                    lib.has("phyai_kernel"),
                    lib.has("triton"),
                    device.vendor == "nvidia",
                    dtype.input.in_({"fp16", "bf16", "fp32"}),
                    dtype.weight == dtype.input,
                    dtype.cos.in_({"fp16", "bf16", "fp32"}),
                    dtype.sin.in_({"fp16", "bf16", "fp32"}),
                    shape.hidden <= 8192,
                ),
                prepare=_triton,
            ),
            Impl(
                kernel_id="torch.qk_rmsnorm_rope",
                op="qk_rmsnorm_rope",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=_torch,
            ),
        )
    )


__all__ = ["QK_RMSNORM_ROPE", "register"]
