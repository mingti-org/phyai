"""HumanWAM-Goal pointwise fusions with explicit BF16 rounding boundaries."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _prepare_qkv(
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_ptr,
    sin_ptr,
    text_k_ptr,
    text_v_ptr,
    out_q_ptr,
    out_k_ptr,
    out_v_ptr,
    TEXT: tl.constexpr,
    Q_TOKEN: tl.constexpr,
    Q_HEAD: tl.constexpr,
    K_TOKEN: tl.constexpr,
    K_HEAD: tl.constexpr,
    V_TOKEN: tl.constexpr,
    V_HEAD: tl.constexpr,
    COS_TOKEN: tl.constexpr,
    SIN_TOKEN: tl.constexpr,
    TEXT_K_TOKEN: tl.constexpr,
    TEXT_K_HEAD: tl.constexpr,
    TEXT_K_COL: tl.constexpr,
    TEXT_V_TOKEN: tl.constexpr,
    TEXT_V_HEAD: tl.constexpr,
    TEXT_V_COL: tl.constexpr,
    Q_EPS: tl.constexpr,
    K_EPS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    INTERLEAVED_KV: tl.constexpr = False,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    heads = tl.arange(0, KV_HEADS)
    cols = tl.arange(0, 128)
    if token < TEXT:
        if group == 0:
            dest = (token * KV_HEADS + heads[:, None]) * 128 + cols[None, :]
            text_key = tl.load(
                text_k_ptr
                + token * TEXT_K_TOKEN
                + heads[:, None] * TEXT_K_HEAD
                + cols[None, :] * TEXT_K_COL
            )
            text_value = tl.load(
                text_v_ptr
                + token * TEXT_V_TOKEN
                + heads[:, None] * TEXT_V_HEAD
                + cols[None, :] * TEXT_V_COL
            )
            tl.store(out_k_ptr + dest, text_key)
            tl.store(out_v_ptr + dest, text_value)
    else:
        gen_token = token - TEXT
        query_heads = group * KV_HEADS + heads
        if group < 4:
            source = (
                q_ptr
                + gen_token * Q_TOKEN
                + query_heads[:, None] * Q_HEAD
                + cols[None, :]
            )
            weight = tl.load(q_weight_ptr + cols)
            eps = Q_EPS
        else:
            source = (
                k_ptr + gen_token * K_TOKEN + heads[:, None] * K_HEAD + cols[None, :]
            )
            weight = tl.load(k_weight_ptr + cols)
            eps = K_EPS
        norm_value = tl.load(source).to(tl.float32)
        variance = tl.sum(norm_value * norm_value, axis=1) / 128
        normalized = (norm_value * tl.rsqrt(variance[:, None] + eps)).to(
            q_ptr.dtype.element_ty
        )
        affine = (normalized.to(tl.float32) * weight[None, :].to(tl.float32)).to(
            q_ptr.dtype.element_ty
        )
        partner_cols = tl.broadcast_to(((cols + 64) % 128)[None, :], (KV_HEADS, 128))
        partner = tl.gather(affine, partner_cols, axis=1)
        rotated = tl.where(cols[None, :] < 64, -partner, partner)
        cos = tl.load(cos_ptr + gen_token * COS_TOKEN + cols)
        sin = tl.load(sin_ptr + gen_token * SIN_TOKEN + cols)
        # Eager BF16 multiplication rounds each product before the addition.
        first = (affine.to(tl.float32) * cos[None, :].to(tl.float32)).to(
            q_ptr.dtype.element_ty
        )
        second = (rotated.to(tl.float32) * sin[None, :].to(tl.float32)).to(
            q_ptr.dtype.element_ty
        )
        result = first.to(tl.float32) + second.to(tl.float32)
        if group < 4:
            dest = (gen_token * (4 * KV_HEADS) + query_heads[:, None]) * 128 + cols[
                None, :
            ]
            tl.store(out_q_ptr + dest, result)
        else:
            stride = KV_HEADS * (2 if INTERLEAVED_KV else 1)
            dest = (token * stride + heads[:, None]) * 128 + cols[None, :]
            tl.store(out_k_ptr + dest, result)
            gen_value = tl.load(
                v_ptr + gen_token * V_TOKEN + heads[:, None] * V_HEAD + cols[None, :]
            )
            tl.store(out_v_ptr + dest, gen_value)


@triton.jit
def _add_rms_norm(residual, projected, weight, hidden, normalized, EPS: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    component = tl.arange(0, 4)
    offsets = row * 4096 + lane[:, None] * 4 + component[None, :]
    acc = tl.full((32, 4), 0, tl.float32)
    # Torch 2.10/2.11 Reduce.cuh uses four serial accumulators per lane for
    # contiguous FP32 means with width 4096 and >=16 rows. Preserve their
    # addition order: a flat tree reduction can change the final BF16 value.
    for step in tl.static_range(32):
        value = (
            (
                tl.load(residual + offsets + step * 128).to(tl.float32)
                + tl.load(projected + offsets + step * 128).to(tl.float32)
            )
            .to(hidden.dtype.element_ty)
            .to(tl.float32)
        )
        acc = acc + value * value
    even, odd = tl.split(tl.reshape(acc, (32, 2, 2)))
    acc0, acc2 = tl.split(even)
    acc1, acc3 = tl.split(odd)
    partial = ((acc0 + acc1) + acc2) + acc3
    variance = tl.sum(partial, axis=0) * (1.0 / 4096)

    col = tl.arange(0, 4096)
    offset = row * 4096 + col
    # Native eager addition first returns a BF16 residual. Its rounded values
    # are both the residual output and the input to the FP32 variance.
    summed = (
        tl.load(residual + offset).to(tl.float32)
        + tl.load(projected + offset).to(tl.float32)
    ).to(hidden.dtype.element_ty)
    value = summed.to(tl.float32)
    before_affine = (value * tl.rsqrt(variance + EPS)).to(normalized.dtype.element_ty)
    result = before_affine * tl.load(weight + col)
    tl.store(hidden + offset, summed)
    tl.store(normalized + offset, result)


@triton.jit
def _silu_mul(
    fused, output, COUNT: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // WIDTH, index % WIDTH
    gate = tl.load(fused + row * (2 * WIDTH) + col, index < COUNT, 0).to(tl.float32)
    up = tl.load(fused + row * (2 * WIDTH) + col + WIDTH, index < COUNT, 0).to(
        tl.float32
    )
    # Native SiLU returns BF16 before the separate multiplication by up.
    activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(output.dtype.element_ty)
    tl.store(output + index, activated.to(tl.float32) * up, index < COUNT)
