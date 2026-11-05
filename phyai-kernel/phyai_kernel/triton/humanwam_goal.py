"""HumanWAM-Goal pointwise fusions with explicit BF16 rounding boundaries."""

import triton
import triton.language as tl


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
            dest = (token * KV_HEADS + heads[:, None]) * 128 + cols[None, :]
            tl.store(out_k_ptr + dest, result)
            gen_value = tl.load(
                v_ptr + gen_token * V_TOKEN + heads[:, None] * V_HEAD + cols[None, :]
            )
            tl.store(out_v_ptr + dest, gen_value)


@triton.jit
def _add_rms_norm(residual, projected, weight, hidden, normalized, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 4096)
    offset = row * 4096 + col
    # Native eager addition first returns a BF16 residual. Its rounded values
    # are both the residual output and the input to the FP32 variance.
    summed = (
        tl.load(residual + offset).to(tl.float32)
        + tl.load(projected + offset).to(tl.float32)
    ).to(hidden.dtype.element_ty)
    value = summed.to(tl.float32)
    variance = tl.sum(value * value, axis=0) * (1.0 / 4096)
    before_affine = (value * tl.rsqrt(variance + EPS)).to(normalized.dtype.element_ty)
    result = before_affine * tl.load(weight + col)
    tl.store(hidden + offset, summed)
    tl.store(normalized + offset, result)
