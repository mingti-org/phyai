from __future__ import annotations

from dataclasses import dataclass, field

# Language-length buckets the pi0.5 scheduler captures graphs for. Must match
# ``PI05WS1Scheduler._lang_buckets`` (the {16,48,112} set intersected with
# ``0 < b < tokenizer_max_length`` plus ``tokenizer_max_length`` as fallback).
DEFAULT_LANG_BUCKETS: tuple[int, ...] = (16, 48, 112, 200)


@dataclass(frozen=True)
class Pi05Dims:
    """Dimensions needed for the FLOP model. Defaults are public ``pi05_base``.

    The driver builds this from the live :class:`PI05Config` so the analysis
    tracks whatever checkpoint was actually profiled; the defaults here exist
    only so this module runs (and self-checks) with no checkpoint present.
    """

    # Vision tower (SigLIP-So400m).
    v_hidden: int = 1152
    v_layers: int = 27
    v_heads: int = 16
    v_intermediate: int = 4304
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3

    # PaliGemma language model (gemma_2b text side, MQA).
    l_hidden: int = 2048
    l_layers: int = 18
    l_heads: int = 8
    l_kv_heads: int = 1
    l_head_dim: int = 256
    l_intermediate: int = 16384

    # Action expert (gemma_300m, MQA).
    e_hidden: int = 1024
    e_layers: int = 18
    e_heads: int = 8
    e_kv_heads: int = 1
    e_head_dim: int = 256
    e_intermediate: int = 4096

    # Flow-matching / layout knobs.
    chunk_size: int = 50
    num_inference_steps: int = 10
    tokenizer_max_length: int = 200
    lang_buckets: tuple[int, ...] = field(default=DEFAULT_LANG_BUCKETS)

    @property
    def patches_per_image(self) -> int:
        return (self.image_size // self.patch_size) ** 2


def gemm_flop(m: int, n: int, k: int) -> int:
    """FLOP for an ``(m,k) @ (k,n)`` matmul: ``2*m*n*k`` (multiply + add)."""
    return 2 * m * n * k


def transformer_layer_flop(
    m: int,
    hidden: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    intermediate: int,
    *,
    kv_len: int | None = None,
    gated_mlp: bool = True,
) -> int:
    """Dominant-GEMM FLOP for one decoder/encoder layer over ``m`` query tokens.

    Counts QKV projection, output projection, attention scores (``q·kᵀ``) and
    context (``a·v``) over ``kv_len`` keys, and the MLP. ``gated_mlp=True`` is
    SwiGLU/GeGLU (gate + up + down = three matmuls); ``False`` is the plain
    two-matmul MLP used by SigLIP. Norms, biases, RoPE and the activation
    elementwise are <1% and omitted (they are launch-bound, not FLOP-bound).
    """
    if kv_len is None:
        kv_len = m
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv = gemm_flop(m, q_dim + 2 * kv_dim, hidden)
    o = gemm_flop(m, hidden, q_dim)
    scores = 2 * m * kv_len * head_dim * heads
    ctx = 2 * m * kv_len * head_dim * heads
    if gated_mlp:
        mlp = gemm_flop(m, intermediate, hidden) * 2 + gemm_flop(
            m, hidden, intermediate
        )
    else:
        mlp = gemm_flop(m, intermediate, hidden) + gemm_flop(m, hidden, intermediate)
    return qkv + o + scores + ctx + mlp


def bucket_lang_len(lang_len: int, dims: Pi05Dims) -> int:
    """Smallest captured language bucket covering ``lang_len`` (scheduler parity)."""
    return next(
        (b for b in sorted(dims.lang_buckets) if b >= lang_len),
        dims.tokenizer_max_length,
    )


def n_per_sample(lang_len: int, dims: Pi05Dims, num_images: int) -> int:
    """Prefix tokens per robot at the bucket covering ``lang_len``."""
    image_tokens = dims.patches_per_image * num_images
    return image_tokens + bucket_lang_len(lang_len, dims)


def stage_flops(
    dims: Pi05Dims,
    *,
    lang_len: int = 1,
    num_images: int = 3,
) -> dict[str, float]:
    """Per-sample (per-robot) FLOP for each pipeline stage.

    Returns a dict keyed by ``vision``, ``llm_prefix``, ``expert_1step`` and
    ``expert_loop`` (the last = ``expert_1step * num_inference_steps``). All
    values are FLOP for **one** robot; the driver multiplies by batch size.
    """
    image_tokens = dims.patches_per_image * num_images

    # --- Vision: num_images cameras through the tower in one forward. ---
    # m = all image tokens (768); attention is per-image so kv_len = patches
    # of a single camera (256), not the full stack.
    vision = 0.0
    for _ in range(dims.v_layers):
        vision += transformer_layer_flop(
            image_tokens,
            dims.v_hidden,
            dims.v_heads,
            dims.v_heads,  # SigLIP is full MHA (kv_heads == heads)
            dims.v_hidden // dims.v_heads,
            dims.v_intermediate,
            kv_len=dims.patches_per_image,
            gated_mlp=False,  # SigLIP MLP is gelu(fc1)·fc2, not gated
        )
    # Patch-embed conv as an im2col GEMM, + multi-modal projector (→ LM hidden).
    vision += (
        2 * image_tokens * dims.v_hidden * (dims.num_channels * dims.patch_size**2)
    )
    vision += gemm_flop(image_tokens, dims.l_hidden, dims.v_hidden)

    # --- LLM prefix: n_per_sample tokens, full self-attention, 18 layers. ---
    n_ps = n_per_sample(lang_len, dims, num_images)
    llm_prefix = 0.0
    for _ in range(dims.l_layers):
        llm_prefix += transformer_layer_flop(
            n_ps,
            dims.l_hidden,
            dims.l_heads,
            dims.l_kv_heads,
            dims.l_head_dim,
            dims.l_intermediate,
            kv_len=n_ps,
            gated_mlp=True,
        )

    # --- Expert one Euler step: chunk_size queries vs (prefix + suffix) kv. ---
    e_kv_len = n_ps + dims.chunk_size
    expert_1step = 0.0
    for _ in range(dims.e_layers):
        expert_1step += transformer_layer_flop(
            dims.chunk_size,
            dims.e_hidden,
            dims.e_heads,
            dims.e_kv_heads,
            dims.e_head_dim,
            dims.e_intermediate,
            kv_len=e_kv_len,
            gated_mlp=True,
        )

    return {
        "vision": vision,
        "llm_prefix": llm_prefix,
        "expert_1step": expert_1step,
        "expert_loop": expert_1step * dims.num_inference_steps,
    }


def analytic_weight_bytes(dims: Pi05Dims, *, dtype_bytes: int = 2) -> dict[str, float]:
    """Rough per-stage resident weight bytes (dominant GEMMs), for the self-check.

    NOT used for the shipped roofline — the driver replaces these with exact
    per-module byte counts from the loaded model. Norms/biases/embeddings are
    omitted, so these undercount the true resident size by a few percent; they
    exist only so a checkpoint-free run can still print a plausible AI.
    """

    def attn_mlp_params(hidden, heads, kv_heads, head_dim, inter, *, gated):
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        attn = hidden * (q_dim + 2 * kv_dim) + q_dim * hidden
        mlp = (3 if gated else 2) * hidden * inter
        return attn + mlp

    v_params = dims.v_layers * attn_mlp_params(
        dims.v_hidden,
        dims.v_heads,
        dims.v_heads,
        dims.v_hidden // dims.v_heads,
        dims.v_intermediate,
        gated=False,
    )
    v_params += dims.v_hidden * (dims.num_channels * dims.patch_size**2)  # patch embed
    v_params += dims.v_hidden * dims.l_hidden  # projector

    l_params = dims.l_layers * attn_mlp_params(
        dims.l_hidden,
        dims.l_heads,
        dims.l_kv_heads,
        dims.l_head_dim,
        dims.l_intermediate,
        gated=True,
    )

    e_params = dims.e_layers * attn_mlp_params(
        dims.e_hidden,
        dims.e_heads,
        dims.e_kv_heads,
        dims.e_head_dim,
        dims.e_intermediate,
        gated=True,
    )

    return {
        "vision": v_params * dtype_bytes,
        "llm_prefix": l_params * dtype_bytes,
        # The expert reads its weights once per Euler step.
        "expert_1step": e_params * dtype_bytes,
        "expert_loop": e_params * dtype_bytes * dims.num_inference_steps,
    }


def _main() -> None:
    dims = Pi05Dims()
    lang_len = 1
    num_images = 3
    flop = stage_flops(dims, lang_len=lang_len, num_images=num_images)
    wbytes = analytic_weight_bytes(dims)
    n_ps = n_per_sample(lang_len, dims, num_images)

    print(
        f"pi05_base analytic FLOP model  (lang_len={lang_len} → bucket "
        f"{bucket_lang_len(lang_len, dims)}, n_per_sample={n_ps}, "
        f"num_images={num_images})"
    )
    print("weight bytes are an ANALYTIC ESTIMATE (driver uses live model bytes)\n")
    print(f"{'stage':<16}{'GFLOP/sample':>14}{'W_MiB(est)':>12}{'AI(est)':>10}")
    for k in ("vision", "llm_prefix", "expert_1step", "expert_loop"):
        ai = flop[k] / wbytes[k]
        print(f"{k:<16}{flop[k] / 1e9:>14.2f}{wbytes[k] / 2**20:>12.1f}{ai:>10.1f}")

    e2e = flop["vision"] + flop["llm_prefix"] + flop["expert_loop"]
    print(
        f"\nper-sample compute (vision + llm_prefix + 10-step expert) = "
        f"{e2e / 1e9:.1f} GFLOP"
    )


if __name__ == "__main__":
    _main()
