from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pi0Dims:
    """Dimensions needed for the pi0 FLOP model.

    Defaults match the public pi0 geometry. The profiling driver replaces these
    values with a live :class:`PI0Config`, so checkpoint-specific changes are
    reflected in the JSON.
    """

    # Vision tower.
    v_hidden: int = 1152
    v_layers: int = 27
    v_heads: int = 16
    v_intermediate: int = 4304
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3

    # PaliGemma language prefix.
    l_hidden: int = 2048
    l_layers: int = 18
    l_heads: int = 8
    l_kv_heads: int = 1
    l_head_dim: int = 256
    l_intermediate: int = 16384

    # Action expert.
    e_hidden: int = 1024
    e_layers: int = 18
    e_heads: int = 8
    e_kv_heads: int = 1
    e_head_dim: int = 256
    e_intermediate: int = 4096

    # Request geometry.
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    tokenizer_max_length: int = 48
    num_images: int = 3

    @property
    def patches_per_image(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def image_tokens(self) -> int:
        return self.patches_per_image * self.num_images

    @property
    def n_per_sample(self) -> int:
        return self.image_tokens + self.tokenizer_max_length


def gemm_flop(m: int, n: int, k: int) -> int:
    """FLOP for an ``(m,k) @ (k,n)`` matmul: multiply plus add."""
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
    """Dominant-GEMM FLOP for one transformer layer."""
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


def stage_flops(dims: Pi0Dims, *, lang_len: int = 1) -> dict[str, float]:
    """Per-sample FLOP for pi0 stages.

    ``llm_prefix`` follows the scheduler's padded prefix graph shape: every
    sample uses ``image_tokens + tokenizer_max_length`` query slots for dense
    projections. Attention reads only the real image and language K/V slots.

    The pi0 expert layer is staged as one state-token pass plus one action-chunk
    pass per layer. State attends over prefix + state. Action attends over prefix
    + state + action tokens.
    """
    image_tokens = dims.image_tokens
    real_prefix = image_tokens + max(0, min(lang_len, dims.tokenizer_max_length))

    vision = 0.0
    for _ in range(dims.v_layers):
        vision += transformer_layer_flop(
            image_tokens,
            dims.v_hidden,
            dims.v_heads,
            dims.v_heads,
            dims.v_hidden // dims.v_heads,
            dims.v_intermediate,
            kv_len=dims.patches_per_image,
            gated_mlp=False,
        )
    vision += (
        2 * image_tokens * dims.v_hidden * (dims.num_channels * dims.patch_size**2)
    )
    vision += gemm_flop(image_tokens, dims.l_hidden, dims.v_hidden)

    n_prefix = dims.n_per_sample
    llm_prefix = 0.0
    for _ in range(dims.l_layers):
        llm_prefix += transformer_layer_flop(
            n_prefix,
            dims.l_hidden,
            dims.l_heads,
            dims.l_kv_heads,
            dims.l_head_dim,
            dims.l_intermediate,
            kv_len=real_prefix,
            gated_mlp=True,
        )

    state_layers_once = 0.0
    action_layers_1step = 0.0
    for _ in range(dims.e_layers):
        state_layers_once += transformer_layer_flop(
            1,
            dims.e_hidden,
            dims.e_heads,
            dims.e_kv_heads,
            dims.e_head_dim,
            dims.e_intermediate,
            kv_len=real_prefix + 1,
            gated_mlp=True,
        )
        action_layers_1step += transformer_layer_flop(
            dims.chunk_size,
            dims.e_hidden,
            dims.e_heads,
            dims.e_kv_heads,
            dims.e_head_dim,
            dims.e_intermediate,
            kv_len=real_prefix + 1 + dims.chunk_size,
            gated_mlp=True,
        )

    state_head_once = gemm_flop(1, dims.e_hidden, dims.max_state_dim)
    action_heads_1step = 0.0
    action_heads_1step += gemm_flop(dims.chunk_size, dims.e_hidden, dims.max_action_dim)
    action_heads_1step += gemm_flop(dims.chunk_size, dims.e_hidden, 2 * dims.e_hidden)
    action_heads_1step += gemm_flop(dims.chunk_size, dims.e_hidden, dims.e_hidden)
    action_heads_1step += gemm_flop(dims.chunk_size, dims.max_action_dim, dims.e_hidden)

    expert_state_prefill = state_head_once + state_layers_once
    expert_action_step = action_heads_1step + action_layers_1step
    expert_loop = expert_action_step * dims.num_inference_steps
    return {
        "vision": vision,
        "llm_prefix": llm_prefix,
        "expert_state_prefill": expert_state_prefill,
        "expert_action_step": expert_action_step,
        "expert_loop": expert_loop,
        "expert_total": expert_state_prefill + expert_loop,
        "real_prefix_tokens": float(real_prefix),
        "padded_prefix_tokens": float(n_prefix),
    }


def analytic_weight_bytes(dims: Pi0Dims, *, dtype_bytes: int = 2) -> dict[str, float]:
    """Rough per-stage resident weight bytes for checkpoint-free self-checks."""

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
    v_params += dims.v_hidden * (dims.num_channels * dims.patch_size**2)
    v_params += dims.v_hidden * dims.l_hidden

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
    state_head_params = dims.max_state_dim * dims.e_hidden
    action_head_params = (
        dims.max_action_dim * dims.e_hidden
        + dims.e_hidden * dims.max_action_dim
        + 2 * dims.e_hidden * dims.e_hidden
        + dims.e_hidden * dims.e_hidden
    )
    expert_state_prefill = (e_params + state_head_params) * dtype_bytes
    expert_action_step = (e_params + action_head_params) * dtype_bytes
    expert_loop = expert_action_step * dims.num_inference_steps
    return {
        "vision": v_params * dtype_bytes,
        "llm_prefix": l_params * dtype_bytes,
        "expert_state_prefill": expert_state_prefill,
        "expert_action_step": expert_action_step,
        "expert_loop": expert_loop,
        "expert_total": expert_state_prefill + expert_loop,
    }


def _main() -> None:
    dims = Pi0Dims()
    flop = stage_flops(dims)
    wbytes = analytic_weight_bytes(dims)
    print("pi0 analytic FLOP model")
    print(
        f"  image_tokens={dims.image_tokens}  prefix={int(flop['padded_prefix_tokens'])}  "
        f"chunk={dims.chunk_size}  steps={dims.num_inference_steps}"
    )
    print(f"{'stage':<18}{'GFLOP':>14}{'W_MiB(est)':>12}{'AI(est)':>10}")
    for key in (
        "vision",
        "llm_prefix",
        "expert_state_prefill",
        "expert_action_step",
        "expert_loop",
        "expert_total",
    ):
        ai = flop[key] / wbytes[key]
        print(
            f"{key:<18}{flop[key] / 1e9:>14.2f}{wbytes[key] / 2**20:>12.1f}{ai:>10.1f}"
        )


if __name__ == "__main__":
    _main()
