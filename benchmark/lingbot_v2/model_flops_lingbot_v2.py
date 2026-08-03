from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LingBotV2Dims:
    """Dimensions used by the LingBot V2 dominant-GEMM FLOP model."""

    vision_hidden: int = 1024
    vision_layers: int = 24
    vision_heads: int = 16
    vision_intermediate: int = 4096
    patch_vector_dim: int = 1536
    spatial_merge_unit: int = 4
    vision_out_hidden: int = 2560
    deepstack_mergers: int = 3

    text_hidden: int = 2560
    text_layers: int = 36
    text_heads: int = 32
    text_kv_heads: int = 8
    text_head_dim: int = 128
    text_intermediate: int = 9728

    expert_hidden: int = 768
    expert_layers: int = 36
    expert_heads: int = 32
    expert_kv_heads: int = 8
    expert_head_dim: int = 128
    num_experts: int = 32
    top_k: int = 4
    moe_intermediate: int = 512
    shared_intermediate: int = 704

    num_images: int = 3
    patches_per_image: int = 256
    vision_boundaries_per_image: int = 2
    current_query_tokens: int = 8
    future_query_tokens: int = 8
    chunk_size: int = 50
    action_dim: int = 55
    max_state_dim: int = 55
    num_inference_steps: int = 10

    @property
    def merged_tokens_per_image(self) -> int:
        return self.patches_per_image // self.spatial_merge_unit

    @property
    def vision_patch_tokens(self) -> int:
        return self.num_images * self.patches_per_image

    @property
    def merged_vision_tokens(self) -> int:
        return self.num_images * self.merged_tokens_per_image

    @property
    def suffix_tokens(self) -> int:
        return 1 + self.chunk_size

    @property
    def active_moe_intermediate(self) -> int:
        return self.top_k * self.moe_intermediate + self.shared_intermediate


def gemm_flop(m: int, n: int, k: int) -> int:
    """FLOPs for ``(m, k) @ (k, n)`` using multiply plus add."""

    return 2 * m * n * k


def prefix_tokens(lang_len: int, dims: LingBotV2Dims) -> int:
    """Number of compact real tokens processed by the text prefix."""

    return (
        dims.merged_vision_tokens
        + dims.num_images * dims.vision_boundaries_per_image
        + lang_len
        + dims.current_query_tokens
        + dims.future_query_tokens
    )


def expert_visible_prefix_tokens(lang_len: int, dims: LingBotV2Dims) -> int:
    """Prefix tokens visible to the action suffix."""

    return prefix_tokens(lang_len, dims) - dims.future_query_tokens


def attention_flop(
    query_tokens: int,
    kv_tokens: int,
    heads: int,
    head_dim: int,
) -> int:
    """QK score plus probability-value matmul FLOPs."""

    return 4 * query_tokens * kv_tokens * heads * head_dim


def stage_flops(
    dims: LingBotV2Dims,
    *,
    lang_len: int,
) -> dict[str, float]:
    """Dominant per-sample FLOPs for the three weight-bearing stages.

    Norms, RoPE, activations, routing elementwise operations, cache writes, and
    residual additions are omitted. The router GEMM and AdaRMSNorm condition
    projections are retained because they are material model-specific work.
    """

    vision_tokens = dims.vision_patch_tokens
    vision_qkv = gemm_flop(
        vision_tokens,
        3 * dims.vision_hidden,
        dims.vision_hidden,
    )
    vision_o = gemm_flop(
        vision_tokens,
        dims.vision_hidden,
        dims.vision_hidden,
    )
    vision_attention = (
        attention_flop(
            dims.patches_per_image,
            dims.patches_per_image,
            dims.vision_heads,
            dims.vision_hidden // dims.vision_heads,
        )
        * dims.num_images
    )
    vision_mlp = gemm_flop(
        vision_tokens,
        dims.vision_intermediate,
        dims.vision_hidden,
    ) + gemm_flop(
        vision_tokens,
        dims.vision_hidden,
        dims.vision_intermediate,
    )
    vision_blocks = dims.vision_layers * (
        vision_qkv + vision_o + vision_attention + vision_mlp
    )
    vision_patch_embed = gemm_flop(
        vision_tokens,
        dims.vision_hidden,
        dims.patch_vector_dim,
    )
    merger_count = 1 + dims.deepstack_mergers
    merged_width = dims.vision_hidden * dims.spatial_merge_unit
    one_merger = gemm_flop(
        dims.merged_vision_tokens,
        merged_width,
        merged_width,
    ) + gemm_flop(
        dims.merged_vision_tokens,
        dims.vision_out_hidden,
        merged_width,
    )
    vision_mergers = merger_count * one_merger
    vision = vision_patch_embed + vision_blocks + vision_mergers

    n_prefix = prefix_tokens(lang_len, dims)
    text_q_dim = dims.text_heads * dims.text_head_dim
    text_kv_dim = dims.text_kv_heads * dims.text_head_dim
    text_qkv = gemm_flop(
        n_prefix,
        text_q_dim + 2 * text_kv_dim,
        dims.text_hidden,
    )
    text_o = gemm_flop(
        n_prefix,
        dims.text_hidden,
        text_q_dim,
    )
    causal_pairs = n_prefix * (n_prefix + 1) // 2
    text_attention = 4 * causal_pairs * dims.text_heads * dims.text_head_dim
    text_mlp = 2 * gemm_flop(
        n_prefix,
        dims.text_intermediate,
        dims.text_hidden,
    ) + gemm_flop(
        n_prefix,
        dims.text_hidden,
        dims.text_intermediate,
    )
    text_prefix = dims.text_layers * (text_qkv + text_o + text_attention + text_mlp)

    suffix_tokens = dims.suffix_tokens
    visible_prefix = expert_visible_prefix_tokens(lang_len, dims)
    expert_q_dim = dims.expert_heads * dims.expert_head_dim
    expert_kv_dim = dims.expert_kv_heads * dims.expert_head_dim
    expert_qkv = gemm_flop(
        suffix_tokens,
        expert_q_dim + 2 * expert_kv_dim,
        dims.expert_hidden,
    )
    expert_o = gemm_flop(
        suffix_tokens,
        dims.expert_hidden,
        expert_q_dim,
    )
    expert_attention = attention_flop(
        1,
        visible_prefix + 1,
        dims.expert_heads,
        dims.expert_head_dim,
    ) + attention_flop(
        dims.chunk_size,
        visible_prefix + suffix_tokens,
        dims.expert_heads,
        dims.expert_head_dim,
    )
    expert_router = gemm_flop(
        suffix_tokens,
        dims.num_experts,
        dims.expert_hidden,
    )
    expert_moe = 2 * gemm_flop(
        suffix_tokens,
        dims.active_moe_intermediate,
        dims.expert_hidden,
    ) + gemm_flop(
        suffix_tokens,
        dims.expert_hidden,
        dims.active_moe_intermediate,
    )
    # Two AdaRMSNorm modules per layer, called once for the state stream and
    # once for the action stream. Each call projects gamma and beta.
    expert_adanorm = 8 * gemm_flop(
        1,
        dims.expert_hidden,
        dims.expert_hidden,
    )
    expert_layers = dims.expert_layers * (
        expert_qkv
        + expert_o
        + expert_attention
        + expert_router
        + expert_moe
        + expert_adanorm
    )
    expert_heads = (
        gemm_flop(1, dims.expert_hidden, dims.max_state_dim)
        + gemm_flop(
            dims.chunk_size,
            dims.expert_hidden,
            dims.action_dim,
        )
        + gemm_flop(
            dims.chunk_size,
            dims.expert_hidden,
            2 * dims.expert_hidden,
        )
        + gemm_flop(
            dims.chunk_size,
            dims.expert_hidden,
            dims.expert_hidden,
        )
        + gemm_flop(
            dims.chunk_size,
            dims.action_dim,
            dims.expert_hidden,
        )
    )
    expert_1step = expert_layers + expert_heads
    expert_loop = expert_1step * dims.num_inference_steps

    return {
        "vision": float(vision),
        "vision_patch_embed": float(vision_patch_embed),
        "vision_blocks": float(vision_blocks),
        "vision_mergers": float(vision_mergers),
        "text_prefix": float(text_prefix),
        "text_attention": float(
            dims.text_layers * (text_qkv + text_o + text_attention)
        ),
        "text_mlp": float(dims.text_layers * text_mlp),
        "expert_1step": float(expert_1step),
        "expert_loop": float(expert_loop),
        "expert_attention_1step": float(
            dims.expert_layers * (expert_qkv + expert_o + expert_attention)
        ),
        "expert_moe_1step": float(dims.expert_layers * expert_moe),
        "expert_router_1step": float(dims.expert_layers * expert_router),
        "expert_adanorm_1step": float(dims.expert_layers * expert_adanorm),
        "expert_heads_1step": float(expert_heads),
        "e2e_compute": float(vision + text_prefix + expert_loop),
    }


def _main() -> None:
    dims = LingBotV2Dims()
    lang_len = 25
    flop = stage_flops(dims, lang_len=lang_len)
    print(
        "LingBot V2 dominant-GEMM FLOP model "
        f"(lang_len={lang_len}, prefix={prefix_tokens(lang_len, dims)})"
    )
    for key in (
        "vision",
        "text_prefix",
        "expert_1step",
        "expert_loop",
        "e2e_compute",
    ):
        print(f"{key:<20} {flop[key] / 1e9:>12.3f} GFLOP/sample")


if __name__ == "__main__":
    _main()
