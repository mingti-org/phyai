from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cosmos3Dims:
    """Dimensions needed for the FLOP model. Defaults are public ``Cosmos3-Nano``.

    The driver builds this from the live :class:`Cosmos3Config` so the analysis
    tracks whatever checkpoint was profiled; the defaults here exist only so this
    module runs (and self-checks) with no checkpoint present.
    """

    # Transformer core (Qwen3-VL text backbone dims; shared by UND + GEN towers).
    hidden: int = 4096
    layers: int = 36
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 12288

    # Latent / patchify.
    latent_channel: int = 48
    latent_patch_size: int = 2
    patch_latent_dim: int = 192  # latent_patch_size**2 * latent_channel

    # Action adapters.
    action_dim: int = 64

    # Flow-matching default (overridable per run; released DROID policy uses 4).
    num_inference_steps: int = 4


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
) -> int:
    """Dominant-GEMM FLOP for one decoder layer over ``m`` query tokens.

    Counts QKV projection, output projection, attention scores (``q·kᵀ``) and
    context (``a·v``) over ``kv_len`` keys, and the gated SwiGLU MLP (gate + up +
    down = three matmuls). Both the UND self-attn layer and the GEN cross-attn
    layer have this exact GEMM shape — they differ only in ``kv_len`` (UND keys =
    ``m``; GEN keys = ``S_text + S_gen``). Norms, biases, RoPE and the activation
    elementwise are <1% and omitted (launch-bound, not FLOP-bound).
    """
    if kv_len is None:
        kv_len = m
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv = gemm_flop(m, q_dim + 2 * kv_dim, hidden)
    o = gemm_flop(m, hidden, q_dim)
    scores = 2 * m * kv_len * head_dim * heads
    ctx = 2 * m * kv_len * head_dim * heads
    mlp = gemm_flop(m, intermediate, hidden) * 2 + gemm_flop(m, hidden, intermediate)
    return qkv + o + scores + ctx + mlp


def video_tokens(t_lat: int, h_lat: int, w_lat: int, patch: int) -> int:
    """GEN video token count: ``t_lat * ceil(h_lat/p) * ceil(w_lat/p)``."""
    hp = (h_lat + patch - 1) // patch
    wp = (w_lat + patch - 1) // patch
    return t_lat * hp * wp


def cfg_branches(guidance_scale: float) -> int:
    """2 transformer forwards per step when classifier-free guidance is on."""
    return 2 if guidance_scale > 1.0 else 1


def stage_flops(
    dims: Cosmos3Dims,
    *,
    s_text: int,
    t_lat: int,
    h_lat: int,
    w_lat: int,
    action_chunk: int,
    num_inference_steps: int | None = None,
    guidance_scale: float = 1.0,
) -> dict[str, float]:
    """Per-request FLOP for each policy phase (batch=1).

    Returns a dict keyed by ``cond_encode`` (UND tower, one branch),
    ``gen_1step`` (one GEN forward), ``gen_loop`` (``gen_1step *
    num_inference_steps * cfg_branches``), and ``cond_encode_total``
    (``cond_encode * cfg_branches``). VAE encode is conv-dominated and not modelled.
    """
    n_steps = (
        dims.num_inference_steps if num_inference_steps is None else num_inference_steps
    )
    branches = cfg_branches(guidance_scale)

    s_video = video_tokens(t_lat, h_lat, w_lat, dims.latent_patch_size)
    s_gen = s_video + action_chunk
    gen_kv_len = s_text + s_gen

    # --- UND text tower: s_text tokens, causal self-attention, `layers` deep. ---
    cond_encode = 0.0
    for _ in range(dims.layers):
        cond_encode += transformer_layer_flop(
            s_text,
            dims.hidden,
            dims.heads,
            dims.kv_heads,
            dims.head_dim,
            dims.intermediate,
            kv_len=s_text,
        )

    # --- GEN tower one step: S_gen queries cross-attend to (UND text + GEN). ---
    gen_1step = 0.0
    # proj_in (video patches → hidden) + action_proj_in (domain-aware bmm).
    gen_1step += gemm_flop(s_video, dims.hidden, dims.patch_latent_dim)
    gen_1step += gemm_flop(action_chunk, dims.hidden, dims.action_dim)
    for _ in range(dims.layers):
        gen_1step += transformer_layer_flop(
            s_gen,
            dims.hidden,
            dims.heads,
            dims.kv_heads,
            dims.head_dim,
            dims.intermediate,
            kv_len=gen_kv_len,
        )
    # proj_out (hidden → video patches) + action_proj_out (domain-aware bmm).
    gen_1step += gemm_flop(s_video, dims.patch_latent_dim, dims.hidden)
    gen_1step += gemm_flop(action_chunk, dims.action_dim, dims.hidden)

    return {
        "cond_encode": cond_encode,
        "cond_encode_total": cond_encode * branches,
        "gen_1step": gen_1step,
        "gen_loop": gen_1step * n_steps * branches,
        "s_text": float(s_text),
        "s_video": float(s_video),
        "s_action": float(action_chunk),
        "s_gen": float(s_gen),
        "gen_kv_len": float(gen_kv_len),
        "cfg_branches": float(branches),
        "num_inference_steps": float(n_steps),
    }


def analytic_weight_bytes(
    dims: Cosmos3Dims, *, dtype_bytes: int = 2
) -> dict[str, float]:
    """Rough per-phase resident weight bytes (dominant GEMMs), for the self-check.

    NOT used for the shipped roofline — the driver replaces these with exact
    per-module byte counts from the loaded model. The UND ``embed_tokens`` (a
    gather, not a GEMM read) and the unused ``norm`` are excluded, matching what
    the driver streams.
    """

    def attn_mlp_params(hidden, heads, kv_heads, head_dim, inter):
        q_dim = heads * head_dim
        kv_dim = kv_heads * head_dim
        attn = hidden * (q_dim + 2 * kv_dim) + q_dim * hidden
        mlp = 3 * hidden * inter
        return attn + mlp

    per_layer = attn_mlp_params(
        dims.hidden, dims.heads, dims.kv_heads, dims.head_dim, dims.intermediate
    )
    und = dims.layers * per_layer

    gen = dims.layers * per_layer
    gen += dims.patch_latent_dim * dims.hidden + dims.hidden  # proj_in (+bias)
    gen += dims.hidden * dims.patch_latent_dim + dims.patch_latent_dim  # proj_out
    # Action adapters: only the active domain's row is streamed per forward.
    gen += dims.hidden * dims.action_dim + dims.action_dim * dims.hidden

    return {
        "cond_encode": und * dtype_bytes,
        "gen_1step": gen * dtype_bytes,
    }


def _main() -> None:
    dims = Cosmos3Dims()
    # Released Cosmos3-Nano-Policy-DROID config (tech report §4.2.5 + cosmos-framework):
    # action chunk 32, video = chunk+1 = 33 frames @ 480x832, N=4 steps, CFG 3.0.
    # Latent grid via the VAE strides (temporal 4, spatial 16): (9, 30, 52).
    s_text = 96
    chunk = 32
    num_frames = chunk + 1
    t_lat, h_lat, w_lat = (num_frames - 1) // 4 + 1, 480 // 16, 832 // 16
    n_steps = 4
    guidance = 3.0

    flop = stage_flops(
        dims,
        s_text=s_text,
        t_lat=t_lat,
        h_lat=h_lat,
        w_lat=w_lat,
        action_chunk=chunk,
        num_inference_steps=n_steps,
        guidance_scale=guidance,
    )
    wbytes = analytic_weight_bytes(dims)

    print(
        f"Cosmos3-Nano-Policy analytic FLOP model  (s_text={s_text}, "
        f"latent grid=({t_lat},{h_lat},{w_lat}), chunk={chunk}, "
        f"steps={n_steps}, cfg_branches={int(flop['cfg_branches'])})"
    )
    print(
        f"  tokens: S_video={int(flop['s_video'])}  S_action={int(flop['s_action'])}  "
        f"S_gen={int(flop['s_gen'])}  gen_kv_len={int(flop['gen_kv_len'])}"
    )
    print("weight bytes are an ANALYTIC ESTIMATE (driver uses live model bytes)\n")
    print(f"{'phase':<18}{'GFLOP':>14}{'W_MiB(est)':>12}{'AI(est)':>10}")
    for k in ("cond_encode", "gen_1step"):
        ai = flop[k] / wbytes[k]
        print(f"{k:<18}{flop[k] / 1e9:>14.2f}{wbytes[k] / 2**20:>12.1f}{ai:>10.1f}")
    print(f"{'gen_loop':<18}{flop['gen_loop'] / 1e9:>14.2f}{'-':>12}{'-':>10}")

    e2e = flop["cond_encode_total"] + flop["gen_loop"]
    print(
        f"\nper-request transformer compute (cond×{int(flop['cfg_branches'])} + "
        f"{n_steps}-step gen loop) = {e2e / 1e9:.1f} GFLOP"
    )


if __name__ == "__main__":
    _main()
