from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYAI_SRC = REPO_ROOT / "phyai" / "src"
if str(PHYAI_SRC) not in sys.path:
    sys.path.insert(0, str(PHYAI_SRC))

from phyai.models.walloss05_native import WallOSS05MRoPENative  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _official_mrope_formula(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype = query_states.dtype
    k_dtype = key_states.dtype

    cos = cos.float()
    sin = sin.float()
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    doubled = list(mrope_section) + list(mrope_section)

    cos_split = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(doubled, dim=-1))],
        dim=-1,
    ).unsqueeze(2)
    sin_split = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(doubled, dim=-1))],
        dim=-1,
    ).unsqueeze(2)

    q_embed = (query_states.float() * cos_split) + (_rotate_half(query_states.float()) * sin_split)
    k_embed = (key_states.float() * cos_split) + (_rotate_half(key_states.float()) * sin_split)
    return q_embed.to(q_dtype), k_embed.to(k_dtype)


def _make_position_cos_sin(
    *,
    batch: int,
    seq: int,
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.int64, device=device).to(torch.float32)
            / head_dim
        )
    )

    base = torch.arange(seq, dtype=torch.long, device=device).unsqueeze(0).expand(batch, seq)
    position_ids = torch.stack([base, base + 1, base + 2], dim=0)

    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, batch, -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_full = emb.cos().to(dtype)
    sin_full = emb.sin().to(dtype)

    # Match JointQwen2VLAttention._apply_rotary_pos_embed:
    # cos[..., : (cos.size(3) // 2)].contiguous().float()
    return (
        cos_full[..., : head_dim // 2].contiguous().float(),
        sin_full[..., : head_dim // 2].contiguous().float(),
    )


def _compare(name: str, native: torch.Tensor, ref: torch.Tensor, *, atol: float) -> None:
    diff = (native.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cosine = float(F.cosine_similarity(native.flatten().float(), ref.flatten().float(), dim=0).item())
    exact_equal = bool(torch.equal(native, ref))
    allclose = bool(torch.allclose(native, ref, atol=atol, rtol=atol))

    print(f"\n--- {name} ---")
    print("shape:", tuple(native.shape), native.dtype, native.device)
    print("max_abs_diff:", max_abs)
    print("mean_abs_diff:", mean_abs)
    print("cosine:", cosine)
    print("exact_equal:", exact_equal)
    print(f"allclose_{atol}:", allclose)

    if max_abs > atol or cosine < 0.999999:
        raise SystemExit(f"FAILED: {name} M-RoPE parity failed")


def _run_formula_case(dtype: torch.dtype, cfg: dict, device: torch.device) -> None:
    print(f"\n========== formula parity dtype={dtype} device={device} ==========")

    batch = 2
    seq = 7
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["hidden_size"]) // num_heads
    rope_theta = float(cfg["rope_theta"])
    mrope_section = cfg["rope_scaling"]["mrope_section"]

    torch.manual_seed(15000 + (0 if dtype == torch.float32 else 1))
    q = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32, device=device).to(dtype).contiguous()
    k = torch.randn(batch, seq, num_kv_heads, head_dim, dtype=torch.float32, device=device).to(dtype).contiguous()
    cos, sin = _make_position_cos_sin(
        batch=batch,
        seq=seq,
        head_dim=head_dim,
        rope_theta=rope_theta,
        dtype=dtype,
        device=device,
    )

    native = WallOSS05MRoPENative(mrope_section).to(device).eval()

    with torch.no_grad():
        q_native, k_native = native(q.clone(), k.clone(), cos.clone(), sin.clone())
        q_ref, k_ref = _official_mrope_formula(q.clone(), k.clone(), cos.clone(), sin.clone(), mrope_section)

    _compare("q_mrope_vs_formula", q_native, q_ref, atol=1e-6)
    _compare("k_mrope_vs_formula", k_native, k_ref, atol=1e-6)


def _run_cuda_backend_case(dtype: torch.dtype, cfg: dict) -> None:
    if not torch.cuda.is_available():
        print("\nCUDA backend parity: SKIPPED because CUDA is not available")
        return

    print(f"\n========== official CUDA backend parity dtype={dtype} ==========")

    # Import only inside the CUDA-only path. Do not call official proxy on CPU,
    # because OpsProxy may auto-resolve to cuda_inline even for CPU tensors.
    sys.path.insert(0, "/phyai_workspace/src/wall-x_main_0p5_clone")
    from wall_x.model.core.ops import m_rope as official_m_rope  # type: ignore

    device = torch.device("cuda:0")
    batch = 2
    seq = 7
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["hidden_size"]) // num_heads
    rope_theta = float(cfg["rope_theta"])
    mrope_section = cfg["rope_scaling"]["mrope_section"]

    torch.manual_seed(18000 + (0 if dtype == torch.float32 else 1))
    q = torch.randn(batch, seq, num_heads, head_dim, dtype=torch.float32, device=device).to(dtype).contiguous()
    k = torch.randn(batch, seq, num_kv_heads, head_dim, dtype=torch.float32, device=device).to(dtype).contiguous()
    cos, sin = _make_position_cos_sin(
        batch=batch,
        seq=seq,
        head_dim=head_dim,
        rope_theta=rope_theta,
        dtype=dtype,
        device=device,
    )

    native = WallOSS05MRoPENative(mrope_section).to(device).eval()

    with torch.no_grad():
        q_native, k_native = native(q.clone(), k.clone(), cos.clone(), sin.clone())
        q_fallback, k_fallback = official_m_rope.call_with_backend(
            "pytorch", q.clone(), k.clone(), cos.clone(), sin.clone(), mrope_section
        )
        q_cuda, k_cuda = official_m_rope.call_with_backend(
            "cuda_inline", q.clone(), k.clone(), cos.clone(), sin.clone(), mrope_section
        )

    # fp32 CUDA kernel differs from fallback only by floating-point operation order.
    atol = 1e-6
    _compare("q_native_vs_official_fallback_cuda", q_native, q_fallback, atol=atol)
    _compare("k_native_vs_official_fallback_cuda", k_native, k_fallback, atol=atol)
    _compare("q_cuda_inline_vs_official_fallback", q_cuda, q_fallback, atol=atol)
    _compare("k_cuda_inline_vs_official_fallback", k_cuda, k_fallback, atol=atol)
    _compare("q_native_vs_cuda_inline", q_native, q_cuda, atol=atol)
    _compare("k_native_vs_cuda_inline", k_native, k_cuda, atol=atol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    cfg = _load_json(args.checkpoint / "config.json")

    print("========== Config ==========")
    print("rope_scaling:", cfg.get("rope_scaling"))
    print("rope_theta:", cfg.get("rope_theta"))
    print("hidden_size:", cfg.get("hidden_size"))
    print("num_attention_heads:", cfg.get("num_attention_heads"))
    print("num_key_value_heads:", cfg.get("num_key_value_heads"))
    print("head_dim:", cfg["hidden_size"] // cfg["num_attention_heads"])

    for dtype in [torch.float32, torch.bfloat16]:
        _run_formula_case(dtype, cfg, torch.device("cpu"))

    for dtype in [torch.float32, torch.bfloat16]:
        _run_cuda_backend_case(dtype, cfg)

    print("\nPASS: native WALL-OSS-0.5 M-RoPE matches official fallback and safe CUDA backend checks.")


if __name__ == "__main__":
    main()
