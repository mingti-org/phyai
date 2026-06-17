from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYAI_SRC = REPO_ROOT / "phyai" / "src"
if str(PHYAI_SRC) not in sys.path:
    sys.path.insert(0, str(PHYAI_SRC))

from phyai.models.walloss05_native import (  # noqa: E402
    WallOSS05NativeConfig,
    WallOSS05NormMoeNative,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _official_qwen2_rmsnorm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    normed = weight.to(torch.float32) * normed
    return normed.to(input_dtype)


def _official_norm_moe(
    hidden_states: torch.Tensor,
    weights: list[torch.Tensor],
    dim_inputs: tuple[int, ...],
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    new_hidden_states = torch.zeros_like(hidden_states)
    for expert_idx, weight in enumerate(weights):
        start = int(start_indices[expert_idx].item())
        end = int(end_indices[expert_idx].item())
        if start == end:
            continue
        dim_input = dim_inputs[expert_idx]
        selected = hidden_states[start:end]
        input_slice = selected[:, :dim_input]
        processed = _official_qwen2_rmsnorm(input_slice, weight, eps)
        new_hidden_states[start:end, :dim_input] = processed.to(hidden_states.dtype)
    return new_hidden_states


def _copy_param(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    params = dict(module.named_parameters())
    if name not in params:
        raise KeyError(f"{type(module).__name__} has no parameter {name!r}; available={list(params)[:20]}")
    param = params[name]
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch for {name}: module={tuple(param.shape)} ckpt={tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value.to(device=param.device, dtype=param.dtype))


def _load_norm_weights(module: WallOSS05NormMoeNative, checkpoint: Path, kind: str) -> list[torch.Tensor]:
    if kind == "input":
        prefix = "model.layers.0.input_layernorms"
    elif kind == "post_attention":
        prefix = "model.layers.0.post_attention_layernorms"
    else:
        raise ValueError(kind)

    weights: list[torch.Tensor] = []
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as sf:
        for idx in [0, 1]:
            key = f"{prefix}.{idx}.weight"
            tensor = sf.get_tensor(key).float()
            _copy_param(module, f"norms.{idx}.weight", tensor)
            weights.append(tensor)
            print(f"[loaded] {key} -> norms.{idx}.weight {tuple(tensor.shape)} {tensor.dtype}")
    return weights


def _run_one_kind(
    cfg: WallOSS05NativeConfig,
    checkpoint: Path,
    kind: str,
    dtype: torch.dtype,
) -> None:
    native = WallOSS05NormMoeNative(cfg, layer_idx=0, kind=kind, dtype=torch.float32, device="cpu")
    native.eval()
    weights = _load_norm_weights(native, checkpoint, kind)

    torch.manual_seed(7000 + (0 if kind == "input" else 100) + (0 if dtype == torch.float32 else 1))
    n0 = 6
    n1 = 5
    total = n0 + n1
    hidden = cfg.hidden_size
    hidden_states = torch.randn(total, hidden, dtype=torch.float32).to(dtype)
    start_indices = torch.tensor([0, n0], dtype=torch.long)
    end_indices = torch.tensor([n0, total], dtype=torch.long)

    with torch.no_grad():
        native_out, gate, gate_mask = native(
            hidden_states.clone(),
            start_indices.clone(),
            end_indices.clone(),
            adarms_conds=[None, None],
        )
        ref_out = _official_norm_moe(
            hidden_states.clone(),
            weights,
            tuple(cfg.dim_inputs),
            start_indices,
            end_indices,
            cfg.rms_norm_eps,
        )

    diff = (native_out.float() - ref_out.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cosine = float(F.cosine_similarity(native_out.flatten().float(), ref_out.flatten().float(), dim=0).item())
    exact_equal = bool(torch.equal(native_out, ref_out))
    allclose_1e_6 = bool(torch.allclose(native_out, ref_out, atol=1e-6, rtol=1e-6))

    tail = native_out[n0:, cfg.dim_inputs[1]:]
    tail_max = float(tail.abs().max().item()) if tail.numel() else 0.0

    print(f"\n===== kind={kind} dtype={dtype} =====")
    print("native_out shape:", tuple(native_out.shape), native_out.dtype)
    print("gate:", gate)
    print("gate_mask:", gate_mask)
    print("max_abs_diff:", max_abs)
    print("mean_abs_diff:", mean_abs)
    print("cosine:", cosine)
    print("exact_equal:", exact_equal)
    print("allclose_1e_6:", allclose_1e_6)
    print("expert1 padded tail max abs:", tail_max)

    if max_abs > 1e-6 or not exact_equal:
        raise SystemExit(f"FAILED: NormMoe parity failed for kind={kind}, dtype={dtype}")
    if tail_max != 0.0:
        raise SystemExit("FAILED: expert1 padded tail should remain zero.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--norm-key", default="x2_normal")
    args = parser.parse_args()

    ckpt_config = _load_json(args.checkpoint / "config.json")
    train_config = _load_yaml(args.train_config)
    cfg = WallOSS05NativeConfig.from_checkpoint_and_train_config(
        ckpt_config,
        train_config,
        norm_key=args.norm_key,
    )

    print("========== Config ==========")
    print("rms_norm_eps:", cfg.rms_norm_eps)
    print("num_experts:", cfg.num_experts)
    print("dim_inputs:", cfg.dim_inputs)
    print("norm_moe:", cfg.norm_moe)
    print("mot_opt:", cfg.mot_opt)
    print("use_adarms:", cfg.use_adarms)

    for kind in ["input", "post_attention"]:
        for dtype in [torch.float32, torch.bfloat16]:
            _run_one_kind(cfg, args.checkpoint, kind, dtype)

    print("\nPASS: native Norm-MoE matches official Qwen2RMSNorm formula for layer-0 input/post-attention norms.")


if __name__ == "__main__":
    main()
