from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import yaml
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYAI_SRC = REPO_ROOT / "phyai" / "src"
WALLX_SRC = Path("/phyai_workspace/src/wall-x_main_0p5_clone")

for p in [PHYAI_SRC, WALLX_SRC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# The official Wall-X SparseMoeBlock only needs transformers.activations.ACT2FN.
# Keep this parity script dependency-light by injecting a minimal shim instead
# of requiring the full transformers package in the PR26 PhyAI environment.
if "transformers.activations" not in sys.modules:
    transformers_mod = types.ModuleType("transformers")
    activations_mod = types.ModuleType("transformers.activations")
    activations_mod.ACT2FN = {
        "silu": torch.nn.functional.silu,
        "gelu": torch.nn.functional.gelu,
        "relu": torch.nn.functional.relu,
    }
    transformers_mod.activations = activations_mod
    sys.modules.setdefault("transformers", transformers_mod)
    sys.modules["transformers.activations"] = activations_mod

import phyai.layers.linear as L  # noqa: E402
from phyai.models.walloss05_native import (  # noqa: E402
    WallOSS05NativeConfig,
    WallOSS05SparseMoeBlockNative,
)
from wall_x.model.core.action.moe import SparseMoeBlock  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _copy_param(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    params = dict(module.named_parameters())
    if name not in params:
        raise KeyError(f"{type(module).__name__} has no parameter {name!r}; available={list(params)[:30]}")
    param = params[name]
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch for {name}: module={tuple(param.shape)} ckpt={tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value.to(device=param.device, dtype=param.dtype))


def _load_layer0_moe_weights(native, official, checkpoint: Path) -> None:
    keys = []
    for expert_idx in [0, 1]:
        for leaf in ["gate_up_proj.weight", "down_proj.weight"]:
            local = f"experts.{expert_idx}.{leaf}"
            ckpt = f"model.layers.0.moe.experts.{expert_idx}.{leaf}"
            keys.append((local, ckpt))

    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as sf:
        for local_name, ckpt_key in keys:
            tensor = sf.get_tensor(ckpt_key).float()
            _copy_param(native, local_name, tensor)
            _copy_param(official, local_name, tensor)
            print(f"[loaded] {ckpt_key} -> {local_name} {tuple(tensor.shape)} {tensor.dtype}")


def _make_official_config(cfg: WallOSS05NativeConfig) -> SimpleNamespace:
    return SimpleNamespace(
        experts=list(cfg.experts),
        dim_inputs=list(cfg.dim_inputs),
        mot_opt=cfg.mot_opt,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--tokens-expert0", type=int, default=5)
    parser.add_argument("--tokens-expert1", type=int, default=4)
    args = parser.parse_args()

    L.init(register_flashinfer=False, validate=True, sample_specs=["bf16"])

    ckpt_config = _load_json(args.checkpoint / "config.json")
    train_config = _load_yaml(args.train_config)
    cfg = WallOSS05NativeConfig.from_checkpoint_and_train_config(
        ckpt_config,
        train_config,
        norm_key=args.norm_key,
    )

    print("========== Config ==========")
    print("num_experts:", cfg.num_experts)
    print("dim_inputs:", cfg.dim_inputs)
    print("experts:", cfg.experts)
    print("mot_opt:", cfg.mot_opt)

    native = WallOSS05SparseMoeBlockNative(cfg, layer_idx=0, params_dtype=torch.float32, device="cpu")
    official = SparseMoeBlock(_make_official_config(cfg), num_experts=cfg.num_experts, use_selective_recompute=False)

    native.eval()
    official.eval()

    _load_layer0_moe_weights(native, official, args.checkpoint)

    print("\n========== Deterministic input ==========")
    torch.manual_seed(2026)
    n0 = args.tokens_expert0
    n1 = args.tokens_expert1
    total = n0 + n1
    hidden = cfg.hidden_size

    hidden_states = torch.randn(total, hidden, dtype=torch.float32)
    start_indices = torch.tensor([0, n0], dtype=torch.long)
    end_indices = torch.tensor([n0, total], dtype=torch.long)
    experts_indices = torch.cat(
        [
            torch.zeros(n0, dtype=torch.long),
            torch.ones(n1, dtype=torch.long),
        ],
        dim=0,
    )

    with torch.no_grad():
        native_out = native(
            hidden_states.clone(),
            experts_indices.clone(),
            start_indices.clone(),
            end_indices.clone(),
        )
        official_out = official(
            hidden_states.clone(),
            experts_indices.clone(),
            start_indices.clone(),
            end_indices.clone(),
        )

    print("native_out shape:", tuple(native_out.shape), native_out.dtype)
    print("official_out shape:", tuple(official_out.shape), official_out.dtype)

    diff = (native_out.float() - official_out.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cosine = float(F.cosine_similarity(native_out.flatten().float(), official_out.flatten().float(), dim=0).item())
    exact_equal = bool(torch.equal(native_out, official_out))
    allclose_1e_6 = bool(torch.allclose(native_out, official_out, atol=1e-6, rtol=1e-6))

    print("\n========== Parity ==========")
    print("max_abs_diff:", max_abs)
    print("mean_abs_diff:", mean_abs)
    print("cosine:", cosine)
    print("exact_equal:", exact_equal)
    print("allclose_1e_6:", allclose_1e_6)

    # Expert 1 only writes the first 1024 dims; the remaining dims must stay zero.
    tail = native_out[n0:, cfg.dim_inputs[1]:]
    tail_max = float(tail.abs().max().item()) if tail.numel() else 0.0
    print("expert1 padded tail max abs:", tail_max)

    if max_abs > 1e-5 or cosine < 0.999999:
        raise SystemExit("FAILED: native SparseMoeBlock does not match official reference tightly enough.")
    if tail_max != 0.0:
        raise SystemExit("FAILED: expert1 padded tail should remain zero.")

    print("\nPASS: native SparseMoeBlock layer-0 matches official Wall-X SparseMoeBlock.")


if __name__ == "__main__":
    main()
