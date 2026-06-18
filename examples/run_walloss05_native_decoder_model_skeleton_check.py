from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYAI_SRC = REPO_ROOT / "phyai" / "src"
if str(PHYAI_SRC) not in sys.path:
    sys.path.insert(0, str(PHYAI_SRC))

import phyai.layers.linear as L  # noqa: E402
from phyai.models.walloss05_native import (  # noqa: E402
    WallOSS05DecoderModelNative,
    WallOSS05NativeConfig,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _check_param_shape(module: torch.nn.Module, param_name: str, ckpt_tensor: torch.Tensor) -> None:
    params = dict(module.named_parameters())
    if param_name not in params:
        raise KeyError(f"missing param {param_name}")
    shape_model = tuple(params[param_name].shape)
    shape_ckpt = tuple(ckpt_tensor.shape)
    if shape_model != shape_ckpt:
        raise ValueError(f"shape mismatch {param_name}: model={shape_model} ckpt={shape_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--norm-key", default="x2_normal")
    args = parser.parse_args()

    L.init(register_flashinfer=False, validate=True, sample_specs=["bf16"])

    cfg = WallOSS05NativeConfig.from_checkpoint_and_train_config(
        _load_json(args.checkpoint / "config.json"),
        _load_yaml(args.train_config),
        norm_key=args.norm_key,
    )

    print("========== Config ==========")
    print("num_hidden_layers:", cfg.num_hidden_layers)
    print("num_experts:", cfg.num_experts)
    print("dim_inputs:", cfg.dim_inputs)
    print("hidden_size:", cfg.hidden_size)
    print("num_attention_heads:", cfg.num_attention_heads)
    print("num_key_value_heads:", cfg.num_key_value_heads)

    print("\n========== Construct decoder model skeleton ==========")
    model = WallOSS05DecoderModelNative(
        cfg,
        params_dtype=torch.bfloat16,
        device="cpu",
    )
    model.eval()

    print("num layers:", len(model.layers))
    print("num final norms:", len(model.norms))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total native decoder params:", total_params)
    print("trainable native decoder params:", trainable_params)

    assert len(model.layers) == cfg.num_hidden_layers
    assert len(model.norms) == cfg.num_experts

    print("\n========== Check all decoder layer parameter shapes against checkpoint ==========")
    checked = 0
    with safe_open(args.checkpoint / "model.safetensors", framework="pt", device="cpu") as sf:
        for layer_idx in range(cfg.num_hidden_layers):
            mapping = {
                f"layers.{layer_idx}.input_norm.norms.0.weight": f"model.layers.{layer_idx}.input_layernorms.0.weight",
                f"layers.{layer_idx}.input_norm.norms.1.weight": f"model.layers.{layer_idx}.input_layernorms.1.weight",
                f"layers.{layer_idx}.self_attn.projections.qkv_proj_experts.0.weight": f"model.layers.{layer_idx}.self_attn.qkv_proj_experts.0.weight",
                f"layers.{layer_idx}.self_attn.projections.qkv_proj_experts.0.bias": f"model.layers.{layer_idx}.self_attn.qkv_proj_experts.0.bias",
                f"layers.{layer_idx}.self_attn.projections.o_proj_experts.0.weight": f"model.layers.{layer_idx}.self_attn.o_proj_experts.0.weight",
                f"layers.{layer_idx}.self_attn.projections.qkv_proj_experts.1.weight": f"model.layers.{layer_idx}.self_attn.qkv_proj_experts.1.weight",
                f"layers.{layer_idx}.self_attn.projections.qkv_proj_experts.1.bias": f"model.layers.{layer_idx}.self_attn.qkv_proj_experts.1.bias",
                f"layers.{layer_idx}.self_attn.projections.o_proj_experts.1.weight": f"model.layers.{layer_idx}.self_attn.o_proj_experts.1.weight",
                f"layers.{layer_idx}.ffn.post_attention_norm.norms.0.weight": f"model.layers.{layer_idx}.post_attention_layernorms.0.weight",
                f"layers.{layer_idx}.ffn.post_attention_norm.norms.1.weight": f"model.layers.{layer_idx}.post_attention_layernorms.1.weight",
                f"layers.{layer_idx}.ffn.moe.experts.0.gate_up_proj.weight": f"model.layers.{layer_idx}.moe.experts.0.gate_up_proj.weight",
                f"layers.{layer_idx}.ffn.moe.experts.0.down_proj.weight": f"model.layers.{layer_idx}.moe.experts.0.down_proj.weight",
                f"layers.{layer_idx}.ffn.moe.experts.1.gate_up_proj.weight": f"model.layers.{layer_idx}.moe.experts.1.gate_up_proj.weight",
                f"layers.{layer_idx}.ffn.moe.experts.1.down_proj.weight": f"model.layers.{layer_idx}.moe.experts.1.down_proj.weight",
            }
            for param_name, ckpt_key in mapping.items():
                _check_param_shape(model, param_name, sf.get_tensor(ckpt_key))
                checked += 1

        for expert_idx in range(cfg.num_experts):
            param_name = f"norms.{expert_idx}.weight"
            ckpt_key = f"model.norms.{expert_idx}.weight"
            _check_param_shape(model, param_name, sf.get_tensor(ckpt_key))
            checked += 1

    print("checked parameter tensors:", checked)

    print("\n========== Tiny final norm smoke test ==========")
    n0, n1 = 4, 3
    hidden = torch.randn(n0 + n1, cfg.hidden_size, dtype=torch.bfloat16)
    starts = torch.tensor([0, n0], dtype=torch.long)
    ends = torch.tensor([n0, n0 + n1], dtype=torch.long)
    out = model.final_norm(hidden, starts, ends)
    print("final_norm out:", tuple(out.shape), out.dtype)
    assert tuple(out.shape) == (n0 + n1, cfg.hidden_size)

    tail = out[n0:, cfg.dim_inputs[1]:]
    tail_max = float(tail.detach().abs().max()) if tail.numel() else 0.0
    print("expert1 final norm padded tail max abs:", tail_max)
    assert tail_max == 0.0

    print("\nPASS: native decoder model skeleton has 36 layers and matches checkpoint tensor shapes.")


if __name__ == "__main__":
    main()
