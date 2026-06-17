from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
import phyai.layers.linear as L
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYAI_SRC = REPO_ROOT / "phyai" / "src"
if str(PHYAI_SRC) not in sys.path:
    sys.path.insert(0, str(PHYAI_SRC))

from phyai.models.walloss05_native import (  # noqa: E402
    WallOSS05ActionProcessorNative,
    WallOSS05NativeConfig,
)


def _first_weight_shape(module: torch.nn.Module) -> tuple[int, ...]:
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            return tuple(param.shape)
    raise RuntimeError(f"no weight parameter found in {type(module).__name__}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _normalizer_keys(path: Path, norm_key: str) -> None:
    state = torch.load(path, map_location="cpu")
    print(f"normalizer file: {path}")
    print(f"normalizer state type: {type(state).__name__}")
    if not isinstance(state, dict):
        print("normalizer is not a dict-like state; skip key inspection")
        return
    keys = sorted(str(k) for k in state.keys())
    print("normalizer first keys:", keys[:12])
    for candidate in [
        f"min.{norm_key}",
        f"delta.{norm_key}",
        f"min.{norm_key}.weight",
        f"delta.{norm_key}.weight",
    ]:
        if candidate in state:
            tensor = state[candidate]
            print(f"  {candidate}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def main() -> None:
    # Standalone module validation bypasses Engine setup, so initialize the
    # PHYAI linear dispatcher explicitly. Use torch fallback only for this CPU
    # smoke test to avoid requiring flashinfer or fp8-capable hardware.
    L.init(register_flashinfer=False, validate=True, sample_specs=["bf16"])

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

    print("========== Config overlay ==========")
    print("hidden_size:", cfg.hidden_size)
    print("action_hidden_size:", cfg.action_hidden_size)
    print("state_hidden_size:", cfg.state_hidden_size)
    print("dim_inputs:", cfg.dim_inputs)
    print("action_dim_internal:", cfg.action_dim_internal)
    print("propri_dim_internal:", cfg.propri_dim_internal)
    print("action_horizon:", cfg.action_horizon)
    print("action_horizon_flow:", cfg.action_horizon_flow)
    print("proj_with_mask:", cfg.proj_with_mask)
    print("use_adarms:", cfg.use_adarms)
    print("norm_key:", cfg.norm_key)
    print("dof_config:", dict(cfg.dof_config))
    print("agent_pos_config:", dict(cfg.agent_pos_config))

    print("\n========== Construct native action processor ==========")
    module = WallOSS05ActionProcessorNative(
        cfg,
        params_dtype=torch.float32,
        device="cpu",
    )

    native_shapes = {
        "action_preprocessor.w1.weight": _first_weight_shape(module.w1),
        "action_preprocessor.w2.weight": _first_weight_shape(module.w2),
        "action_preprocessor.w3.weight": _first_weight_shape(module.w3),
        "action_preprocessor.action_proj_back.weight": _first_weight_shape(
            module.action_proj_back
        ),
    }

    print("native parameter shapes:")
    for key, shape in native_shapes.items():
        print(f"  {key}: {shape}")

    print("\n========== Compare against checkpoint shapes ==========")
    st_path = args.checkpoint / "model.safetensors"
    failures = []
    with safe_open(st_path, framework="pt", device="cpu") as sf:
        for key, native_shape in native_shapes.items():
            if key not in sf.keys():
                failures.append(f"missing checkpoint key: {key}")
                print(f"  [MISS] {key}")
                continue
            ckpt_shape = tuple(sf.get_tensor(key).shape)
            ok = ckpt_shape == native_shape
            print(f"  [{'OK' if ok else 'BAD'}] {key}: native={native_shape} ckpt={ckpt_shape}")
            if not ok:
                failures.append(f"shape mismatch: {key}: native={native_shape} ckpt={ckpt_shape}")

    print("\n========== Normalizer files ==========")
    _normalizer_keys(args.checkpoint / "normalizer_action.pth", args.norm_key)
    _normalizer_keys(args.checkpoint / "normalizer_propri.pth", args.norm_key)

    print("\n========== Tiny forward shape smoke test ==========")
    batch = 2
    horizon = cfg.action_horizon
    action_dim = cfg.action_dim_internal
    torch.manual_seed(0)
    noisy_action = torch.randn(batch, horizon, action_dim)
    dof_mask = torch.ones(batch, horizon, action_dim)
    timestep = torch.linspace(0.0, 1.0, batch)
    out, adarms_cond = module.step(timestep, noisy_action, dof_mask=dof_mask)
    print("step output shape:", tuple(out.shape))
    print("adarms_cond:", adarms_cond)
    expected = (batch, horizon, cfg.hidden_size)
    if tuple(out.shape) != expected:
        failures.append(f"step output shape mismatch: got={tuple(out.shape)} expected={expected}")

    if failures:
        print("\nFAILED:")
        for item in failures:
            print(" -", item)
        raise SystemExit(1)

    print("\nPASS: config overlay, action processor construction, checkpoint shapes, and smoke forward all succeeded.")


if __name__ == "__main__":
    main()
