from __future__ import annotations

import argparse
import json
import sys
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

import phyai.layers.linear as L  # noqa: E402
from phyai.models.walloss05_native import (  # noqa: E402
    WallOSS05ActionProcessorNative,
    WallOSS05NativeConfig,
)
from wall_x.model.core.action.processor import ActionProcessor  # noqa: E402


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _copy_param(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    params = dict(module.named_parameters())
    if name not in params:
        raise KeyError(f"{type(module).__name__} has no parameter {name!r}; available={list(params)[:20]}")
    param = params[name]
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"shape mismatch for {name}: module={tuple(param.shape)} ckpt={tuple(value.shape)}")
    with torch.no_grad():
        param.copy_(value.to(device=param.device, dtype=param.dtype))


def _load_action_processor_weights(native, official, checkpoint: Path) -> None:
    keys = {
        "w1.weight": "action_preprocessor.w1.weight",
        "w2.weight": "action_preprocessor.w2.weight",
        "w3.weight": "action_preprocessor.w3.weight",
        "action_proj_back.weight": "action_preprocessor.action_proj_back.weight",
    }

    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as sf:
        for local_name, ckpt_key in keys.items():
            tensor = sf.get_tensor(ckpt_key).float()
            _copy_param(native, local_name, tensor)
            _copy_param(official, local_name, tensor)
            print(f"[loaded] {ckpt_key} -> {local_name} {tuple(tensor.shape)} {tensor.dtype}")


def _make_official_config(cfg: WallOSS05NativeConfig) -> SimpleNamespace:
    return SimpleNamespace(
        dof_config=dict(cfg.dof_config),
        agent_pos_config=dict(cfg.agent_pos_config),
        action_hidden_size=cfg.action_hidden_size,
        state_hidden_size=cfg.state_hidden_size,
        hidden_size=cfg.hidden_size,
        dim_inputs=list(cfg.dim_inputs),
        use_state_string_representation=cfg.use_state_string_representation,
        proj_with_mask=cfg.proj_with_mask,
        use_flow_action_expert=cfg.use_flow_action_expert,
        noise_scheduler=dict(cfg.noise_scheduler),
        use_adarms=cfg.use_adarms,
        use_x_pred=cfg.use_x_pred,
        use_x_loss=cfg.use_x_loss,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--norm-key", default="x2_normal")
    parser.add_argument("--batch", type=int, default=2)
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
    print("action_dim_internal:", cfg.action_dim_internal)
    print("propri_dim_internal:", cfg.propri_dim_internal)
    print("action_horizon:", cfg.action_horizon)
    print("action_hidden_size:", cfg.action_hidden_size)
    print("hidden_size:", cfg.hidden_size)
    print("proj_with_mask:", cfg.proj_with_mask)
    print("use_adarms:", cfg.use_adarms)

    native = WallOSS05ActionProcessorNative(cfg, params_dtype=torch.float32, device="cpu")
    official = ActionProcessor(_make_official_config(cfg))

    native.eval()
    official.eval()

    _load_action_processor_weights(native, official, args.checkpoint)

    print("\n========== Deterministic input ==========")
    torch.manual_seed(1234)
    batch = args.batch
    horizon = cfg.action_horizon
    action_dim = cfg.action_dim_internal

    noisy_action = torch.randn(batch, horizon, action_dim, dtype=torch.float32)
    dof_mask = torch.ones(batch, horizon, action_dim, dtype=torch.float32)
    timestep = torch.linspace(0.05, 0.95, batch, dtype=torch.float32)

    with torch.no_grad():
        native_out, native_adarms = native.step(timestep, noisy_action.clone(), dof_mask=dof_mask.clone())
        official_out, official_adarms = official.step(timestep, noisy_action.clone(), dof_mask=dof_mask.clone())

    print("native_out shape:", tuple(native_out.shape), native_out.dtype)
    print("official_out shape:", tuple(official_out.shape), official_out.dtype)
    print("native_adarms:", native_adarms)
    print("official_adarms:", official_adarms)

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

    if max_abs > 1e-5 or cosine < 0.999999:
        raise SystemExit("FAILED: native ActionProcessor.step does not match official reference tightly enough.")

    print("\nPASS: native ActionProcessor.step matches official Wall-X ActionProcessor.step.")


if __name__ == "__main__":
    main()
