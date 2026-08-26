from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file


WEIGHT_SUFFIXES = {
    "router": "qwen_expert.model.layers.0.mlp.gate.weight",
    "correction": "qwen_expert.model.layers.0.mlp.e_score_correction_bias",
    "expert_gate": "qwen_expert.model.layers.0.mlp.experts.gate_proj",
    "expert_up": "qwen_expert.model.layers.0.mlp.experts.up_proj",
    "expert_down": "qwen_expert.model.layers.0.mlp.experts.down_proj",
    "shared_gate": "qwen_expert.model.layers.0.mlp.shared_expert.gate_proj.weight",
    "shared_up": "qwen_expert.model.layers.0.mlp.shared_expert.up_proj.weight",
    "shared_down": "qwen_expert.model.layers.0.mlp.shared_expert.down_proj.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare official Robby Triton MoE and PHYAI FlashInfer CUTLASS "
            "MoE on identical LingBot V2 weights, inputs, and routes."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--hidden-artifact", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-timed", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/lingbot_v2/lingbot_v2_moe_thor.json"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution: str, module_name: str | None = None) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        if module_name is None:
            return "unknown"
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))


def resolve_checkpoint_keys(
    checkpoint: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"checkpoint index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    resolved: dict[str, str] = {}
    for name, suffix in WEIGHT_SUFFIXES.items():
        matches = [key for key in weight_map if key.endswith(suffix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one checkpoint key ending with {suffix!r}, "
                f"found {matches}."
            )
        resolved[name] = matches[0]
    return resolved, weight_map


def load_checkpoint_moe_config(checkpoint: Path) -> tuple[int, float]:
    """Load the routed-MoE contract recorded by the checkpoint."""

    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config is missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"checkpoint config must be a JSON object: {config_path}")

    root_containers = [config]
    for parent_key in ("model", "model_config"):
        parent = config.get(parent_key)
        if isinstance(parent, dict):
            root_containers.append(parent)
    moe_containers = []
    for parent in root_containers:
        for moe_key in ("moe", "moe_config"):
            moe = parent.get(moe_key)
            if isinstance(moe, dict):
                moe_containers.append(moe)

    top_k = next(
        (
            container["token_top_k"]
            for container in root_containers
            if "token_top_k" in container
        ),
        None,
    )
    if top_k is None:
        top_k = next(
            (
                container[key]
                for container in moe_containers
                for key in ("top_k", "token_top_k")
                if key in container
            ),
            None,
        )
    routed_scaling_factor = next(
        (
            container["routed_scaling_factor"]
            for container in (*root_containers, *moe_containers)
            if "routed_scaling_factor" in container
        ),
        None,
    )
    missing = [
        name
        for name, value in (
            ("token_top_k/top_k", top_k),
            ("routed_scaling_factor", routed_scaling_factor),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"checkpoint config {config_path} is missing required MoE fields: "
            + ", ".join(missing)
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"checkpoint MoE top_k must be an integer, got {top_k!r}")
    if top_k <= 0:
        raise ValueError(
            f"checkpoint MoE top_k must be a positive integer, got {top_k!r}"
        )
    if isinstance(routed_scaling_factor, bool) or not isinstance(
        routed_scaling_factor, (int, float)
    ):
        raise TypeError(
            "checkpoint routed_scaling_factor must be numeric, "
            f"got {routed_scaling_factor!r}"
        )
    routed_scaling_factor = float(routed_scaling_factor)
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
        raise ValueError(
            "checkpoint routed_scaling_factor must be finite and positive, "
            f"got {routed_scaling_factor!r}"
        )
    return top_k, routed_scaling_factor


def load_layer0_weights(
    checkpoint: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    resolved, weight_map = resolve_checkpoint_keys(checkpoint)
    weights: dict[str, torch.Tensor] = {}
    for name, key in resolved.items():
        shard = checkpoint / weight_map[key]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(key)
        weights[name] = value.to(device=device, dtype=torch.bfloat16).contiguous()
    return weights, resolved


def load_robby_moe(official_repo: Path) -> tuple[Callable, Path]:
    source = official_repo / "lingbotvla" / "ops" / "robby_moe.py"
    if not source.is_file():
        raise FileNotFoundError(f"official Robby MoE source is missing: {source}")
    spec = importlib.util.spec_from_file_location("lingbot_v2_robbby_moe", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official Robby MoE from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.robby_moe_forward, source


def make_hidden_inputs(
    hidden_artifact: Path | None,
    *,
    hidden_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, str | None]]:
    if hidden_artifact is not None:
        if not hidden_artifact.is_file():
            raise FileNotFoundError(
                f"hidden-state artifact is missing: {hidden_artifact}"
            )
        tensors = load_file(hidden_artifact, device="cpu")
        keys = {
            "state": "expert.layer.0.post_norm.state",
            "action": "expert.layer.0.post_norm.action",
        }
        missing = [key for key in keys.values() if key not in tensors]
        if missing:
            raise ValueError(
                f"hidden-state artifact does not contain required tensors: {missing}"
            )
        result = {
            stream: tensors[key]
            .reshape(-1, hidden_size)
            .to(device=device, dtype=torch.bfloat16)
            .contiguous()
            for stream, key in keys.items()
        }
        return result, {
            "kind": "captured_layer0_post_norm",
            "path": str(hidden_artifact),
            "sha256": sha256_file(hidden_artifact),
        }

    generator = torch.Generator(device=device)
    generator.manual_seed(20260731)
    return {
        "state": torch.randn(
            1,
            hidden_size,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ),
        "action": torch.randn(
            50,
            hidden_size,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ),
    }, {"kind": "deterministic_synthetic", "path": None, "sha256": None}


def route_tokens(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast(hidden.device.type, enabled=False):
        logits = F.linear(hidden.float(), router_weight.float())
    scores = logits.sigmoid()
    selected = torch.topk(
        scores + correction_bias.float().unsqueeze(0),
        top_k,
        dim=-1,
    ).indices
    routing_weights = scores.gather(1, selected)
    routing_weights = routing_weights / (
        routing_weights.sum(dim=-1, keepdim=True) + 1e-20
    )
    if routed_scaling_factor != 1.0:
        routing_weights = routing_weights * routed_scaling_factor
    return routing_weights.to(hidden.dtype), selected


def make_robby_workspace(
    hidden: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
) -> dict[str, torch.Tensor]:
    tokens, hidden_size = hidden.shape
    max_routes = tokens * top_k
    return {
        "counts": torch.empty(num_experts, dtype=torch.int32, device=hidden.device),
        "rows": torch.empty(
            num_experts, max_routes, dtype=torch.int32, device=hidden.device
        ),
        "slots": torch.empty(
            num_experts, max_routes, dtype=torch.int32, device=hidden.device
        ),
        "inter": torch.empty(
            tokens,
            top_k,
            intermediate_size,
            dtype=hidden.dtype,
            device=hidden.device,
        ),
        "out": torch.empty(
            tokens, hidden_size, dtype=torch.float32, device=hidden.device
        ),
    }


def time_cuda(
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    *,
    n_warmup: int,
    n_timed: int,
) -> float:
    if n_warmup < 0 or n_timed <= 0:
        raise ValueError("n_warmup must be non-negative and n_timed must be positive.")
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_timed):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / n_timed


def tensor_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    reference32 = reference.float().reshape(-1)
    candidate32 = candidate.float().reshape(-1)
    return {
        "cosine": float(F.cosine_similarity(reference32, candidate32, dim=0).item()),
        "max_abs": float((reference32 - candidate32).abs().max().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(reference32 - candidate32).div(
                torch.linalg.vector_norm(reference32).clamp_min(1e-12)
            )
        ),
    }


@torch.inference_mode()
def benchmark_stream(
    name: str,
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    robby_moe_forward: Callable,
    *,
    top_k: int,
    routed_scaling_factor: float,
    n_warmup: int,
    n_timed: int,
) -> dict:
    from flashinfer.fused_moe import cutlass_fused_moe

    num_experts, intermediate_size, hidden_size = weights["expert_gate"].shape
    routing_weights, selected_experts = route_tokens(
        hidden,
        weights["router"],
        weights["correction"],
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
    )
    fc1 = torch.cat(
        [weights["expert_up"], weights["expert_gate"]],
        dim=1,
    ).contiguous()
    workspace = make_robby_workspace(
        hidden,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
    )

    def router() -> tuple[torch.Tensor, torch.Tensor]:
        return route_tokens(
            hidden,
            weights["router"],
            weights["correction"],
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
        )

    def flashinfer_routed() -> torch.Tensor:
        result = cutlass_fused_moe(
            hidden,
            selected_experts.to(torch.int32),
            routing_weights.float(),
            fc1,
            weights["expert_down"],
            hidden.dtype,
            quant_scales=None,
            tp_size=1,
            tp_rank=0,
        )
        return result[0]

    def robby_routed() -> torch.Tensor:
        return robby_moe_forward(
            hidden,
            routing_weights,
            selected_experts,
            weights["expert_gate"],
            weights["expert_up"],
            weights["expert_down"],
            workspace=workspace,
        )

    def shared() -> torch.Tensor:
        gate = F.linear(hidden, weights["shared_gate"])
        up = F.linear(hidden, weights["shared_up"])
        return F.linear(F.silu(gate) * up, weights["shared_down"])

    def flashinfer_complete() -> torch.Tensor:
        current_weights, current_selected = router()
        result = cutlass_fused_moe(
            hidden,
            current_selected.to(torch.int32),
            current_weights.float(),
            fc1,
            weights["expert_down"],
            hidden.dtype,
            quant_scales=None,
            tp_size=1,
            tp_rank=0,
        )[0]
        return result + shared()

    def robby_complete() -> torch.Tensor:
        current_weights, current_selected = router()
        result = robby_moe_forward(
            hidden,
            current_weights,
            current_selected,
            weights["expert_gate"],
            weights["expert_up"],
            weights["expert_down"],
            workspace=workspace,
        )
        return result + shared()

    flash_output = flashinfer_routed()
    robby_output = robby_routed()
    torch.cuda.synchronize(hidden.device)
    routed_metrics = tensor_metrics(robby_output, flash_output)

    timings = {
        "router": time_cuda(router, n_warmup=n_warmup, n_timed=n_timed),
        "shared_expert": time_cuda(shared, n_warmup=n_warmup, n_timed=n_timed),
        "flashinfer_routed": time_cuda(
            flashinfer_routed, n_warmup=n_warmup, n_timed=n_timed
        ),
        "robby_routed": time_cuda(robby_routed, n_warmup=n_warmup, n_timed=n_timed),
        "flashinfer_complete": time_cuda(
            flashinfer_complete, n_warmup=n_warmup, n_timed=n_timed
        ),
        "robby_complete": time_cuda(robby_complete, n_warmup=n_warmup, n_timed=n_timed),
    }
    histogram = torch.bincount(
        selected_experts.reshape(-1), minlength=num_experts
    ).cpu()
    return {
        "name": name,
        "tokens": int(hidden.shape[0]),
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_experts": num_experts,
        "top_k": top_k,
        "route_histogram": histogram.tolist(),
        "timing_ms": timings,
        "speedup": {
            "flashinfer_vs_robby_routed": (
                timings["robby_routed"] / timings["flashinfer_routed"]
            ),
            "flashinfer_vs_robby_complete": (
                timings["robby_complete"] / timings["flashinfer_complete"]
            ),
        },
        "routed_output_parity": routed_metrics,
        "output_dtype": {
            "flashinfer": str(flash_output.dtype),
            "robby": str(robby_output.dtype),
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the LingBot V2 MoE benchmark.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    top_k, routed_scaling_factor = load_checkpoint_moe_config(args.checkpoint)
    weights, checkpoint_keys = load_layer0_weights(args.checkpoint, device)
    robby_moe_forward, robby_source = load_robby_moe(args.official_repo)
    hidden_size = int(weights["router"].shape[1])
    hidden_inputs, input_info = make_hidden_inputs(
        args.hidden_artifact,
        hidden_size=hidden_size,
        device=device,
    )

    streams = {
        name: benchmark_stream(
            name,
            hidden,
            weights,
            robby_moe_forward,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
        )
        for name, hidden in hidden_inputs.items()
    }
    combined = {
        metric: sum(stream["timing_ms"][metric] for stream in streams.values())
        for metric in (
            "router",
            "shared_expert",
            "flashinfer_routed",
            "robby_routed",
            "flashinfer_complete",
            "robby_complete",
        )
    }
    combined["estimated_flashinfer_expert_loop"] = (
        combined["flashinfer_complete"] * 36 * 10
    )
    combined["estimated_robby_expert_loop"] = combined["robby_complete"] * 36 * 10

    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer": package_version("flashinfer-python", "flashinfer"),
            "triton": package_version("triton"),
        },
        "checkpoint": str(args.checkpoint),
        "moe_config": {
            "top_k": top_k,
            "routed_scaling_factor": routed_scaling_factor,
        },
        "checkpoint_keys": checkpoint_keys,
        "robby_source": str(robby_source),
        "robby_source_sha256": sha256_file(robby_source),
        "input": input_info,
        "n_warmup": args.n_warmup,
        "n_timed": args.n_timed,
        "streams": streams,
        "combined_state_action_ms": combined,
        "combined_speedup": {
            "flashinfer_vs_robby_routed": (
                combined["robby_routed"] / combined["flashinfer_routed"]
            ),
            "flashinfer_vs_robby_complete": (
                combined["robby_complete"] / combined["flashinfer_complete"]
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("LingBot V2 MoE A/B benchmark")
    print(f"GPU: {result['gpu']['name']}")
    print(f"Input: {input_info['kind']}")
    for name, stream in streams.items():
        timing = stream["timing_ms"]
        print(f"\n{name} stream: T={stream['tokens']}")
        print(f"  Router                : {timing['router']:.6f} ms")
        print(f"  Shared expert         : {timing['shared_expert']:.6f} ms")
        print(f"  FlashInfer routed     : {timing['flashinfer_routed']:.6f} ms")
        print(f"  Robby routed          : {timing['robby_routed']:.6f} ms")
        print(f"  FlashInfer complete   : {timing['flashinfer_complete']:.6f} ms")
        print(f"  Robby complete        : {timing['robby_complete']:.6f} ms")
        print(
            "  FlashInfer speedup    : "
            f"{stream['speedup']['flashinfer_vs_robby_complete']:.3f}x"
        )
        print(
            "  Routed output cosine  : "
            f"{stream['routed_output_parity']['cosine']:.9f}"
        )
    print("\nCombined state + action per layer/step")
    print(f"  FlashInfer complete   : {combined['flashinfer_complete']:.6f} ms")
    print(f"  Robby complete        : {combined['robby_complete']:.6f} ms")
    print(
        "  FlashInfer speedup    : "
        f"{result['combined_speedup']['flashinfer_vs_robby_complete']:.3f}x"
    )
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
