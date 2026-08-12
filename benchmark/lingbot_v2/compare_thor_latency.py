from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CONTRACT_KEYS = (
    "batch_size",
    "num_images",
    "max_patches_per_image",
    "patches_per_image",
    "patch_vector_dim",
    "lang_len",
    "input_ids",
    "chunk_size",
    "num_inference_steps",
    "action_dim",
)

META_KEYS = (
    "input_sha256",
    "params_dtype",
    "vision_dtype",
    "use_cuda_graph",
    "torch_compile",
    "n_warmup",
    "n_timed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one Official/PHYAI Thor benchmark pair and write a "
            "machine-readable CSV plus a Markdown latency table."
        )
    )
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--phyai", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def required_meta_string(meta: dict[str, Any], key: str) -> str:
    """Return one required non-empty string from benchmark metadata."""

    value = meta.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"meta.{key} must be a non-empty string")
    return value.strip()


def patch_embed_operator(meta: dict[str, Any]) -> str:
    """Canonicalize old and new report labels to Conv3D or GEMM."""

    value = required_meta_string(meta, "vision_patch_embed_backend").lower()
    if "gemm" in value:
        return "gemm"
    if "conv3d" in value:
        return "conv3d"
    return value


def validate_pair(
    official: dict[str, Any],
    phyai: dict[str, Any],
) -> dict[str, Any]:
    official_meta = official["meta"]
    phyai_meta = phyai["meta"]
    mismatches: list[str] = []
    for name, meta in (("official", official_meta), ("phyai", phyai_meta)):
        try:
            required_meta_string(meta, "attention_backend")
        except ValueError as error:
            mismatches.append(f"{name}.{error}")
        try:
            patch_embed_operator(meta)
        except ValueError as error:
            mismatches.append(f"{name}.{error}")
    for key in META_KEYS:
        if official_meta.get(key) != phyai_meta.get(key):
            mismatches.append(
                f"meta.{key}: official={official_meta.get(key)!r}, "
                f"phyai={phyai_meta.get(key)!r}"
            )
    for key in CONTRACT_KEYS:
        official_value = official_meta["contract"].get(key)
        phyai_value = phyai_meta["contract"].get(key)
        if official_value != phyai_value:
            mismatches.append(
                f"contract.{key}: official={official_value!r}, "
                f"phyai={phyai_value!r}"
            )
    official_gpu = official["hardware"].get("name")
    phyai_gpu = phyai["hardware"].get("name")
    if official_gpu != phyai_gpu:
        mismatches.append(
            f"hardware.name: official={official_gpu!r}, phyai={phyai_gpu!r}"
        )
    if "Thor" not in str(official_gpu):
        mismatches.append(f"hardware.name is not a Thor GPU: {official_gpu!r}")
    for name, report in (("official", official), ("phyai", phyai)):
        if not bool(report.get("checks", {}).get("all_finite")):
            mismatches.append(f"{name}.checks.all_finite is not true")
    if mismatches:
        raise ValueError(
            "benchmark contracts are not comparable:\n- " + "\n- ".join(mismatches)
        )

    official_mean = float(official["latency_ms"]["mean"])
    phyai_mean = float(phyai["latency_ms"]["mean"])
    return {
        "official_mean_ms": official_mean,
        "phyai_mean_ms": phyai_mean,
        "phyai_speedup_vs_official": official_mean / phyai_mean,
        "official_speedup_vs_phyai": phyai_mean / official_mean,
        "phyai_slowdown_pct": (100.0 * (phyai_mean - official_mean) / official_mean),
        "phyai_latency_reduction_pct": (
            100.0 * (official_mean - phyai_mean) / official_mean
        ),
    }


def implementation_row(
    name: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    latency = report["latency_ms"]
    checks = report["checks"]
    meta = report["meta"]
    return {
        "implementation": name,
        "gpu": report["hardware"]["name"],
        "mean_ms": latency["mean"],
        "p50_ms": latency["p50"],
        "p90_ms": latency["p90"],
        "p99_ms": latency["p99"],
        "std_ms": latency["std"],
        "peak_allocated_gib": (int(checks["allocated_peak_bytes"]) / 2**30),
        "params_dtype": meta["params_dtype"],
        "vision_dtype": meta["vision_dtype"],
        "attention_backend": required_meta_string(meta, "attention_backend"),
        "moe_backend": meta.get("moe_backend", meta.get("linear_kernel")),
        "input_sha256": meta["input_sha256"],
    }


def phyai_latency_change_text(summary: dict[str, Any]) -> str:
    change = float(summary["phyai_slowdown_pct"])
    if change >= 0:
        return f"+{change:.2f}% slower"
    return f"{-change:.2f}% faster"


def comparison_contract_text(report: dict[str, Any]) -> str:
    """Render the already-validated comparison contract from report data."""

    meta = report["meta"]
    contract = meta["contract"]
    graph = "on" if bool(meta["use_cuda_graph"]) else "off"
    compile_mode = "on" if bool(meta["torch_compile"]) else "off"
    return (
        "Comparison contract: "
        f"B={contract['batch_size']}, {contract['num_images']} active views, "
        f"{contract['patches_per_image']} patches/view, "
        f"parameters={meta['params_dtype']}, vision={meta['vision_dtype']}, "
        "prompt IDs identical, "
        f"chunk={contract['chunk_size']}, "
        f"{contract['num_inference_steps']} Euler steps, "
        f"CUDA Graph {graph}, torch.compile {compile_mode}, and identical "
        f"input SHA256={meta['input_sha256']}."
    )


def write_outputs(
    official: dict[str, Any],
    phyai: dict[str, Any],
    summary: dict[str, Any],
    csv_path: Path,
    markdown_path: Path,
) -> None:
    rows = (
        implementation_row("Official Thor-compatible", official),
        implementation_row("PHYAI", phyai),
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LingBot V2 Thor latency",
        "",
        "| Implementation | Mean (ms) | P50 (ms) | P90 (ms) | "
        "P99 (ms) | Std (ms) | Peak allocated (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['implementation']} | {float(row['mean_ms']):.3f} | "
            f"{float(row['p50_ms']):.3f} | "
            f"{float(row['p90_ms']):.3f} | "
            f"{float(row['p99_ms']):.3f} | "
            f"{float(row['std_ms']):.3f} | "
            f"{float(row['peak_allocated_gib']):.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "PHYAI speedup vs Official: "
                f"**{summary['phyai_speedup_vs_official']:.3f}x**"
            ),
            "",
            (
                "Official speedup vs PHYAI: "
                f"**{summary['official_speedup_vs_phyai']:.3f}x**"
            ),
            "",
            (
                "PHYAI latency change vs Official: "
                f"**{phyai_latency_change_text(summary)}**"
            ),
            "",
            comparison_contract_text(official),
            "",
            "Official label means the official LingBot model code and native "
            "Robby MoE, with only the hard-coded FlashAttention2 selector "
            "changed to the repository's existing eager-attention config so "
            "that it can run on the Thor software stack.",
        ]
    )
    markdown_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    args = parse_args()
    official = read_report(args.official)
    phyai = read_report(args.phyai)
    summary = validate_pair(official, phyai)
    write_outputs(
        official,
        phyai,
        summary,
        args.csv,
        args.markdown,
    )
    print(f"Official mean : {summary['official_mean_ms']:.3f} ms")
    print(f"PHYAI mean    : {summary['phyai_mean_ms']:.3f} ms")
    print("PHYAI speedup : " f"{summary['phyai_speedup_vs_official']:.3f}x")
    print("Official speedup: " f"{summary['official_speedup_vs_phyai']:.3f}x")
    print("PHYAI change   : " f"{phyai_latency_change_text(summary)}")
    print(f"CSV           : {args.csv}")
    print(f"Markdown      : {args.markdown}")


if __name__ == "__main__":
    main()
