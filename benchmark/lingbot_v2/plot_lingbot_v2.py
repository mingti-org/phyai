from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STAGES = (
    "lingbot_v2.vision",
    "lingbot_v2.prefix_pack",
    "lingbot_v2.prefix_plan",
    "lingbot_v2.prefix_forward",
    "lingbot_v2.expert_plan",
    "lingbot_v2.euler",
)

STAGE_LABEL = {
    "lingbot_v2.vision": "Vision",
    "lingbot_v2.prefix_pack": "Prefix pack",
    "lingbot_v2.prefix_plan": "Prefix plan",
    "lingbot_v2.prefix_forward": "Text prefix",
    "lingbot_v2.expert_plan": "Expert plan",
    "lingbot_v2.euler": "10-step expert",
}

STAGE_COLOR = {
    "lingbot_v2.vision": "#4C78A8",
    "lingbot_v2.prefix_pack": "#72B7B2",
    "lingbot_v2.prefix_plan": "#B9A0CF",
    "lingbot_v2.prefix_forward": "#F58518",
    "lingbot_v2.expert_plan": "#FFBF79",
    "lingbot_v2.euler": "#E45756",
}

DETAILS = (
    "lingbot_v2.detail.vision_patch_embed",
    "lingbot_v2.detail.vision_blocks",
    "lingbot_v2.detail.vision_mergers",
    "lingbot_v2.detail.text_attention",
    "lingbot_v2.detail.text_mlp",
    "lingbot_v2.detail.expert_attention",
    "lingbot_v2.detail.expert_moe",
    "lingbot_v2.detail.expert_adanorm",
    "lingbot_v2.detail.expert_heads",
)

DETAIL_LABEL = (
    "PatchEmbed",
    "Vision blocks",
    "Vision mergers",
    "Text attention",
    "Text MLP",
    "Expert attention",
    "Expert MoE",
    "Expert AdaNorm",
    "Action heads",
)

plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def profile_label(profile: dict) -> str:
    gpu = profile["hardware"]["name"]
    dtype = profile["meta"]["vision_dtype"].replace("torch.", "")
    return f"{gpu}\nViT {dtype}"


def load_profiles(paths: list[Path]) -> list[dict]:
    profiles = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    hashes = {profile["meta"]["input_sha256"] for profile in profiles}
    if len(hashes) != 1:
        raise ValueError(
            "refusing to compare profiles generated from different input artifacts: "
            f"{sorted(hashes)}"
        )
    contracts = {
        json.dumps(profile["meta"]["contract"], sort_keys=True) for profile in profiles
    }
    if len(contracts) != 1:
        raise ValueError("refusing to compare profiles with different model contracts.")
    return profiles


def component_profile_is_valid(profile: dict) -> bool:
    diagnostics = profile.get("profile_diagnostics")
    if diagnostics is not None:
        return bool(diagnostics.get("valid", False))
    baseline = float(profile["latency_ms"]["mean"])
    profiled = sum(float(value) for value in profile["stage_gpu_ms"].values())
    return baseline > 0 and abs(profiled - baseline) / baseline <= 0.15


def detail_profile_is_valid(profile: dict) -> bool:
    diagnostics = profile.get("detail_profile_diagnostics")
    if diagnostics is None:
        return component_profile_is_valid(profile)
    return bool(diagnostics.get("valid", False))


def save_figure(figure: plt.Figure, out_dir: Path, name: str) -> None:
    figure.tight_layout()
    path = out_dir / name
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    print("wrote", path)


def plot_e2e(profiles: list[dict], out_dir: Path) -> None:
    labels = [profile_label(profile) for profile in profiles]
    means = [profile["latency_ms"]["mean"] for profile in profiles]
    p99 = [profile["latency_ms"]["p99"] for profile in profiles]
    errors = [max(0.0, tail - mean) for tail, mean in zip(p99, means)]
    x = np.arange(len(profiles))
    figure, axis = plt.subplots(figsize=(max(6.4, 2.0 * len(profiles)), 4.4))
    bars = axis.bar(
        x,
        means,
        yerr=errors,
        capsize=5,
        color="#4C78A8",
        width=0.62,
        label="Mean with P99 upper error",
    )
    axis.bar_label(bars, fmt="%.1f ms", padding=4)
    axis.set_xticks(x, labels)
    axis.set_ylabel("engine.step latency (ms)")
    axis.set_title("LingBot V2 fixed-input latency")
    axis.legend(frameon=False)
    save_figure(figure, out_dir, "fig1_e2e_latency.svg")


def plot_stage_latency(profiles: list[dict], out_dir: Path) -> None:
    labels = [profile_label(profile) for profile in profiles]
    x = np.arange(len(profiles))
    bottom = np.zeros(len(profiles))
    figure, axis = plt.subplots(figsize=(max(7.2, 2.1 * len(profiles)), 4.8))
    for stage in STAGES:
        values = np.array([profile["stage_gpu_ms"][stage] for profile in profiles])
        axis.bar(
            x,
            values,
            bottom=bottom,
            width=0.64,
            label=STAGE_LABEL[stage],
            color=STAGE_COLOR[stage],
        )
        bottom += values
    axis.set_xticks(x, labels)
    axis.set_ylabel("GPU timeline latency (ms)")
    axis.set_title("LingBot V2 latency by scheduler component")
    axis.legend(frameon=False, ncols=2)
    save_figure(figure, out_dir, "fig2_stage_latency.svg")


def plot_stage_share(profiles: list[dict], out_dir: Path) -> None:
    labels = [profile_label(profile) for profile in profiles]
    x = np.arange(len(profiles))
    bottom = np.zeros(len(profiles))
    figure, axis = plt.subplots(figsize=(max(7.2, 2.1 * len(profiles)), 4.8))
    for stage in STAGES:
        values = np.array(
            [profile["stage_latency_share_pct"][stage] for profile in profiles]
        )
        axis.bar(
            x,
            values,
            bottom=bottom,
            width=0.64,
            label=STAGE_LABEL[stage],
            color=STAGE_COLOR[stage],
        )
        bottom += values
    other = np.maximum(0.0, 100.0 - bottom)
    axis.bar(
        x,
        other,
        bottom=bottom,
        width=0.64,
        label="Unattributed",
        color="#BAB0AC",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Share of profiled scheduler stages (%)")
    axis.set_ylim(0, max(100.0, float((bottom + other).max()) * 1.05))
    axis.set_title("LingBot V2 component latency share")
    axis.legend(frameon=False, ncols=2)
    save_figure(figure, out_dir, "fig3_stage_share.svg")


def plot_mfu(profiles: list[dict], out_dir: Path) -> None:
    keys = (
        "lingbot_v2.vision",
        "lingbot_v2.prefix_forward",
        "lingbot_v2.euler",
        "e2e",
    )
    key_labels = ("Vision", "Text prefix", "10-step expert", "E2E")
    if not any(
        profile["mfu_pct_bf16_peak"].get("e2e") is not None for profile in profiles
    ):
        print("skip fig4_mfu.svg: profiles do not contain a measured BF16 peak")
        return
    x = np.arange(len(keys))
    width = 0.8 / len(profiles)
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for index, profile in enumerate(profiles):
        values = [profile["mfu_pct_bf16_peak"].get(key) or 0.0 for key in keys]
        axis.bar(
            x - 0.4 + width / 2 + index * width,
            values,
            width=width,
            label=profile_label(profile).replace("\n", " / "),
        )
    axis.set_xticks(x, key_labels)
    axis.set_ylabel("MFU (% of measured BF16 peak)")
    axis.set_title("LingBot V2 component MFU")
    axis.legend(frameon=False)
    save_figure(figure, out_dir, "fig4_component_mfu.svg")


def plot_detail_latency(profiles: list[dict], out_dir: Path) -> None:
    x = np.arange(len(DETAILS))
    width = 0.8 / len(profiles)
    figure, axis = plt.subplots(figsize=(11.2, 5.0))
    for index, profile in enumerate(profiles):
        values = [profile["detail_gpu_ms"][key] for key in DETAILS]
        axis.bar(
            x - 0.4 + width / 2 + index * width,
            values,
            width=width,
            label=profile_label(profile).replace("\n", " / "),
        )
    axis.set_xticks(x, DETAIL_LABEL, rotation=24, ha="right")
    axis.set_ylabel("GPU timeline latency (ms)")
    axis.set_title("LingBot V2 detailed component latency")
    axis.legend(frameon=False)
    save_figure(figure, out_dir, "fig5_detail_latency.svg")


def plot_detail_mfu(profiles: list[dict], out_dir: Path) -> None:
    if not any(
        profile["mfu_pct_bf16_peak"].get("e2e") is not None for profile in profiles
    ):
        print("skip fig6_detail_mfu.svg: profiles do not contain a measured BF16 peak")
        return
    x = np.arange(len(DETAILS))
    width = 0.8 / len(profiles)
    figure, axis = plt.subplots(figsize=(11.2, 5.0))
    for index, profile in enumerate(profiles):
        values = [profile["mfu_pct_bf16_peak"].get(key) or 0.0 for key in DETAILS]
        axis.bar(
            x - 0.4 + width / 2 + index * width,
            values,
            width=width,
            label=profile_label(profile).replace("\n", " / "),
        )
    axis.set_xticks(x, DETAIL_LABEL, rotation=24, ha="right")
    axis.set_ylabel("MFU (% of measured BF16 peak)")
    axis.set_title("LingBot V2 detailed component MFU")
    axis.legend(frameon=False)
    save_figure(figure, out_dir, "fig6_detail_mfu.svg")


def write_summary(profiles: list[dict], out_dir: Path) -> None:
    path = out_dir / "summary.csv"
    fields = [
        "gpu",
        "vision_dtype",
        "torch",
        "cuda",
        "flashinfer",
        "input_sha256",
        "mean_ms",
        "p50_ms",
        "p99_ms",
        "std_ms",
        "component_profile_valid",
        "profile_overhead_ratio",
        "vision_ms",
        "text_prefix_ms",
        "expert_10step_ms",
        "expert_1step_ms",
        "peak_allocated_gib",
        "e2e_mfu_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            component_valid = component_profile_is_valid(profile)
            diagnostics = profile.get("profile_diagnostics", {})
            writer.writerow(
                {
                    "gpu": profile["hardware"]["name"],
                    "vision_dtype": profile["meta"]["vision_dtype"],
                    "torch": profile["software"]["torch"],
                    "cuda": profile["software"]["cuda"],
                    "flashinfer": profile["software"]["flashinfer"],
                    "input_sha256": profile["meta"]["input_sha256"],
                    "mean_ms": profile["latency_ms"]["mean"],
                    "p50_ms": profile["latency_ms"]["p50"],
                    "p99_ms": profile["latency_ms"]["p99"],
                    "std_ms": profile["latency_ms"]["std"],
                    "component_profile_valid": component_valid,
                    "profile_overhead_ratio": diagnostics.get("overhead_ratio"),
                    "vision_ms": (
                        profile["stage_gpu_ms"]["lingbot_v2.vision"]
                        if component_valid
                        else None
                    ),
                    "text_prefix_ms": (
                        profile["stage_gpu_ms"]["lingbot_v2.prefix_forward"]
                        if component_valid
                        else None
                    ),
                    "expert_10step_ms": (
                        profile["stage_gpu_ms"]["lingbot_v2.euler"]
                        if component_valid
                        else None
                    ),
                    "expert_1step_ms": (
                        profile["euler_step_mean_ms"] if component_valid else None
                    ),
                    "peak_allocated_gib": profile["checks"]["allocated_peak_bytes"]
                    / 2**30,
                    "e2e_mfu_pct": profile["mfu_pct_bf16_peak"].get("e2e"),
                }
            )
    print("wrote", path)


def main() -> None:
    args = parse_args()
    profiles = load_profiles(args.inputs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_e2e(profiles, args.out_dir)
    component_profiles = [
        profile for profile in profiles if component_profile_is_valid(profile)
    ]
    detail_profiles = [
        profile for profile in component_profiles if detail_profile_is_valid(profile)
    ]
    invalid_profiles = [
        profile_label(profile).replace("\n", " / ")
        for profile in profiles
        if not component_profile_is_valid(profile)
    ]
    if invalid_profiles:
        print(
            "skip invalid component profiles:",
            ", ".join(invalid_profiles),
        )
    if component_profiles:
        plot_stage_latency(component_profiles, args.out_dir)
        plot_stage_share(component_profiles, args.out_dir)
        plot_mfu(component_profiles, args.out_dir)
    else:
        print("skip component figures: no profile passed the perturbation check")
    if detail_profiles:
        plot_detail_latency(detail_profiles, args.out_dir)
        plot_detail_mfu(detail_profiles, args.out_dir)
    else:
        print("skip detailed figures: no detail profile passed the perturbation check")
    write_summary(profiles, args.out_dir)


if __name__ == "__main__":
    main()
