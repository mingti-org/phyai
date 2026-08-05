from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
    }
)

COLOR = {
    "vision": "#4C78A8",
    "llm": "#F58518",
    "expert": "#E45756",
    "throughput": "#54A24B",
    "alloc": "#4C78A8",
    "reserved": "#B279A2",
}


class Profile:
    """Thin typed view over the pi0 profile JSON."""

    def __init__(self, data: dict):
        self.meta = data["meta"]
        self.hw = data["hardware"]
        self.flop = data["stages_flop"]["flop_per_sample"]
        self.sweep = sorted(data["sweep"], key=lambda r: r["bs"])
        self.bs = [r["bs"] for r in self.sweep]
        self.x = np.arange(len(self.bs))

    @property
    def gpu(self) -> str:
        return self.hw.get("name", "GPU")

    @property
    def peak(self) -> float | None:
        return self.hw.get("peak_bf16_tflops")

    @property
    def bw(self) -> float | None:
        return self.hw.get("hbm_tb_s")

    @property
    def ridge(self) -> float | None:
        return self.hw.get("ridge_point_flop_per_byte")

    def stage(self, scope: str) -> np.ndarray:
        return np.array(
            [r["stage_gpu_ms"].get(scope, 0.0) for r in self.sweep], dtype=float
        )

    def field(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.sweep], dtype=float)

    def expert_metric(self, section: str) -> np.ndarray:
        return np.array(
            [
                r[section].get("expert_loop", r[section].get("expert", 0.0))
                for r in self.sweep
            ],
            dtype=float,
        )

    def expert_total_stage(self) -> np.ndarray:
        return self.stage("pi0.expert_state_prefill") + self.stage("pi0.expert_loop")

    def hw_caption(self) -> str:
        base = self.gpu
        if self.peak and self.bw:
            base += f" · {self.peak:g} TFLOPS BF16 / {self.bw:g} TB/s (measured)"
        return (
            f"{base}\n{self.meta.get('num_images')} cameras, "
            f"prefix {self.meta.get('n_per_sample')} tokens, "
            f"chunk {self.meta.get('chunk_size')}, "
            f"N={self.meta.get('num_inference_steps')}, "
            f"vision {self.meta.get('vision_dtype')}"
        )


def _xticks(ax, p: Profile):
    ax.set_xticks(p.x)
    ax.set_xticklabels([f"bs={b}" for b in p.bs])


def _save(fig, out_dir: Path, name: str):
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig1_stage_latency(p: Profile, out_dir: Path):
    vis = p.stage("pi0.vision_loop")
    llm = p.stage("pi0.llm_prefix_fwd")
    exp = p.expert_total_stage()
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(p.x, vis, 0.6, label="vision (SigLIP)", color=COLOR["vision"])
    ax.bar(p.x, llm, 0.6, bottom=vis, label="LLM prefix", color=COLOR["llm"])
    ax.bar(
        p.x,
        exp,
        0.6,
        bottom=vis + llm,
        label=f"expert prefill + {p.meta.get('num_inference_steps')}-step Euler",
        color=COLOR["expert"],
    )
    total = vis + llm + exp
    for i, t in enumerate(total):
        ax.text(i, t + total.max() * 0.01, f"{t:.0f}", ha="center", fontsize=9)
    _xticks(ax, p)
    ax.set_ylabel("GPU time / step (ms)")
    ax.set_title(
        f"pi0 per-step latency by stage\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig1_stage_latency.svg")


def fig2_roofline(p: Profile, out_dir: Path):
    if not (p.peak and p.bw):
        print("skip fig2_roofline: no measured peak/bandwidth in JSON")
        return
    r0 = p.sweep[0]
    ai0 = r0["arithmetic_intensity"]
    ach0 = r0["achieved_tflops"]
    exp_ai = p.expert_metric("arithmetic_intensity").tolist()
    exp_ach = p.expert_metric("achieved_tflops").tolist()
    x_hi = 10 ** np.ceil(np.log10(max(exp_ai + [ai0["vision"], ai0["llm_prefix"]]) * 3))
    ai_axis = np.logspace(0, np.log10(x_hi), 200)
    roof = np.minimum(p.bw * ai_axis, p.peak)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(ai_axis, roof, "k-", lw=2, label="roofline")
    ax.axvline(p.ridge, color="gray", ls="--", lw=1)
    ax.text(
        p.ridge * 1.1, p.peak * 0.012, f"ridge {p.ridge:g}", color="gray", fontsize=9
    )
    for key, label, color in (
        ("vision", "vision", COLOR["vision"]),
        ("llm_prefix", "LLM prefix", COLOR["llm"]),
    ):
        ax.scatter(
            [ai0[key]],
            [ach0[key]],
            s=90,
            color=color,
            zorder=4,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.annotate(
            label,
            (ai0[key], ach0[key]),
            textcoords="offset points",
            xytext=(0, 13),
            ha="center",
            color=color,
            fontsize=9,
            weight="bold",
        )
    ax.plot(
        exp_ai,
        exp_ach,
        "-o",
        color=COLOR["expert"],
        lw=1.8,
        zorder=3,
        label="expert loop (bs sweep)",
    )
    for i, (r, x, y) in enumerate(zip(p.sweep, exp_ai, exp_ach)):
        last = i == len(p.sweep) - 1
        ax.annotate(
            f"bs={r['bs']}",
            (x, y),
            textcoords="offset points",
            xytext=(-10 if last else 6, -13),
            ha="right" if last else "left",
            color=COLOR["expert"],
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, x_hi)
    ax.set_ylim(
        max(1.0, min(exp_ach + [ach0["vision"], ach0["llm_prefix"]]) * 0.4),
        p.peak * 1.4,
    )
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achieved TFLOPS")
    ax.set_title(
        f"pi0 roofline placement\n{p.hw_caption()}", fontsize=11.5, weight="bold"
    )
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out_dir, "fig2_roofline.svg")


def fig3_per_sample_stage(p: Profile, out_dir: Path):
    bs = np.array(p.bs, dtype=float)
    vis = p.stage("pi0.vision_loop") / bs
    llm = p.stage("pi0.llm_prefix_fwd") / bs
    exp = p.expert_total_stage() / bs
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for y, key, label in (
        (vis, "vision", "vision"),
        (llm, "llm", "LLM prefix"),
        (exp, "expert", "expert total"),
    ):
        ax.plot(p.x, y, "-o", color=COLOR[key], lw=2.2, label=label)
    if exp[0] > 0 and exp[-1] > 0:
        ax.annotate(
            f"{exp[0] / exp[-1]:.1f}x drop",
            (p.x[-1], exp[-1]),
            textcoords="offset points",
            xytext=(-10, 14),
            color=COLOR["expert"],
            fontsize=10,
            weight="bold",
            ha="right",
        )
    _xticks(ax, p)
    ax.set_ylabel("per-sample GPU time (ms)")
    ax.set_title(
        f"pi0 per-sample stage time vs batch\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    ax.legend(frameon=False)
    _save(fig, out_dir, "fig3_per_sample_stage.svg")


def fig4_latency_throughput(p: Profile, out_dir: Path):
    ps = p.field("per_sample_ms")
    tp = p.field("throughput_sps")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(p.x, ps, 0.5, color=COLOR["expert"], alpha=0.55, label="per-sample latency")
    for i, v in enumerate(ps):
        ax.text(
            i,
            v + ps.max() * 0.01,
            f"{v:.1f}",
            ha="center",
            fontsize=9,
            color=COLOR["expert"],
        )
    ax.set_ylabel("per-sample latency (ms)", color=COLOR["expert"])
    ax.tick_params(axis="y", labelcolor=COLOR["expert"])
    _xticks(ax, p)
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.plot(p.x, tp, "-o", color=COLOR["throughput"], lw=2.4)
    for i, v in enumerate(tp):
        ax2.text(
            i,
            v + tp.max() * 0.02,
            f"{v:.0f}",
            ha="center",
            fontsize=9,
            color=COLOR["throughput"],
        )
    ax2.set_ylabel("action chunks/s", color=COLOR["throughput"])
    ax2.tick_params(axis="y", labelcolor=COLOR["throughput"])
    ax.set_title(
        f"pi0 latency and throughput vs batch\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    _save(fig, out_dir, "fig4_latency_throughput.svg")


def fig5_expert_mfu(p: Profile, out_dir: Path):
    mfu = [r.get("expert_loop_mfu_pct", r.get("expert_mfu_pct")) for r in p.sweep]
    if any(value is None for value in mfu):
        print("skip fig5_expert_mfu: no MFU in JSON")
        return
    y = np.array(mfu, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(p.x, y, "-o", color=COLOR["expert"], lw=2.4)
    for i, v in enumerate(y):
        ax.text(i, v + y.max() * 0.02, f"{v:.1f}%", ha="center", fontsize=9)
    _xticks(ax, p)
    ax.set_ylabel(f"expert loop MFU (% of {p.peak:g} TFLOPS)")
    ax.set_ylim(0, y.max() * 1.18)
    ax.set_title(
        f"pi0 expert loop MFU vs batch\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    _save(fig, out_dir, "fig5_expert_mfu.svg")


def fig6_stage_share(p: Profile, out_dir: Path):
    vis = p.stage("pi0.vision_loop")
    llm = p.stage("pi0.llm_prefix_fwd")
    exp = p.expert_total_stage()
    other = (p.field("stage_sum_ms") - vis - llm - exp).clip(min=0)
    total = vis + llm + exp + other
    total[total == 0] = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bottom = np.zeros(len(p.bs))
    for y, key, label in (
        (vis, "vision", "vision"),
        (llm, "llm", "LLM prefix"),
        (exp, "expert", "expert total"),
        (other, None, "plan / other"),
    ):
        frac = 100 * y / total
        ax.bar(
            p.x, frac, 0.6, bottom=bottom, label=label, color=COLOR.get(key, "#BAB0AC")
        )
        bottom += frac
    _xticks(ax, p)
    ax.set_ylabel("share of stage GPU time (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"pi0 stage share vs batch\n{p.hw_caption()}", fontsize=11.5, weight="bold"
    )
    ax.legend(frameon=False, loc="lower center", ncol=4, fontsize=9)
    _save(fig, out_dir, "fig6_stage_share.svg")


def fig7_memory(p: Profile, out_dir: Path):
    alloc = p.field("mem_alloc_mib") / 1024.0
    resv = p.field("mem_reserved_mib") / 1024.0
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    width = 0.38
    ax.bar(p.x - width / 2, alloc, width, label="allocated", color=COLOR["alloc"])
    ax.bar(p.x + width / 2, resv, width, label="reserved", color=COLOR["reserved"])
    total = p.hw.get("total_mem_gb")
    if total:
        ax.axhline(total, color="gray", ls="--", lw=1)
        ax.text(
            0, total * 0.97, f"{total:g} GB total", color="gray", fontsize=9, va="top"
        )
    _xticks(ax, p)
    ax.set_ylabel("peak CUDA memory (GB)")
    ax.set_title(
        f"pi0 peak memory vs batch\n{p.hw_caption()}", fontsize=11.5, weight="bold"
    )
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig7_memory.svg")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark/pi0/figures"))
    args = ap.parse_args()

    p = Profile(json.loads(args.in_path.read_text()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device: {p.gpu} | peak {p.peak} TFLOPS | bw {p.bw} TB/s | "
        f"{len(p.bs)} batch points"
    )
    fig1_stage_latency(p, args.out_dir)
    fig2_roofline(p, args.out_dir)
    fig3_per_sample_stage(p, args.out_dir)
    fig4_latency_throughput(p, args.out_dir)
    fig5_expert_mfu(p, args.out_dir)
    fig6_stage_share(p, args.out_dir)
    fig7_memory(p, args.out_dir)


if __name__ == "__main__":
    main()
