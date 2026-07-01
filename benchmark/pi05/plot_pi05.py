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

# Consistent stage colors across every figure.
COLOR = {
    "vision": "#4C78A8",
    "llm": "#F58518",
    "expert": "#E45756",
    "throughput": "#54A24B",
    "alloc": "#4C78A8",
    "reserved": "#B279A2",
}


class Profile:
    """Thin typed view over the profile JSON + derived per-batch arrays."""

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
        return np.array([r["stage_gpu_ms"][scope] for r in self.sweep])

    def field(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.sweep])

    def hw_caption(self) -> str:
        """Short device + measured-roofline caption for figure titles."""
        if self.peak and self.bw:
            return (
                f"{self.gpu} · {self.peak:g} TFLOPS BF16 / {self.bw:g} TB/s (measured)"
            )
        return self.gpu


def _xticks(ax, p: Profile):
    ax.set_xticks(p.x)
    ax.set_xticklabels([f"bs={b}" for b in p.bs])


def _save(fig, out_dir: Path, name: str):
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# --------------------------------------------------------------------------- #
# Figure 1 — stacked per-step latency by stage                                #
# --------------------------------------------------------------------------- #
def fig1_stage_latency(p: Profile, out_dir: Path):
    vis = p.stage("pi05.vision_loop")
    llm = p.stage("pi05.llm_prefix_fwd")
    exp = p.stage("pi05.expert_loop")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(p.x, vis, 0.6, label="vision (SigLIP)", color=COLOR["vision"])
    ax.bar(p.x, llm, 0.6, bottom=vis, label="LLM prefix", color=COLOR["llm"])
    ax.bar(
        p.x,
        exp,
        0.6,
        bottom=vis + llm,
        label="expert (10-step Euler)",
        color=COLOR["expert"],
    )
    tot = vis + llm + exp
    for i, t in enumerate(tot):
        ax.text(
            i, t + tot.max() * 0.01, f"{t:.0f}", ha="center", va="bottom", fontsize=9
        )
    _xticks(ax, p)
    ax.set_ylabel("GPU time / step (ms)")
    ax.set_title(
        f"Per-step latency by stage (bf16 + CUDA graph)\n{p.hw_caption()}",
        fontsize=12,
        weight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig1_stage_latency.svg")


# --------------------------------------------------------------------------- #
# Figure 2 — roofline with the three modules across batch                     #
# --------------------------------------------------------------------------- #
def fig2_roofline(p: Profile, out_dir: Path):
    if not (p.peak and p.bw):
        print("skip fig2_roofline: no measured peak/bandwidth in JSON")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ai_axis = np.logspace(0, 4, 200)
    roof = np.minimum(p.bw * ai_axis, p.peak)
    ax.plot(ai_axis, roof, "k-", lw=2, label="roofline")
    ax.axvline(p.ridge, color="gray", ls="--", lw=1)
    ax.text(
        p.ridge * 1.1, p.peak * 0.012, f"ridge {p.ridge:g}", color="gray", fontsize=9
    )

    # Vision / LLM: compute-bound, ~batch-flat — plot the bs=1 point.
    r0 = p.sweep[0]
    ai0, ach0 = r0["arithmetic_intensity"], r0["achieved_tflops"]
    ax.scatter(
        [ai0["vision"]],
        [ach0["vision"]],
        s=90,
        color=COLOR["vision"],
        zorder=4,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.annotate(
        "vision",
        (ai0["vision"], ach0["vision"]),
        textcoords="offset points",
        xytext=(0, 13),
        ha="center",
        color=COLOR["vision"],
        fontsize=9,
        weight="bold",
    )
    ax.scatter(
        [ai0["llm_prefix"]],
        [ach0["llm_prefix"]],
        s=90,
        color=COLOR["llm"],
        zorder=4,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.annotate(
        "LLM prefix",
        (ai0["llm_prefix"], ach0["llm_prefix"]),
        textcoords="offset points",
        xytext=(0, 13),
        ha="center",
        color=COLOR["llm"],
        fontsize=9,
        weight="bold",
    )

    # Expert: sweep batch — the point marches right (AI↑) and up (TFLOPS↑).
    exp_ai = [r["arithmetic_intensity"]["expert"] for r in p.sweep]
    exp_ach = [r["achieved_tflops"]["expert"] for r in p.sweep]
    ax.plot(
        exp_ai,
        exp_ach,
        "-o",
        color=COLOR["expert"],
        lw=1.8,
        zorder=3,
        label="expert (bs sweep)",
    )
    # Label below-right for the low-batch points; below-left for the last two
    # so they don't crash into the vision point sitting in the same band.
    for i, (r, x, y) in enumerate(zip(p.sweep, exp_ai, exp_ach)):
        below_left = i >= len(p.sweep) - 2
        ax.annotate(
            f"bs={r['bs']}",
            (x, y),
            textcoords="offset points",
            xytext=(-30 if below_left else 6, -12),
            color=COLOR["expert"],
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e4)
    ax.set_ylim(max(1.0, min(exp_ach) * 0.4), p.peak * 1.4)
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achieved TFLOPS")
    ax.set_title(
        f"Roofline placement of the three modules\n{p.hw_caption()}",
        fontsize=12,
        weight="bold",
    )
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out_dir, "fig2_roofline.svg")


# --------------------------------------------------------------------------- #
# Figure 3 — per-sample stage time (the mechanism plot)                       #
# --------------------------------------------------------------------------- #
def fig3_per_sample_stage(p: Profile, out_dir: Path):
    bs = np.array(p.bs, dtype=float)
    vis = p.stage("pi05.vision_loop") / bs
    llm = p.stage("pi05.llm_prefix_fwd") / bs
    exp = p.stage("pi05.expert_loop") / bs
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for y, key, lab in (
        (vis, "vision", "vision (SigLIP)"),
        (llm, "llm", "LLM prefix"),
        (exp, "expert", "expert (10-step Euler)"),
    ):
        ax.plot(p.x, y, "-o", color=COLOR[key], lw=2.2, label=lab)
    # Annotate the expert collapse (the whole story).
    if exp[0] > 0 and exp[-1] > 0:
        ax.annotate(
            f"{exp[0] / exp[-1]:.1f}× drop",
            (p.x[-1], exp[-1]),
            textcoords="offset points",
            xytext=(-10, 14),
            color=COLOR["expert"],
            fontsize=10,
            weight="bold",
        )
    _xticks(ax, p)
    ax.set_ylabel("per-sample GPU time (ms)")
    ax.set_title(
        f"Per-sample stage time vs batch — vision/LLM flat, expert collapses"
        f"\n{p.hw_caption()}",
        fontsize=12,
        weight="bold",
    )
    ax.legend(frameon=False)
    _save(fig, out_dir, "fig3_per_sample_stage.svg")


# --------------------------------------------------------------------------- #
# Figure 4 — per-sample latency + throughput (dual axis)                      #
# --------------------------------------------------------------------------- #
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
            va="bottom",
            fontsize=9,
            color=COLOR["expert"],
        )
    ax.set_ylabel("per-sample latency (ms)", color=COLOR["expert"])
    ax.tick_params(axis="y", labelcolor=COLOR["expert"])
    _xticks(ax, p)
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.plot(p.x, tp, "-o", color=COLOR["throughput"], lw=2.4, label="throughput")
    for i, v in enumerate(tp):
        ax2.text(
            i,
            v + tp.max() * 0.02,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR["throughput"],
        )
    ax2.set_ylabel("throughput (samples/s)", color=COLOR["throughput"])
    ax2.tick_params(axis="y", labelcolor=COLOR["throughput"])
    ax.set_title(
        f"Per-sample latency ↓ and throughput ↑ vs batch\n{p.hw_caption()}",
        fontsize=12,
        weight="bold",
    )
    _save(fig, out_dir, "fig4_latency_throughput.svg")


# --------------------------------------------------------------------------- #
# Figure 5 — expert MFU climb                                                 #
# --------------------------------------------------------------------------- #
def fig5_expert_mfu(p: Profile, out_dir: Path):
    mfu = [r.get("expert_mfu_pct") for r in p.sweep]
    if any(m is None for m in mfu):
        print("skip fig5_expert_mfu: no MFU (peak missing) in JSON")
        return
    mfu = np.array(mfu, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(p.x, mfu, "-o", color=COLOR["expert"], lw=2.4)
    for i, v in enumerate(mfu):
        ax.text(
            i, v + mfu.max() * 0.02, f"{v:.1f}%", ha="center", va="bottom", fontsize=9
        )
    _xticks(ax, p)
    ax.set_ylabel(f"expert MFU (% of {p.peak:g} TFLOPS)")
    ax.set_ylim(0, mfu.max() * 1.18)
    ax.set_title(
        f"Action-expert MFU climbs with batch (launch-bound → compute-bound)"
        f"\n{p.hw_caption()}",
        fontsize=12,
        weight="bold",
    )
    _save(fig, out_dir, "fig5_expert_mfu.svg")


# --------------------------------------------------------------------------- #
# Figure 6 — 100%-stacked stage share                                         #
# --------------------------------------------------------------------------- #
def fig6_stage_share(p: Profile, out_dir: Path):
    vis = p.stage("pi05.vision_loop")
    llm = p.stage("pi05.llm_prefix_fwd")
    exp = p.stage("pi05.expert_loop")
    other = (p.field("stage_sum_ms") - vis - llm - exp).clip(min=0)
    tot = vis + llm + exp + other
    tot[tot == 0] = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    parts = [
        (vis, "vision", "vision"),
        (llm, "llm", "LLM prefix"),
        (exp, "expert", "expert"),
        (other, None, "plan / other"),
    ]
    bottom = np.zeros(len(p.bs))
    for y, key, lab in parts:
        frac = 100 * y / tot
        ax.bar(
            p.x, frac, 0.6, bottom=bottom, label=lab, color=COLOR.get(key, "#BAB0AC")
        )
        bottom += frac
    _xticks(ax, p)
    ax.set_ylabel("share of stage GPU time (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Where the step goes: stage share vs batch\n{p.hw_caption()}",
        fontsize=12,
        weight="bold",
    )
    ax.legend(frameon=False, loc="lower center", ncol=4, fontsize=9)
    _save(fig, out_dir, "fig6_stage_share.svg")


# --------------------------------------------------------------------------- #
# Figure 7 — peak memory                                                      #
# --------------------------------------------------------------------------- #
def fig7_memory(p: Profile, out_dir: Path):
    alloc = p.field("mem_alloc_mib") / 1024.0
    resv = p.field("mem_reserved_mib") / 1024.0
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    w = 0.38
    ax.bar(p.x - w / 2, alloc, w, label="allocated", color=COLOR["alloc"])
    ax.bar(p.x + w / 2, resv, w, label="reserved", color=COLOR["reserved"])
    total = p.hw.get("total_mem_gb")
    if total:
        ax.axhline(total, color="gray", ls="--", lw=1)
        ax.text(
            0, total * 0.97, f"{total:g} GB total", color="gray", fontsize=9, va="top"
        )
    _xticks(ax, p)
    ax.set_ylabel("peak CUDA memory (GB)")
    ax.set_title(
        f"Peak memory vs batch (resident weights + KV + workspace)\n{p.hw_caption()}",
        fontsize=12,
        weight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig7_memory.svg")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        required=True,
        help="profile JSON from profile_pi05.py",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark/pi05/figures"))
    args = ap.parse_args()

    p = Profile(json.loads(args.in_path.read_text()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device: {p.gpu}  | peak {p.peak} TFLOPS  bw {p.bw} TB/s  "
        f"| {len(p.bs)} batch points"
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
