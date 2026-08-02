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

# Consistent phase colors across every figure.
COLOR = {
    "vae": "#4C78A8",
    "cond": "#F58518",
    "gen": "#E45756",
    "throughput": "#54A24B",
    "alloc": "#4C78A8",
    "reserved": "#B279A2",
}


class Profile:
    """Thin typed view over the profile JSON + derived per-batch arrays."""

    def __init__(self, data: dict):
        self.meta = data["meta"]
        self.hw = data["hardware"]
        self.pf = data["phases_flop"]
        self.sweep = sorted(data["sweep"], key=lambda r: r["bs"])
        self.bs = [r["bs"] for r in self.sweep]
        self.x = np.arange(len(self.bs))
        self.branches = self.meta.get("cfg_branches", 1)
        self.num_steps = self.meta.get("num_steps", 4)

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

    def field(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.sweep], dtype=float)

    def vae(self) -> np.ndarray:
        return np.array(
            [(r.get("vae_encode_ms") or 0.0) for r in self.sweep], dtype=float
        )

    def hw_caption(self) -> str:
        cfg = f"CFG×{int(self.branches)}" if self.branches > 1 else "no CFG"
        grid = "×".join(str(v) for v in self.meta.get("latent_grid", []))
        base = f"{self.gpu}"
        if self.peak and self.bw:
            base += f" · {self.peak:g} TFLOPS BF16 / {self.bw:g} TB/s (measured)"
        return (
            f"{base}\nN={self.num_steps} steps, latent {grid}, "
            f"chunk {self.meta.get('action_chunk')}, {cfg}, bf16"
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


# --------------------------------------------------------------------------- #
# Figure 1 — stacked e2e latency by phase                                     #
# --------------------------------------------------------------------------- #
def fig1_phase_latency(p: Profile, out_dir: Path):
    vae = p.vae()
    cond = p.field("cond_encode_total_ms")
    gen = p.field("gen_loop_ms")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(p.x, vae, 0.6, label="VAE encode (obs)", color=COLOR["vae"])
    ax.bar(
        p.x,
        cond,
        0.6,
        bottom=vae,
        label=f"cond encode (UND ×{int(p.branches)})",
        color=COLOR["cond"],
    )
    ax.bar(
        p.x,
        gen,
        0.6,
        bottom=vae + cond,
        label=f"GEN denoise loop (N={p.num_steps})",
        color=COLOR["gen"],
    )
    tot = vae + cond + gen
    for i, t in enumerate(tot):
        ax.text(
            i, t + tot.max() * 0.01, f"{t:.0f}", ha="center", va="bottom", fontsize=9
        )
    _xticks(ax, p)
    ax.set_ylabel("batch latency (ms)")
    ax.set_title(
        f"Per-batch latency by phase (no video decode)\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig1_phase_latency.svg")


# --------------------------------------------------------------------------- #
# Figure 2 — roofline: GEN per-step point swept across batch                  #
# --------------------------------------------------------------------------- #
def fig2_roofline(p: Profile, out_dir: Path):
    if not (p.peak and p.bw):
        print("skip fig2_roofline: no measured peak/bandwidth in JSON")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # GEN per-step: sweep batch — the point marches right (AI↑) and up (TFLOPS↑).
    gen_ai = [r["arithmetic_intensity"]["gen_step"] for r in p.sweep]
    gen_ach = [r["achieved_tflops"]["gen_step"] for r in p.sweep]
    r0 = p.sweep[0]
    cond_ai, cond_ach = (
        r0["arithmetic_intensity"]["cond_encode"],
        r0["achieved_tflops"]["cond_encode"],
    )
    # x-axis upper bound follows the data: AI scales with batch (weights amortized),
    # so the highest-batch GEN point can sit well past 1e4. Pad ~3x past the max.
    x_hi = 10 ** np.ceil(np.log10(max(gen_ai + [cond_ai]) * 3))
    ai_axis = np.logspace(0, np.log10(x_hi), 200)
    roof = np.minimum(p.bw * ai_axis, p.peak)
    ax.plot(ai_axis, roof, "k-", lw=2, label="roofline")
    ax.axvline(p.ridge, color="gray", ls="--", lw=1)
    ax.text(
        p.ridge * 1.1, p.peak * 0.012, f"ridge {p.ridge:g}", color="gray", fontsize=9
    )

    ax.plot(
        gen_ai,
        gen_ach,
        "-o",
        color=COLOR["gen"],
        lw=1.8,
        zorder=3,
        label="GEN per-step (bs sweep)",
    )
    for i, (r, x, y) in enumerate(zip(p.sweep, gen_ai, gen_ach)):
        # Label left of the last point so it doesn't run off the right edge.
        dx = -10 if i == len(p.sweep) - 1 else 6
        ha = "right" if i == len(p.sweep) - 1 else "left"
        ax.annotate(
            f"bs={r['bs']}",
            (x, y),
            textcoords="offset points",
            xytext=(dx, -13),
            color=COLOR["gen"],
            fontsize=8,
            ha=ha,
        )
    # cond encode at bs=1 (compute-light, small seq).
    ax.scatter(
        [cond_ai],
        [cond_ach],
        s=90,
        color=COLOR["cond"],
        zorder=4,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.annotate(
        "cond encode",
        (cond_ai, cond_ach),
        textcoords="offset points",
        xytext=(0, 13),
        ha="center",
        color=COLOR["cond"],
        fontsize=9,
        weight="bold",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, x_hi)
    ax.set_ylim(max(1.0, min(gen_ach + [cond_ach]) * 0.4), p.peak * 1.4)
    ax.set_xlabel("arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("achieved TFLOPS")
    ax.set_title(
        f"Roofline: GEN per-step vs batch (already compute-bound)\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out_dir, "fig2_roofline.svg")


# --------------------------------------------------------------------------- #
# Figure 3 — per-sample latency (the batching payoff)                         #
# --------------------------------------------------------------------------- #
def fig3_per_sample(p: Profile, out_dir: Path):
    ps = p.field("per_sample_ms")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(p.x, ps, "-o", color=COLOR["gen"], lw=2.4)
    for i, v in enumerate(ps):
        ax.text(
            i, v + ps.max() * 0.015, f"{v:.0f}", ha="center", va="bottom", fontsize=9
        )
    if ps[0] > 0 and ps[-1] > 0:
        ax.annotate(
            f"{ps[0] / ps[-1]:.2f}× per-sample drop\n(bs=1 → bs={p.bs[-1]})",
            (p.x[-1], ps[-1]),
            textcoords="offset points",
            xytext=(-16, 26),
            color=COLOR["gen"],
            fontsize=10,
            weight="bold",
            ha="right",
        )
    _xticks(ax, p)
    ax.set_ylim(0, ps.max() * 1.18)
    ax.set_ylabel("per-sample latency (ms)")
    ax.set_title(
        f"Per-sample latency vs batch — the batching payoff\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    _save(fig, out_dir, "fig3_per_sample.svg")


# --------------------------------------------------------------------------- #
# Figure 4 — e2e latency + action throughput (dual axis)                      #
# --------------------------------------------------------------------------- #
def fig4_latency_throughput(p: Profile, out_dir: Path):
    e2e = p.field("e2e_ms")
    tp = p.field("action_chunks_per_s")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(p.x, e2e, 0.5, color=COLOR["gen"], alpha=0.55, label="batch latency")
    for i, v in enumerate(e2e):
        ax.text(
            i,
            v + e2e.max() * 0.01,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR["gen"],
        )
    ax.set_ylabel("batch latency (ms)", color=COLOR["gen"])
    ax.tick_params(axis="y", labelcolor=COLOR["gen"])
    _xticks(ax, p)
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.plot(p.x, tp, "-o", color=COLOR["throughput"], lw=2.4, label="throughput")
    for i, v in enumerate(tp):
        ax2.text(
            i,
            v + tp.max() * 0.02,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR["throughput"],
        )
    ax2.set_ylabel("action chunks / s", color=COLOR["throughput"])
    ax2.tick_params(axis="y", labelcolor=COLOR["throughput"])
    ax.set_title(
        f"Latency and throughput vs batch\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    _save(fig, out_dir, "fig4_latency_throughput.svg")


# --------------------------------------------------------------------------- #
# Figure 5 — GEN MFU climb                                                    #
# --------------------------------------------------------------------------- #
def fig5_gen_mfu(p: Profile, out_dir: Path):
    mfu = [r.get("gen_mfu_pct") for r in p.sweep]
    if any(m is None for m in mfu):
        print("skip fig5_gen_mfu: no MFU (peak missing) in JSON")
        return
    mfu = np.array(mfu, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(p.x, mfu, "-o", color=COLOR["gen"], lw=2.4)
    for i, v in enumerate(mfu):
        ax.text(
            i, v + mfu.max() * 0.02, f"{v:.1f}%", ha="center", va="bottom", fontsize=9
        )
    _xticks(ax, p)
    ax.set_ylabel(f"GEN per-step MFU (% of {p.peak:g} TFLOPS)")
    ax.set_ylim(0, max(100.0, mfu.max() * 1.18))
    ax.set_title(
        f"GEN-tower MFU vs batch (starts compute-bound at bs=1)\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    _save(fig, out_dir, "fig5_gen_mfu.svg")


# --------------------------------------------------------------------------- #
# Figure 6 — 100%-stacked phase share                                         #
# --------------------------------------------------------------------------- #
def fig6_phase_share(p: Profile, out_dir: Path):
    vae = p.vae()
    cond = p.field("cond_encode_total_ms")
    gen = p.field("gen_loop_ms")
    tot = vae + cond + gen
    tot[tot == 0] = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bottom = np.zeros(len(p.bs))
    for y, key, lab in (
        (vae, "vae", "VAE encode"),
        (cond, "cond", "cond encode"),
        (gen, "gen", "GEN denoise loop"),
    ):
        frac = 100 * y / tot
        ax.bar(p.x, frac, 0.6, bottom=bottom, label=lab, color=COLOR[key])
        bottom += frac
    _xticks(ax, p)
    ax.set_ylabel("share of latency (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        f"Where the action step goes: phase share vs batch\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    ax.legend(frameon=False, loc="lower center", ncol=3, fontsize=9)
    _save(fig, out_dir, "fig6_phase_share.svg")


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
        f"Peak memory vs batch (resident weights + activations)\n{p.hw_caption()}",
        fontsize=11.5,
        weight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "fig7_memory.svg")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--in", dest="in_path", type=Path, required=True, help="profile JSON"
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("benchmark/cosmos3_policy/figures")
    )
    args = ap.parse_args()

    p = Profile(json.loads(args.in_path.read_text()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device: {p.gpu}  | peak {p.peak} TFLOPS  bw {p.bw} TB/s  "
        f"| N={p.num_steps}  | {len(p.bs)} batch points"
    )

    fig1_phase_latency(p, args.out_dir)
    fig2_roofline(p, args.out_dir)
    fig3_per_sample(p, args.out_dir)
    fig4_latency_throughput(p, args.out_dir)
    fig5_gen_mfu(p, args.out_dir)
    fig6_phase_share(p, args.out_dir)
    fig7_memory(p, args.out_dir)


if __name__ == "__main__":
    main()
