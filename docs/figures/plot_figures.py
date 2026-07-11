#!/usr/bin/env python3
"""Render the MC-sampling guide figures from docs/figures/data/*.csv to
docs/src/assets/mc_*.svg. Regenerate the data first with
    julia --project docs/figures/generate_data.jl
"""
import csv
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "..", "src", "assets")
os.makedirs(OUT, exist_ok=True)

# palette (validated): series-1 blue, series-2 aqua, neutral ink/guides
BLUE = "#2a78d6"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e7e6e2"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 10,
    "svg.fonttype": "path",       # embed glyphs as paths (robust across viewers)
})


def read(name, skip=0):
    with open(os.path.join(DATA, name)) as f:
        rows = list(csv.reader(f))[skip:]
    return rows[0], [[float(x) for x in r] for r in rows[1:]]


def fig1():
    _, curves = read("fig1_curves.csv")
    _, mc = read("fig1_mc.csv")
    t = [r[0] for r in curves]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.grid(True, axis="y", zorder=0)
    ax.plot(t, [r[1] for r in curves], color=MUTED, lw=2, zorder=2)
    ax.plot(t, [r[2] for r in curves], color=AQUA, lw=2, ls=(0, (5, 3)), zorder=2)
    ax.errorbar([r[0] for r in mc], [r[1] for r in mc], yerr=[3 * r[2] for r in mc],
                fmt="o", color=BLUE, ms=5, lw=1.2, capsize=2, zorder=3)

    ax.annotate("exact  $L(\\beta|J|)$", xy=(0.83, 0.43), color=MUTED)
    ax.annotate("Metropolis MC", xy=(0.27, 0.87), color=BLUE)
    ax.annotate("mean field  $m(\\tau)^2$", xy=(0.03, 0.10), color=AQUA)
    ax.annotate("$T_\\mathrm{MF}$: spurious MFA transition", xy=(1 / 3, 0.0),
                xytext=(0.42, 0.12), color=MUTED, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$k_B T\,/\,|J|$")
    ax.set_ylabel(r"$\langle \mathbf{e}_1\cdot \mathbf{e}_2\rangle$")
    ax.set_title("Two coupled spins: MC samples the exact correlation",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mc_dimer_correlation.svg"))
    plt.close(fig)


def fig2():
    with open(os.path.join(DATA, "fig2_trace.csv")) as f:
        first = f.readline()
    e0 = float(first.split("E0 =")[1].split()[0])
    _, rows = read("fig2_trace.csv", skip=1)
    idx = [r[0] for r in rows]
    tj = [r[1] for r in rows]
    energy = [r[2] for r in rows]
    acc = [r[3] for r in rows]

    # temperature-block boundaries and labels
    bounds, labels, start = [], [], 0
    for k in range(1, len(tj) + 1):
        if k == len(tj) or tj[k] != tj[start]:
            labels.append((0.5 * (idx[start] + idx[k - 1]), tj[start]))
            k < len(tj) and bounds.append(idx[k - 1] + 0.5)
            start = k

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True,
                                  height_ratios=[3, 1])
    for b in bounds:
        for a in (ax, ax2):
            a.axvline(b, color=GRID, lw=1, zorder=0)
    ax.axhline(e0, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.annotate("aligned ground state $E_0$", xy=(idx[2], e0 + 0.05), color=MUTED,
                fontsize=9)
    ax.plot(idx, energy, color=BLUE, lw=1.2, alpha=0.85, zorder=2)
    # per-block mean: the equilibrated level the eye should read off the noisy trace
    start = 0
    for k in range(1, len(tj) + 1):
        if k == len(tj) or tj[k] != tj[start]:
            m = sum(energy[start:k]) / (k - start)
            ax.plot([idx[start], idx[k - 1]], [m, m], color=INK, lw=2.2, zorder=3,
                    solid_capstyle="butt")
            start = k
    for j, (x, t) in enumerate(labels):
        lab = f"$k_BT/|J| = {t:g}$" if j == 0 else f"${t:g}$"
        ax.annotate(lab, xy=(x, 1.02), xycoords=("data", "axes fraction"),
                    ha="center", color=MUTED, fontsize=8.5)
    ax.set_ylabel("energy (eV)")

    ax2.plot(idx, acc, color=AQUA, lw=1.4, zorder=2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("acceptance")
    ax2.set_xlabel("stored configuration")
    ax2.set_xlim(idx[0], idx[-1])

    fig.suptitle("Annealing sweep on an 8-site ferromagnet (warm-started ladder)",
                 x=0.13, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(os.path.join(OUT, "mc_annealing.svg"))
    plt.close(fig)


def fig3():
    _, rows = read("fig3_axis.csv")
    idx = [r[0] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.grid(True, axis="y", zorder=0)
    ax.scatter(idx, [r[2] for r in rows], s=14, color=AQUA, alpha=0.75, lw=0, zorder=2)
    ax.scatter(idx, [r[1] for r in rows], s=14, color=BLUE, alpha=0.9, lw=0, zorder=3)
    box = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5)
    ax.annotate("randomize = false — the axis stays where the chain started",
                xy=(8, 0.83), color=BLUE, fontsize=9.5, bbox=box)
    ax.annotate("randomize = true — Haar-uniform orientation", xy=(8, -0.88),
                color=AQUA, fontsize=9.5, bbox=box)
    ax.set_xlim(0, idx[-1] + 1)
    ax.set_ylim(-1.1, 1.18)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("stored configuration")
    ax.set_ylabel(r"$\hat{\mathbf{n}}\cdot\hat{\mathbf{z}}$  (mean axis)")
    ax.set_title("The absolute orientation is a zero mode local updates barely move",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mc_randomize.svg"))
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("wrote", os.path.abspath(OUT))
