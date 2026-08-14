from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from v2_registry import cultivar_code_map


COLORS = {
    "navy": "#193b57",
    "blue": "#3f7ca6",
    "light_blue": "#dceaf3",
    "green": "#4f8766",
    "light_green": "#e0eee5",
    "gold": "#c58c2a",
    "light_gold": "#f4ead4",
    "gray": "#5b6570",
    "light_gray": "#eef1f3",
}


def box(axis, xy, width, height, title, detail, facecolor, edgecolor, fontsize=10):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    axis.add_patch(patch)
    x, y = xy
    axis.text(x + width / 2, y + height * 0.62, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=COLORS["navy"])
    axis.text(x + width / 2, y + height * 0.28, detail, ha="center", va="center", fontsize=fontsize - 1, color=COLORS["gray"])
    return patch


def arrow(axis, start, end):
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.1, color="#71808b"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(args.samples) if args.samples.suffix.lower() == ".parquet" else pd.read_csv(args.samples)
    counts = samples.groupby("cultivar_ascii").size().sort_values()
    counts.index = counts.index.map(cultivar_code_map())

    fig = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
    mosaic = fig.subplot_mosaic([["A", "B"], ["C", "B"]], width_ratios=[1.2, 1.0], height_ratios=[1.08, 0.92])

    ax = mosaic["A"]
    ax.set_title("A  Cohort construction", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ((0.08, 0.76), "Linked spectro-mechanical cohort", "5,502 fruits  |  11,004 ARC curves", COLORS["light_gray"], "#a9b1b7"),
        ((0.08, 0.53), "High-confidence analysis", "5,430 fruits  |  98.7% retained", COLORS["light_blue"], COLORS["blue"]),
        ((0.08, 0.29), "Strict texture-QC sensitivity", "4,952 fruits  |  90.0% retained", COLORS["light_gold"], COLORS["gold"]),
        ((0.08, 0.05), "Research-ready release", "4,941 complete fruits  |  16 cultivars  |  19 batches", COLORS["light_green"], COLORS["green"]),
    ]
    for idx, (xy, title, detail, face, edge) in enumerate(stages):
        box(ax, xy, 0.84, 0.14, title, detail, face, edge, fontsize=10)
        if idx < len(stages) - 1:
            arrow(ax, (0.50, xy[1] - 0.015), (0.50, stages[idx + 1][0][1] + 0.155))
    ax.text(0.88, 0.64, "72 high-confidence exclusions", ha="right", va="center", fontsize=8.0, color="#8d4e4e")
    ax.text(0.88, 0.18, "11 incomplete conventional traits", ha="right", va="center", fontsize=8.0, color="#8d4e4e")

    ax = mosaic["B"]
    ax.set_title("B  Strict-core cultivar composition", loc="left", fontsize=13, fontweight="bold")
    bars = ax.barh(counts.index, counts.values, color=COLORS["blue"], edgecolor="white", height=0.72)
    ax.bar_label(bars, padding=3, fontsize=8.5, color=COLORS["gray"])
    ax.set_xlabel("Individual fruits")
    ax.set_xlim(0, counts.max() * 1.15)
    ax.grid(axis="x", color="#e2e5e8", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.text(0.99, 0.01, f"Total = {len(samples):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, fontweight="bold", color=COLORS["navy"])

    ax = mosaic["C"]
    ax.set_title("C  Leakage-safe modelling and deployment design", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    first = [
        ((0.02, 0.64), 0.27, "NIR spectra", "901-1701 nm; 228 bands"),
        ((0.365, 0.64), 0.27, "Single-trait target", "SRF / RD / PFD / MFF / F6\nLS / LW / PRW / AF"),
        ((0.71, 0.64), 0.27, "Texture reference", "2 ARC penetrations per fruit"),
    ]
    for xy, width, title, detail in first:
        box(ax, xy, width, 0.21, title, detail, COLORS["light_blue"], COLORS["blue"], fontsize=9.5)
        arrow(ax, (xy[0] + width / 2, 0.63), (0.5, 0.50))
    box(ax, (0.22, 0.31), 0.56, 0.17, "Leave-one-cultivar-out evaluation", "16 outer folds; all tuning confined to training cultivars", COLORS["light_gold"], COLORS["gold"], fontsize=9.5)
    arrow(ax, (0.5, 0.30), (0.5, 0.235))
    box(ax, (0.02, 0.03), 0.30, 0.18, "Chemometric anchor", "Nested single-trait\nPLSR", COLORS["light_gray"], "#9da7ae", fontsize=9.3)
    box(ax, (0.35, 0.03), 0.30, 0.18, "PlumRAC-Net", "Residual anchor +\ntrait-specific RAC tail", COLORS["light_green"], COLORS["green"], fontsize=9.3)
    box(ax, (0.68, 0.03), 0.30, 0.18, "Few-shot adaptation", "0/5/10/20/40 labelled\ntarget fruits", COLORS["light_green"], COLORS["green"], fontsize=9.3)
    arrow(ax, (0.5, 0.235), (0.17, 0.215))
    arrow(ax, (0.5, 0.235), (0.50, 0.215))
    arrow(ax, (0.5, 0.235), (0.83, 0.215))

    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig01_study_design.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
