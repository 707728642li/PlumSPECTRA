from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "navy": "#183B56",
    "blue": "#3E7DA6",
    "blue_light": "#DFECF4",
    "green": "#4E8A68",
    "green_light": "#E1EFE6",
    "gold": "#C68B26",
    "gold_light": "#F5EAD2",
    "purple": "#7A6599",
    "purple_light": "#ECE7F2",
    "gray": "#63717C",
    "gray_light": "#EFF2F4",
}


def node(axis, x, y, width, height, title, detail, face, edge, title_size=9.5):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        facecolor=face,
        edgecolor=edge,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["navy"],
    )
    axis.text(
        x + width / 2,
        y + height * 0.29,
        detail,
        ha="center",
        va="center",
        fontsize=8.1,
        color=COLORS["gray"],
        linespacing=1.15,
    )
    return patch


def arrow(axis, start, end, color=None, connectionstyle="arc3"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            color=color or COLORS["gray"],
            connectionstyle=connectionstyle,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.2, 6.8), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, "PlumRAC-Net", fontsize=18, fontweight="bold", color=COLORS["navy"], va="top")
    ax.text(
        0.02,
        0.905,
        "Plum Residual-Anchored Cross-cultivar Network - one trait, one independently tuned model",
        fontsize=10.5,
        color=COLORS["gray"],
        va="top",
    )

    node(ax, 0.025, 0.56, 0.13, 0.20, "NIR", "228 bands\n901-1701 nm", COLORS["gray_light"], "#AAB4BB")
    node(
        ax,
        0.205,
        0.56,
        0.15,
        0.20,
        "Physical front end",
        "Raw-space scatter/noise\nthen RAW + SNV + SG1",
        COLORS["blue_light"],
        COLORS["blue"],
    )
    arrow(ax, (0.155, 0.66), (0.205, 0.66))

    node(ax, 0.41, 0.69, 0.17, 0.17, "PLSR anchor", "Fold-safe preprocessing\nand latent variables", COLORS["gold_light"], COLORS["gold"])
    node(ax, 0.41, 0.42, 0.17, 0.19, "Residual encoder", "Compact dilated 1D-ResNet\n72.5k trainable parameters", COLORS["blue_light"], COLORS["blue"])
    arrow(ax, (0.355, 0.66), (0.41, 0.775), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (0.355, 0.64), (0.41, 0.515), connectionstyle="arc3,rad=0.08")

    node(ax, 0.635, 0.42, 0.18, 0.19, "Attentive RAC tail", "Spectral attention + final-value\nabsolute / centred / rank loss", COLORS["purple_light"], COLORS["purple"])
    arrow(ax, (0.58, 0.515), (0.635, 0.515), color=COLORS["blue"])

    ax.text(0.61, 0.79, r"$\hat{y}_{PLS}$", fontsize=12, color=COLORS["gold"], ha="center")
    ax.text(0.84, 0.54, r"$\Delta\hat{y}$", fontsize=12, color=COLORS["purple"], ha="center")
    arrow(ax, (0.58, 0.775), (0.87, 0.70), color=COLORS["gold"], connectionstyle="arc3,rad=0.12")
    arrow(ax, (0.815, 0.515), (0.87, 0.64), color=COLORS["purple"], connectionstyle="arc3,rad=-0.12")
    node(
        ax,
        0.87,
        0.59,
        0.105,
        0.17,
        "Gated sum",
        r"$\hat{y}=\hat{y}_{PLS}+g\Delta\hat{y}$" + "\n" + r"$g\in\{0,.25,.50\}$; 5/5 wins",
        COLORS["green_light"],
        COLORS["green"],
        title_size=9.2,
    )

    node(ax, 0.205, 0.17, 0.15, 0.16, "Cultivar-safe CV", "16 LOCO outer folds\nall tuning inside train", COLORS["gray_light"], "#AAB4BB")
    arrow(ax, (0.28, 0.33), (0.28, 0.55))
    node(ax, 0.41, 0.14, 0.17, 0.19, "Full outer refit", "Selected epoch/profile refit\non all 15 source cultivars", COLORS["green_light"], COLORS["green"])
    arrow(ax, (0.355, 0.25), (0.41, 0.235))
    node(ax, 0.635, 0.14, 0.18, 0.19, "Affine adapter", "Optional 5-40-shot\ntarget-cultivar calibration", COLORS["gold_light"], COLORS["gold"])
    arrow(ax, (0.58, 0.235), (0.635, 0.235))
    arrow(ax, (0.815, 0.235), (0.922, 0.585), color=COLORS["gold"], connectionstyle="arc3,rad=-0.18")

    ax.text(0.025, 0.035, "Zero-shot safeguard", fontsize=9.3, fontweight="bold", color=COLORS["navy"])
    ax.text(
        0.14,
        0.035,
        "A nonzero half-amplitude correction requires improvement in every source validation cultivar; otherwise g = 0 recovers PLSR exactly.",
        fontsize=8.8,
        color=COLORS["gray"],
        va="center",
    )

    for suffix in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig_plumrac_architecture.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
