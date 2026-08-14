from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

from v2_registry import add_cultivar_code


PROJECT = Path(__file__).resolve().parents[1]
TRAITS = ["SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"]
MODELS = ["global_pls", "domain_pls", "domain_svr", "deep", "deep_kernel"]
MODEL_LABELS = {
    "global_pls": "Global PLSR",
    "domain_pls": "Cultivar-aware PLSR",
    "domain_svr": "Nested RBF-SVR",
    "deep": "Trait-specific deep",
    "deep_kernel": "Deep–kernel ensemble",
}
BLUE = "#2878B5"
ORANGE = "#F28E2B"
GREEN = "#2A9D8F"
PURPLE = "#7A5195"
RED = "#C44E52"
GREY = "#68737D"


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 9.0,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.3,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        transform=ax.transAxes,
    )


def build_figure1(
    ledger_path: Path,
    fold_counts_path: Path,
    output_dir: Path,
) -> None:
    ledger = pd.read_parquet(ledger_path)
    targets = [
        "skin_break_force_g_mean",
        "skin_break_displacement_raw_mean",
        "skin_break_drop_g_mean",
        "flesh_force_mean_g_mean",
        "force_at_6_rawpos_g_mean",
        "loading_stiffness_g_per_rawpos_mean",
        "loading_work_g_rawpos_mean",
        "post_break_work_g_rawpos_mean",
        "adhesive_force_g_mean",
    ]
    complete = ledger[targets].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    primary = ledger[ledger["qc_primary_include"].astype(bool) & complete].copy()
    final = primary[primary["cultivar_ascii"].astype(str) != "6.11"].copy()
    final = add_cultivar_code(final)
    cultivar_counts = final.groupby("cultivar_code", observed=True).size().sort_values()
    fold_counts = pd.read_csv(fold_counts_path)
    fold_totals = fold_counts.groupby("outer_fold", observed=True)["samples"].sum()

    fig = plt.figure(figsize=(13.2, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.05])

    ax_a = fig.add_subplot(grid[0, :])
    ax_a.axis("off")
    ax_a.set_title("a  Fruit-level measurement order and inference boundary", loc="left")
    sequence = [
        ("Intact fruit", "#EAF2F8", BLUE),
        ("NIR spectrum\n901–1701 nm; 228 bands", "#D9EEF7", BLUE),
        ("Fruit mass", "#F7E9D4", ORANGE),
        ("Texture analyzer\n2 penetration curves", "#FADBD8", RED),
        ("SSC", "#E8F4EA", GREEN),
        ("pH", "#E8F4EA", GREEN),
    ]
    x_positions = np.linspace(0.015, 0.84, len(sequence))
    for index, ((label, face, edge), x) in enumerate(zip(sequence, x_positions, strict=True)):
        rounded_box(ax_a, (float(x), 0.47), 0.14, 0.25, label, face, edge, fontsize=9.2)
        if index < len(sequence) - 1:
            ax_a.annotate(
                "",
                xy=(x_positions[index + 1] - 0.007, 0.595),
                xytext=(x + 0.147, 0.595),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.4},
            )
    ax_a.text(
        0.19,
        0.25,
        "Prediction-time input",
        color=BLUE,
        weight="bold",
        ha="center",
        transform=ax_a.transAxes,
    )
    ax_a.annotate(
        "",
        xy=(0.25, 0.45),
        xytext=(0.25, 0.30),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.4},
    )
    ax_a.text(
        0.61,
        0.25,
        "Reference phenotypes acquired after NIR",
        color=RED,
        weight="bold",
        ha="center",
        transform=ax_a.transAxes,
    )

    ax_b = fig.add_subplot(grid[1, 0])
    ax_b.axis("off")
    ax_b.set_title("b  Model-independent cohort construction and frozen audit", loc="left")
    rounded_box(ax_b, (0.02, 0.70), 0.34, 0.17, f"Matched fruit ledger\nn = {len(ledger):,}", "#EEF1F3", GREY)
    rounded_box(ax_b, (0.02, 0.43), 0.34, 0.17, f"Primary texture QC\nn = {len(primary):,}", "#EAF2F8", BLUE)
    rounded_box(ax_b, (0.02, 0.16), 0.34, 0.17, f"Final modelling cohort\nn = {len(final):,}; 15 cultivars", "#E8F4EA", GREEN)
    for start, end in [(0.70, 0.60), (0.43, 0.33)]:
        ax_b.annotate(
            "",
            xy=(0.19, end + 0.08),
            xytext=(0.19, start),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.4},
        )
    ax_b.text(0.38, 0.235, "exclude 6.11\nmeasurement QC", color=RED, fontsize=8.5, transform=ax_b.transAxes)
    rounded_box(ax_b, (0.53, 0.62), 0.43, 0.22, "Frozen non-overlapping five-fold audit\nevery fruit tested exactly once", "#F7E9D4", ORANGE)
    rounded_box(ax_b, (0.53, 0.32), 0.20, 0.17, "Deep model\none per trait", "#EAF2F8", BLUE)
    rounded_box(ax_b, (0.76, 0.32), 0.20, 0.17, "Nested RBF-SVR\none per trait", "#EEE7F4", PURPLE)
    rounded_box(ax_b, (0.62, 0.06), 0.26, 0.14, "Fixed 0.5 / 0.5\ndeep–kernel ensemble", "#E8F4EA", GREEN)
    ax_b.annotate("", xy=(0.73, 0.20), xytext=(0.63, 0.32), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": GREY})
    ax_b.annotate("", xy=(0.78, 0.20), xytext=(0.86, 0.32), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": GREY})

    ax_c = fig.add_subplot(grid[1, 1])
    y = np.arange(len(cultivar_counts))
    ax_c.barh(y, cultivar_counts.to_numpy(), color=BLUE, alpha=0.88)
    ax_c.set_yticks(y, cultivar_counts.index)
    ax_c.set_xlabel("Fruits in final cohort")
    ax_c.set_title("c  Large and deliberately unbalanced cultivar panel", loc="left")
    for index, value in enumerate(cultivar_counts.to_numpy()):
        ax_c.text(value + 8, index, str(int(value)), va="center", fontsize=7.5)
    ax_c.text(
        0.98,
        0.03,
        "Outer-fold n = " + "/".join(str(int(value)) for value in fold_totals),
        ha="right",
        transform=ax_c.transAxes,
        fontsize=8.5,
        color=GREY,
    )
    save(fig, output_dir, "fig01_v20_study_design")


def build_figure3(analysis_dir: Path, output_dir: Path) -> None:
    pooled = pd.read_csv(analysis_dir / "pooled_metrics.csv")
    comparisons = pd.read_csv(analysis_dir / "paired_cluster_bootstrap_comparisons.csv")
    rmse = pooled.pivot(index="model", columns="trait", values="rmse").reindex(index=MODELS, columns=TRAITS)
    gain = 100.0 * (1.0 - rmse.div(rmse.loc["global_pls"], axis=1))
    r2 = pooled.pivot(index="model", columns="trait", values="r2").reindex(index=MODELS, columns=TRAITS)

    fig = plt.figure(figsize=(13.2, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.25, 1.0])
    ax_a = fig.add_subplot(grid[0, :])
    display_gain = gain.loc[["domain_pls", "domain_svr", "deep", "deep_kernel"]]
    display_gain.index = [MODEL_LABELS[index] for index in display_gain.index]
    sns.heatmap(
        display_gain,
        cmap="YlGnBu",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "RMSE reduction vs global PLSR (%)"},
        ax=ax_a,
    )
    ax_a.set_xlabel("Texture endpoint")
    ax_a.set_ylabel("")
    ax_a.set_title("a  Every nonlinear or cultivar-aware model improves the conventional baseline", loc="left")

    ax_b = fig.add_subplot(grid[1, 0])
    baselines = ["global_pls", "domain_pls", "domain_svr"]
    colors = {"global_pls": ORANGE, "domain_pls": BLUE, "domain_svr": PURPLE}
    offsets = {"global_pls": -0.22, "domain_pls": 0.0, "domain_svr": 0.22}
    y = np.arange(len(TRAITS))
    for baseline in baselines:
        subset = (
            comparisons[
                (comparisons["candidate"] == "deep_kernel")
                & (comparisons["baseline"] == baseline)
            ]
            .set_index("trait")
            .loc[TRAITS]
        )
        values = subset["relative_rmse_improvement_pct"].to_numpy(float)
        low = values - subset["relative_improvement_ci_low"].to_numpy(float)
        high = subset["relative_improvement_ci_high"].to_numpy(float) - values
        ax_b.errorbar(
            values,
            y + offsets[baseline],
            xerr=np.vstack([low, high]),
            fmt="o",
            ms=4.5,
            capsize=2,
            color=colors[baseline],
            label=MODEL_LABELS[baseline],
        )
    ax_b.axvline(0, color="black", lw=0.8)
    ax_b.set_yticks(y, TRAITS)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Ensemble RMSE reduction (%) with cultivar-cluster 95% CI")
    ax_b.set_title("b  Prespecified ensemble outperforms three increasingly strong comparators", loc="left")
    ax_b.legend(frameon=False, ncol=3, fontsize=8, loc="lower right")

    ax_c = fig.add_subplot(grid[1, 1])
    r2_display = r2.copy()
    r2_display.index = [MODEL_LABELS[index] for index in r2_display.index]
    sns.heatmap(
        r2_display,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.45,
        vmin=0.25,
        vmax=0.70,
        cbar_kws={"label": "Pooled out-of-fold $R^2$"},
        ax=ax_c,
    )
    ax_c.set_xlabel("Texture endpoint")
    ax_c.set_ylabel("")
    ax_c.set_title("c  Predictive strength across nine mechanical phenotypes", loc="left")
    save(fig, output_dir, "fig03_v20_model_performance")


def build_figure4(analysis_dir: Path, output_dir: Path) -> None:
    predictions = pd.read_parquet(analysis_dir / "v20_merged_predictions.parquet")
    pooled = pd.read_csv(analysis_dir / "pooled_metrics.csv").set_index(["trait", "model"])
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 10.0), constrained_layout=True)
    for ax, trait in zip(axes.ravel(), TRAITS, strict=True):
        frame = predictions[predictions["trait"] == trait]
        hb = ax.hexbin(
            frame["y_true"],
            frame["y_deep_kernel"],
            gridsize=42,
            mincnt=1,
            cmap="mako",
            bins="log",
        )
        limits = [
            float(min(frame["y_true"].min(), frame["y_deep_kernel"].min())),
            float(max(frame["y_true"].max(), frame["y_deep_kernel"].max())),
        ]
        ax.plot(limits, limits, linestyle="--", color="white", lw=1.0)
        metric = pooled.loc[(trait, "deep_kernel")]
        ax.set_title(f"{trait}   $R^2$={metric['r2']:.2f}, r={metric['pearson_r']:.2f}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.text(
            0.03,
            0.96,
            "n = 4,839",
            va="top",
            transform=ax.transAxes,
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 2},
        )
    fig.suptitle(
        "Deep–kernel out-of-fold predictions for all 43,551 fruit–trait pairs",
        fontsize=14,
        weight="bold",
    )
    save(fig, output_dir, "fig04_v20_all_trait_predictions")


def build_figure5(analysis_dir: Path, output_dir: Path) -> None:
    centered = pd.read_csv(analysis_dir / "within_cultivar_centered_metrics.csv")
    cultivar = pd.read_csv(analysis_dir / "cultivar_metrics.csv")
    complementarity = pd.read_csv(analysis_dir / "deep_svr_complementarity.csv").set_index("trait").loc[TRAITS]
    centered_r = (
        centered.pivot(index="model", columns="trait", values="pearson_r")
        .reindex(index=["domain_pls", "domain_svr", "deep", "deep_kernel"], columns=TRAITS)
    )
    centered_r2 = centered[centered["model"] == "deep_kernel"].set_index("trait").loc[TRAITS, "r2"]

    domain = cultivar[cultivar["model"] == "domain_pls"].set_index(["cultivar_ascii", "trait"])["rmse"]
    hybrid = cultivar[cultivar["model"] == "deep_kernel"].set_index(["cultivar_ascii", "trait"])["rmse"]
    cultivar_gain = (100.0 * (1.0 - hybrid / domain)).rename("gain").reset_index()
    cultivar_gain = add_cultivar_code(cultivar_gain)
    gain_matrix = cultivar_gain.pivot(index="cultivar_code", columns="trait", values="gain").reindex(columns=TRAITS)
    gain_matrix = gain_matrix.loc[gain_matrix.mean(axis=1).sort_values().index]

    fig = plt.figure(figsize=(13.2, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[0.8, 1.2])
    ax_a = fig.add_subplot(grid[0, 0])
    display = centered_r.copy()
    display.index = [MODEL_LABELS[index] for index in display.index]
    sns.heatmap(
        display,
        cmap="crest",
        annot=True,
        fmt=".2f",
        linewidths=0.45,
        vmin=0,
        vmax=0.46,
        cbar_kws={"label": "Within-cultivar Pearson r"},
        ax=ax_a,
    )
    ax_a.set_xlabel("Texture endpoint")
    ax_a.set_ylabel("")
    ax_a.set_title("a  Fruit-level ranking after cultivar means are removed", loc="left")

    ax_b = fig.add_subplot(grid[0, 1])
    colors = [GREEN if value > 0 else RED for value in centered_r2]
    ax_b.bar(np.arange(len(TRAITS)), centered_r2.to_numpy(), color=colors)
    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.set_xticks(np.arange(len(TRAITS)), TRAITS, rotation=45, ha="right")
    ax_b.set_ylabel("Within-cultivar $R^2$")
    ax_b.set_title("b  Correlation persists, but absolute individual-fruit skill is harder", loc="left")

    ax_c = fig.add_subplot(grid[1, 0])
    sns.heatmap(
        gain_matrix,
        cmap=sns.diverging_palette(15, 220, as_cmap=True),
        center=0,
        annot=True,
        fmt=".0f",
        linewidths=0.35,
        cbar_kws={"label": "Ensemble RMSE reduction vs cultivar-aware PLSR (%)"},
        ax=ax_c,
    )
    ax_c.set_xlabel("Texture endpoint")
    ax_c.set_ylabel("Cultivar code")
    ax_c.set_title("c  Cultivar-level heterogeneity is visible rather than averaged away", loc="left")

    ax_d = fig.add_subplot(grid[1, 1])
    x = np.arange(len(TRAITS))
    ax_d.bar(
        x - 0.18,
        complementarity["deep_svr_error_correlation"],
        width=0.36,
        color=PURPLE,
        label="Error correlation",
    )
    ax_d.bar(
        x + 0.18,
        complementarity["opposite_error_sign_fraction"],
        width=0.36,
        color=GREEN,
        label="Opposite-sign fraction",
    )
    ax_d.set_xticks(x, TRAITS, rotation=45, ha="right")
    ax_d.set_ylim(0, 1.0)
    ax_d.set_ylabel("Fraction / correlation")
    ax_d.legend(frameon=False, fontsize=8)
    ax_d.set_title("d  Partial error diversity supports fixed averaging", loc="left")
    save(fig, output_dir, "fig05_v20_within_cultivar_heterogeneity")


def build_figure6(v21_analysis_dir: Path, output_dir: Path) -> None:
    pooled = pd.read_csv(v21_analysis_dir / "pooled_and_batch_macro_metrics.csv")
    batches = pd.read_csv(v21_analysis_dir / "per_batch_metrics.csv")
    comparisons = pd.read_csv(v21_analysis_dir / "descriptive_batch_bootstrap_comparisons.csv")
    hybrid_comparison = comparisons[comparisons["candidate"] == "deep_kernel"]
    comparison_matrix = (
        hybrid_comparison.pivot(index="baseline", columns="trait", values="relative_batch_macro_improvement_pct")
        .reindex(index=["global_pls", "domain_pls", "domain_svr"], columns=TRAITS)
    )
    comparison_matrix.index = [MODEL_LABELS[index] for index in comparison_matrix.index]

    domain = batches[batches["model"] == "domain_pls"].set_index(["batch_id", "trait"])["rmse"]
    hybrid = batches[batches["model"] == "deep_kernel"].set_index(["batch_id", "trait"])["rmse"]
    batch_gain = (100.0 * (1.0 - hybrid / domain)).rename("gain").reset_index()
    batch_gain_matrix = batch_gain.pivot(index="batch_id", columns="trait", values="gain").reindex(columns=TRAITS)
    batch_counts = batches[
        (batches["model"] == "deep_kernel") & (batches["trait"] == "SRF")
    ].set_index("batch_id")["n"].reindex(batch_gain_matrix.index)

    fig = plt.figure(figsize=(13.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15], width_ratios=[1.0, 1.25])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_a.barh(np.arange(len(batch_counts)), batch_counts.to_numpy(), color=[BLUE] * 3 + [ORANGE] * 2)
    ax_a.set_yticks(np.arange(len(batch_counts)), batch_counts.index)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Held-out fruits")
    ax_a.set_title("a  Five complete batches from two multi-batch cultivars", loc="left")
    ax_a.text(0.98, 0.04, "KLD: 3 batches   WW: 2 batches", transform=ax_a.transAxes, ha="right", fontsize=8.5, color=GREY)

    ax_b = fig.add_subplot(grid[0, 1])
    sns.heatmap(
        comparison_matrix,
        cmap=sns.diverging_palette(15, 220, as_cmap=True),
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.45,
        cbar_kws={"label": "Batch-macro RMSE reduction (%)"},
        ax=ax_b,
    )
    ax_b.set_xlabel("Texture endpoint")
    ax_b.set_ylabel("")
    ax_b.set_title("b  Same-cultivar leave-one-batch-out performance", loc="left")

    ax_c = fig.add_subplot(grid[1, 0])
    sns.heatmap(
        batch_gain_matrix,
        cmap=sns.diverging_palette(15, 220, as_cmap=True),
        center=0,
        annot=True,
        fmt=".0f",
        linewidths=0.4,
        cbar_kws={"label": "Ensemble RMSE reduction vs cultivar-aware PLSR (%)"},
        ax=ax_c,
    )
    ax_c.set_xlabel("Texture endpoint")
    ax_c.set_ylabel("Held-out batch")
    ax_c.set_title("c  Batch-specific gains and failures", loc="left")

    ax_d = fig.add_subplot(grid[1, 1])
    ax_d.axis("off")
    ax_d.set_title("d  Evidence boundary after V20 and V21", loc="left")
    boxes = [
        (0.72, "V20: new fruit, known cultivar\n15 cultivars; 4,839 fruit\nSUPPORTED INTERNAL INTERPOLATION", GREEN),
        (0.40, "V21: held-out batch, same cultivar\n2 cultivars; 5 batches; 1,236 fruit\nLIMITED TRANSFER AUDIT", ORANGE),
        (0.08, "New year / orchard / instrument / cultivar\nNO IDENTIFIABLE EXTERNAL TEST\nPROSPECTIVE VALIDATION REQUIRED", RED),
    ]
    for y, label, color in boxes:
        rounded_box(ax_d, (0.13, y), 0.74, 0.20, label, color, "white", fontsize=9.3)
    ax_d.annotate("", xy=(0.50, 0.61), xytext=(0.50, 0.72), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.3})
    ax_d.annotate("", xy=(0.50, 0.29), xytext=(0.50, 0.40), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.3})
    save(fig, output_dir, "fig06_v21_crossbatch_boundary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--fold-counts", type=Path, required=True)
    parser.add_argument("--v21-analysis-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    analysis_dir = args.analysis_dir.resolve()
    build_figure1(args.qc_ledger.resolve(), args.fold_counts.resolve(), output_dir)
    build_figure3(analysis_dir, output_dir)
    build_figure4(analysis_dir, output_dir)
    build_figure5(analysis_dir, output_dir)
    # Figure 2 is generated by the dedicated phenotype-atlas workflow.  Copy
    # both publication formats into the final figure directory so a submission
    # package is complete and sequentially named in one location.
    figure2_source = PROJECT / "results/texture_atlas/figures/fig_texture_phenotype_atlas"
    for suffix in (".pdf", ".png"):
        shutil.copy2(
            figure2_source.with_suffix(suffix),
            output_dir / f"fig02_texture_phenotype_atlas{suffix}",
        )
    figures = [
        "fig01_v20_study_design",
        "fig02_texture_phenotype_atlas",
        "fig03_v20_model_performance",
        "fig04_v20_all_trait_predictions",
        "fig05_v20_within_cultivar_heterogeneity",
    ]
    if args.v21_analysis_dir is not None:
        build_figure6(args.v21_analysis_dir.resolve(), output_dir)
        figures.append("fig06_v21_crossbatch_boundary")
    manifest = {
        "figures": figures,
        "figure_2_source": "results/texture_atlas/figures/fig_texture_phenotype_atlas",
        "formats": ["pdf", "png"],
        "png_dpi": 450,
        "quantitative_panels_generated_from_frozen_predictions": True,
        "image_generation_used": False,
        "image_generation_reason": "All main figures encode quantitative or protocol facts and therefore require reproducible vector graphics.",
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
