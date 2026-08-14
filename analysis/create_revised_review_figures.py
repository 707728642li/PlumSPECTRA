from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TRAITS = ["LS", "SRF", "PFD", "PRW", "LW", "MFF", "RD", "F6", "AF"]
BLUE = "#3B73B9"
ORANGE = "#E58B2A"
GREEN = "#2A9D8F"
RED = "#C44E52"
GREY = "#8A8A8A"


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def load_selected(project: Path, manifest: dict[str, object], trait: str) -> pd.DataFrame:
    source = manifest["traits"][trait]
    paths = [source] if isinstance(source, str) else source
    return pd.concat([pd.read_parquet(project / path) for path in paths], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    evidence_dir = args.evidence_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
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

    # Figure 3: equal-information model benchmark and paired uncertainty.
    benchmark = pd.read_csv(evidence_dir / "revised_model_benchmark.csv")
    labels = {
        "pls_domain": "Domain PLSR",
        "ridge": "Ridge",
        "svr": "RBF-SVR (untuned)",
        "rf": "Random forest",
        "hgb": "Hist. gradient boost",
        "ai_selected": "Selected residual AI",
    }
    models = list(labels)
    heat = (
        benchmark[benchmark["model"].isin(models)]
        .pivot(index="model", columns="trait", values="vs_global_pls_pct")
        .reindex(index=models, columns=TRAITS)
    )
    heat.index = [labels[value] for value in heat.index]
    primary = pd.read_csv(evidence_dir / "revised_primary_statistics.csv").set_index("trait").loc[TRAITS]
    ai_svr = pd.read_csv(evidence_dir / "revised_ai_vs_untuned_svr.csv").set_index("trait").loc[TRAITS]

    fig = plt.figure(figsize=(12.2, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])
    ax_a = fig.add_subplot(grid[0, :])
    sns.heatmap(
        heat,
        cmap=sns.diverging_palette(15, 220, as_cmap=True),
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "RMSE reduction vs ordinary global PLSR (%)"},
        ax=ax_a,
    )
    ax_a.set_xlabel("Mechanical endpoint")
    ax_a.set_ylabel("")
    ax_a.set_title("a  Equal-split model benchmark")

    y = np.arange(len(TRAITS))
    ax_b = fig.add_subplot(grid[1, 0])
    values = primary["ai_vs_domain_pls_pct"].to_numpy(float)
    low = values - primary["ai_vs_domain_ci_low"].to_numpy(float)
    high = primary["ai_vs_domain_ci_high"].to_numpy(float) - values
    ax_b.errorbar(values, y, xerr=np.vstack([low, high]), fmt="o", color=BLUE, ecolor=BLUE, capsize=2)
    ax_b.axvline(0, color="black", linewidth=0.8)
    ax_b.set_yticks(y, TRAITS)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("AI RMSE reduction vs domain PLSR (%)")
    ax_b.set_title("b  Cultivar-cluster 95% intervals")

    ax_c = fig.add_subplot(grid[1, 1])
    values = ai_svr["ai_vs_untuned_svr_pct"].to_numpy(float)
    low = values - ai_svr["cluster_ci_low"].to_numpy(float)
    high = ai_svr["cluster_ci_high"].to_numpy(float) - values
    colors = [GREEN if lo > 0 else GREY for lo in ai_svr["cluster_ci_low"]]
    for index, (value, lo, hi, color) in enumerate(zip(values, low, high, colors, strict=True)):
        ax_c.errorbar(value, index, xerr=np.asarray([[lo], [hi]]), fmt="o", color=color, ecolor=color, capsize=2)
    ax_c.axvline(0, color="black", linewidth=0.8)
    ax_c.set_yticks(y, TRAITS)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("AI RMSE reduction vs untuned RBF-SVR (%)")
    ax_c.set_title("c  Kernel comparator (post-review)")
    save(fig, output_dir, "fig03_revised_model_benchmark")

    # Figure 4: total-gain decomposition and within-cultivar information.
    within = pd.read_csv(evidence_dir / "revised_within_cultivar_statistics.csv").set_index("trait").loc[TRAITS]
    domain_gain = primary["domain_pls_vs_global_pls_pct"].to_numpy(float)
    residual_gain = primary["ai_vs_global_pls_pct"].to_numpy(float) - domain_gain
    x = np.arange(len(TRAITS))
    fig = plt.figure(figsize=(12.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax_a = fig.add_subplot(grid[0, :])
    ax_a.bar(x, domain_gain, color=ORANGE, label="Cultivar-domain correction")
    ax_a.bar(x, residual_gain, bottom=domain_gain, color=BLUE, label="Neural residual increment")
    for index, share in enumerate(primary["domain_share_of_total_gain_pct"].to_numpy(float)):
        ax_a.text(index, domain_gain[index] + residual_gain[index] + 0.45, f"{share:.0f}%", ha="center", fontsize=8)
    ax_a.set_xticks(x, TRAITS)
    ax_a.set_ylabel("RMSE reduction vs ordinary global PLSR (%)")
    ax_a.set_ylim(0, max(primary["ai_vs_global_pls_pct"]) + 4.5)
    ax_a.legend(frameon=False, ncol=2, loc="upper right")
    ax_a.set_title("a  Total system gain decomposed by model component\n(labels: domain share of total gain)")

    ax_b = fig.add_subplot(grid[1, 0])
    ax_b.scatter(within["pooled_r2_ai"], within["within_cultivar_r2_ai"], s=55, color=BLUE)
    for trait, row in within.iterrows():
        ax_b.annotate(trait, (row["pooled_r2_ai"], row["within_cultivar_r2_ai"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax_b.axhline(0, color="black", linewidth=0.8)
    ax_b.set_xlabel("Pooled $R^2$ (means retained)")
    ax_b.set_ylabel("Within-cultivar $R^2$ (means removed)")
    ax_b.set_title("b  Domain-level and fruit-level skill differ")

    ax_c = fig.add_subplot(grid[1, 1])
    width = 0.37
    ax_c.bar(x - width / 2, within["within_cultivar_pearson_domain_pls"], width, color=ORANGE, label="Domain PLSR")
    ax_c.bar(x + width / 2, within["within_cultivar_pearson_ai"], width, color=BLUE, label="Residual AI")
    ax_c.set_xticks(x, TRAITS)
    ax_c.set_ylabel("Within-cultivar Pearson $r$")
    ax_c.set_ylim(0, 0.50)
    ax_c.legend(frameon=False)
    ax_c.set_title("c  AI preserves more within-cultivar ranking signal")
    save(fig, output_dir, "fig04_gain_decomposition_within_cultivar")

    # Figure 5: cultivar/batch nesting and deployment boundaries.
    cross = pd.read_csv(evidence_dir / "batch_by_cultivar_counts.csv", index_col=0)
    lobo = pd.read_csv(project / "results/reviewer_recompute/recompute_within_domain_ceiling.csv").set_index("trait").loc[TRAITS]
    fig = plt.figure(figsize=(12.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    sns.heatmap(np.log1p(cross), cmap="Blues", cbar_kws={"label": "log(1 + fruit count)"}, ax=ax_a)
    ax_a.set_xlabel("Cultivar")
    ax_a.set_ylabel("Acquisition batch")
    ax_a.set_title("a  Every retained batch contains one cultivar")

    ax_b = fig.add_subplot(grid[0, 1])
    values = lobo["leave_one_BATCH_out_PLS_R2"].to_numpy(float)
    ax_b.barh(np.arange(len(TRAITS)), values, color=RED)
    ax_b.axvline(0, color="black", linewidth=0.8)
    ax_b.set_yticks(np.arange(len(TRAITS)), TRAITS)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Leave-one-batch-out PLSR $R^2$")
    ax_b.set_title("b  New-session transfer failed")

    ax_c = fig.add_subplot(grid[0, 2])
    ax_c.axis("off")
    boxes = [
        (0.82, "Same recorded session\nNew fruit, known cultivar\nSUPPORTED", GREEN),
        (0.50, "New session\nKnown cultivar\nNOT SUPPORTED", ORANGE),
        (0.18, "New cultivar\nZero-shot transfer\nRD directional only", RED),
    ]
    for y_value, text, color in boxes:
        ax_c.text(
            0.5,
            y_value,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": color, "edgecolor": "white", "alpha": 0.88},
            color="white",
            weight="bold",
            transform=ax_c.transAxes,
        )
    ax_c.annotate("", xy=(0.5, 0.63), xytext=(0.5, 0.70), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": GREY})
    ax_c.annotate("", xy=(0.5, 0.31), xytext=(0.5, 0.38), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": GREY})
    ax_c.set_title("c  Three deployment boundaries")
    save(fig, output_dir, "fig05_batch_structure_deployment_boundary")

    # Figure 6: pooled and cultivar-centred predictions for representative endpoints.
    chosen = ["LS", "SRF", "RD", "AF"]
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.6), constrained_layout=True)
    for column, trait in enumerate(chosen):
        frame = load_selected(project, manifest, trait)
        sample = frame.sample(min(len(frame), 3500), random_state=20260807)
        axes[0, column].hexbin(sample["y_true"], sample["y_pred"], gridsize=38, mincnt=1, cmap="viridis")
        limits = [
            min(sample["y_true"].min(), sample["y_pred"].min()),
            max(sample["y_true"].max(), sample["y_pred"].max()),
        ]
        axes[0, column].plot(limits, limits, linestyle="--", color="white", linewidth=1)
        axes[0, column].set_title(f"{trait}: pooled")
        axes[0, column].set_xlabel("Observed")
        if column == 0:
            axes[0, column].set_ylabel("Predicted")

        centred = frame.copy()
        centred["truth_c"] = centred["y_true"] - centred.groupby(["repeat", "cultivar_ascii"])["y_true"].transform("mean")
        centred["pred_c"] = centred["y_pred"] - centred.groupby(["repeat", "cultivar_ascii"])["y_pred"].transform("mean")
        centred = centred.sample(min(len(centred), 3500), random_state=20260808)
        axes[1, column].hexbin(centred["truth_c"], centred["pred_c"], gridsize=38, mincnt=1, cmap="magma")
        axes[1, column].axhline(0, color="white", linewidth=0.7)
        axes[1, column].axvline(0, color="white", linewidth=0.7)
        row = within.loc[trait]
        axes[1, column].set_title(f"within cultivar: r={row['within_cultivar_pearson_ai']:.2f}")
        axes[1, column].set_xlabel("Observed minus cultivar mean")
        if column == 0:
            axes[1, column].set_ylabel("Predicted minus cultivar mean")
    save(fig, output_dir, "fig06_pooled_and_within_cultivar_predictions")

    manifest_out = {
        "figures": [
            "fig03_revised_model_benchmark",
            "fig04_gain_decomposition_within_cultivar",
            "fig05_batch_structure_deployment_boundary",
            "fig06_pooled_and_within_cultivar_predictions",
        ],
        "formats": ["pdf", "png"],
        "png_dpi": 400,
        "statistical_text_source": str(evidence_dir),
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")
    print(json.dumps(manifest_out, indent=2), flush=True)


if __name__ == "__main__":
    main()
