from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


LABELS = {
    "skin_break_force_g_mean": "Rupture force",
    "skin_break_displacement_raw_mean": "Rupture displacement",
    "skin_break_drop_g_mean": "Force drop",
    "flesh_force_mean_g_mean": "Flesh resistance",
    "force_at_6_rawpos_g_mean": "Force at 6 units",
    "loading_stiffness_g_per_rawpos_mean": "Loading stiffness",
    "loading_work_g_rawpos_mean": "Loading work",
    "post_break_work_g_rawpos_mean": "Post-rupture work",
    "adhesive_force_g_mean": "Adhesive force",
}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--strict-predictions", type=Path)
    parser.add_argument("--hardvalid-predictions", type=Path)
    parser.add_argument("--fewshot", type=Path, required=True)
    parser.add_argument("--axis-fewshot", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction = pd.read_parquet(args.predictions)
    rows: list[dict[str, float | int | str]] = []
    fold_rows: list[dict[str, float | int | str]] = []
    for target, frame in prediction.groupby("target", observed=True):
        correlations: list[float] = []
        fold_r2: list[float] = []
        fold_weights: list[int] = []
        for cultivar, fold in frame.groupby("cultivar_ascii", observed=True):
            r = float(fold["y_true"].corr(fold["y_pred"]))
            r2 = float(r2_score(fold["y_true"], fold["y_pred"]))
            correlations.append(r)
            fold_r2.append(r2)
            fold_weights.append(len(fold))
            fold_rows.append(
                {
                    "target": target,
                    "target_label": LABELS[target],
                    "cultivar_ascii": cultivar,
                    "n": len(fold),
                    "r2": r2,
                    "pearson_r": r,
                }
            )
        centered_true = frame["y_true"] - frame.groupby("cultivar_ascii", observed=True)["y_true"].transform("mean")
        centered_pred = frame["y_pred"] - frame.groupby("cultivar_ascii", observed=True)["y_pred"].transform("mean")
        centered_r2 = 1.0 - float(((centered_true - centered_pred) ** 2).sum() / (centered_true**2).sum())
        rows.append(
            {
                "target": target,
                "target_label": LABELS[target],
                "n": len(frame),
                "pooled_r2": float(r2_score(frame["y_true"], frame["y_pred"])),
                "within_cultivar_centered_r2": centered_r2,
                "cultivar_macro_r2_mean": float(np.mean(fold_r2)),
                "median_within_cultivar_r": float(np.median(correlations)),
                "weighted_within_cultivar_r": float(np.average(correlations, weights=fold_weights)),
                "min_fold_r2": float(np.min(fold_r2)),
                "max_fold_r2": float(np.max(fold_r2)),
            }
        )
    validity = pd.DataFrame(rows).sort_values("pooled_r2", ascending=False)
    validity.to_csv(output / "endpoint_transfer_validity_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "endpoint_per_cultivar_metrics.csv", index=False)

    cohort_rows: list[dict[str, float | int | str]] = []
    cohort_sources = [
        ("High-confidence analysis", args.predictions),
        ("Strict 10%", args.strict_predictions),
        ("Hard-valid sensitivity", args.hardvalid_predictions),
    ]
    for cohort_name, path in cohort_sources:
        if path is None:
            continue
        cohort_prediction = pd.read_parquet(path)
        for target, frame in cohort_prediction.groupby("target", observed=True):
            cohort_rows.append(
                {
                    "cohort": cohort_name,
                    "target": target,
                    "target_label": LABELS[target],
                    "n": len(frame),
                    "r2": float(r2_score(frame["y_true"], frame["y_pred"])),
                }
            )
    pd.DataFrame(cohort_rows).to_csv(output / "cohort_sensitivity_metrics.csv", index=False)

    fewshot = pd.read_csv(args.fewshot)
    fewshot = fewshot.loc[fewshot["shots"].isin([0, 5, 20])].copy()
    fewshot["target_label"] = fewshot["target"].map(LABELS)
    fewshot.to_csv(output / "endpoint_fewshot_review_table.csv", index=False)
    axis_fewshot = pd.read_csv(args.axis_fewshot)
    axis_fewshot.loc[axis_fewshot["shots"].isin([0, 5, 20])].to_csv(
        output / "axis_fewshot_review_table.csv", index=False
    )

    ledger = pd.read_parquet(args.ledger)
    cohort_counts = (
        ledger.assign(
            strict_complete=(
                ledger["qc_primary_include"].astype(bool)
                & ledger["include_primary_multitask"].astype(bool)
                & ledger["texture_dual_valid"].astype(bool)
            )
        )
        .groupby(["cultivar_ascii", "cultivar_original"], observed=True)
        .agg(
            matched=("sample_id", "size"),
            hard_valid=("qc_sensitivity_include", "sum"),
            analysis=("qc_analysis_include", "sum"),
            strict=("qc_primary_include", "sum"),
            strict_complete=("strict_complete", "sum"),
        )
        .reset_index()
    )
    cohort_counts.to_csv(output / "cultivar_cohort_counts.csv", index=False, encoding="utf-8-sig")
    domain = (
        ledger.groupby("cultivar_ascii", observed=True)
        .agg(fruits=("sample_id", "size"), batches=("batch_id", "nunique"))
        .reset_index()
        .sort_values(["batches", "fruits"], ascending=[False, False])
    )
    domain.to_csv(output / "cultivar_batch_replication.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    plot = validity.sort_values("pooled_r2")
    y = np.arange(len(plot))
    ax.hlines(y, plot["within_cultivar_centered_r2"], plot["pooled_r2"], color="0.75", lw=1.4)
    ax.scatter(plot["pooled_r2"], y, color="#4E79A7", s=32, label="Pooled LOCO R2")
    ax.scatter(plot["within_cultivar_centered_r2"], y, color="#E15759", s=32, label="Within-cultivar centered R2")
    ax.axvline(0, color="0.45", lw=0.8, ls="--")
    ax.set_yticks(y, plot["target_label"])
    ax.set_xlabel("R2")
    ax.set_title("Pooled transfer masks weak within-cultivar fit")
    ax.legend(frameon=False, fontsize=7)
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    shot_plot = fewshot.pivot(index="target_label", columns="shots", values="r2_mean").loc[plot["target_label"]]
    y = np.arange(len(shot_plot))
    ax.hlines(y, shot_plot[0], shot_plot[5], color="0.75", lw=1.4)
    ax.scatter(shot_plot[0], y, color="#4E79A7", s=30, label="0-shot")
    ax.scatter(shot_plot[5], y, color="#59A14F", s=30, label="5-shot intercept")
    ax.axvline(0, color="0.45", lw=0.8, ls="--")
    ax.set_yticks(y, shot_plot.index)
    ax.set_xlabel("Pooled R2")
    ax.set_title("Intercept calibration raises pooled scores")
    ax.legend(frameon=False, fontsize=7)
    add_panel_label(ax, "b")

    ax = axes[1, 0]
    rank_plot = validity.sort_values("median_within_cultivar_r")
    colors = np.where(rank_plot["median_within_cultivar_r"] >= 0.3, "#59A14F", "#F28E2B")
    ax.barh(rank_plot["target_label"], rank_plot["median_within_cultivar_r"], color=colors)
    ax.axvline(0.3, color="0.45", lw=0.8, ls="--")
    ax.set_xlim(0, 0.42)
    ax.set_xlabel("Median Pearson r across held-out cultivars")
    ax.set_title("Fruit-ranking signal remains modest")
    add_panel_label(ax, "c")

    ax = axes[1, 1]
    domain_plot = domain.sort_values(["batches", "fruits"])
    colors = domain_plot["batches"].map({1: "#BAB0AC", 2: "#F28E2B", 3: "#E15759"})
    ax.barh(domain_plot["cultivar_ascii"], domain_plot["batches"], color=colors)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Recorded acquisition batches")
    ax.set_title("Cultivar and batch are largely confounded")
    add_panel_label(ax, "d")

    fig.savefig(output / "fig_critical_validity_audit.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "fig_critical_validity_audit.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "endpoint_count": len(validity),
        "pooled_r2_range": [float(validity["pooled_r2"].min()), float(validity["pooled_r2"].max())],
        "within_cultivar_centered_r2_range": [
            float(validity["within_cultivar_centered_r2"].min()),
            float(validity["within_cultivar_centered_r2"].max()),
        ],
        "median_within_cultivar_r_range": [
            float(validity["median_within_cultivar_r"].min()),
            float(validity["median_within_cultivar_r"].max()),
        ],
        "cultivars_with_one_batch": int((domain["batches"] == 1).sum()),
        "cultivars_with_multiple_batches": int((domain["batches"] > 1).sum()),
        "fewshot_warning": "Intercept-only calibration changes bias but cannot change within-cultivar ranks or Pearson correlation.",
    }
    (output / "critical_validity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
