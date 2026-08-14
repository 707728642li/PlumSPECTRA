from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ENDPOINTS = [
    "skin_break_force_g",
    "skin_break_displacement_raw",
    "skin_break_drop_g",
    "force_at_6_rawpos_g",
    "flesh_force_mean_g",
    "loading_stiffness_g_per_rawpos",
    "loading_work_g_rawpos",
    "post_break_work_g_rawpos",
    "adhesive_force_g",
]

LABELS = {
    "skin_break_force_g": "Maximum penetration force",
    "skin_break_displacement_raw": "Peak-force displacement",
    "skin_break_drop_g": "Post-peak force drop",
    "force_at_6_rawpos_g": "Force at 6 units",
    "flesh_force_mean_g": "Flesh resistance",
    "loading_stiffness_g_per_rawpos": "Loading stiffness",
    "loading_work_g_rawpos": "Loading work",
    "post_break_work_g_rawpos": "Post-peak work",
    "adhesive_force_g": "Adhesive force",
}


def finite_pair(a: pd.Series, b: pd.Series) -> np.ndarray:
    return pd.DataFrame({"rep1": pd.to_numeric(a, errors="coerce"), "rep2": pd.to_numeric(b, errors="coerce")}).dropna().to_numpy(float)


def icc_a1(a: pd.Series, b: pd.Series) -> float:
    values = finite_pair(a, b)
    n, k = values.shape
    if n < 3:
        return np.nan
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_columns = n * np.sum((column_means - grand) ** 2)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else np.nan


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": groups}).dropna()
    if frame.empty:
        return np.nan
    grand = frame["value"].mean()
    total = np.sum((frame["value"] - grand) ** 2)
    between = sum(len(group) * (group["value"].mean() - grand) ** 2 for _, group in frame.groupby("group", observed=True))
    return float(between / total) if total > 0 else np.nan


def add_panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(args.ledger)
    batch = pd.read_csv(args.batch_summary)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_theme(style="ticks", context="paper", rc=mpl.rcParams)

    reliability_rows: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        for cohort, mask in {
            "Full parsed cohort": ledger["texture_valid_replicates"].ge(2),
            "High-confidence analysis cohort": ledger["qc_analysis_include"],
            "Strict 10% QC release cohort": ledger["qc_primary_include"],
        }.items():
            subset = ledger.loc[mask]
            pair = finite_pair(subset[f"rep01_{endpoint}"], subset[f"rep02_{endpoint}"])
            mean_values = pd.to_numeric(subset[f"{endpoint}_mean"], errors="coerce")
            reliability_rows.append(
                {
                    "endpoint": endpoint,
                    "label": LABELS[endpoint],
                    "cohort": cohort,
                    "n": int(len(pair)),
                    "icc_a1": icc_a1(subset[f"rep01_{endpoint}"], subset[f"rep02_{endpoint}"]),
                    "median_cv": float(pd.to_numeric(subset[f"{endpoint}_cv"], errors="coerce").replace([np.inf, -np.inf], np.nan).median()),
                    "iqr": float(mean_values.quantile(0.75) - mean_values.quantile(0.25)),
                    "range_2_98": float(mean_values.quantile(0.98) - mean_values.quantile(0.02)),
                    "cultivar_eta_squared": eta_squared(mean_values, subset["cultivar_ascii"]),
                }
            )
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(table_dir / "texture_qc_reliability_before_after.csv", index=False)

    before = reliability[reliability["cohort"] == "Full parsed cohort"].set_index("endpoint")
    analysis = reliability[reliability["cohort"] == "High-confidence analysis cohort"].set_index("endpoint")
    after = reliability[reliability["cohort"] == "Strict 10% QC release cohort"].set_index("endpoint")
    impact = after[["label", "n", "icc_a1", "median_cv", "iqr", "range_2_98", "cultivar_eta_squared"]].join(
        before[["n", "icc_a1", "median_cv", "iqr", "range_2_98", "cultivar_eta_squared"]],
        lsuffix="_primary",
        rsuffix="_full",
    )
    impact["icc_change"] = impact["icc_a1_primary"] - impact["icc_a1_full"]
    impact["iqr_retained_fraction"] = impact["iqr_primary"] / impact["iqr_full"]
    impact["range_2_98_retained_fraction"] = impact["range_2_98_primary"] / impact["range_2_98_full"]
    impact.to_csv(table_dir / "texture_qc_impact_summary.csv")
    analysis_impact = analysis[["label", "n", "icc_a1", "median_cv", "iqr", "range_2_98", "cultivar_eta_squared"]].join(
        before[["n", "icc_a1", "median_cv", "iqr", "range_2_98", "cultivar_eta_squared"]],
        lsuffix="_analysis",
        rsuffix="_full",
    )
    analysis_impact["icc_change"] = analysis_impact["icc_a1_analysis"] - analysis_impact["icc_a1_full"]
    analysis_impact["iqr_retained_fraction"] = analysis_impact["iqr_analysis"] / analysis_impact["iqr_full"]
    analysis_impact["range_2_98_retained_fraction"] = analysis_impact["range_2_98_analysis"] / analysis_impact["range_2_98_full"]
    analysis_impact.to_csv(table_dir / "texture_high_confidence_qc_impact_summary.csv")

    evidence_cols = [
        "acquisition_moderate",
        "replicate_moderate",
        "texture_multivariate_moderate",
        "spectral_moderate",
        "chemical_moderate",
    ]
    evidence_labels = {
        "acquisition_moderate": "Acquisition",
        "replicate_moderate": "Replicate",
        "texture_multivariate_moderate": "Texture MV",
        "spectral_moderate": "Spectral",
        "chemical_moderate": "Chemical",
    }
    excluded = ledger.loc[ledger["qc_consensus_exclude"]].copy()
    excluded["evidence_pattern"] = excluded[evidence_cols].apply(
        lambda row: " + ".join(evidence_labels[col] for col in evidence_cols if bool(row[col])), axis=1
    )
    patterns = excluded["evidence_pattern"].value_counts().rename_axis("evidence_pattern").reset_index(name="fruit_count")
    patterns.to_csv(table_dir / "texture_qc_evidence_combinations.csv", index=False)

    cultivar = ledger.groupby(["cultivar_ascii", "cultivar_original"], observed=True).agg(
        fruit_count=("sample_id", "size"),
        primary_included=("qc_primary_include", "sum"),
        excluded=("qc_consensus_exclude", "sum"),
    ).reset_index()
    cultivar["excluded_fraction"] = cultivar["excluded"] / cultivar["fruit_count"]
    cultivar.to_csv(table_dir / "texture_qc_cultivar_retention.csv", index=False)

    multi_batch_rows: list[dict[str, object]] = []
    for cultivar_name, group in ledger.groupby("cultivar_ascii", observed=True):
        if group["batch_id"].nunique() < 2:
            continue
        for endpoint in ENDPOINTS:
            for cohort, mask in {
                "Full parsed cohort": group["texture_valid_replicates"].ge(2),
                "QC-primary cohort": group["qc_primary_include"],
            }.items():
                subset = group.loc[mask]
                multi_batch_rows.append(
                    {
                        "cultivar_ascii": cultivar_name,
                        "endpoint": endpoint,
                        "label": LABELS[endpoint],
                        "cohort": cohort,
                        "n": int(len(subset)),
                        "batch_eta_squared": eta_squared(subset[f"{endpoint}_mean"], subset["batch_id"]),
                    }
                )
    multi_batch = pd.DataFrame(multi_batch_rows)
    multi_batch.to_csv(table_dir / "texture_qc_multibatch_effects.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    flow_labels = ["All matched", "Hard-valid", "Analysis", "Strict release"]
    flow_values = [
        len(ledger),
        int(ledger["qc_sensitivity_include"].sum()),
        int(ledger["qc_analysis_include"].sum()),
        int(ledger["qc_primary_include"].sum()),
    ]
    colors = ["#4C78A8", "#72B7B2", "#59A14F", "#B07AA1"]
    ax.bar(flow_labels, flow_values, color=colors, width=0.65)
    for idx, value in enumerate(flow_values):
        ax.text(idx, value + len(ledger) * 0.018, f"{value:,}", ha="center", fontweight="bold")
    ax.set_ylim(0, len(ledger) * 1.13)
    ax.set_ylabel("Fruit samples")
    ax.set_title("Auditable cohort flow")
    ax.tick_params(axis="x", rotation=15)
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    top_patterns = patterns.head(9).sort_values("fruit_count")
    ax.barh(top_patterns["evidence_pattern"], top_patterns["fruit_count"], color="#E15759")
    ax.set_xlabel("Excluded fruit samples")
    ax.set_title("Concordant QC evidence")
    add_panel_label(ax, "b")

    ax = axes[0, 2]
    order = impact.sort_values("icc_a1_primary").index.tolist()
    y = np.arange(len(order))
    ax.hlines(y, impact.loc[order, "icc_a1_full"], impact.loc[order, "icc_a1_primary"], color="0.75", lw=1.2)
    ax.scatter(impact.loc[order, "icc_a1_full"], y, color="#F28E2B", s=25, label="Full")
    ax.scatter(analysis.loc[order, "icc_a1"], y, color="#59A14F", s=25, label="High-confidence")
    ax.scatter(impact.loc[order, "icc_a1_primary"], y, color="#4E79A7", s=25, label="Strict 10%")
    ax.set_yticks(y, [LABELS[item] for item in order])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Absolute-agreement ICC(A,1)")
    ax.set_title("Replicate reliability")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "c")

    ax = axes[1, 0]
    batch_plot = batch.sort_values("consensus_excluded_fraction")
    ax.barh(batch_plot["batch_id"], 100 * batch_plot["consensus_excluded_fraction"], color="#B07AA1")
    ax.axvspan(10, 20, color="#59A14F", alpha=0.10)
    ax.set_xlabel("Strict-tier exclusion (%)")
    ax.set_title("Batch retention balance")
    add_panel_label(ax, "d")

    ax = axes[1, 1]
    size = 25 + 160 * batch["consensus_excluded_fraction"]
    speed_deviation_ppm = 1e6 * (batch["median_loading_speed"] - 1.0)
    scatter = ax.scatter(
        speed_deviation_ppm,
        batch["median_baseline_noise_g"],
        s=size,
        c=100 * batch["consensus_excluded_fraction"],
        cmap="magma",
        edgecolor="white",
        linewidth=0.5,
    )
    for _, row in batch.iterrows():
        speed_x = 1e6 * (row["median_loading_speed"] - 1.0)
        ax.annotate(row["batch_id"], (speed_x, row["median_baseline_noise_g"]), fontsize=5, alpha=0.8)
    ax.set_xlabel("Loading-speed deviation from 1 unit/s (ppm)")
    ax.set_ylabel("Batch median baseline noise (g)")
    ax.set_title("Batch acquisition context")
    fig.colorbar(scatter, ax=ax, label="Excluded (%)", shrink=0.75)
    add_panel_label(ax, "e")

    ax = axes[1, 2]
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    order = impact.sort_values("iqr_retained_fraction").index.tolist()
    x = np.arange(len(order))
    ax.scatter(x, impact.loc[order, "iqr_retained_fraction"], color="#4E79A7", label="IQR", s=28)
    ax.scatter(x, impact.loc[order, "range_2_98_retained_fraction"], color="#E15759", label="2-98% range", s=28)
    ax.set_xticks(x, [LABELS[item] for item in order], rotation=65, ha="right")
    ax.set_ylabel("Retained spread / full-cohort spread")
    ax.set_title("Biological range preservation")
    ax.legend(frameon=False)
    add_panel_label(ax, "f")
    fig.savefig(figure_dir / "fig_texture_qc_audit.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / "fig_texture_qc_audit.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "fruit_total": int(len(ledger)),
        "primary_included": int(ledger["qc_primary_include"].sum()),
        "primary_excluded": int(ledger["qc_consensus_exclude"].sum()),
        "primary_excluded_fraction": float(ledger["qc_consensus_exclude"].mean()),
        "median_icc_full": float(before["icc_a1"].median()),
        "median_icc_primary": float(after["icc_a1"].median()),
        "median_icc_high_confidence_analysis": float(analysis["icc_a1"].median()),
        "median_icc_change": float(impact["icc_change"].median()),
        "median_iqr_retained_fraction": float(impact["iqr_retained_fraction"].median()),
        "median_analysis_iqr_retained_fraction": float(analysis_impact["iqr_retained_fraction"].median()),
        "median_2_98_range_retained_fraction": float(impact["range_2_98_retained_fraction"].median()),
        "median_analysis_2_98_range_retained_fraction": float(analysis_impact["range_2_98_retained_fraction"].median()),
        "batch_exclusion_rate_range": [
            float(batch["consensus_excluded_fraction"].min()),
            float(batch["consensus_excluded_fraction"].max()),
        ],
        "median_multibatch_eta_squared_full": float(
            multi_batch.loc[multi_batch["cohort"] == "Full parsed cohort", "batch_eta_squared"].median()
        ),
        "median_multibatch_eta_squared_primary": float(
            multi_batch.loc[multi_batch["cohort"] == "QC-primary cohort", "batch_eta_squared"].median()
        ),
    }
    (output_dir / "texture_qc_impact_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
