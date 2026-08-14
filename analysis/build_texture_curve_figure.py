from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

from v2_registry import add_cultivar_code, cultivar_code_map


def loading_curve(curve: pd.Series, features: pd.Series) -> tuple[np.ndarray, np.ndarray, int]:
    time = np.asarray(curve["relative_time_s"], dtype=float)
    force = np.asarray(curve["force_g"], dtype=float)
    position = np.asarray(curve["position_raw"], dtype=float)
    baseline_samples = min(len(force), max(20, int(round(0.2 * float(features["sampling_rate_hz"])))))
    force = force - np.median(force[:baseline_samples])
    contact = int(np.argmin(np.abs(time - float(features["contact_time_s"]))))
    reversal = int(np.argmin(position))
    displacement = np.abs(position[contact : reversal + 1] - position[contact])
    displacement = np.maximum.accumulate(displacement)
    loading_force = force[contact : reversal + 1]
    peak = int(np.argmax(loading_force))
    return displacement, loading_force, peak


def full_cycle(curve: pd.Series, features: pd.Series) -> tuple[np.ndarray, np.ndarray, int, int]:
    time = np.asarray(curve["relative_time_s"], dtype=float)
    force = np.asarray(curve["force_g"], dtype=float)
    baseline_samples = min(len(force), max(20, int(round(0.2 * float(features["sampling_rate_hz"])))))
    force = force - np.median(force[:baseline_samples])
    contact = int(np.argmin(np.abs(time - float(features["contact_time_s"]))))
    reversal = int(np.argmin(np.abs(time - float(features["reversal_time_s"]))))
    return time, force, contact, reversal


def read_selected_curves(curves_dir: Path, selection: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for batch_id, group in selection.groupby("batch_id", observed=True):
        path = curves_dir / f"{batch_id}.parquet"
        table = pq.read_table(path, filters=[("sample_id", "in", group["sample_id"].unique().tolist())])
        parts.append(table.to_pandas())
    return pd.concat(parts, ignore_index=True).sort_values(["sample_id", "replicate"])


def add_panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--curve-features", type=Path, required=True)
    parser.add_argument("--curves-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = add_cultivar_code(pd.read_parquet(args.ledger))
    curve_features = pd.read_parquet(args.curve_features).set_index(["sample_id", "replicate"])
    primary = ledger.loc[ledger["qc_analysis_include"]].copy()
    reference_cultivar = primary["cultivar_ascii"].value_counts().index[0]
    cultivar = primary.loc[primary["cultivar_ascii"].eq(reference_cultivar)].sort_values("skin_break_force_g_mean")
    quantile_rows = []
    for quantile in [0.05, 0.50, 0.95]:
        target = cultivar["skin_break_force_g_mean"].quantile(quantile)
        row = cultivar.iloc[(cultivar["skin_break_force_g_mean"] - target).abs().argsort().iloc[0]].copy()
        row["curve_role"] = f"{int(quantile * 100)}th percentile"
        quantile_rows.append(row)
    typical = quantile_rows[1]
    replicate_outlier = ledger.loc[ledger["qc_consensus_exclude"]].sort_values("replicate_stat", ascending=False).iloc[0].copy()
    replicate_outlier["curve_role"] = "replicate-discordant"
    acquisition_outliers = ledger.loc[ledger["qc_consensus_exclude"]].sort_values("acquisition_stat", ascending=False).head(3).copy()
    acquisition_outliers["curve_role"] = ["technical anomaly 1", "technical anomaly 2", "technical anomaly 3"]
    selection = pd.concat([pd.DataFrame(quantile_rows), pd.DataFrame([replicate_outlier]), acquisition_outliers], ignore_index=True)
    curves = read_selected_curves(args.curves_dir.resolve(), selection)
    selection[["sample_id", "batch_id", "cultivar_ascii", "cultivar_code", "curve_role", "qc_reason"]].to_csv(
        output_dir / "texture_curve_examples.csv", index=False, encoding="utf-8-sig"
    )

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
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), constrained_layout=True)

    typical_curve = curves.loc[(curves["sample_id"] == typical["sample_id"]) & (curves["replicate"] == 1)].iloc[0]
    typical_features = curve_features.loc[(typical["sample_id"], 1)]
    displacement, force, peak = loading_curve(typical_curve, typical_features)
    ax = axes[0, 0]
    ax.plot(displacement, force, color="#4E79A7", lw=1.5)
    ax.fill_between(displacement, 0, np.maximum(force, 0), color="#4E79A7", alpha=0.16, label="LW")
    ax.scatter(displacement[peak], force[peak], color="#E15759", s=32, zorder=4)
    ax.annotate(
        "SRF",
        xy=(displacement[peak], force[peak]),
        xytext=(displacement[peak] + 0.7, force[peak] * 0.84),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    post_mask = displacement > displacement[peak] + 0.75
    if post_mask.any():
        ax.plot(displacement[post_mask], force[post_mask], color="#59A14F", lw=1.2, label="MFF")
    ax.set_xlabel("Displacement (raw position unit)")
    ax.set_ylabel("Force (g)")
    ax.set_title("Mechanically interpretable loading curve")
    ax.legend(frameon=False)
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    time, full_force, contact, reversal = full_cycle(typical_curve, typical_features)
    ax.plot(time, full_force, color="#4E79A7", lw=1.1)
    ax.axvspan(time[contact], time[reversal], color="#F28E2B", alpha=0.14, label="Loading")
    negative = (np.arange(len(time)) >= reversal) & (full_force < 0)
    ax.fill_between(time, full_force, 0, where=negative, color="#B07AA1", alpha=0.35, label="Adhesion")
    ax.axhline(0, color="0.55", lw=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Baseline-corrected force (g)")
    ax.set_title("Complete load-unload cycle")
    ax.legend(frameon=False)
    add_panel_label(ax, "b")

    ax = axes[0, 2]
    colors = ["#59A14F", "#4E79A7", "#E15759"]
    for row, color in zip(quantile_rows, colors):
        curve = curves.loc[(curves["sample_id"] == row["sample_id"]) & (curves["replicate"] == 1)].iloc[0]
        features = curve_features.loc[(row["sample_id"], 1)]
        x, y, _ = loading_curve(curve, features)
        ax.plot(x, y, color=color, lw=1.2, label=f"{row['curve_role']} ({row['skin_break_force_g_mean']:.0f} g)")
    ax.set_xlabel("Displacement (raw position unit)")
    ax.set_ylabel("Force (g)")
    ax.set_title(f"Biological range within {cultivar_code_map()[reference_cultivar]}")
    ax.legend(frameon=False)
    add_panel_label(ax, "c")

    ax = axes[1, 0]
    for replicate, color in [(1, "#4E79A7"), (2, "#F28E2B")]:
        curve = curves.loc[(curves["sample_id"] == typical["sample_id"]) & (curves["replicate"] == replicate)].iloc[0]
        features = curve_features.loc[(typical["sample_id"], replicate)]
        x, y, _ = loading_curve(curve, features)
        ax.plot(x, y, color=color, lw=1.2, label=f"Replicate {replicate}")
    ax.set_xlabel("Displacement (raw position unit)")
    ax.set_ylabel("Force (g)")
    ax.set_title("Typical dual-position agreement")
    ax.legend(frameon=False)
    add_panel_label(ax, "d")

    ax = axes[1, 1]
    for replicate, color in [(1, "#4E79A7"), (2, "#E15759")]:
        curve = curves.loc[(curves["sample_id"] == replicate_outlier["sample_id"]) & (curves["replicate"] == replicate)].iloc[0]
        features = curve_features.loc[(replicate_outlier["sample_id"], replicate)]
        x, y, _ = loading_curve(curve, features)
        ax.plot(x, y, color=color, lw=1.2, label=f"Replicate {replicate}")
    ax.set_xlabel("Displacement (raw position unit)")
    ax.set_ylabel("Force (g)")
    ax.set_title("QC-excluded replicate discordance")
    ax.legend(frameon=False)
    add_panel_label(ax, "e")

    ax = axes[1, 2]
    concordance_features = [
        ("skin_break_force_g", "SRF", "#4E79A7"),
        ("loading_stiffness_g_per_rawpos", "LS", "#F28E2B"),
        ("post_break_work_g_rawpos", "PRW", "#59A14F"),
    ]
    for feature, label, color in concordance_features:
        rep1 = primary[f"rep01_{feature}"].astype(float)
        rep2 = primary[f"rep02_{feature}"].astype(float)
        combined = pd.concat([rep1, rep2], ignore_index=True)
        median_value = combined.median()
        scale = 1.4826 * (combined - median_value).abs().median()
        ax.scatter((rep1 - median_value) / scale, (rep2 - median_value) / scale, s=4, alpha=0.12, color=color, linewidths=0, label=label)
    ax.plot([-4, 6], [-4, 6], color="0.35", ls="--", lw=0.8)
    ax.set_xlim(-4, 6)
    ax.set_ylim(-4, 6)
    ax.set_xlabel("Replicate 1 (robust z)")
    ax.set_ylabel("Replicate 2 (robust z)")
    ax.set_title(f"Dual-position reliability (n={len(primary):,} fruit)")
    ax.legend(frameon=False, markerscale=2.5)
    add_panel_label(ax, "f")

    fig.savefig(output_dir / "fig_texture_curve_mechanics.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "fig_texture_curve_mechanics.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
