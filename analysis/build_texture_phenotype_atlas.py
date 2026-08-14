from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from v2_registry import add_cultivar_code, cultivar_code_map, trait_abbreviation_map


ENDPOINTS = [
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
LABELS = trait_abbreviation_map()
AXIS_NAMES = ["Flesh resistance-energy", "Deformation-compliance", "Skin rupture resistance"]
AXIS_LABELS = dict(zip(AXIS_NAMES, ["FRE", "DC", "SRR"], strict=True))


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 100, tolerance: float = 1e-6) -> np.ndarray:
    rows, columns = loadings.shape
    rotation = np.eye(columns)
    objective = 0.0
    for _ in range(max_iter):
        previous = objective
        rotated = loadings @ rotation
        u, singular, vh = np.linalg.svd(
            loadings.T
            @ (rotated**3 - (gamma / rows) * rotated @ np.diag(np.diag(rotated.T @ rotated)))
        )
        rotation = u @ vh
        objective = float(singular.sum())
        if previous and objective / previous < 1.0 + tolerance:
            break
    return rotation


def centered_within(values: pd.Series, groups: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric - numeric.groupby(groups, observed=True).transform("median")


def add_panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    ledger = add_cultivar_code(pd.read_parquet(args.ledger))
    primary = ledger.loc[ledger["qc_analysis_include"]].copy()

    median = primary[ENDPOINTS].median()
    iqr = primary[ENDPOINTS].quantile(0.75) - primary[ENDPOINTS].quantile(0.25)
    scaled = ((primary[ENDPOINTS] - median) / iqr.replace(0, 1.0)).fillna(0.0)
    pca = PCA(n_components=3, random_state=20260806)
    pca_scores = pca.fit_transform(scaled)
    pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    rotation = varimax(pca_loadings)
    rotated_loadings = pca_loadings @ rotation
    rotated_weights = pca.components_.T @ rotation
    rotated_scores = scaled.to_numpy() @ rotated_weights

    # Match factors to preregistered mechanical meanings without using NIR predictions.
    flesh_indices = [ENDPOINTS.index(item) for item in [
        "flesh_force_mean_g_mean", "force_at_6_rawpos_g_mean", "loading_work_g_rawpos_mean",
        "post_break_work_g_rawpos_mean", "adhesive_force_g_mean"
    ]]
    deformation_index = ENDPOINTS.index("skin_break_displacement_raw_mean")
    flesh_factor = int(np.argmax(np.abs(rotated_loadings[flesh_indices]).sum(axis=0)))
    remaining = [index for index in range(3) if index != flesh_factor]
    deformation_factor = remaining[int(np.argmax(np.abs(rotated_loadings[deformation_index, remaining])))]
    rupture_factor = next(index for index in remaining if index != deformation_factor)
    order = [flesh_factor, deformation_factor, rupture_factor]
    rotated_loadings = rotated_loadings[:, order]
    rotated_scores = rotated_scores[:, order]
    orientation_features = [
        "post_break_work_g_rawpos_mean",
        "skin_break_displacement_raw_mean",
        "skin_break_force_g_mean",
    ]
    for axis, feature in enumerate(orientation_features):
        feature_index = ENDPOINTS.index(feature)
        if rotated_loadings[feature_index, axis] < 0:
            rotated_loadings[:, axis] *= -1
            rotated_scores[:, axis] *= -1

    loadings = pd.DataFrame(rotated_loadings, index=[LABELS[item] for item in ENDPOINTS], columns=AXIS_NAMES)
    score_variance = np.var(rotated_scores, axis=0, ddof=1)
    explained = score_variance / float(np.var(scaled.to_numpy(), axis=0, ddof=1).sum())
    axes = primary[["sample_id", "batch_id", "cultivar_ascii", "cultivar_code", "cultivar_original"]].copy()
    for index, axis_name in enumerate(AXIS_NAMES):
        axes[axis_name] = rotated_scores[:, index]
    axes.to_parquet(table_dir / "descriptive_texture_axis_scores.parquet", index=False, compression="zstd")
    loadings.to_csv(table_dir / "descriptive_texture_axis_loadings.csv")
    pd.DataFrame(
        {"axis": AXIS_NAMES, "rotated_score_variance_fraction": explained}
    ).to_csv(table_dir / "descriptive_texture_axis_variance.csv", index=False)
    pd.DataFrame(
        {
            "endpoint": ENDPOINTS,
            "training_fold_scaling": "median/IQR fitted on outer-training fruit only",
            "flesh_resistance_energy_weight": [0, 0, 0, 1, 1, 0, 1, 1, 1],
            "deformation_compliance_weight": [0, 1, 0, 0, 0, -1, 0, 0, 0],
            "skin_rupture_resistance_weight": [1, 0, 1, 0, 0, 1, 1, 0, 0],
        }
    ).to_csv(table_dir / "fold_safe_texture_axis_registry.csv", index=False)

    counts = ledger.groupby(["cultivar_ascii", "cultivar_original"], observed=True).agg(
        all_fruit=("sample_id", "size"),
        primary_fruit=("qc_analysis_include", "sum"),
    ).reset_index()
    counts["qc_excluded"] = counts["all_fruit"] - counts["primary_fruit"]
    counts["cultivar_code"] = counts["cultivar_ascii"].map(cultivar_code_map())
    counts = counts.sort_values("all_fruit")
    counts.to_csv(table_dir / "texture_atlas_cultivar_counts.csv", index=False)

    cultivar_order = counts["cultivar_ascii"].tolist()
    endpoint_z = (primary[ENDPOINTS] - primary[ENDPOINTS].median()) / (
        1.4826 * (primary[ENDPOINTS] - primary[ENDPOINTS].median()).abs().median()
    ).replace(0, 1.0)
    fingerprint = endpoint_z.assign(cultivar_ascii=primary["cultivar_ascii"]).groupby("cultivar_ascii", observed=True).median()
    fingerprint = fingerprint.loc[cultivar_order]
    fingerprint.index = fingerprint.index.map(cultivar_code_map())
    fingerprint.columns = [LABELS[item] for item in ENDPOINTS]
    fingerprint.to_csv(table_dir / "texture_cultivar_robust_fingerprint.csv")

    traits = ENDPOINTS + ["fruit_weight_g", "soluble_solids_pct", "ph"]
    centered = pd.DataFrame(
        {trait: centered_within(primary[trait], primary["cultivar_ascii"]) for trait in traits}
    )
    correlation = centered.corr(method="spearman")
    correlation.index = [LABELS[item] for item in traits]
    correlation.columns = [LABELS[item] for item in traits]
    correlation.to_csv(table_dir / "within_cultivar_trait_correlations.csv")

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
    palette_values = sns.color_palette("husl", n_colors=len(cultivar_order))
    palette = dict(zip(cultivar_order, palette_values))
    fig, axes_plot = plt.subplots(2, 3, figsize=(14.4, 8.8), constrained_layout=True)

    ax = axes_plot[0, 0]
    ax.barh(counts["cultivar_code"], counts["primary_fruit"], color="#4E79A7", label="Analysis cohort")
    ax.barh(
        counts["cultivar_code"], counts["qc_excluded"], left=counts["primary_fruit"], color="#E15759", label="QC-excluded"
    )
    ax.set_xlabel("Fruit samples")
    ax.set_title(f"Large-scale mechanical phenotyping (n={len(ledger):,})")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "a")

    ax = axes_plot[0, 1]
    long_z = endpoint_z.copy()
    long_z.columns = [LABELS[item] for item in ENDPOINTS]
    long_z = long_z.melt(var_name="endpoint", value_name="robust_z")
    sns.violinplot(data=long_z, x="endpoint", y="robust_z", color="#76B7B2", inner="quart", cut=0, linewidth=0.6, ax=ax)
    ax.set_ylim(-4, 6)
    ax.set_xlabel("")
    ax.set_ylabel("Robust standardized value")
    ax.tick_params(axis="x", rotation=70, labelsize=6)
    ax.set_title("Mechanical phenotype distributions")
    add_panel_label(ax, "b")

    ax = axes_plot[0, 2]
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-0.65,
        vmax=0.65,
        square=True,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Within-cultivar Spearman rho", "shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Trait association atlas")
    ax.tick_params(axis="x", rotation=90, labelsize=5.5)
    ax.tick_params(axis="y", labelsize=6.5)
    add_panel_label(ax, "c")

    ax = axes_plot[1, 0]
    sns.heatmap(
        fingerprint,
        cmap="vlag",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Cultivar median robust z", "shrink": 0.7},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Cultivar mechanical fingerprints")
    ax.tick_params(axis="x", rotation=65)
    add_panel_label(ax, "d")

    ax = axes_plot[1, 1]
    sns.heatmap(
        loadings.rename(columns=AXIS_LABELS),
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6},
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Varimax loading", "shrink": 0.7},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Three-axis texture representation ({100 * pca.explained_variance_ratio_.sum():.1f}% variance)")
    ax.tick_params(axis="x", rotation=25)
    add_panel_label(ax, "e")

    ax = axes_plot[1, 2]
    for cultivar in cultivar_order:
        mask_cultivar = axes["cultivar_ascii"].eq(cultivar).to_numpy()
        ax.scatter(
            axes.loc[mask_cultivar, AXIS_NAMES[0]],
            axes.loc[mask_cultivar, AXIS_NAMES[2]],
            s=7,
            alpha=0.45,
            color=palette[cultivar],
            linewidths=0,
            label=cultivar_code_map()[cultivar],
        )
    ax.set_xlabel(AXIS_LABELS[AXIS_NAMES[0]])
    ax.set_ylabel(AXIS_LABELS[AXIS_NAMES[2]])
    ax.set_title("Fruit-level texture landscape")
    add_panel_label(ax, "f")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False)
    fig.savefig(figure_dir / "fig_texture_phenotype_atlas.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / "fig_texture_phenotype_atlas.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "fruit_total": int(len(ledger)),
        "analysis_fruit": int(len(primary)),
        "texture_curve_total": int(len(ledger) * 2),
        "cultivar_labels": int(ledger["cultivar_ascii"].nunique()),
        "batches": int(ledger["batch_id"].nunique()),
        "three_component_variance_fraction": float(pca.explained_variance_ratio_.sum()),
        "axis_names": AXIS_NAMES,
        "note": "Descriptive axes use the high-confidence-QC analysis cohort. Predictive axis evaluation must refit scaling within each outer training fold.",
    }
    (output_dir / "texture_atlas_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
