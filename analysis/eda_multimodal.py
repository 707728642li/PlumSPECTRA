from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


TARGETS = {
    "fruit_weight_g": "Fruit weight (g)",
    "soluble_solids_pct": "Soluble solids (%)",
    "ph": "pH",
}
TEXTURE_BASES = [
    "skin_break_force_g",
    "skin_break_displacement_raw",
    "skin_break_drop_g",
    "max_loading_force_g",
    "force_at_2_rawpos_g",
    "force_at_4_rawpos_g",
    "force_at_6_rawpos_g",
    "flesh_force_mean_g",
    "loading_stiffness_g_per_rawpos",
    "loading_work_g_rawpos",
    "post_break_work_g_rawpos",
    "adhesive_force_g",
    "fracture_peak_count",
]
TEXTURE_LABELS = {
    "skin_break_force_g": "Skin-break force",
    "skin_break_displacement_raw": "Skin-break displacement",
    "skin_break_drop_g": "Skin-break force drop",
    "max_loading_force_g": "Maximum loading force",
    "force_at_2_rawpos_g": "Force at 2 position units",
    "force_at_4_rawpos_g": "Force at 4 position units",
    "force_at_6_rawpos_g": "Force at 6 position units",
    "flesh_force_mean_g": "Mean flesh force",
    "loading_stiffness_g_per_rawpos": "Loading stiffness",
    "loading_work_g_rawpos": "Loading work",
    "post_break_work_g_rawpos": "Post-break work",
    "adhesive_force_g": "Adhesive force",
    "fracture_peak_count": "Fracture peak count",
}


def finite_pair(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    pair = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    return pair["a"].to_numpy(float), pair["b"].to_numpy(float)


def safe_correlation(a: pd.Series, b: pd.Series, method: str) -> tuple[float, float, int]:
    x, y = finite_pair(a, b)
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, len(x)
    result = spearmanr(x, y) if method == "spearman" else pearsonr(x, y)
    return float(result.statistic), float(result.pvalue), len(x)


def icc_absolute_agreement(rep1: pd.Series, rep2: pd.Series) -> float:
    x, y = finite_pair(rep1, rep2)
    values = np.column_stack([x, y])
    n, k = values.shape
    if n < 3:
        return np.nan
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_columns = n * np.sum((column_means - grand) ** 2)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ss_error = np.sum(residual**2)
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator != 0 else np.nan


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": groups}).dropna()
    if frame.empty:
        return np.nan
    grand = frame["value"].mean()
    total = np.sum((frame["value"] - grand) ** 2)
    between = sum(len(group) * (group["value"].mean() - grand) ** 2 for _, group in frame.groupby("group"))
    return float(between / total) if total > 0 else np.nan


def centered_within_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric - numeric.groupby(groups).transform("median")


def add_panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


def save_figure(fig: mpl.figure.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_theme(style="ticks", context="paper", rc=mpl.rcParams)

    master = pd.read_parquet(multimodal_dir / "master_samples.parquet")
    cultivar_counts = (
        master.groupby(["cultivar_ascii", "cultivar_original"])
        .agg(total=("sample_id", "size"), nir_valid=("nir_c_valid", "sum"))
        .reset_index()
        .sort_values("nir_valid", ascending=True)
    )
    cultivar_order = cultivar_counts["cultivar_ascii"].tolist()
    palette_values = sns.color_palette("colorblind", n_colors=len(cultivar_order))
    palette = dict(zip(cultivar_order, palette_values))

    target_summary_rows: list[dict[str, object]] = []
    variance_rows: list[dict[str, object]] = []
    for target in TARGETS:
        valid = master[f"{target}_valid"]
        for cultivar, group in master.loc[valid].groupby("cultivar_ascii"):
            values = group[target]
            target_summary_rows.append(
                {
                    "target": target,
                    "cultivar_ascii": cultivar,
                    "n": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "sd": float(values.std()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
        variance_rows.append(
            {
                "target": target,
                "cultivar_eta_squared": eta_squared(master.loc[valid, target], master.loc[valid, "cultivar_ascii"]),
                "n": int(valid.sum()),
            }
        )
    pd.DataFrame(target_summary_rows).to_csv(table_dir / "cultivar_target_summary.csv", index=False)
    pd.DataFrame(variance_rows).to_csv(table_dir / "phenotype_cultivar_effects.csv", index=False)

    reliability_rows: list[dict[str, object]] = []
    texture_effect_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    dual = master.loc[master["texture_dual_valid"]].copy()
    for base in TEXTURE_BASES:
        rep1 = dual[f"rep01_{base}"]
        rep2 = dual[f"rep02_{base}"]
        pearson, pearson_p, n = safe_correlation(rep1, rep2, "pearson")
        spearman, spearman_p, _ = safe_correlation(rep1, rep2, "spearman")
        mean_absdiff = float((rep1 - rep2).abs().mean())
        reliability_rows.append(
            {
                "feature": base,
                "label": TEXTURE_LABELS[base],
                "n": n,
                "icc_a1": icc_absolute_agreement(rep1, rep2),
                "pearson_r": pearson,
                "pearson_p": pearson_p,
                "spearman_rho": spearman,
                "spearman_p": spearman_p,
                "mean_absolute_difference": mean_absdiff,
                "median_cv": float(dual[f"{base}_cv"].replace([np.inf, -np.inf], np.nan).median()),
            }
        )
        mean_column = f"{base}_mean"
        texture_effect_rows.append(
            {
                "feature": base,
                "label": TEXTURE_LABELS[base],
                "cultivar_eta_squared": eta_squared(dual[mean_column], dual["cultivar_ascii"]),
                "n": int(dual[mean_column].notna().sum()),
            }
        )
        for target in TARGETS:
            valid = dual[f"{target}_valid"]
            global_r, global_p, n_global = safe_correlation(dual.loc[valid, mean_column], dual.loc[valid, target], "spearman")
            feature_centered = centered_within_group(dual.loc[valid, mean_column], dual.loc[valid, "cultivar_ascii"])
            target_centered = centered_within_group(dual.loc[valid, target], dual.loc[valid, "cultivar_ascii"])
            within_r, within_p, n_within = safe_correlation(feature_centered, target_centered, "spearman")
            association_rows.append(
                {
                    "feature": base,
                    "target": target,
                    "global_spearman_rho": global_r,
                    "global_p": global_p,
                    "within_cultivar_centered_spearman_rho": within_r,
                    "within_cultivar_p": within_p,
                    "n": min(n_global, n_within),
                }
            )
    reliability = pd.DataFrame(reliability_rows).sort_values("icc_a1", ascending=False)
    texture_effects = pd.DataFrame(texture_effect_rows).sort_values("cultivar_eta_squared", ascending=False)
    associations = pd.DataFrame(association_rows)
    reliability.to_csv(table_dir / "texture_replicate_reliability.csv", index=False)
    texture_effects.to_csv(table_dir / "texture_cultivar_effects.csv", index=False)
    associations.to_csv(table_dir / "texture_target_associations.csv", index=False)

    absorbance = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelengths = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    spectral_meta = row_index.merge(
        master[["sample_id", "cultivar_ascii", "batch_id", *TARGETS.keys(), *[f"{target}_valid" for target in TARGETS]]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    snv = (absorbance - absorbance.mean(axis=1, keepdims=True)) / absorbance.std(axis=1, ddof=1, keepdims=True)
    pca = PCA(n_components=20, random_state=20260806)
    scores = pca.fit_transform(snv)
    for component in range(scores.shape[1]):
        spectral_meta[f"snv_pc{component + 1}"] = scores[:, component]
    spectral_meta.to_parquet(table_dir / "spectral_pca_scores.parquet", index=False, compression="zstd")
    pca_effects = [
        {
            "component": i + 1,
            "explained_variance_ratio": float(pca.explained_variance_ratio_[i]),
            "cultivar_eta_squared": eta_squared(pd.Series(scores[:, i]), spectral_meta["cultivar_ascii"]),
        }
        for i in range(scores.shape[1])
    ]
    pd.DataFrame(pca_effects).to_csv(table_dir / "spectral_pca_cultivar_effects.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.barh(cultivar_counts["cultivar_ascii"], cultivar_counts["nir_valid"], color=[palette[c] for c in cultivar_order])
    ax.set_xlabel("Valid NIR fruit samples")
    ax.set_ylabel("")
    ax.set_title("Cohort composition")
    add_panel_label(ax, "a")
    for panel_index, (target, label) in enumerate(TARGETS.items(), start=1):
        ax = axes.flat[panel_index]
        plot_data = master.loc[master[f"{target}_valid"]]
        sns.boxplot(
            data=plot_data,
            x="cultivar_ascii",
            y=target,
            order=cultivar_order,
            palette=palette,
            hue="cultivar_ascii",
            legend=False,
            showfliers=False,
            linewidth=0.6,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=70)
        ax.set_title(label + " diversity")
        add_panel_label(ax, chr(ord("a") + panel_index))
    ax = axes[1, 1]
    for cultivar in cultivar_order:
        idx = spectral_meta["cultivar_ascii"].eq(cultivar).to_numpy()
        ax.scatter(scores[idx, 0], scores[idx, 1], s=5, alpha=0.45, color=palette[cultivar], label=cultivar, linewidths=0)
    ax.set_xlabel(f"SNV-PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"SNV-PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Spectral population structure")
    add_panel_label(ax, "e")
    ax = axes[1, 2]
    for cultivar in cultivar_order:
        idx = spectral_meta["cultivar_ascii"].eq(cultivar).to_numpy()
        ax.plot(wavelengths, np.median(absorbance[idx], axis=0), color=palette[cultivar], lw=0.9, alpha=0.9, label=cultivar)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorbance (a.u.)")
    ax.set_title("Cultivar median spectra")
    add_panel_label(ax, "f")
    handles, labels = axes[1, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False)
    save_figure(fig, figure_dir / "fig02_phenotypic_spectral_diversity")

    reliability_plot = reliability.sort_values("icc_a1", ascending=True)
    cultivar_medians = dual.groupby("cultivar_ascii")[[f"{base}_mean" for base in TEXTURE_BASES]].median()
    cultivar_medians.columns = [TEXTURE_LABELS[base] for base in TEXTURE_BASES]
    standardized = (cultivar_medians - cultivar_medians.mean(axis=0)) / cultivar_medians.std(axis=0, ddof=1)
    standardized = standardized.loc[cultivar_order]
    standardized.to_csv(table_dir / "texture_cultivar_standardized_medians.csv")
    within_matrix = associations.pivot(index="feature", columns="target", values="within_cultivar_centered_spearman_rho").loc[TEXTURE_BASES]
    global_matrix = associations.pivot(index="feature", columns="target", values="global_spearman_rho").loc[TEXTURE_BASES]

    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.8), constrained_layout=True)
    ax = axes[0, 0]
    colors = ["#4C78A8" if value >= 0.75 else "#F2A541" if value >= 0.5 else "#D1495B" for value in reliability_plot["icc_a1"]]
    ax.barh(reliability_plot["label"], reliability_plot["icc_a1"], color=colors)
    ax.axvline(0.5, color="0.45", lw=0.7, ls="--")
    ax.axvline(0.75, color="0.45", lw=0.7, ls=":")
    ax.set_xlim(min(-0.05, reliability_plot["icc_a1"].min() - 0.05), 1.0)
    ax.set_xlabel("Absolute-agreement ICC(A,1)")
    ax.set_title("Two-position texture reliability")
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    sns.heatmap(
        standardized,
        cmap="vlag",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Cultivar median z-score", "shrink": 0.7},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Cultivar texture fingerprints")
    ax.tick_params(axis="x", rotation=70)
    add_panel_label(ax, "b")

    for ax, matrix, title, panel in [
        (axes[1, 0], global_matrix, "Global texture–quality associations", "c"),
        (axes[1, 1], within_matrix, "Within-cultivar associations", "d"),
    ]:
        display = matrix.copy()
        display.index = [TEXTURE_LABELS[index] for index in display.index]
        display.columns = [TARGETS[column] for column in display.columns]
        sns.heatmap(
            display,
            cmap="vlag",
            center=0,
            vmin=-0.6,
            vmax=0.6,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 6.5},
            linewidths=0.25,
            linecolor="white",
            cbar_kws={"label": "Spearman rho", "shrink": 0.7},
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(title)
        add_panel_label(ax, panel)
    save_figure(fig, figure_dir / "fig03_texture_reliability_biology")

    summary = {
        "samples": int(len(master)),
        "valid_nir_c": int(master["nir_c_valid"].sum()),
        "dual_texture": int(master["texture_dual_valid"].sum()),
        "cultivars": int(master["cultivar_ascii"].nunique()),
        "target_cultivar_eta_squared": {row["target"]: row["cultivar_eta_squared"] for row in variance_rows},
        "spectral_pca_explained_variance_first_2": float(pca.explained_variance_ratio_[:2].sum()),
        "spectral_pca_explained_variance_first_20": float(pca.explained_variance_ratio_.sum()),
        "spectral_pc1_cultivar_eta_squared": float(pca_effects[0]["cultivar_eta_squared"]),
        "texture_icc_median": float(reliability["icc_a1"].median()),
        "texture_icc_range": [float(reliability["icc_a1"].min()), float(reliability["icc_a1"].max())],
        "texture_cultivar_eta_squared_median": float(texture_effects["cultivar_eta_squared"].median()),
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
