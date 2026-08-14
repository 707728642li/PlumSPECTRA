from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from train_texture_pls_loco import regression_metrics
from v2_registry import cultivar_code_map, trait_abbreviation_map


TARGETS = [
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
LABELS = {
    **trait_abbreviation_map(),
    "deformation_compliance": "DC",
    "flesh_resistance_energy": "FRE",
    "skin_rupture_resistance": "SRR",
}
FAMILY_COLORS = {
    "skin_break_force_g_mean": "#E15759",
    "skin_break_displacement_raw_mean": "#E15759",
    "skin_break_drop_g_mean": "#E15759",
    "flesh_force_mean_g_mean": "#59A14F",
    "force_at_6_rawpos_g_mean": "#59A14F",
    "loading_stiffness_g_per_rawpos_mean": "#4E79A7",
    "loading_work_g_rawpos_mean": "#F28E2B",
    "post_break_work_g_rawpos_mean": "#F28E2B",
    "adhesive_force_g_mean": "#B07AA1",
}


def model_metrics(path: Path, model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(path)
    pooled_rows = []
    fold_rows = []
    for target, group in frame.groupby("target", observed=True):
        pooled_rows.append({"model": model, "target": target, **regression_metrics(group["y_true"], group["y_pred"])})
    for (target, cultivar), group in frame.groupby(["target", "cultivar_ascii"], observed=True):
        fold_rows.append(
            {
                "model": model,
                "target": target,
                "cultivar_ascii": cultivar,
                "cultivar_code": cultivar_code_map()[cultivar],
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )
    return frame.assign(model=model), pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)


def add_panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pls", type=Path, required=True)
    parser.add_argument("--cnn", type=Path, required=True)
    parser.add_argument("--transformer", type=Path)
    parser.add_argument("--random-metrics", type=Path, required=True)
    parser.add_argument("--fewshot-pls", type=Path, required=True)
    parser.add_argument("--fewshot-cnn", type=Path)
    parser.add_argument("--axis-predictions", type=Path, required=True)
    parser.add_argument("--axis-fewshot-pls", type=Path, required=True)
    parser.add_argument("--axis-fewshot-cnn", type=Path, required=True)
    parser.add_argument("--multibatch-metrics", type=Path, required=True)
    parser.add_argument("--reliability", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames = []
    pooled_frames = []
    fold_frames = []
    for path, model in [(args.pls, "PLSR"), (args.cnn, "1D-CNN")]:
        prediction, pooled, fold = model_metrics(path, model)
        prediction_frames.append(prediction)
        pooled_frames.append(pooled)
        fold_frames.append(fold)
    if args.transformer and args.transformer.exists():
        prediction, pooled, fold = model_metrics(args.transformer, "Transformer")
        prediction_frames.append(prediction)
        pooled_frames.append(pooled)
        fold_frames.append(fold)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = pd.concat(pooled_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)

    random_metrics = pd.read_csv(args.random_metrics).groupby("target", observed=True).mean(numeric_only=True).reset_index()
    random_metrics["setting"] = "Random-split PLSR"
    loco_pls = pooled.loc[pooled["model"].eq("PLSR")].copy()
    loco_pls["setting"] = "Zero-shot LOCO PLSR"
    loco_cnn = pooled.loc[pooled["model"].eq("1D-CNN")].copy()
    loco_cnn["setting"] = "Zero-shot LOCO CNN"
    fewshot_pls = pd.read_csv(args.fewshot_pls)
    setting_parts = [random_metrics, loco_pls, loco_cnn]
    for shots in [5, 20]:
        part = fewshot_pls.loc[fewshot_pls["shots"].eq(shots), ["target", "r2_mean", "ccc_mean"]].rename(
            columns={"r2_mean": "r2", "ccc_mean": "ccc"}
        )
        part["setting"] = f"{shots}-shot LOCO PLSR"
        setting_parts.append(part)
    settings = pd.concat(setting_parts, ignore_index=True)
    setting_order = ["Random-split PLSR", "Zero-shot LOCO PLSR", "Zero-shot LOCO CNN", "5-shot LOCO PLSR", "20-shot LOCO PLSR"]
    r2_matrix = settings.pivot(index="setting", columns="target", values="r2").reindex(index=setting_order, columns=TARGETS)
    r2_matrix.columns = [LABELS[item] for item in r2_matrix.columns]
    r2_matrix.to_csv(output_dir / "texture_prediction_validation_matrix.csv")
    pooled.to_csv(output_dir / "texture_prediction_model_metrics.csv", index=False)
    folds.to_csv(output_dir / "texture_prediction_fold_metrics.csv", index=False)

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
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    sns.heatmap(
        r2_matrix,
        cmap="vlag",
        center=0,
        vmin=-0.2,
        vmax=0.65,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "R²", "shrink": 0.75},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=65)
    ax.set_title("Validation hierarchy reveals cultivar shift")
    add_panel_label(ax, "a")

    ax = axes[0, 1]
    random_r2 = random_metrics.set_index("target")["r2"]
    loco_r2 = loco_pls.set_index("target")["r2"]
    for target in TARGETS:
        ax.plot([loco_r2[target], random_r2[target]], [LABELS[target], LABELS[target]], color="0.75", lw=1.2)
        ax.scatter(loco_r2[target], LABELS[target], color="#E15759", s=28)
        ax.scatter(random_r2[target], LABELS[target], color="#4E79A7", s=28)
    ax.axvline(0, color="0.5", lw=0.7, ls="--")
    ax.set_xlabel("R²")
    ax.set_title("Interpolation-to-transfer gap")
    ax.scatter([], [], color="#4E79A7", label="Random split")
    ax.scatter([], [], color="#E15759", label="Zero-shot LOCO")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "b")

    ax = axes[0, 2]
    model_order = pooled.groupby("model")["ccc"].mean().sort_values(ascending=False).index.tolist()
    ccc_matrix = pooled.pivot(index="target", columns="model", values="ccc").reindex(index=TARGETS, columns=model_order)
    ccc_matrix.index = [LABELS[item] for item in ccc_matrix.index]
    sns.heatmap(
        ccc_matrix,
        cmap="crest",
        vmin=0,
        vmax=0.55,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "CCC", "shrink": 0.75},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Trait-specific model advantage")
    add_panel_label(ax, "c")

    ax = axes[1, 0]
    for target in TARGETS:
        group = fewshot_pls.loc[fewshot_pls["target"].eq(target)].sort_values("shots")
        ax.plot(group["shots"], group["r2_mean"], marker="o", ms=3, lw=1.2, color=FAMILY_COLORS[target], alpha=0.85, label=LABELS[target])
    ax.axhline(0, color="0.5", lw=0.7, ls="--")
    ax.set_xlabel("Calibration fruit per new cultivar")
    ax.set_ylabel("Held-out evaluation R²")
    ax.set_title("Rapid few-shot recovery")
    add_panel_label(ax, "d")

    ax = axes[1, 1]
    best_model_by_target = pooled.loc[pooled.groupby("target")["ccc"].idxmax(), ["target", "model"]].set_index("target")["model"]
    chosen_folds = []
    for target, model in best_model_by_target.items():
        chosen_folds.append(folds.loc[folds["target"].eq(target) & folds["model"].eq(model)])
    chosen = pd.concat(chosen_folds)
    fold_matrix = chosen.pivot(index="cultivar_code", columns="target", values="ccc").reindex(columns=TARGETS)
    fold_matrix.columns = [LABELS[item] for item in fold_matrix.columns]
    sns.heatmap(
        fold_matrix,
        cmap="vlag",
        center=0,
        vmin=-0.4,
        vmax=0.8,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Held-out cultivar CCC", "shrink": 0.75},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=65)
    ax.set_title("Cultivar-specific transfer landscape")
    add_panel_label(ax, "e")

    ax = axes[1, 2]
    axis_predictions = pd.read_parquet(args.axis_predictions)
    deformation = axis_predictions.loc[
        axis_predictions["model"].eq("1D-CNN") & axis_predictions["target"].eq("deformation_compliance")
    ]
    extent = [
        float(min(deformation["y_true"].quantile(0.005), deformation["y_pred"].quantile(0.005))),
        float(max(deformation["y_true"].quantile(0.995), deformation["y_pred"].quantile(0.995))),
    ]
    hb = ax.hexbin(deformation["y_true"], deformation["y_pred"], gridsize=45, mincnt=1, cmap="viridis", bins="log")
    ax.plot(extent, extent, color="white", lw=1.0, ls="--")
    axis_metric = regression_metrics(deformation["y_true"], deformation["y_pred"])
    ax.text(0.04, 0.94, f"R²={axis_metric['r2']:.2f}\nCCC={axis_metric['ccc']:.2f}", transform=ax.transAxes, va="top", color="white", fontweight="bold")
    ax.set_xlabel("Observed deformation-compliance")
    ax.set_ylabel("Predicted deformation-compliance")
    ax.set_title("Best zero-shot mechanical axis")
    fig.colorbar(hb, ax=ax, label="log10 fruit density", shrink=0.75)
    add_panel_label(ax, "f")
    fig.savefig(output_dir / "fig_texture_prediction_transfer.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "fig_texture_prediction_transfer.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Axis-focused transfer figure.
    axis_fewshot = pd.concat(
        [
            pd.read_csv(args.axis_fewshot_pls).assign(model="PLSR"),
            pd.read_csv(args.axis_fewshot_cnn).assign(model="1D-CNN"),
        ],
        ignore_index=True,
    )
    axis_metric_rows = []
    for (model, target), group in axis_predictions.groupby(["model", "target"], observed=True):
        axis_metric_rows.append({"model": model, "target": target, **regression_metrics(group["y_true"], group["y_pred"])})
    axis_metrics = pd.DataFrame(axis_metric_rows)
    reliability = pd.read_csv(args.reliability)
    endpoint_best = pooled.loc[pooled.groupby("target")["r2"].idxmax(), ["target", "r2", "model"]]
    reliability_primary = reliability.loc[
        reliability["cohort"].eq("High-confidence analysis cohort"), ["endpoint", "icc_a1"]
    ].copy()
    reliability_primary["target"] = reliability_primary["endpoint"] + "_mean"
    endpoint_best = endpoint_best.merge(
        reliability_primary[["target", "icc_a1"]],
        left_on="target",
        right_on="target",
    )
    multibatch = pd.read_csv(args.multibatch_metrics)
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 3.9), constrained_layout=True)
    ax = axes[0]
    zero_axis = axis_metrics.pivot(index="target", columns="model", values="r2")
    zero_axis.index = [LABELS[item] for item in zero_axis.index]
    sns.heatmap(zero_axis, cmap="crest", vmin=0, vmax=0.45, annot=True, fmt=".2f", cbar_kws={"label": "Zero-shot R²", "shrink": 0.75}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Fold-safe mechanical axes")
    add_panel_label(ax, "a")
    ax = axes[1]
    styles = {"PLSR": "-", "1D-CNN": "--"}
    axis_colors = {"deformation_compliance": "#4E79A7", "flesh_resistance_energy": "#59A14F", "skin_rupture_resistance": "#E15759"}
    for (model, target), group in axis_fewshot.groupby(["model", "target"], observed=True):
        ax.plot(group["shots"], group["r2_mean"], styles[model], marker="o", ms=3, color=axis_colors[target], label=f"{LABELS[target]} ({model})")
    ax.set_xlabel("Calibration fruit per new cultivar")
    ax.set_ylabel("Held-out evaluation R²")
    ax.set_title("Few-shot texture-axis transfer")
    ax.legend(frameon=False, fontsize=6)
    add_panel_label(ax, "b")
    ax = axes[2]
    for _, row in endpoint_best.iterrows():
        ax.scatter(row["icc_a1"], row["r2"], s=55, color=FAMILY_COLORS[row["target"]], edgecolor="white", linewidth=0.5)
        ax.annotate(LABELS[row["target"]], (row["icc_a1"], row["r2"]), xytext=(3, 2), textcoords="offset points", fontsize=6)
    ax.axhline(0, color="0.5", lw=0.7, ls="--")
    ax.set_xlabel("Duplicate-penetration ICC(A,1)")
    ax.set_ylabel("Best-model zero-shot R²")
    ax.set_title("Reliable measurement ≠ universal transfer")
    add_panel_label(ax, "c")
    fig.savefig(output_dir / "fig_texture_axis_transfer.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / "fig_texture_axis_transfer.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
