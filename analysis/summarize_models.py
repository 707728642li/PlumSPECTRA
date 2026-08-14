from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from train_pls_loco import metrics


MODEL_LABELS = {
    "pls_direct": "Direct PLSR",
    "hierarchical": "Hierarchical PLSR",
    "cnn": "1D-CNN",
    "cnn_texture_aux": "1D-CNN + texture auxiliary",
    "transformer": "Patch Transformer",
}
MODEL_ORDER = ["pls_direct", "hierarchical", "cnn", "cnn_texture_aux", "transformer"]
TARGET_LABELS = {
    "fruit_weight_g": "Fruit weight",
    "soluble_solids_pct": "Soluble solids",
    "ph": "pH",
}
TARGET_UNITS = {"fruit_weight_g": "g", "soluble_solids_pct": "%", "ph": "pH units"}


def stable_seed(*values: object) -> int:
    return int(hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()[:8], 16)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def save_figure(fig: mpl.figure.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_panel(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=11, fontweight="bold", va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", required=True, help="MODEL=predictions.parquet")
    parser.add_argument("--fewshot-dir", type=Path, required=True)
    parser.add_argument("--pls-random-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
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
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_theme(style="ticks", context="paper", rc=mpl.rcParams)
    colors = dict(zip(MODEL_ORDER, sns.color_palette("colorblind", len(MODEL_ORDER))))

    frames: dict[str, pd.DataFrame] = {}
    for specification in args.predictions:
        model, path = specification.split("=", 1)
        frames[model] = pd.read_parquet(Path(path).resolve())

    pooled_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for model, frame in frames.items():
        for target, group in frame.groupby("target"):
            result = metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
            pooled_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "target": target,
                    **result,
                    "normalized_rmse": result["rmse"] / result["y_sd"],
                }
            )
            for cultivar, cultivar_group in group.groupby("cultivar_ascii"):
                fold_result = metrics(cultivar_group["y_true"].to_numpy(), cultivar_group["y_pred"].to_numpy())
                fold_rows.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "target": target,
                        "cultivar_ascii": cultivar,
                        **fold_result,
                        "normalized_rmse": fold_result["rmse"] / fold_result["y_sd"],
                    }
                )
    pooled = pd.DataFrame(pooled_rows)
    folds = pd.DataFrame(fold_rows)
    pooled.to_csv(table_dir / "zero_shot_pooled_metrics.csv", index=False)
    folds.to_csv(table_dir / "zero_shot_cultivar_metrics.csv", index=False)

    fewshot_dir = args.fewshot_dir.resolve()
    fewshot_summary = pd.read_csv(fewshot_dir / "fewshot_summary.csv")
    fewshot_folds = pd.read_parquet(fewshot_dir / "fewshot_fold_metrics.parquet")
    fewshot_summary.to_csv(table_dir / "fewshot_summary.csv", index=False)

    comparison_rows: list[dict[str, object]] = []
    for target in TARGET_LABELS:
        for shots in [0, 5, 10, 20, 50]:
            averaged = (
                fewshot_folds.loc[(fewshot_folds["target"] == target) & (fewshot_folds["shots"] == shots)]
                .groupby(["model", "cultivar_ascii"], as_index=False)["rmse"]
                .mean()
            )
            reference = "pls_direct" if shots == 0 else "hierarchical"
            competitors = [model for model in MODEL_ORDER if model != reference]
            reference_values = averaged.loc[averaged["model"] == reference].set_index("cultivar_ascii")["rmse"]
            for competitor in competitors:
                competitor_values = averaged.loc[averaged["model"] == competitor].set_index("cultivar_ascii")["rmse"]
                paired = pd.concat([reference_values.rename("reference"), competitor_values.rename("competitor")], axis=1).dropna()
                difference = paired["competitor"] - paired["reference"]
                test = wilcoxon(paired["reference"], paired["competitor"], alternative="less", zero_method="wilcox")
                comparison_rows.append(
                    {
                        "target": target,
                        "shots": shots,
                        "reference_model": reference,
                        "competitor_model": competitor,
                        "cultivars": len(paired),
                        "reference_mean_cultivar_rmse": float(paired["reference"].mean()),
                        "competitor_mean_cultivar_rmse": float(paired["competitor"].mean()),
                        "mean_rmse_reduction": float(difference.mean()),
                        "median_rmse_reduction": float(difference.median()),
                        "relative_mean_reduction_pct": float(100 * difference.mean() / paired["competitor"].mean()),
                        "wilcoxon_statistic": float(test.statistic),
                        "p_value_one_sided": float(test.pvalue),
                    }
                )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["p_holm_within_target_shots"] = comparisons.groupby(["target", "shots"])["p_value_one_sided"].transform(holm_adjust)
    comparisons.to_csv(table_dir / "paired_cultivar_model_tests.csv", index=False)

    random_metrics = pd.read_csv(args.pls_random_metrics.resolve())
    random_mean = random_metrics.groupby("target", as_index=False)[["rmse", "r2", "ccc"]].mean()
    random_mean["validation"] = "Random fruit split"
    loco_pls = pooled.loc[pooled["model"] == "pls_direct", ["target", "rmse", "r2", "ccc"]].copy()
    loco_pls["validation"] = "Leave-one-cultivar-out"
    validation_comparison = pd.concat([random_mean, loco_pls], ignore_index=True)
    validation_comparison.to_csv(table_dir / "pls_validation_regime_comparison.csv", index=False)

    target_order = list(TARGET_LABELS)
    pooled["target_label"] = pooled["target"].map(TARGET_LABELS)
    pooled["model"] = pd.Categorical(pooled["model"], categories=MODEL_ORDER, ordered=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    sns.barplot(data=pooled, x="target_label", y="r2", hue="model", palette=colors, ax=ax)
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("Pooled leave-one-cultivar-out $R^2$")
    ax.set_title("Zero-shot predictive accuracy")
    ax.legend_.remove()
    add_panel(ax, "a")
    ax = axes[0, 1]
    sns.barplot(data=pooled, x="target_label", y="normalized_rmse", hue="model", palette=colors, ax=ax)
    ax.axhline(1, color="0.4", lw=0.7, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE / observed SD")
    ax.set_title("Scale-free zero-shot error")
    ax.legend_.remove()
    add_panel(ax, "b")
    ax = axes[1, 0]
    display_validation = validation_comparison.copy()
    display_validation["target_label"] = display_validation["target"].map(TARGET_LABELS)
    sns.barplot(data=display_validation, x="target_label", y="r2", hue="validation", palette=["#4C78A8", "#E45756"], ax=ax)
    ax.axhline(0, color="0.3", lw=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("PLSR $R^2$")
    ax.set_title("Validation design changes the conclusion")
    ax.legend(title="", frameon=False, loc="best")
    add_panel(ax, "c")
    ax = axes[1, 1]
    heatmap = folds.loc[folds["target"] == "soluble_solids_pct"].pivot(index="model", columns="cultivar_ascii", values="normalized_rmse").loc[MODEL_ORDER]
    heatmap.index = [MODEL_LABELS[index] for index in heatmap.index]
    sns.heatmap(heatmap, cmap="rocket_r", vmin=0.5, vmax=2.0, linewidths=0.2, linecolor="white", cbar_kws={"label": "SSC RMSE / cultivar SD", "shrink": 0.75}, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Cultivar-specific SSC transfer error")
    ax.tick_params(axis="x", rotation=70)
    add_panel(ax, "d")
    handles = [mpl.patches.Patch(color=colors[model], label=MODEL_LABELS[model]) for model in MODEL_ORDER]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, figure_dir / "fig04_zero_shot_model_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True, sharex=True)
    for axis, target, panel in zip(axes, target_order, ["a", "b", "c"]):
        for model in MODEL_ORDER:
            group = fewshot_summary.loc[(fewshot_summary["model"] == model) & (fewshot_summary["target"] == target)].sort_values("shots")
            axis.plot(group["shots"], group["r2_mean"], marker="o", ms=3.5, lw=1.3, color=colors[model], label=MODEL_LABELS[model])
            axis.fill_between(group["shots"], group["r2_ci025"], group["r2_ci975"], color=colors[model], alpha=0.12, linewidth=0)
        axis.axhline(0, color="0.5", lw=0.6)
        axis.set_xscale("symlog", linthresh=1, linscale=0.7)
        axis.set_xticks([0, 1, 3, 5, 10, 20, 50])
        axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.set_xlabel("Calibration fruit per new cultivar")
        axis.set_ylabel("Pooled $R^2$")
        axis.set_title(TARGET_LABELS[target])
        add_panel(axis, panel)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, figure_dir / "fig05_fewshot_calibration_curves")

    hierarchical = frames["hierarchical"]
    representative_parts: list[pd.DataFrame] = []
    shots = 10
    repeat = 1
    for (target, cultivar), group in hierarchical.groupby(["target", "cultivar_ascii"]):
        group = group.reset_index(drop=True)
        rng = np.random.default_rng(stable_seed(target, cultivar, shots, repeat, 20260806))
        calibration_positions = rng.choice(len(group), size=shots, replace=False)
        calibration_mask = np.zeros(len(group), dtype=bool)
        calibration_mask[calibration_positions] = True
        intercept = float((group.loc[calibration_mask, "y_true"] - group.loc[calibration_mask, "y_pred"]).mean())
        evaluation = group.loc[~calibration_mask].copy()
        evaluation["y_pred"] += intercept
        representative_parts.append(evaluation)
    representative = pd.concat(representative_parts, ignore_index=True)
    representative.to_parquet(table_dir / "hierarchical_10shot_repeat1_predictions.parquet", index=False, compression="zstd")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    cultivar_names = sorted(representative["cultivar_ascii"].unique())
    cultivar_palette = dict(zip(cultivar_names, sns.color_palette("husl", len(cultivar_names))))
    for axis, target, panel in zip(axes, target_order, ["a", "b", "c"]):
        group = representative.loc[representative["target"] == target]
        for cultivar, cultivar_group in group.groupby("cultivar_ascii"):
            axis.scatter(cultivar_group["y_true"], cultivar_group["y_pred"], s=7, alpha=0.45, color=cultivar_palette[cultivar], linewidths=0)
        lower = float(min(group["y_true"].min(), group["y_pred"].min()))
        upper = float(max(group["y_true"].max(), group["y_pred"].max()))
        axis.plot([lower, upper], [lower, upper], color="0.2", lw=0.8, ls="--")
        result = metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        axis.text(0.04, 0.95, f"$R^2$ = {result['r2']:.3f}\nRMSE = {result['rmse']:.2f} {TARGET_UNITS[target]}", transform=axis.transAxes, va="top", fontsize=7.5)
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")
        axis.set_title(TARGET_LABELS[target])
        add_panel(axis, panel)
    save_figure(fig, figure_dir / "fig06_hierarchical_10shot_predictions")

    top_results = fewshot_summary.loc[(fewshot_summary["model"] == "hierarchical") & (fewshot_summary["shots"].isin([5, 10, 20])), ["target", "shots", "rmse_mean", "rmse_ci025", "rmse_ci975", "r2_mean", "r2_ci025", "r2_ci975", "ccc_mean"]]
    report = {
        "zero_shot_best_by_target_r2": pooled.sort_values("r2", ascending=False).groupby("target", observed=False).first().reset_index()[["target", "model_label", "r2", "rmse"]].to_dict(orient="records"),
        "hierarchical_fewshot_key_results": top_results.to_dict(orient="records"),
        "paired_tests_significant_holm_0_05": int((comparisons["p_holm_within_target_shots"] < 0.05).sum()),
    }
    (output_dir / "model_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
