from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


DEFAULT_DEVELOPMENT_CODES = ["L313", "CHL", "KLD", "WW", "WX"]


def summarize(frame: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (trait, model), group in frame.groupby(["trait", "model"], observed=True):
        work = group.copy()
        work["residual"] = work["y_pred"] - work["y_true"]
        fold_bias = work.groupby("cultivar_code")["residual"].transform("mean")
        work["residual_centered"] = work["residual"] - fold_bias
        work["y_pred_oracle_intercept"] = work["y_pred"] - fold_bias
        total_sse = float(np.sum(work["residual"] ** 2))
        bias_sse = float(np.sum(fold_bias**2))
        within_sse = float(np.sum(work["residual_centered"] ** 2))
        rmse = float(np.sqrt(total_sse / len(work)))
        oracle_rmse = float(np.sqrt(mean_squared_error(work["y_true"], work["y_pred_oracle_intercept"])))
        true_centered = work["y_true"] - work.groupby("cultivar_code")["y_true"].transform("mean")
        pred_centered = work["y_pred"] - work.groupby("cultivar_code")["y_pred"].transform("mean")
        rows.append(
            {
                "scope": scope,
                "trait": trait,
                "model": model,
                "n": int(len(work)),
                "cultivars": int(work["cultivar_code"].nunique()),
                "rmse": rmse,
                "r2": float(r2_score(work["y_true"], work["y_pred"])),
                "centred_r2": float(r2_score(true_centered, pred_centered)),
                "oracle_intercept_rmse": oracle_rmse,
                "oracle_intercept_r2": float(r2_score(work["y_true"], work["y_pred_oracle_intercept"])),
                "rmse_recoverable_by_intercept_pct": 100.0 * (rmse - oracle_rmse) / rmse,
                "bias_sse_fraction": bias_sse / total_sse if total_sse > 0 else np.nan,
                "within_sse_fraction": within_sse / total_sse if total_sse > 0 else np.nan,
                "sse_decomposition_error": (bias_sse + within_sse - total_sse) / max(total_sse, 1e-12),
            }
        )
    return rows


def fold_metrics(frame: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (trait, model, cultivar_code), group in frame.groupby(
        ["trait", "model", "cultivar_code"], observed=True
    ):
        residual = group["y_pred"].to_numpy(float) - group["y_true"].to_numpy(float)
        bias = float(np.mean(residual))
        rows.append(
            {
                "scope": scope,
                "trait": trait,
                "model": model,
                "cultivar_code": cultivar_code,
                "n": int(len(group)),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "bias": bias,
                "centred_rmse": float(np.sqrt(np.mean((residual - bias) ** 2))),
                "absolute_bias_fraction_of_rmse": abs(bias) / max(float(np.sqrt(np.mean(residual**2))), 1e-12),
            }
        )
    return rows


def residual_correlations(frame: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trait, group in frame.groupby("trait", observed=True):
        wide = group.assign(residual=group["y_pred"] - group["y_true"]).pivot(
            index="sample_id", columns="model", values="residual"
        )
        correlation = wide.corr()
        for left in correlation.index:
            for right in correlation.columns:
                if str(left) >= str(right):
                    continue
                rows.append(
                    {
                        "scope": scope,
                        "trait": trait,
                        "model_left": left,
                        "model_right": right,
                        "residual_pearson_r": float(correlation.loc[left, right]),
                        "paired_samples": int(wide[[left, right]].dropna().shape[0]),
                    }
                )
    return rows


def anchor_oracle(frame: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    anchors = frame.loc[frame["model"].isin(["PLSR", "Ridge"])]
    for (trait, cultivar_code), group in anchors.groupby(["trait", "cultivar_code"], observed=True):
        metrics = []
        for model, model_group in group.groupby("model", observed=True):
            rmse = float(np.sqrt(mean_squared_error(model_group["y_true"], model_group["y_pred"])))
            metrics.append((model, rmse))
        if len(metrics) != 2:
            continue
        metrics.sort(key=lambda item: item[1])
        rows.append(
            {
                "scope": scope,
                "trait": trait,
                "cultivar_code": cultivar_code,
                "oracle_best_anchor": metrics[0][0],
                "oracle_best_rmse": metrics[0][1],
                "other_anchor": metrics[1][0],
                "other_rmse": metrics[1][1],
                "oracle_anchor_gain_pct": 100.0 * (metrics[1][1] - metrics[0][1]) / metrics[1][1],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-codes", default=",".join(DEFAULT_DEVELOPMENT_CODES))
    args = parser.parse_args()

    predictions = pd.read_parquet(args.predictions.resolve())
    required = {"sample_id", "trait", "model", "cultivar_code", "y_true", "y_pred"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(predictions.columns))}")
    development_codes = [value.strip().upper() for value in args.development_codes.split(",")]
    development = predictions.loc[predictions["cultivar_code"].isin(development_codes)].copy()
    if development["cultivar_code"].nunique() != len(development_codes):
        observed = sorted(development["cultivar_code"].unique())
        raise ValueError(f"Development cultivar mismatch: requested {development_codes}, observed {observed}")

    scopes = {"development": development, "descriptive_all": predictions}
    summaries: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    correlations: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    for scope, frame in scopes.items():
        summaries.extend(summarize(frame, scope))
        folds.extend(fold_metrics(frame, scope))
        correlations.extend(residual_correlations(frame, scope))
        oracle_rows.extend(anchor_oracle(frame, scope))

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    fold_table = pd.DataFrame(folds)
    correlation_table = pd.DataFrame(correlations)
    oracle_table = pd.DataFrame(oracle_rows)
    summary.to_csv(output / "error_decomposition.csv", index=False)
    fold_table.to_csv(output / "fold_error_components.csv", index=False)
    correlation_table.to_csv(output / "residual_correlations.csv", index=False)
    oracle_table.to_csv(output / "anchor_oracle_by_fold.csv", index=False)

    plot = summary.loc[summary["scope"] == "development"].copy()
    traits = sorted(plot["trait"].unique())
    models = [model for model in ["PLSR", "Ridge", "PlumRAC-Net"] if model in set(plot["model"])]
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)
    sns.barplot(data=plot, x="trait", y="bias_sse_fraction", hue="model", hue_order=models, ax=axes[0])
    axes[0].set_title("A  Error attributable to cultivar bias", loc="left", fontweight="bold")
    axes[0].set_ylabel("Fraction of squared error")
    axes[0].set_xlabel("Trait")
    axes[0].legend(frameon=False, fontsize=7)
    sns.barplot(data=plot, x="trait", y="rmse_recoverable_by_intercept_pct", hue="model", hue_order=models, ax=axes[1])
    axes[1].set_title("B  Oracle intercept ceiling", loc="left", fontweight="bold")
    axes[1].set_ylabel("RMSE recoverable (%)")
    axes[1].set_xlabel("Trait")
    if axes[1].legend_:
        axes[1].legend_.remove()
    pair = oracle_table.loc[oracle_table["scope"] == "development"]
    if not pair.empty:
        share = pair.groupby(["trait", "oracle_best_anchor"]).size().rename("folds").reset_index()
        sns.barplot(data=share, x="trait", y="folds", hue="oracle_best_anchor", ax=axes[2])
    axes[2].set_title("C  Fold-wise better linear anchor", loc="left", fontweight="bold")
    axes[2].set_ylabel("Development folds")
    axes[2].set_xlabel("Trait")
    axes[2].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig_v3_error_diagnosis.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "design_use": "Only the prespecified development cultivar folds may inform V3 architecture choices.",
        "development_codes": development_codes,
        "descriptive_all_warning": "All-fold outputs are descriptive and cannot serve as clean V3 confirmation evidence after inspection.",
        "traits": traits,
        "models": models,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
