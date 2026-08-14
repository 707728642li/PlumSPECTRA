from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from v2_registry import add_cultivar_code, trait_registry


MODEL_ORDER = ["PLSR", "Ridge", "PlumRAC-Net"]
MODEL_COLORS = {"PLSR": "#C58C2A", "Ridge": "#75818A", "PlumRAC-Net": "#3F7CA6"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_specs(values: list[str] | None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values or []:
        abbreviation, path = value.split("=", 1)
        key = abbreviation.strip().upper()
        if key in result:
            raise ValueError(f"Duplicate trait specification: {key}")
        result[key] = Path(path).resolve()
    return result


def metrics(frame: pd.DataFrame, prediction: str = "y_pred") -> dict[str, float | int]:
    truth = frame["y_true"].to_numpy(float)
    pred = frame[prediction].to_numpy(float)
    residual = pred - truth
    truth_mean = float(np.mean(truth))
    pred_mean = float(np.mean(pred))
    truth_var = float(np.var(truth, ddof=1))
    pred_var = float(np.var(pred, ddof=1))
    covariance = float(np.cov(truth, pred, ddof=1)[0, 1])
    ccc = 2.0 * covariance / max(truth_var + pred_var + (truth_mean - pred_mean) ** 2, np.finfo(float).eps)
    rmse = float(np.sqrt(mean_squared_error(truth, pred)))
    pearson = pearsonr(truth, pred).statistic if len(truth) > 2 else np.nan
    spearman = spearmanr(truth, pred).statistic if len(truth) > 2 else np.nan
    return {
        "n": int(len(frame)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(truth, pred)),
        "bias": float(np.mean(residual)),
        "r2": float(r2_score(truth, pred)),
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "ccc": float(ccc),
        "rpd": float(np.std(truth, ddof=1) / rmse),
        "rpiq": float((np.quantile(truth, 0.75) - np.quantile(truth, 0.25)) / rmse),
    }


def centred_metrics(frame: pd.DataFrame) -> dict[str, float]:
    work = frame.copy()
    work["true_c"] = work["y_true"] - work.groupby("cultivar_ascii")["y_true"].transform("mean")
    work["pred_c"] = work["y_pred"] - work.groupby("cultivar_ascii")["y_pred"].transform("mean")
    return {
        "centred_r2": float(r2_score(work["true_c"], work["pred_c"])),
        "centred_rmse": float(np.sqrt(mean_squared_error(work["true_c"], work["pred_c"]))),
        "centred_pearson_r": float(pearsonr(work["true_c"], work["pred_c"]).statistic),
        "centred_spearman_rho": float(spearmanr(work["true_c"], work["pred_c"]).statistic),
    }


def load_prediction(path: Path, target: str, model: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"sample_id", "cultivar_ascii", "target", "y_true", "y_pred"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame.columns))}")
    frame = frame.loc[frame["target"] == target, list(required)].copy()
    if len(frame) == 0:
        raise ValueError(f"No predictions for {target} in {path}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample predictions for {model}/{target}")
    frame["model"] = model
    return add_cultivar_code(frame)


def cluster_bootstrap_delta(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    paired = reference[["sample_id", "cultivar_ascii", "y_true", "y_pred"]].merge(
        candidate[["sample_id", "y_pred"]], on="sample_id", suffixes=("_reference", "_candidate"), validate="one_to_one"
    )
    cultivars = np.asarray(sorted(paired["cultivar_ascii"].unique()))
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        sampled = rng.choice(cultivars, size=len(cultivars), replace=True)
        parts = [paired.loc[paired["cultivar_ascii"] == cultivar] for cultivar in sampled]
        boot = pd.concat(parts, ignore_index=True)
        rmse_ref = float(np.sqrt(mean_squared_error(boot["y_true"], boot["y_pred_reference"])))
        rmse_cand = float(np.sqrt(mean_squared_error(boot["y_true"], boot["y_pred_candidate"])))
        deltas.append(100.0 * (rmse_ref - rmse_cand) / rmse_ref)
    values = np.asarray(deltas)
    return {
        "rmse_improvement_pct": float(np.mean(values)),
        "ci025": float(np.quantile(values, 0.025)),
        "ci975": float(np.quantile(values, 0.975)),
        "probability_improvement": float(np.mean(values > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pls-predictions", type=Path, required=True)
    parser.add_argument("--plumrac", action="append", help="ABBR=predictions_ensemble.parquet")
    parser.add_argument("--ridge", action="append", help="ABBR=predictions.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry = trait_registry().set_index("abbreviation")
    plumrac_specs = parse_specs(args.plumrac)
    ridge_specs = parse_specs(args.ridge)
    requested = [abbr for abbr in registry.index if abbr in plumrac_specs]
    if not requested:
        raise ValueError("At least one --plumrac ABBR=path prediction set is required")

    frames: list[pd.DataFrame] = []
    for abbreviation in requested:
        target = str(registry.loc[abbreviation, "target"])
        frames.append(load_prediction(args.pls_predictions.resolve(), target, "PLSR"))
        if abbreviation in ridge_specs:
            frames.append(load_prediction(ridge_specs[abbreviation], target, "Ridge"))
        frames.append(load_prediction(plumrac_specs[abbreviation], target, "PlumRAC-Net"))
    predictions = pd.concat(frames, ignore_index=True)
    target_to_abbr = dict(zip(registry["target"], registry.index, strict=True))
    predictions["trait"] = predictions["target"].map(target_to_abbr)
    predictions.to_parquet(output / "all_predictions.parquet", index=False, compression="zstd")
    for model, model_frame in predictions.groupby("model", observed=True):
        slug = str(model).lower().replace("-", "_")
        model_frame.to_parquet(output / f"predictions_{slug}.parquet", index=False, compression="zstd")

    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for (trait, model), group in predictions.groupby(["trait", "model"], observed=True):
        summary_rows.append({"trait": trait, "model": model, **metrics(group), **centred_metrics(group)})
        for (cultivar, cultivar_code), fold in group.groupby(["cultivar_ascii", "cultivar_code"], observed=True):
            fold_rows.append(
                {
                    "trait": trait,
                    "model": model,
                    "cultivar_ascii": cultivar,
                    "cultivar_code": cultivar_code,
                    **metrics(fold),
                }
            )
    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values(["trait", "model"])
    summary.to_csv(output / "model_summary.csv", index=False)
    folds.to_csv(output / "fold_metrics.csv", index=False)

    comparison_rows: list[dict[str, object]] = []
    heatmap_rows: list[dict[str, object]] = []
    for trait in requested:
        pls = predictions.loc[(predictions["trait"] == trait) & (predictions["model"] == "PLSR")]
        rac = predictions.loc[(predictions["trait"] == trait) & (predictions["model"] == "PlumRAC-Net")]
        pls_pooled_rmse = float(np.sqrt(mean_squared_error(pls["y_true"], pls["y_pred"])))
        rac_pooled_rmse = float(np.sqrt(mean_squared_error(rac["y_true"], rac["y_pred"])))
        pooled_rmse_improvement_pct = 100.0 * (pls_pooled_rmse - rac_pooled_rmse) / pls_pooled_rmse
        boot = cluster_bootstrap_delta(pls, rac, args.bootstrap_repeats, 20260806 + len(comparison_rows))
        pls_fold = folds.loc[(folds["trait"] == trait) & (folds["model"] == "PLSR")].set_index("cultivar_code")
        rac_fold = folds.loc[(folds["trait"] == trait) & (folds["model"] == "PlumRAC-Net")].set_index("cultivar_code")
        common = pls_fold.index.intersection(rac_fold.index)
        relative = 100.0 * (pls_fold.loc[common, "rmse"] - rac_fold.loc[common, "rmse"]) / pls_fold.loc[common, "rmse"]
        comparison_rows.append(
            {
                "trait": trait,
                "fold_wins": int((relative > 1e-6).sum()),
                "folds": int(len(relative)),
                "plsr_pooled_rmse": pls_pooled_rmse,
                "plumrac_pooled_rmse": rac_pooled_rmse,
                "pooled_rmse_improvement_pct": pooled_rmse_improvement_pct,
                **boot,
            }
        )
        for cultivar_code, value in relative.items():
            heatmap_rows.append({"trait": trait, "cultivar_code": cultivar_code, "rmse_improvement_pct": float(value)})
    comparison = pd.DataFrame(comparison_rows)
    def interpretation(row: pd.Series) -> str:
        if row["rmse_improvement_pct"] > 0 and row["fold_wins"] >= 12 and row["ci025"] > 0:
            return "clear improvement"
        if abs(row["rmse_improvement_pct"]) <= 1.0 and row["ci025"] <= 0 <= row["ci975"]:
            return "safeguarded parity"
        if row["rmse_improvement_pct"] > 0 or row["fold_wins"] >= 9:
            return "promising but heterogeneous"
        return "no improvement"

    comparison["interpretation"] = comparison.apply(interpretation, axis=1)
    heatmap = pd.DataFrame(heatmap_rows)
    comparison.to_csv(output / "plumrac_vs_plsr.csv", index=False)
    heatmap.to_csv(output / "cultivar_rmse_improvement.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    trait_order = requested
    present_models = [model for model in MODEL_ORDER if model in summary["model"].astype(str).unique()]
    x = np.arange(len(trait_order))
    width = 0.78 / len(present_models)
    for index, model in enumerate(present_models):
        values = summary.loc[summary["model"].astype(str) == model].set_index("trait").reindex(trait_order)
        axes[0, 0].bar(x + (index - (len(present_models) - 1) / 2) * width, values["r2"], width, label=model, color=MODEL_COLORS[model])
        axes[1, 0].bar(x + (index - (len(present_models) - 1) / 2) * width, values["centred_r2"], width, color=MODEL_COLORS[model])
    axes[0, 0].set_title("A  Zero-shot pooled performance", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel(r"$R^2$")
    axes[0, 0].legend(frameon=False, ncol=len(present_models))
    axes[1, 0].set_title("C  Within-cultivar discrimination", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel(r"Centred $R^2$")
    for axis in [axes[0, 0], axes[1, 0]]:
        axis.axhline(0, color="#5B6570", linewidth=0.8)
        axis.set_xticks(x, trait_order)
        axis.spines[["top", "right"]].set_visible(False)

    comp = comparison.set_index("trait").reindex(trait_order)
    yerr = np.vstack([comp["rmse_improvement_pct"] - comp["ci025"], comp["ci975"] - comp["rmse_improvement_pct"]])
    axes[0, 1].errorbar(x, comp["rmse_improvement_pct"], yerr=yerr, fmt="o", color=MODEL_COLORS["PlumRAC-Net"], ecolor="#75818A", capsize=3)
    axes[0, 1].axhline(0, color="#5B6570", linewidth=0.8)
    axes[0, 1].set_xticks(x, trait_order)
    axes[0, 1].set_ylabel("RMSE improvement over PLSR (%)")
    axes[0, 1].set_title("B  Cultivar-cluster bootstrap", loc="left", fontweight="bold")
    axes[0, 1].spines[["top", "right"]].set_visible(False)

    matrix = heatmap.pivot(index="cultivar_code", columns="trait", values="rmse_improvement_pct").reindex(columns=trait_order)
    matrix = matrix.loc[matrix.mean(axis=1).sort_values().index]
    sns.heatmap(matrix, cmap="vlag", center=0, linewidths=0.25, linecolor="white", cbar_kws={"label": "RMSE improvement (%)"}, ax=axes[1, 1])
    axes[1, 1].set_title("D  Held-out cultivar gains", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Trait")
    axes[1, 1].set_ylabel("Cultivar / selection")
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig_v2_model_comparison.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "validation": "strict nested leave-one-cultivar-out",
        "traits": requested,
        "models": present_models,
        "model_summary": summary.assign(model=summary["model"].astype(str)).to_dict(orient="records"),
        "plumrac_vs_plsr": comparison.to_dict(orient="records"),
        "bootstrap_unit": "held-out cultivar/selection",
        "bootstrap_repeats": args.bootstrap_repeats,
    }
    project_root = Path(__file__).resolve().parents[1]
    provenance_paths = [
        Path(__file__).resolve(),
        args.pls_predictions.resolve(),
        project_root / "configs" / "v2_nomenclature.csv",
        project_root / "configs" / "v2_trait_registry.csv",
        project_root / "environment-lock.txt",
        *plumrac_specs.values(),
        *ridge_specs.values(),
    ]
    report["provenance_sha256"] = {
        str(path.relative_to(project_root) if path.is_relative_to(project_root) else path): sha256_file(path)
        for path in provenance_paths
        if path.exists()
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"traits": requested, "models": present_models, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
