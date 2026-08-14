from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from train_texture_pls_loco import DEFAULT_TARGETS, concordance_correlation, preprocess_all
from v2_registry import add_cultivar_code, abbreviated_trait


RIDGE_GRID = [
    {"preprocessing": preprocessing, "alpha": alpha}
    for preprocessing in ["raw", "snv", "sg1", "snv_sg1"]
    for alpha in [0.1, 1.0, 10.0, 100.0, 1_000.0]
]
SVR_PROFILES = [
    {"C": 1.0, "gamma": "scale", "epsilon": 0.10},
    {"C": 10.0, "gamma": "scale", "epsilon": 0.10},
    {"C": 100.0, "gamma": "scale", "epsilon": 0.10},
    {"C": 10.0, "gamma": 0.001, "epsilon": 0.10},
    {"C": 100.0, "gamma": 0.001, "epsilon": 0.10},
    {"C": 10.0, "gamma": 0.01, "epsilon": 0.10},
    {"C": 100.0, "gamma": 0.01, "epsilon": 0.10},
]
SVR_GRID = [
    {"preprocessing": preprocessing, **profile}
    for preprocessing in ["raw", "snv", "sg1"]
    for profile in SVR_PROFILES
]


def build_estimator(model: str, config: dict[str, Any], y_mean: float, y_sd: float):
    if model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(config["alpha"])))
    if model == "svr":
        return make_pipeline(
            StandardScaler(),
            SVR(C=float(config["C"]), gamma=config["gamma"], epsilon=float(config["epsilon"])),
        )
    raise ValueError(f"Unknown model: {model}")


def fit_predict(
    model: str,
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
) -> tuple[Any, np.ndarray]:
    y_mean = float(np.mean(y[fit_indices]))
    y_sd = max(float(np.std(y[fit_indices], ddof=1)), 1e-8)
    estimator = build_estimator(model, config, y_mean, y_sd)
    y_fit = y[fit_indices]
    if model == "svr":
        y_fit = (y_fit - y_mean) / y_sd
    estimator.fit(arrays[config["preprocessing"]][fit_indices], y_fit)
    prediction = estimator.predict(arrays[config["preprocessing"]][predict_indices])
    if model == "svr":
        prediction = prediction * y_sd + y_mean
    return estimator, np.asarray(prediction, dtype=float)


def candidate_grid(model: str) -> list[dict[str, Any]]:
    if model == "ridge":
        return RIDGE_GRID
    if model == "svr":
        return SVR_GRID
    raise ValueError(model)


def select_configuration(
    model: str,
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    inner_splits: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_groups = groups[train_indices]
    splitter = GroupKFold(n_splits=min(inner_splits, len(np.unique(train_groups))))
    target_sd = max(float(np.std(y[train_indices], ddof=1)), 1e-12)
    rows = []
    for config in candidate_grid(model):
        cultivar_scores = []
        for fold, (relative_fit, relative_valid) in enumerate(
            splitter.split(train_indices, groups=train_groups), start=1
        ):
            fit_indices = train_indices[relative_fit]
            valid_indices = train_indices[relative_valid]
            _, prediction = fit_predict(model, config, arrays, y, fit_indices, valid_indices)
            for cultivar in np.unique(groups[valid_indices]):
                mask = groups[valid_indices] == cultivar
                rmse = float(np.sqrt(np.mean((y[valid_indices][mask] - prediction[mask]) ** 2)))
                cultivar_scores.append(
                    {
                        "inner_fold": fold,
                        "validation_cultivar": cultivar,
                        "normalized_rmse": rmse / target_sd,
                    }
                )
        rows.append(
            {
                **config,
                "macro_normalized_rmse": float(np.mean([item["normalized_rmse"] for item in cultivar_scores])),
                "details": cultivar_scores,
            }
        )
    rows.sort(key=lambda row: (row["macro_normalized_rmse"], json.dumps({k: v for k, v in row.items() if k != "details"}, sort_keys=True)))
    return rows[0], rows


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 2 and np.std(y_pred) > 0 else np.nan,
        "spearman_rho": float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 2 and np.std(y_pred) > 0 else np.nan,
        "ccc": concordance_correlation(y_true, y_pred),
    }


def run_outer_fold(
    model: str,
    target: str,
    heldout: str,
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    eligible: np.ndarray,
    output_dir: Path,
    inner_splits: int,
) -> dict[str, Any]:
    fold_dir = output_dir / "runs" / abbreviated_trait(target) / heldout.replace(" ", "_")
    prediction_path = fold_dir / "predictions.parquet"
    metadata_path = fold_dir / "metadata.json"
    if prediction_path.exists() and metadata_path.exists():
        return {"predictions": pd.read_parquet(prediction_path), "metadata": json.loads(metadata_path.read_text(encoding="utf-8"))}
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_indices = np.flatnonzero(eligible & (groups != heldout))
    test_indices = np.flatnonzero(eligible & (groups == heldout))
    selected, candidates = select_configuration(model, arrays, y, groups, train_indices, inner_splits)
    estimator, prediction = fit_predict(model, selected, arrays, y, train_indices, test_indices)
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids[test_indices],
            "cultivar_ascii": groups[test_indices],
            "target": target,
            "model": model,
            "y_true": y[test_indices],
            "y_pred": prediction,
        }
    )
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame = add_cultivar_code(frame)
    frame.to_parquet(prediction_path, index=False, compression="zstd")
    joblib.dump(estimator, fold_dir / "model.joblib", compress=3)
    inner_rows = []
    for candidate in candidates:
        candidate_values = {key: value for key, value in candidate.items() if key != "details"}
        if "gamma" in candidate_values:
            candidate_values["gamma"] = str(candidate_values["gamma"])
        for detail in candidate["details"]:
            inner_rows.append({**candidate_values, **detail})
    pd.DataFrame(inner_rows).to_parquet(fold_dir / "inner_scores.parquet", index=False, compression="zstd")
    metadata = {
        "model": model,
        "target": target,
        "trait_abbreviation": abbreviated_trait(target),
        "heldout_cultivar": heldout,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "selected": {key: value for key, value in selected.items() if key != "details"},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"predictions": frame, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["ridge", "svr"], required=True)
    parser.add_argument("--target", choices=DEFAULT_TARGETS, required=True)
    parser.add_argument("--cohort", choices=["analysis", "primary", "sensitivity"], default="analysis")
    parser.add_argument(
        "--exclude-cultivars",
        default="",
        help="Comma-separated model-independent whole-cultivar QC exclusions.",
    )
    parser.add_argument("--heldout", default="all")
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    multimodal_dir = args.multimodal_dir.resolve()
    raw = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    y = pd.to_numeric(aligned[args.target], errors="coerce").to_numpy(float)
    cohort_column = {
        "analysis": "qc_analysis_include",
        "primary": "qc_primary_include",
        "sensitivity": "qc_sensitivity_include",
    }[args.cohort]
    eligible = aligned[cohort_column].to_numpy(bool) & np.isfinite(y)
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    sample_ids = aligned["sample_id"].to_numpy()
    arrays = preprocess_all(raw, wavelength)
    all_cultivars = sorted(np.unique(groups).tolist())
    excluded_cultivars = sorted(
        {value.strip() for value in args.exclude_cultivars.split(",") if value.strip()}
    )
    unknown_exclusions = sorted(set(excluded_cultivars) - set(all_cultivars))
    if unknown_exclusions:
        raise ValueError(f"Unknown cultivar exclusions: {unknown_exclusions}")
    if excluded_cultivars:
        eligible &= ~np.isin(groups, excluded_cultivars)
    cultivars = (
        sorted(set(all_cultivars) - set(excluded_cultivars))
        if args.heldout == "all"
        else [value.strip() for value in args.heldout.split(",")]
    )
    requested_excluded = sorted(set(cultivars) & set(excluded_cultivars))
    if requested_excluded:
        raise ValueError(f"Held-out cultivars are excluded by QC: {requested_excluded}")

    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for heldout in cultivars:
            futures.append(
                executor.submit(
                    run_outer_fold,
                    args.model,
                    args.target,
                    heldout,
                    arrays,
                    y,
                    groups,
                    sample_ids,
                    eligible,
                    output_dir,
                    args.inner_splits,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"completed {args.model} {result['metadata']['trait_abbreviation']} / {result['metadata']['heldout_cultivar']}", flush=True)

    predictions = pd.concat([result["predictions"] for result in results], ignore_index=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    fold_rows = []
    for cultivar, group in predictions.groupby("cultivar_ascii", observed=True):
        fold_rows.append({"heldout_cultivar": cultivar, **regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    centred_true = predictions["y_true"] - predictions.groupby("cultivar_ascii")["y_true"].transform("mean")
    centred_pred = predictions["y_pred"] - predictions.groupby("cultivar_ascii")["y_pred"].transform("mean")
    summary = {
        "model": args.model,
        "target": args.target,
        "trait_abbreviation": abbreviated_trait(args.target),
        "validation": "nested leave-one-cultivar-out",
        "cohort": args.cohort,
        "model_independent_excluded_cultivars": excluded_cultivars,
        "pooled_metrics": regression_metrics(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()),
        "cultivar_macro_metrics": {column: float(folds[column].mean()) for column in ["rmse", "mae", "bias", "r2", "pearson_r", "spearman_rho", "ccc"]},
        "within_cultivar_centered_r2": float(r2_score(centred_true, centred_pred)),
        "candidate_count": len(candidate_grid(args.model)),
        "inner_splits": args.inner_splits,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
