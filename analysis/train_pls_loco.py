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
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


TARGETS = ["fruit_weight_g", "soluble_solids_pct", "ph"]
PREPROCESSING = ["raw", "snv", "sg1", "snv_sg1"]
COMPONENTS = [4, 8, 12, 16, 24]


def preprocess_all(x: np.ndarray, wavelength: np.ndarray) -> dict[str, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    snv_scale = x64.std(axis=1, ddof=1, keepdims=True)
    snv = (x64 - x64.mean(axis=1, keepdims=True)) / np.where(snv_scale > 1e-12, snv_scale, 1.0)
    delta = float(np.median(np.diff(wavelength)))
    sg1 = savgol_filter(x64, window_length=11, polyorder=2, deriv=1, delta=delta, axis=1, mode="interp")
    snv_sg1 = savgol_filter(snv, window_length=11, polyorder=2, deriv=1, delta=delta, axis=1, mode="interp")
    return {"raw": x64, "snv": snv, "sg1": sg1, "snv_sg1": snv_sg1}


def concordance_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sd = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan
    iqr = float(np.quantile(y_true, 0.75) - np.quantile(y_true, 0.25))
    correlation = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 2 and np.std(y_pred) > 0 else np.nan
    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "pearson_r": correlation,
        "ccc": concordance_correlation(y_true, y_pred),
        "rpd": float(sd / rmse) if rmse > 0 else np.nan,
        "rpiq": float(iqr / rmse) if rmse > 0 else np.nan,
        "y_mean": float(np.mean(y_true)),
        "y_sd": sd,
    }


def evaluate_configuration(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    preprocessing: str,
    n_components: int,
) -> tuple[float, list[dict[str, Any]]]:
    outer_train_groups = groups[train_indices]
    unique_groups = np.unique(outer_train_groups)
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    cultivar_rmse: list[float] = []
    detail: list[dict[str, Any]] = []
    outer_train_sd = float(np.std(y[train_indices], ddof=1))
    for inner_fold, (relative_train, relative_valid) in enumerate(
        splitter.split(train_indices, groups=outer_train_groups), start=1
    ):
        inner_train = train_indices[relative_train]
        inner_valid = train_indices[relative_valid]
        estimator = PLSRegression(n_components=n_components, scale=True, max_iter=1000, tol=1e-7)
        estimator.fit(arrays[preprocessing][inner_train], y[inner_train])
        prediction = estimator.predict(arrays[preprocessing][inner_valid]).ravel()
        for cultivar in np.unique(groups[inner_valid]):
            cultivar_mask = groups[inner_valid] == cultivar
            score = float(np.sqrt(mean_squared_error(y[inner_valid][cultivar_mask], prediction[cultivar_mask])))
            cultivar_rmse.append(score)
            detail.append(
                {
                    "inner_fold": inner_fold,
                    "validation_cultivar": cultivar,
                    "n": int(cultivar_mask.sum()),
                    "rmse": score,
                    "normalized_rmse_outer_train_sd": score / outer_train_sd,
                }
            )
    return float(np.mean(cultivar_rmse) / outer_train_sd), detail


def run_outer_fold(
    heldout_cultivar: str,
    target: str,
    arrays: dict[str, np.ndarray],
    sample_ids: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    model_dir: Path,
) -> dict[str, Any]:
    train_indices = np.flatnonzero(valid & (groups != heldout_cultivar))
    test_indices = np.flatnonzero(valid & (groups == heldout_cultivar))
    if len(test_indices) == 0:
        raise ValueError(f"No valid test samples for {target}/{heldout_cultivar}")
    configuration_results: list[dict[str, Any]] = []
    for preprocessing in PREPROCESSING:
        for n_components in COMPONENTS:
            score, detail = evaluate_configuration(
                arrays, y, groups, train_indices, preprocessing, n_components
            )
            configuration_results.append(
                {
                    "preprocessing": preprocessing,
                    "n_components": n_components,
                    "macro_normalized_rmse": score,
                    "detail": detail,
                }
            )
    configuration_results.sort(
        key=lambda row: (row["macro_normalized_rmse"], row["n_components"], PREPROCESSING.index(row["preprocessing"]))
    )
    selected = configuration_results[0]
    estimator = PLSRegression(
        n_components=selected["n_components"], scale=True, max_iter=1000, tol=1e-7
    )
    estimator.fit(arrays[selected["preprocessing"]][train_indices], y[train_indices])
    prediction = estimator.predict(arrays[selected["preprocessing"]][test_indices]).ravel()
    model_path = model_dir / target / f"heldout_{heldout_cultivar.replace(' ', '_')}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_family": "PLSRegression",
            "target": target,
            "heldout_cultivar": heldout_cultivar,
            "preprocessing": selected["preprocessing"],
            "n_components": selected["n_components"],
            "estimator": estimator,
            "train_sample_ids": sample_ids[train_indices],
        },
        model_path,
        compress=3,
    )
    return {
        "target": target,
        "heldout_cultivar": heldout_cultivar,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "prediction": prediction,
        "selected": selected,
        "configuration_results": configuration_results,
        "model_path": model_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    master = pd.read_parquet(multimodal_dir / "master_samples.parquet").set_index("sample_id")
    aligned = master.loc[row_index["sample_id"]].reset_index()
    if not np.array_equal(aligned["sample_id"].to_numpy(), row_index["sample_id"].to_numpy()):
        raise ValueError("Sample ordering mismatch")
    arrays = preprocess_all(x, wavelength)
    sample_ids = aligned["sample_id"].to_numpy()
    groups = aligned["cultivar_ascii"].to_numpy()
    cultivars = sorted(np.unique(groups))

    futures = []
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for target in TARGETS:
            y = aligned[target].to_numpy(float)
            valid = aligned[f"include_nir_{target}"].to_numpy(bool)
            for cultivar in cultivars:
                futures.append(
                    executor.submit(
                        run_outer_fold,
                        cultivar,
                        target,
                        arrays,
                        sample_ids,
                        groups,
                        y,
                        valid,
                        model_dir,
                    )
                )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"completed {result['target']} held-out {result['heldout_cultivar']} "
                f"with {result['selected']['preprocessing']}/{result['selected']['n_components']}"
            )

    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda row: (row["target"], row["heldout_cultivar"])):
        target = result["target"]
        test = result["test_indices"]
        y_true = aligned[target].to_numpy(float)[test]
        y_pred = result["prediction"]
        selected = result["selected"]
        for index, truth, prediction in zip(test, y_true, y_pred):
            prediction_rows.append(
                {
                    "sample_id": sample_ids[index],
                    "cultivar_ascii": groups[index],
                    "target": target,
                    "y_true": float(truth),
                    "y_pred": float(prediction),
                    "residual": float(prediction - truth),
                    "preprocessing": selected["preprocessing"],
                    "n_components": int(selected["n_components"]),
                }
            )
        fold_metric_rows.append(
            {
                "target": target,
                "heldout_cultivar": result["heldout_cultivar"],
                **metrics(y_true, y_pred),
            }
        )
        selected_rows.append(
            {
                "target": target,
                "heldout_cultivar": result["heldout_cultivar"],
                "preprocessing": selected["preprocessing"],
                "n_components": int(selected["n_components"]),
                "inner_macro_normalized_rmse": float(selected["macro_normalized_rmse"]),
                "model_path": result["model_path"],
            }
        )
        for configuration in result["configuration_results"]:
            for detail in configuration["detail"]:
                inner_rows.append(
                    {
                        "target": target,
                        "outer_heldout_cultivar": result["heldout_cultivar"],
                        "preprocessing": configuration["preprocessing"],
                        "n_components": int(configuration["n_components"]),
                        "configuration_macro_normalized_rmse": float(configuration["macro_normalized_rmse"]),
                        **detail,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    selected_parameters = pd.DataFrame(selected_rows)
    inner_scores = pd.DataFrame(inner_rows)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    selected_parameters.to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    inner_scores.to_parquet(output_dir / "inner_cv_scores.parquet", index=False, compression="zstd")

    aggregate_rows: list[dict[str, Any]] = []
    for target, group in predictions.groupby("target"):
        aggregate_rows.append({"target": target, "scope": "pooled_loco", **metrics(group["y_true"], group["y_pred"])})
        target_folds = fold_metrics.loc[fold_metrics["target"].eq(target)]
        for metric in ["rmse", "mae", "bias", "r2", "pearson_r", "ccc", "rpd", "rpiq"]:
            aggregate_rows.append(
                {
                    "target": target,
                    "scope": f"cultivar_macro_{metric}",
                    "n": int(len(target_folds)),
                    metric: float(target_folds[metric].mean()),
                    f"{metric}_sd_across_cultivars": float(target_folds[metric].std(ddof=1)),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    summary = {
        "model": "PLSRegression",
        "validation": "nested leave-one-cultivar-out",
        "outer_folds": len(cultivars),
        "targets": TARGETS,
        "preprocessing_candidates": PREPROCESSING,
        "component_candidates": COMPONENTS,
        "predictions": int(len(predictions)),
        "pooled_metrics": {
            target: metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
            for target, group in predictions.groupby("target")
        },
        "selected_preprocessing_counts": selected_parameters.groupby(["target", "preprocessing"]).size().rename("folds").reset_index().to_dict(orient="records"),
        "selected_component_counts": selected_parameters.groupby(["target", "n_components"]).size().rename("folds").reset_index().to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
