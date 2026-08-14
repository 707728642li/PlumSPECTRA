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


DEFAULT_TARGETS = [
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
PREPROCESSING = ["raw", "snv", "sg1", "snv_sg1"]
# V25: include the low-complexity region explicitly. Several texture targets
# select one to three latent variables, so starting at four biases both the
# standalone baseline and the residual-CNN anchor toward over-complex models.
COMPONENTS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24]


def preprocess_all(x: np.ndarray, wavelength: np.ndarray) -> dict[str, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    snv_scale = x64.std(axis=1, ddof=1, keepdims=True)
    snv = (x64 - x64.mean(axis=1, keepdims=True)) / np.where(snv_scale > 1e-12, snv_scale, 1.0)
    delta = float(np.median(np.diff(wavelength)))
    sg1 = savgol_filter(x64, window_length=11, polyorder=2, deriv=1, delta=delta, axis=1, mode="interp")
    snv_sg1 = savgol_filter(snv, window_length=11, polyorder=2, deriv=1, delta=delta, axis=1, mode="interp")
    return {"raw": x64, "snv": snv, "sg1": sg1, "snv_sg1": snv_sg1}


def concordance_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    variance_true, variance_pred = np.var(y_true), np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denominator = variance_true + variance_pred + (mean_true - mean_pred) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    standard_deviation = float(np.std(y_true, ddof=1)) if len(y_true) > 1 else np.nan
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
        "rpd": float(standard_deviation / rmse) if rmse > 0 else np.nan,
        "rpiq": float(iqr / rmse) if rmse > 0 else np.nan,
        "y_mean": float(np.mean(y_true)),
        "y_sd": standard_deviation,
    }


def select_configuration(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    inner_splits: int,
    components: list[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    components = COMPONENTS if components is None else components
    train_groups = groups[train_indices]
    splitter = GroupKFold(n_splits=min(inner_splits, len(np.unique(train_groups))))
    target_sd = max(float(np.std(y[train_indices], ddof=1)), 1e-12)
    configurations: list[dict[str, Any]] = []
    for preprocessing in PREPROCESSING:
        for n_components in components:
            fold_scores: list[float] = []
            fold_details: list[dict[str, Any]] = []
            for inner_fold, (relative_train, relative_valid) in enumerate(
                splitter.split(train_indices, groups=train_groups), start=1
            ):
                fit_indices = train_indices[relative_train]
                valid_indices = train_indices[relative_valid]
                estimator = PLSRegression(n_components=n_components, scale=True, max_iter=1000, tol=1e-7)
                estimator.fit(arrays[preprocessing][fit_indices], y[fit_indices])
                prediction = estimator.predict(arrays[preprocessing][valid_indices]).ravel()
                for cultivar in np.unique(groups[valid_indices]):
                    mask = groups[valid_indices] == cultivar
                    rmse = float(np.sqrt(mean_squared_error(y[valid_indices][mask], prediction[mask])))
                    fold_scores.append(rmse / target_sd)
                    fold_details.append(
                        {
                            "inner_fold": inner_fold,
                            "validation_cultivar": cultivar,
                            "n": int(mask.sum()),
                            "rmse": rmse,
                            "normalized_rmse_outer_train_sd": rmse / target_sd,
                        }
                    )
            configurations.append(
                {
                    "preprocessing": preprocessing,
                    "n_components": n_components,
                    "macro_normalized_rmse": float(np.mean(fold_scores)),
                    "details": fold_details,
                }
            )
    configurations.sort(
        key=lambda row: (
            row["macro_normalized_rmse"],
            row["n_components"],
            PREPROCESSING.index(row["preprocessing"]),
        )
    )
    return configurations[0], configurations


def run_outer_fold(
    target: str,
    heldout: str,
    arrays: dict[str, np.ndarray],
    sample_ids: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    eligible: np.ndarray,
    model_dir: Path,
    inner_splits: int,
    components: list[int],
) -> dict[str, Any]:
    train_indices = np.flatnonzero(eligible & (groups != heldout))
    test_indices = np.flatnonzero(eligible & (groups == heldout))
    if len(test_indices) < 3:
        raise ValueError(f"Insufficient test observations for {target}/{heldout}: {len(test_indices)}")
    selected, configurations = select_configuration(
        arrays, y, groups, train_indices, inner_splits, components
    )
    estimator = PLSRegression(n_components=selected["n_components"], scale=True, max_iter=1000, tol=1e-7)
    estimator.fit(arrays[selected["preprocessing"]][train_indices], y[train_indices])
    prediction = estimator.predict(arrays[selected["preprocessing"]][test_indices]).ravel()
    model_path = model_dir / target / f"heldout_{heldout.replace(' ', '_')}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_family": "PLSRegression",
            "validation": "nested_leave_one_cultivar_out",
            "target": target,
            "heldout_cultivar": heldout,
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
        "heldout_cultivar": heldout,
        "test_indices": test_indices,
        "prediction": prediction,
        "selected": selected,
        "configurations": configurations,
        "model_path": model_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="all", help="Comma-separated target columns or 'all'.")
    parser.add_argument("--cohort", choices=["analysis", "primary", "sensitivity"], default="analysis")
    parser.add_argument(
        "--exclude-cultivars",
        default="",
        help="Comma-separated model-independent whole-cultivar QC exclusions.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument(
        "--components",
        default=",".join(str(value) for value in COMPONENTS),
        help="Comma-separated PLS component grid.",
    )
    args = parser.parse_args()

    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    absorbance = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    if not np.array_equal(aligned["sample_id"].to_numpy(), row_index["sample_id"].to_numpy()):
        raise ValueError("Sample ordering mismatch")
    arrays = preprocess_all(absorbance, wavelength)
    sample_ids = aligned["sample_id"].to_numpy()
    groups = aligned["cultivar_ascii"].to_numpy()
    all_cultivars = sorted(np.unique(groups).tolist())
    excluded_cultivars = sorted(
        {value.strip() for value in args.exclude_cultivars.split(",") if value.strip()}
    )
    unknown_exclusions = sorted(set(excluded_cultivars) - set(all_cultivars))
    if unknown_exclusions:
        raise ValueError(f"Unknown cultivar exclusions: {unknown_exclusions}")
    targets = DEFAULT_TARGETS if args.targets == "all" else [item.strip() for item in args.targets.split(",")]
    missing = [target for target in targets if target not in aligned.columns]
    if missing:
        raise ValueError(f"Missing targets: {missing}")
    cultivars = sorted(set(all_cultivars) - set(excluded_cultivars))
    components = sorted(
        {int(value.strip()) for value in args.components.split(",") if value.strip()}
    )
    if not components or min(components) < 1:
        raise ValueError("PLS components must be positive integers")

    futures = []
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for target in targets:
            y = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
            cohort_column = {
                "analysis": "qc_analysis_include",
                "primary": "qc_primary_include",
                "sensitivity": "qc_sensitivity_include",
            }[args.cohort]
            eligible = aligned[cohort_column].to_numpy(bool) & np.isfinite(y)
            if excluded_cultivars:
                eligible &= ~np.isin(groups, excluded_cultivars)
            for cultivar in cultivars:
                futures.append(
                    executor.submit(
                        run_outer_fold,
                        target,
                        cultivar,
                        arrays,
                        sample_ids,
                        groups,
                        y,
                        eligible,
                        model_dir,
                        args.inner_splits,
                        components,
                    )
                )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            selected = result["selected"]
            print(
                f"completed {result['target']} held-out {result['heldout_cultivar']} "
                f"with {selected['preprocessing']}/{selected['n_components']}"
            )

    prediction_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda row: (row["target"], row["heldout_cultivar"])):
        target = result["target"]
        indices = result["test_indices"]
        truth = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)[indices]
        prediction = result["prediction"]
        selected = result["selected"]
        for index, y_true, y_pred in zip(indices, truth, prediction):
            prediction_rows.append(
                {
                    "sample_id": sample_ids[index],
                    "cultivar_ascii": groups[index],
                    "target": target,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "residual": float(y_pred - y_true),
                    "preprocessing": selected["preprocessing"],
                    "n_components": int(selected["n_components"]),
                }
            )
        fold_metric_rows.append(
            {
                "target": target,
                "heldout_cultivar": result["heldout_cultivar"],
                **regression_metrics(truth, prediction),
            }
        )
        selected_rows.append(
            {
                "target": target,
                "heldout_cultivar": result["heldout_cultivar"],
                "preprocessing": selected["preprocessing"],
                "n_components": int(selected["n_components"]),
                "inner_macro_normalized_rmse": selected["macro_normalized_rmse"],
                "model_path": result["model_path"],
            }
        )
        for configuration in result["configurations"]:
            for detail in configuration["details"]:
                inner_rows.append(
                    {
                        "target": target,
                        "outer_heldout_cultivar": result["heldout_cultivar"],
                        "preprocessing": configuration["preprocessing"],
                        "n_components": int(configuration["n_components"]),
                        "configuration_macro_normalized_rmse": configuration["macro_normalized_rmse"],
                        **detail,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    selected_parameters = pd.DataFrame(selected_rows)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    selected_parameters.to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(inner_rows).to_parquet(output_dir / "inner_cv_scores.parquet", index=False, compression="zstd")

    pooled_metrics = {
        target: regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        for target, group in predictions.groupby("target", observed=True)
    }
    macro_metrics = {
        target: {
            metric: float(group[metric].mean())
            for metric in ["rmse", "mae", "bias", "r2", "pearson_r", "ccc", "rpd", "rpiq"]
        }
        for target, group in fold_metrics.groupby("target", observed=True)
    }
    summary = {
        "model": "PLSRegression",
        "validation": "nested leave-one-cultivar-out",
        "cohort": {
            "analysis": "high-confidence-QC analysis cohort",
            "primary": "V25 primary model-independent QC cohort",
            "sensitivity": "hard-valid full sensitivity",
        }[args.cohort],
        "model_independent_excluded_cultivars": excluded_cultivars,
        "component_candidates": components,
        "eligible_cultivars": len(cultivars),
        "outer_folds": len(cultivars),
        "inner_splits": args.inner_splits,
        "targets": targets,
        "preprocessing_candidates": PREPROCESSING,
        "predictions": int(len(predictions)),
        "pooled_metrics": pooled_metrics,
        "cultivar_macro_metrics": macro_metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
