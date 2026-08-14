from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from train_texture_pls_loco import (
    COMPONENTS,
    DEFAULT_TARGETS,
    PREPROCESSING,
    preprocess_all,
    regression_metrics,
)


def select_hyperparameters(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    cultivars: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    cultivar_calibration: bool = False,
) -> tuple[str, int, list[dict[str, float | int | str]]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    rows: list[dict[str, float | int | str]] = []
    for preprocessing in PREPROCESSING:
        for n_components in COMPONENTS:
            squared_errors: list[np.ndarray] = []
            for relative_train, relative_valid in splitter.split(train_indices, cultivars[train_indices]):
                fit_indices = train_indices[relative_train]
                valid_indices = train_indices[relative_valid]
                model = PLSRegression(n_components=n_components, scale=True, max_iter=1000, tol=1e-7)
                model.fit(arrays[preprocessing][fit_indices], y[fit_indices])
                prediction = model.predict(arrays[preprocessing][valid_indices]).ravel()
                if cultivar_calibration:
                    fit_prediction = model.predict(arrays[preprocessing][fit_indices]).ravel()
                    fit_residual = y[fit_indices] - fit_prediction
                    offsets = {
                        str(cultivar): float(
                            np.mean(fit_residual[cultivars[fit_indices] == cultivar])
                        )
                        for cultivar in np.unique(cultivars[fit_indices])
                    }
                    prediction = prediction + np.asarray(
                        [offsets[str(cultivar)] for cultivar in cultivars[valid_indices]],
                        dtype=float,
                    )
                squared_errors.append((prediction - y[valid_indices]) ** 2)
            rmse = float(np.sqrt(np.mean(np.concatenate(squared_errors))))
            rows.append(
                {
                    "preprocessing": preprocessing,
                    "n_components": n_components,
                    "cv_rmse": rmse,
                    "selection_mode": "cultivar_aware" if cultivar_calibration else "global",
                }
            )
    rows.sort(key=lambda row: (row["cv_rmse"], row["n_components"], PREPROCESSING.index(str(row["preprocessing"]))))
    selected = rows[0]
    return str(selected["preprocessing"]), int(selected["n_components"]), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--cohort", choices=["analysis", "primary", "sensitivity"], default="analysis")
    parser.add_argument(
        "--exclude-cultivars",
        default="",
        help="Comma-separated model-independent whole-cultivar QC exclusions.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    absorbance = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    arrays = preprocess_all(absorbance, wavelength)
    sample_ids = aligned["sample_id"].to_numpy()
    cultivars = aligned["cultivar_ascii"].to_numpy()
    cultivar_names = sorted(np.unique(cultivars).tolist())
    excluded_cultivars = sorted(
        {value.strip() for value in args.exclude_cultivars.split(",") if value.strip()}
    )
    unknown_exclusions = sorted(set(excluded_cultivars) - set(cultivar_names))
    if unknown_exclusions:
        raise ValueError(f"Unknown cultivar exclusions: {unknown_exclusions}")
    targets = DEFAULT_TARGETS if args.targets == "all" else [item.strip() for item in args.targets.split(",")]

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    for target in targets:
        y = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
        cohort_column = {
            "analysis": "qc_analysis_include",
            "primary": "qc_primary_include",
            "sensitivity": "qc_sensitivity_include",
        }[args.cohort]
        eligible = aligned[cohort_column].to_numpy(bool) & np.isfinite(y)
        if excluded_cultivars:
            eligible &= ~np.isin(cultivars, excluded_cultivars)
        eligible_indices = np.flatnonzero(eligible)
        for repeat in range(1, args.repeats + 1):
            seed = 20260806 + repeat
            train_indices, test_indices = train_test_split(
                eligible_indices,
                test_size=0.2,
                random_state=seed,
                shuffle=True,
                stratify=cultivars[eligible_indices],
            )
            preprocessing, n_components, cv_results = select_hyperparameters(
                arrays, y, cultivars, train_indices, seed
            )
            model = PLSRegression(n_components=n_components, scale=True, max_iter=1000, tol=1e-7)
            model.fit(arrays[preprocessing][train_indices], y[train_indices])
            prediction = model.predict(arrays[preprocessing][test_indices]).ravel()
            selection_rows.append(
                {
                    "target": target,
                    "repeat": repeat,
                    "preprocessing": preprocessing,
                    "n_components": n_components,
                    "test_samples": len(test_indices),
                }
            )
            for row in cv_results:
                cv_rows.append({"target": target, "repeat": repeat, **row})
            metric_rows.append({"target": target, "repeat": repeat, **regression_metrics(y[test_indices], prediction)})
            for index, truth, estimate in zip(test_indices, y[test_indices], prediction):
                prediction_rows.append(
                    {
                        "sample_id": sample_ids[index],
                        "cultivar_ascii": cultivars[index],
                        "target": target,
                        "repeat": repeat,
                        "y_true": float(truth),
                        "y_pred": float(estimate),
                        "residual": float(estimate - truth),
                    }
                )
            print(f"completed {target} repeat {repeat}: {preprocessing}/{n_components}")

    predictions = pd.DataFrame(prediction_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    metrics_frame.to_csv(output_dir / "repeat_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(cv_rows).to_parquet(output_dir / "inner_cv_scores.parquet", index=False, compression="zstd")
    summary = {
        "model": "PLSRegression",
        "validation": "repeated cultivar-stratified random fruit split",
        "cohort": {
            "analysis": "high-confidence-QC analysis cohort",
            "primary": "strict 10-percent consensus-QC cohort",
            "sensitivity": "hard-valid full sensitivity",
        }[args.cohort],
        "model_independent_excluded_cultivars": excluded_cultivars,
        "test_fraction": 0.2,
        "repeats": args.repeats,
        "metrics_mean": metrics_frame.groupby("target")[
            ["rmse", "mae", "bias", "r2", "pearson_r", "ccc", "rpd", "rpiq"]
        ].mean().to_dict(orient="index"),
        "metrics_sd": metrics_frame.groupby("target")[
            ["rmse", "mae", "bias", "r2", "pearson_r", "ccc", "rpd", "rpiq"]
        ].std(ddof=1).to_dict(orient="index"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
