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

from train_pls_loco import COMPONENTS, PREPROCESSING, TARGETS, metrics, preprocess_all


def select_hyperparameters(
    arrays: dict[str, np.ndarray], y: np.ndarray, cultivar: np.ndarray, train_indices: np.ndarray, seed: int
) -> tuple[str, int, list[dict[str, float | int | str]]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows: list[dict[str, float | int | str]] = []
    for preprocessing in PREPROCESSING:
        for n_components in COMPONENTS:
            squared_error: list[np.ndarray] = []
            for relative_train, relative_valid in splitter.split(train_indices, cultivar[train_indices]):
                inner_train = train_indices[relative_train]
                inner_valid = train_indices[relative_valid]
                model = PLSRegression(n_components=n_components, scale=True, max_iter=1000, tol=1e-7)
                model.fit(arrays[preprocessing][inner_train], y[inner_train])
                prediction = model.predict(arrays[preprocessing][inner_valid]).ravel()
                squared_error.append((prediction - y[inner_valid]) ** 2)
            rmse = float(np.sqrt(np.mean(np.concatenate(squared_error))))
            rows.append({"preprocessing": preprocessing, "n_components": n_components, "cv_rmse": rmse})
    rows.sort(key=lambda row: (row["cv_rmse"], row["n_components"], PREPROCESSING.index(str(row["preprocessing"]))))
    selected = rows[0]
    return str(selected["preprocessing"]), int(selected["n_components"]), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelengths = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    master = pd.read_parquet(multimodal_dir / "master_samples.parquet").set_index("sample_id")
    aligned = master.loc[row_index["sample_id"]].reset_index()
    arrays = preprocess_all(x, wavelengths)
    cultivars = aligned["cultivar_ascii"].to_numpy()
    sample_ids = aligned["sample_id"].to_numpy()

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    for target in TARGETS:
        y = aligned[target].to_numpy(float)
        valid_indices = np.flatnonzero(aligned[f"include_nir_{target}"].to_numpy(bool))
        for repeat in range(1, args.repeats + 1):
            seed = 20260806 + repeat
            train_indices, test_indices = train_test_split(
                valid_indices,
                test_size=0.2,
                random_state=seed,
                shuffle=True,
                stratify=cultivars[valid_indices],
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
            metric_rows.append({"target": target, "repeat": repeat, **metrics(y[test_indices], prediction)})
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
    repeat_metrics = pd.DataFrame(metric_rows)
    selections = pd.DataFrame(selection_rows)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    repeat_metrics.to_csv(output_dir / "repeat_metrics.csv", index=False)
    selections.to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(cv_rows).to_parquet(output_dir / "inner_cv_scores.parquet", index=False, compression="zstd")
    summary = {
        "model": "PLSRegression",
        "validation": "repeated stratified random fruit split",
        "test_fraction": 0.2,
        "repeats": args.repeats,
        "metrics_mean": repeat_metrics.groupby("target")[["rmse", "mae", "bias", "r2", "pearson_r", "ccc", "rpd", "rpiq"]].mean().to_dict(orient="index"),
        "metrics_sd": repeat_metrics.groupby("target")[["rmse", "mae", "bias", "r2", "pearson_r", "ccc", "rpd", "rpiq"]].std(ddof=1).to_dict(orient="index"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
