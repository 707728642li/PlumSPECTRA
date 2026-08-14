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
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from train_pls_loco import COMPONENTS, PREPROCESSING, TARGETS, metrics, preprocess_all


MEAN_COMPONENTS = [1, 2, 3, 4, 5]


def group_centroids(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, indices: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = sorted(np.unique(groups[indices]).tolist())
    x_mean = np.stack([x[indices[groups[indices] == name]].mean(axis=0) for name in names])
    y_mean = np.asarray([y[indices[groups[indices] == name]].mean() for name in names])
    return names, x_mean, y_mean


def select_mean_model(
    arrays: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, train_indices: np.ndarray
) -> tuple[str, int, list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    training_groups = sorted(np.unique(groups[train_indices]).tolist())
    for preprocessing in PREPROCESSING:
        names, centroids, target_means = group_centroids(arrays[preprocessing], y, groups, train_indices)
        for n_components in MEAN_COMPONENTS:
            errors: list[float] = []
            detail: list[dict[str, Any]] = []
            for heldout in training_groups:
                validation_position = names.index(heldout)
                fit_mask = np.asarray([name != heldout for name in names])
                allowed_components = min(n_components, int(fit_mask.sum()) - 1, centroids.shape[1])
                model = PLSRegression(n_components=allowed_components, scale=True, max_iter=1000, tol=1e-7)
                model.fit(centroids[fit_mask], target_means[fit_mask])
                prediction = float(model.predict(centroids[[validation_position]]).ravel()[0])
                error = prediction - float(target_means[validation_position])
                errors.append(error)
                detail.append(
                    {
                        "validation_cultivar": heldout,
                        "y_true_mean": float(target_means[validation_position]),
                        "y_pred_mean": prediction,
                        "error": error,
                    }
                )
            scores.append(
                {
                    "preprocessing": preprocessing,
                    "n_components": n_components,
                    "rmse_cultivar_means": float(np.sqrt(np.mean(np.square(errors)))),
                    "detail": detail,
                }
            )
    scores.sort(key=lambda row: (row["rmse_cultivar_means"], row["n_components"], PREPROCESSING.index(row["preprocessing"])))
    selected = scores[0]
    return selected["preprocessing"], int(selected["n_components"]), scores


def centered_values(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    centered_x = np.empty((len(indices), x.shape[1]), dtype=np.float64)
    centered_y = np.empty(len(indices), dtype=np.float64)
    selected_groups = groups[indices]
    for cultivar in np.unique(selected_groups):
        local = np.flatnonzero(selected_groups == cultivar)
        centered_x[local] = x[indices[local]] - x[indices[local]].mean(axis=0, keepdims=True)
        centered_y[local] = y[indices[local]] - y[indices[local]].mean()
    return centered_x, centered_y


def select_within_model(
    arrays: dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, train_indices: np.ndarray
) -> tuple[str, int, list[dict[str, Any]]]:
    outer_groups = groups[train_indices]
    splitter = GroupKFold(n_splits=min(5, len(np.unique(outer_groups))))
    scores: list[dict[str, Any]] = []
    for preprocessing in PREPROCESSING:
        for n_components in COMPONENTS:
            cultivar_errors: list[float] = []
            detail: list[dict[str, Any]] = []
            for inner_fold, (relative_train, relative_valid) in enumerate(
                splitter.split(train_indices, groups=outer_groups), start=1
            ):
                inner_train = train_indices[relative_train]
                inner_valid = train_indices[relative_valid]
                x_train_centered, y_train_centered = centered_values(
                    arrays[preprocessing], y, groups, inner_train
                )
                model = PLSRegression(n_components=n_components, scale=True, max_iter=1000, tol=1e-7)
                model.fit(x_train_centered, y_train_centered)
                for cultivar in np.unique(groups[inner_valid]):
                    cultivar_indices = inner_valid[groups[inner_valid] == cultivar]
                    x_valid = arrays[preprocessing][cultivar_indices]
                    x_valid_centered = x_valid - x_valid.mean(axis=0, keepdims=True)
                    y_valid_centered = y[cultivar_indices] - y[cultivar_indices].mean()
                    prediction = model.predict(x_valid_centered).ravel()
                    rmse = float(np.sqrt(mean_squared_error(y_valid_centered, prediction)))
                    cultivar_errors.append(rmse)
                    detail.append(
                        {
                            "inner_fold": inner_fold,
                            "validation_cultivar": cultivar,
                            "n": int(len(cultivar_indices)),
                            "within_rmse": rmse,
                        }
                    )
            scores.append(
                {
                    "preprocessing": preprocessing,
                    "n_components": n_components,
                    "macro_within_rmse": float(np.mean(cultivar_errors)),
                    "detail": detail,
                }
            )
    scores.sort(key=lambda row: (row["macro_within_rmse"], row["n_components"], PREPROCESSING.index(row["preprocessing"])))
    selected = scores[0]
    return selected["preprocessing"], int(selected["n_components"]), scores


def run_fold(
    target: str,
    heldout: str,
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    valid: np.ndarray,
    groups: np.ndarray,
    sample_ids: np.ndarray,
    model_dir: Path,
) -> dict[str, Any]:
    train_indices = np.flatnonzero(valid & (groups != heldout))
    test_indices = np.flatnonzero(valid & (groups == heldout))
    mean_preprocessing, mean_components, mean_scores = select_mean_model(arrays, y, groups, train_indices)
    within_preprocessing, within_components, within_scores = select_within_model(arrays, y, groups, train_indices)

    cultivar_names, train_centroids, train_target_means = group_centroids(
        arrays[mean_preprocessing], y, groups, train_indices
    )
    mean_components_fit = min(mean_components, len(cultivar_names) - 1, train_centroids.shape[1])
    mean_model = PLSRegression(n_components=mean_components_fit, scale=True, max_iter=1000, tol=1e-7)
    mean_model.fit(train_centroids, train_target_means)
    test_centroid = arrays[mean_preprocessing][test_indices].mean(axis=0, keepdims=True)
    predicted_test_mean = float(mean_model.predict(test_centroid).ravel()[0])

    x_train_centered, y_train_centered = centered_values(
        arrays[within_preprocessing], y, groups, train_indices
    )
    within_model = PLSRegression(n_components=within_components, scale=True, max_iter=1000, tol=1e-7)
    within_model.fit(x_train_centered, y_train_centered)
    x_test = arrays[within_preprocessing][test_indices]
    test_deviation = within_model.predict(x_test - x_test.mean(axis=0, keepdims=True)).ravel()
    prediction = predicted_test_mean + test_deviation

    path = model_dir / target / f"heldout_{heldout.replace(' ', '_')}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_family": "hierarchical_transductive_pls",
            "target": target,
            "heldout_cultivar": heldout,
            "mean_preprocessing": mean_preprocessing,
            "mean_components": mean_components,
            "mean_model": mean_model,
            "within_preprocessing": within_preprocessing,
            "within_components": within_components,
            "within_model": within_model,
            "train_sample_ids": sample_ids[train_indices],
        },
        path,
        compress=3,
    )
    return {
        "target": target,
        "heldout_cultivar": heldout,
        "test_indices": test_indices,
        "prediction": prediction,
        "predicted_test_mean": predicted_test_mean,
        "observed_test_mean": float(np.mean(y[test_indices])),
        "mean_preprocessing": mean_preprocessing,
        "mean_components": mean_components,
        "within_preprocessing": within_preprocessing,
        "within_components": within_components,
        "mean_scores": mean_scores,
        "within_scores": within_scores,
        "model_path": path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    absorbance = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    arrays = preprocess_all(absorbance, wavelength)
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    master = pd.read_parquet(multimodal_dir / "master_samples.parquet").set_index("sample_id")
    aligned = master.loc[row_index["sample_id"]].reset_index()
    sample_ids = aligned["sample_id"].to_numpy()
    groups = aligned["cultivar_ascii"].to_numpy()
    cultivars = sorted(np.unique(groups).tolist())

    futures = []
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for target in TARGETS:
            y = aligned[target].to_numpy(float)
            valid = aligned[f"include_nir_{target}"].to_numpy(bool)
            for heldout in cultivars:
                futures.append(
                    executor.submit(
                        run_fold,
                        target,
                        heldout,
                        arrays,
                        y,
                        valid,
                        groups,
                        sample_ids,
                        output_dir / "models",
                    )
                )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"completed {result['target']} held-out {result['heldout_cultivar']}: "
                f"mean {result['mean_preprocessing']}/{result['mean_components']}, "
                f"within {result['within_preprocessing']}/{result['within_components']}"
            )

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    mean_cv_rows: list[dict[str, Any]] = []
    within_cv_rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda row: (row["target"], row["heldout_cultivar"])):
        test = result["test_indices"]
        truth = aligned[result["target"]].to_numpy(float)[test]
        prediction = result["prediction"]
        for index, y_true, y_pred in zip(test, truth, prediction):
            prediction_rows.append(
                {
                    "sample_id": sample_ids[index],
                    "cultivar_ascii": groups[index],
                    "target": result["target"],
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "residual": float(y_pred - y_true),
                }
            )
        fold_rows.append(
            {
                "target": result["target"],
                "heldout_cultivar": result["heldout_cultivar"],
                "predicted_test_mean": result["predicted_test_mean"],
                "observed_test_mean": result["observed_test_mean"],
                "mean_calibration_error": result["predicted_test_mean"] - result["observed_test_mean"],
                **metrics(truth, prediction),
            }
        )
        selection_rows.append(
            {
                "target": result["target"],
                "heldout_cultivar": result["heldout_cultivar"],
                "mean_preprocessing": result["mean_preprocessing"],
                "mean_components": result["mean_components"],
                "within_preprocessing": result["within_preprocessing"],
                "within_components": result["within_components"],
                "model_path": result["model_path"],
            }
        )
        for score in result["mean_scores"]:
            for detail in score["detail"]:
                mean_cv_rows.append(
                    {
                        "target": result["target"],
                        "outer_heldout_cultivar": result["heldout_cultivar"],
                        "preprocessing": score["preprocessing"],
                        "n_components": score["n_components"],
                        "configuration_rmse_cultivar_means": score["rmse_cultivar_means"],
                        **detail,
                    }
                )
        for score in result["within_scores"]:
            for detail in score["detail"]:
                within_cv_rows.append(
                    {
                        "target": result["target"],
                        "outer_heldout_cultivar": result["heldout_cultivar"],
                        "preprocessing": score["preprocessing"],
                        "n_components": score["n_components"],
                        "configuration_macro_within_rmse": score["macro_within_rmse"],
                        **detail,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    selections = pd.DataFrame(selection_rows)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    selections.to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(mean_cv_rows).to_parquet(output_dir / "mean_inner_cv.parquet", index=False, compression="zstd")
    pd.DataFrame(within_cv_rows).to_parquet(output_dir / "within_inner_cv.parquet", index=False, compression="zstd")
    pooled = {
        target: metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        for target, group in predictions.groupby("target")
    }
    summary = {
        "model": "hierarchical_transductive_pls",
        "validation": "nested leave-one-cultivar-out; held-out cultivar spectra used only to compute an unlabeled centroid",
        "pooled_metrics": pooled,
        "mean_calibration_rmse_across_cultivars": {
            target: float(np.sqrt(np.mean(group["mean_calibration_error"] ** 2)))
            for target, group in fold_metrics.groupby("target")
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
