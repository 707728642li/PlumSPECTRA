from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold

from train_texture_pls_loco import DEFAULT_TARGETS, PREPROCESSING, preprocess_all, regression_metrics
from v2_registry import abbreviated_trait


QUALITY_TARGETS = ["fruit_weight_g", "soluble_solids_pct", "ph"]
TARGETS = [*QUALITY_TARGETS, *DEFAULT_TARGETS]
COMPONENTS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24]


def offsets(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    return {
        str(cultivar): np.mean(y_true[groups == cultivar] - y_pred[groups == cultivar], axis=0)
        for cultivar in np.unique(groups)
    }


def apply_offsets(prediction: np.ndarray, groups: np.ndarray, values: dict[str, np.ndarray]) -> np.ndarray:
    return prediction + np.vstack([values[str(group)] for group in groups])


def macro_scaled_rmse(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> float:
    trait_rmse = np.sqrt(np.mean(np.square(y_true - y_pred), axis=0))
    return float(np.mean(trait_rmse / np.maximum(scale, 1e-12)))


def choose_configuration(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    splits = [
        (train_indices[fit], train_indices[valid])
        for fit, valid in splitter.split(train_indices, groups[train_indices])
    ]
    rows = []
    for preprocessing in PREPROCESSING:
        for n_components in COMPONENTS:
            global_scores = []
            domain_scores = []
            for fit_indices, valid_indices in splits:
                model = PLSRegression(
                    n_components=n_components, scale=True, max_iter=1000, tol=1e-7
                )
                model.fit(arrays[preprocessing][fit_indices], y[fit_indices])
                fit_prediction = model.predict(arrays[preprocessing][fit_indices])
                valid_prediction = model.predict(arrays[preprocessing][valid_indices])
                y_scale = np.std(y[fit_indices], axis=0, ddof=1)
                domain_prediction = apply_offsets(
                    valid_prediction,
                    groups[valid_indices],
                    offsets(y[fit_indices], fit_prediction, groups[fit_indices]),
                )
                global_scores.append(macro_scaled_rmse(y[valid_indices], valid_prediction, y_scale))
                domain_scores.append(macro_scaled_rmse(y[valid_indices], domain_prediction, y_scale))
            rows.append(
                {
                    "preprocessing": preprocessing,
                    "n_components": n_components,
                    "global_macro_scaled_rmse": float(np.mean(global_scores)),
                    "domain_macro_scaled_rmse": float(np.mean(domain_scores)),
                }
            )
    global_choice = min(
        rows,
        key=lambda row: (
            row["global_macro_scaled_rmse"], row["n_components"], PREPROCESSING.index(row["preprocessing"])
        ),
    )
    domain_choice = min(
        rows,
        key=lambda row: (
            row["domain_macro_scaled_rmse"], row["n_components"], PREPROCESSING.index(row["preprocessing"])
        ),
    )
    return global_choice, domain_choice, rows


def fit_predict(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    choice: dict[str, Any],
    domain: bool,
) -> np.ndarray:
    model = PLSRegression(
        n_components=int(choice["n_components"]), scale=True, max_iter=1000, tol=1e-7
    )
    x = arrays[str(choice["preprocessing"])]
    model.fit(x[train_indices], y[train_indices])
    prediction = model.predict(x[test_indices])
    if domain:
        prediction = apply_offsets(
            prediction,
            groups[test_indices],
            offsets(y[train_indices], model.predict(x[train_indices]), groups[train_indices]),
        )
    return prediction


def run_fold(
    outer_fold: int,
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    sample_ids: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    run_dir = output / f"fold_{outer_fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "predictions.parquet"
    if completed.exists() and (run_dir / "metadata.json").exists():
        return {"outer_fold": int(outer_fold), "status": "skipped"}
    train = np.flatnonzero(folds != outer_fold)
    test = np.flatnonzero(folds == outer_fold)
    global_choice, domain_choice, rows = choose_configuration(
        arrays, y, groups, train, 20260810 + outer_fold
    )
    global_prediction = fit_predict(arrays, y, groups, train, test, global_choice, False)
    domain_prediction = fit_predict(arrays, y, groups, train, test, domain_choice, True)
    long_rows = []
    for trait_index, target in enumerate(TARGETS):
        for row_index, sample_index in enumerate(test):
            long_rows.append(
                {
                    "sample_id": sample_ids[sample_index],
                    "cultivar_ascii": groups[sample_index],
                    "target": target,
                    "trait": abbreviated_trait(target),
                    "outer_fold": outer_fold,
                    "y_true": float(y[sample_index, trait_index]),
                    "y_global_pls2": float(global_prediction[row_index, trait_index]),
                    "y_domain_pls2": float(domain_prediction[row_index, trait_index]),
                }
            )
    prediction = pd.DataFrame(long_rows)
    prediction.to_parquet(completed, index=False, compression="zstd")
    pd.DataFrame(rows).to_csv(run_dir / "inner_pls2_cv.csv", index=False)
    metadata = {
        "protocol": "V25 equal-information 12-response PLS2 common-case baseline",
        "outer_fold": int(outer_fold),
        "train_fruits": int(len(train)),
        "test_fruits": int(len(test)),
        "targets": TARGETS,
        "global_choice": global_choice,
        "domain_choice": domain_choice,
        "component_grid": COMPONENTS,
        "preprocessing_grid": PREPROCESSING,
        "test_labels_used_for_selection": False,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"outer_fold": int(outer_fold), "status": "completed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    multimodal = args.multimodal_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    absorbance = np.load(multimodal / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal / "nir_c_row_index.csv", dtype={"sample_id": str})
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    manifest_path = args.fold_manifest.resolve()
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str, "cultivar_ascii": str})
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    aligned = aligned.set_index("sample_id").loc[manifest["sample_id"]].reset_index()
    if not aligned[TARGETS].apply(pd.to_numeric, errors="coerce").notna().all().all():
        raise ValueError("PLS2 common-case manifest contains missing targets")
    arrays_all = preprocess_all(absorbance, wavelength)
    row_lookup = pd.Series(np.arange(len(row_index)), index=row_index["sample_id"])
    indices = row_lookup.loc[manifest["sample_id"]].to_numpy(int)
    arrays = {name: values[indices] for name, values in arrays_all.items()}
    y = aligned[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    sample_ids = manifest["sample_id"].to_numpy(str)
    groups = manifest["cultivar_ascii"].to_numpy(str)
    folds = manifest["outer_fold"].to_numpy(int)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_fold, fold, arrays, y, sample_ids, groups, folds, output): fold
            for fold in sorted(np.unique(folds))
        }
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)

    predictions = pd.concat(
        [pd.read_parquet(path) for path in sorted(output.glob("fold_*/predictions.parquet"))],
        ignore_index=True,
    )
    if len(predictions) != len(manifest) * len(TARGETS):
        raise RuntimeError("PLS2 prediction coverage is incomplete")
    predictions.to_parquet(output / "predictions.parquet", index=False, compression="zstd")
    metric_rows = []
    for (target, trait), group in predictions.groupby(["target", "trait"], observed=True):
        for model, column in (("global_pls2", "y_global_pls2"), ("cultivar_aware_pls2", "y_domain_pls2")):
            metric_rows.append(
                {"target": target, "trait": trait, "model": model, **regression_metrics(group["y_true"], group[column])}
            )
    pd.DataFrame(metric_rows).to_csv(output / "pooled_metrics.csv", index=False)
    summary = {
        "protocol": "V25 equal-information multi-response PLS2 common-case baseline",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "fruits": int(len(manifest)),
        "traits": len(TARGETS),
        "prediction_rows": int(len(predictions)),
        "outer_folds": [int(value) for value in sorted(np.unique(folds))],
        "test_labels_used_for_selection": False,
        "claim_boundary": "Common complete-case comparison; masked auxiliary labels and PLS2 both use all 12 training responses.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
