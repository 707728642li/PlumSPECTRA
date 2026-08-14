from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from run_v20_nested_baselines import (
    PREPROCESSING,
    V25_SVR_C_VALUES,
    V25_SVR_EPSILON_VALUES,
    V25_SVR_GAMMA_FACTORS,
    fit_svr_prediction,
    prepare_svr_folds,
    score_svr_config,
)
from train_texture_pls_loco import preprocess_all, regression_metrics
from v2_registry import add_cultivar_code


EXTENDED_EPSILON = [1.5, 2.4, 4.0, 8.0, 10.0]


def selected_row(metadata: dict[str, Any], cv: pd.DataFrame) -> pd.Series:
    choice = metadata["domain_svr_choice"]
    rows = cv.loc[
        cv["stage"].ne("preprocessing_screen")
        & cv["preprocessing"].eq(choice["preprocessing"])
        & np.isclose(cv["C"].astype(float), float(choice["C"]))
        & np.isclose(cv["gamma_factor"].astype(float), float(choice["gamma_factor"]))
        & np.isclose(cv["epsilon_z"].astype(float), float(choice["epsilon_z"]))
    ]
    if len(rows) != 1:
        raise RuntimeError("Recorded formal choice is not unique in the saved inner-CV table")
    return rows.iloc[0]


def boundary_axes(metadata: dict[str, Any], cv: pd.DataFrame) -> list[str]:
    choice = metadata["domain_svr_choice"]
    relevant = cv.loc[
        cv["stage"].ne("preprocessing_screen")
        & cv["preprocessing"].eq(choice["preprocessing"])
    ]
    axes: list[str] = []
    for parameter in ("C", "gamma_factor", "epsilon_z"):
        values = sorted(relevant[parameter].astype(float).unique().tolist())
        value = float(choice[parameter])
        if np.isclose(value, values[0]) or np.isclose(value, values[-1]):
            axes.append(parameter)
    return axes


def run_job(
    old_dir: Path,
    output_root: Path,
    arrays: dict[str, np.ndarray],
    aligned: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, Any]:
    old_metadata = json.loads((old_dir / "metadata.json").read_text(encoding="utf-8"))
    old_cv = pd.read_csv(old_dir / "inner_svr_cv.csv")
    axes = boundary_axes(old_metadata, old_cv)
    if "epsilon_z" not in axes:
        raise RuntimeError(
            f"This targeted extender only accepts epsilon-boundary folds: {old_dir}, axes={axes}"
        )
    if bool(old_metadata.get("test_labels_used_for_selection", True)):
        raise RuntimeError(f"Old formal metadata does not certify train-only selection: {old_dir}")
    target = str(old_metadata["target"])
    trait = str(old_metadata["trait"])
    fold = int(old_metadata["outer_fold"])
    old_choice = old_metadata["domain_svr_choice"]

    sample_ids = aligned["sample_id"].astype(str).to_numpy()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    y = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
    sample_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    manifest_indices = np.asarray(
        [sample_to_index[sample_id] for sample_id in manifest["sample_id"]], dtype=int
    )
    fold_values = manifest["outer_fold"].to_numpy(int)
    train_indices = manifest_indices[fold_values != fold]
    test_indices = manifest_indices[fold_values == fold]
    seed = 20260820 + fold
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + 20_000)
    inner_splits = [
        (train_indices[relative_train], train_indices[relative_valid])
        for relative_train, relative_valid in splitter.split(train_indices, groups[train_indices])
    ]
    preprocessing = str(old_choice["preprocessing"])
    prepared = {preprocessing: prepare_svr_folds(arrays[preprocessing], y, groups, inner_splits)}

    combined_rows = old_cv.to_dict("records")
    evaluated = {
        (
            str(row["preprocessing"]),
            float(row["C"]),
            float(row["gamma_factor"]),
            float(row["epsilon_z"]),
        )
        for row in combined_rows
        if str(row["stage"]) != "preprocessing_screen"
    }
    new_rows = 0
    for epsilon in EXTENDED_EPSILON:
        config = {
            "C": float(old_choice["C"]),
            "gamma_factor": float(old_choice["gamma_factor"]),
            "epsilon_z": float(epsilon),
        }
        key = (preprocessing, config["C"], config["gamma_factor"], config["epsilon_z"])
        if key in evaluated:
            continue
        global_rmse, domain_rmse = score_svr_config(prepared[preprocessing], config)
        combined_rows.append(
            {
                "stage": "strict_superset_epsilon_extension",
                "preprocessing": preprocessing,
                **config,
                "global_cv_rmse": global_rmse,
                "domain_cv_rmse": domain_rmse,
            }
        )
        evaluated.add(key)
        new_rows += 1

    tuning_rows = [row for row in combined_rows if str(row["stage"]) != "preprocessing_screen"]
    choice = min(
        tuning_rows,
        key=lambda row: (
            float(row["domain_cv_rmse"]),
            float(row["C"]),
            float(row["gamma_factor"]),
            float(row["epsilon_z"]),
            PREPROCESSING.index(str(row["preprocessing"])),
        ),
    )
    old_selected = selected_row(old_metadata, old_cv)
    if float(choice["domain_cv_rmse"]) > float(old_selected["domain_cv_rmse"]) + max(
        1e-10, abs(float(old_selected["domain_cv_rmse"])) * 1e-10
    ):
        raise RuntimeError("Strict-superset selection unexpectedly worsened inner-CV RMSE")

    global_svr, domain_svr = fit_svr_prediction(
        arrays, y, groups, train_indices, test_indices, choice
    )
    old_prediction = pd.read_parquet(old_dir / "predictions.parquet").copy()
    expected_ids = sample_ids[test_indices].tolist()
    if old_prediction["sample_id"].astype(str).tolist() != expected_ids:
        raise RuntimeError(f"Saved formal test ordering differs from frozen manifest: {old_dir}")
    if not np.allclose(old_prediction["y_true"], y[test_indices], rtol=0, atol=1e-10):
        raise RuntimeError(f"Saved formal truths differ from the QC ledger: {old_dir}")
    prediction = old_prediction.copy()
    prediction["y_global_svr"] = global_svr
    prediction["y_domain_svr"] = domain_svr
    prediction = add_cultivar_code(prediction)

    run_dir = output_root / trait / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=False)
    prediction.to_parquet(run_dir / "predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(combined_rows).to_csv(run_dir / "inner_svr_cv.csv", index=False)
    shutil_source = old_dir / "inner_pls_cv.csv"
    if not shutil_source.is_file():
        raise FileNotFoundError(shutil_source)
    (run_dir / "inner_pls_cv.csv").write_bytes(shutil_source.read_bytes())
    metrics = {
        name: regression_metrics(prediction["y_true"], prediction[column])
        for name, column in {
            "global_pls": "y_global_pls",
            "domain_pls": "y_domain_pls",
            "global_svr_selected_for_domain": "y_global_svr",
            "domain_svr": "y_domain_svr",
        }.items()
    }
    metadata = {
        **old_metadata,
        "domain_svr_choice": {
            "stage": str(choice["stage"]),
            "preprocessing": str(choice["preprocessing"]),
            "C": float(choice["C"]),
            "gamma_factor": float(choice["gamma_factor"]),
            "epsilon_z": float(choice["epsilon_z"]),
            "global_cv_rmse": float(choice["global_cv_rmse"]),
            "domain_cv_rmse": float(choice["domain_cv_rmse"]),
        },
        "svr_search_space": {
            "C": V25_SVR_C_VALUES,
            "gamma_factor": V25_SVR_GAMMA_FACTORS,
            "epsilon_z": V25_SVR_EPSILON_VALUES,
            "strategy": (
                "strict-superset extension of the saved train-internal V25 search; all old CV "
                "rows were retained and epsilon 2.4/4/8/10 was evaluated along the old boundary winner"
            ),
            "boundary_extension": "saved search plus explicit high-epsilon one-axis extension",
            "hard_limits": {
                "C": [0.0003, 3000.0],
                "gamma_factor": [0.0001, 20.0],
                "epsilon_z": [0.001, 10.0],
            },
        },
        "metrics": metrics,
        "strict_superset_search": True,
        "old_inner_cv_rows_reused": int(len(old_cv)),
        "new_inner_cv_rows_evaluated": int(new_rows),
        "source_boundary_run": str(old_dir.resolve()),
        "test_labels_used_for_selection": False,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "trait": trait,
        "outer_fold": fold,
        "old_epsilon": float(old_choice["epsilon_z"]),
        "new_epsilon": float(choice["epsilon_z"]),
        "old_inner_cv_rmse": float(old_selected["domain_cv_rmse"]),
        "new_inner_cv_rmse": float(choice["domain_cv_rmse"]),
        "new_rows_evaluated": new_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extend completed V25 epsilon-boundary baseline folds by reusing their saved inner-CV "
            "rows and evaluating only the strict-superset high-epsilon candidates."
        )
    )
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix with a nonempty candidate directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    multimodal = args.multimodal_dir.resolve()
    raw = np.load(multimodal / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal / "nir_c_row_index.csv", dtype={"sample_id": str})
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    aligned["sample_id"] = aligned["sample_id"].astype(str)
    arrays = preprocess_all(raw, wavelength)
    manifest = pd.read_csv(
        args.fold_manifest.resolve(), dtype={"sample_id": str, "cultivar_ascii": str}
    )
    formal = args.formal_dir.resolve()
    boundary_dirs: list[Path] = []
    for metadata_path in sorted(formal.glob("*/fold_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cv = pd.read_csv(metadata_path.with_name("inner_svr_cv.csv"))
        if "epsilon_z" in boundary_axes(metadata, cv):
            boundary_dirs.append(metadata_path.parent)
    if not boundary_dirs:
        raise RuntimeError("No epsilon-boundary formal folds were found")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_job, path, output, arrays, aligned, manifest): path
            for path in boundary_dirs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    pd.DataFrame(results).sort_values(["trait", "outer_fold"]).to_csv(
        output / "strict_superset_extension_summary.csv", index=False
    )
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "V25 saved-inner-CV strict-superset epsilon extension",
                "formal_source": str(formal),
                "folds": len(results),
                "test_labels_used_for_selection": False,
                "results": sorted(results, key=lambda row: (row["trait"], row["outer_fold"])),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
