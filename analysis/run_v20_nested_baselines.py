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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from train_texture_pls_loco import (
    COMPONENTS,
    DEFAULT_TARGETS,
    PREPROCESSING,
    preprocess_all,
    regression_metrics,
)
from v2_registry import abbreviated_trait, add_cultivar_code


SVR_CONFIGS = [
    {"C": 3.0, "gamma_factor": 0.5, "epsilon_z": 0.08},
    {"C": 3.0, "gamma_factor": 1.0, "epsilon_z": 0.08},
    {"C": 10.0, "gamma_factor": 0.25, "epsilon_z": 0.08},
    {"C": 10.0, "gamma_factor": 0.5, "epsilon_z": 0.03},
    {"C": 10.0, "gamma_factor": 1.0, "epsilon_z": 0.03},
    {"C": 10.0, "gamma_factor": 1.0, "epsilon_z": 0.08},
    {"C": 10.0, "gamma_factor": 2.0, "epsilon_z": 0.08},
    {"C": 30.0, "gamma_factor": 0.5, "epsilon_z": 0.03},
    {"C": 30.0, "gamma_factor": 1.0, "epsilon_z": 0.03},
    {"C": 100.0, "gamma_factor": 0.5, "epsilon_z": 0.03},
]

V25_PLS_COMPONENTS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24]
V25_SVR_C_VALUES = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
V25_SVR_GAMMA_FACTORS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
# The wide audit must be a strict superset of the first V25 correction search.
# In particular, keep 1.5 explicitly: merely lifting the hard limit changed the
# adaptive proposals from 1.5 to 2.4/8.0 and accidentally skipped the former
# boundary winner.  The additional high-epsilon points let inner CV decide
# whether a nearly constant residual model is preferable without losing any
# previously evaluated candidate.
V25_SVR_EPSILON_VALUES = [
    0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20,
    0.30, 0.40, 0.60, 0.80, 1.50, 2.40, 4.00, 8.00,
]

QUALITY_TARGETS = ["fruit_weight_g", "soluble_solids_pct", "ph"]
SUPPORTED_TARGETS = [*DEFAULT_TARGETS, *QUALITY_TARGETS]


def cultivar_offsets(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    residual = y_true - y_pred
    return {
        str(cultivar): float(np.mean(residual[groups == cultivar]))
        for cultivar in np.unique(groups)
    }


def apply_offsets(
    prediction: np.ndarray, groups: np.ndarray, offsets: dict[str, float]
) -> np.ndarray:
    return prediction + np.asarray([offsets[str(group)] for group in groups], dtype=float)


def choose_pls(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    components: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    components = COMPONENTS if components is None else components
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    inner_splits = [
        (train_indices[relative_train], train_indices[relative_valid])
        for relative_train, relative_valid in splitter.split(
            train_indices, groups[train_indices]
        )
    ]
    rows: list[dict[str, Any]] = []
    for preprocessing in PREPROCESSING:
        for n_components in components:
            global_errors: list[np.ndarray] = []
            domain_errors: list[np.ndarray] = []
            for fit_indices, valid_indices in inner_splits:
                model = PLSRegression(
                    n_components=n_components, scale=True, max_iter=1000, tol=1e-7
                )
                model.fit(arrays[preprocessing][fit_indices], y[fit_indices])
                fit_prediction = model.predict(arrays[preprocessing][fit_indices]).ravel()
                valid_prediction = model.predict(arrays[preprocessing][valid_indices]).ravel()
                offsets = cultivar_offsets(
                    y[fit_indices], fit_prediction, groups[fit_indices]
                )
                domain_prediction = apply_offsets(
                    valid_prediction, groups[valid_indices], offsets
                )
                global_errors.append((valid_prediction - y[valid_indices]) ** 2)
                domain_errors.append((domain_prediction - y[valid_indices]) ** 2)
            rows.append(
                {
                    "preprocessing": preprocessing,
                    "n_components": int(n_components),
                    "global_cv_rmse": float(
                        np.sqrt(np.mean(np.concatenate(global_errors)))
                    ),
                    "domain_cv_rmse": float(
                        np.sqrt(np.mean(np.concatenate(domain_errors)))
                    ),
                }
            )
    global_choice = min(
        rows,
        key=lambda row: (
            row["global_cv_rmse"],
            row["n_components"],
            PREPROCESSING.index(row["preprocessing"]),
        ),
    )
    domain_choice = min(
        rows,
        key=lambda row: (
            row["domain_cv_rmse"],
            row["n_components"],
            PREPROCESSING.index(row["preprocessing"]),
        ),
    )
    return global_choice, domain_choice, rows


def fit_pls_prediction(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    choice: dict[str, Any],
    domain_calibration: bool,
) -> np.ndarray:
    preprocessing = str(choice["preprocessing"])
    model = PLSRegression(
        n_components=int(choice["n_components"]),
        scale=True,
        max_iter=1000,
        tol=1e-7,
    )
    model.fit(arrays[preprocessing][train_indices], y[train_indices])
    prediction = model.predict(arrays[preprocessing][test_indices]).ravel()
    if domain_calibration:
        fit_prediction = model.predict(arrays[preprocessing][train_indices]).ravel()
        offsets = cultivar_offsets(y[train_indices], fit_prediction, groups[train_indices])
        prediction = apply_offsets(prediction, groups[test_indices], offsets)
    return prediction


def prepare_svr_folds(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for fit_indices, valid_indices in inner_splits:
        scaler = StandardScaler()
        x_fit = scaler.fit_transform(x[fit_indices])
        x_valid = scaler.transform(x[valid_indices])
        y_mean = float(np.mean(y[fit_indices]))
        y_sd = max(float(np.std(y[fit_indices], ddof=1)), 1e-12)
        prepared.append(
            {
                "x_fit": x_fit,
                "x_valid": x_valid,
                "y_fit_z": (y[fit_indices] - y_mean) / y_sd,
                "y_fit": y[fit_indices],
                "y_valid": y[valid_indices],
                "fit_groups": groups[fit_indices],
                "valid_groups": groups[valid_indices],
                "y_mean": y_mean,
                "y_sd": y_sd,
            }
        )
    return prepared


def score_svr_config(
    prepared_folds: list[dict[str, Any]], config: dict[str, float]
) -> tuple[float, float]:
    global_errors: list[np.ndarray] = []
    domain_errors: list[np.ndarray] = []
    for fold in prepared_folds:
        gamma = float(config["gamma_factor"]) / fold["x_fit"].shape[1]
        model = SVR(
            kernel="rbf",
            C=float(config["C"]),
            gamma=gamma,
            epsilon=float(config["epsilon_z"]),
            cache_size=1600,
        )
        model.fit(fold["x_fit"], fold["y_fit_z"])
        fit_prediction = model.predict(fold["x_fit"]) * fold["y_sd"] + fold["y_mean"]
        valid_prediction = (
            model.predict(fold["x_valid"]) * fold["y_sd"] + fold["y_mean"]
        )
        offsets = cultivar_offsets(
            fold["y_fit"], fit_prediction, fold["fit_groups"]
        )
        domain_prediction = apply_offsets(
            valid_prediction, fold["valid_groups"], offsets
        )
        global_errors.append((valid_prediction - fold["y_valid"]) ** 2)
        domain_errors.append((domain_prediction - fold["y_valid"]) ** 2)
    return (
        float(np.sqrt(np.mean(np.concatenate(global_errors)))),
        float(np.sqrt(np.mean(np.concatenate(domain_errors)))),
    )


def choose_svr(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    seed: int,
    search_profile: str = "legacy",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    inner_splits = [
        (train_indices[relative_train], train_indices[relative_valid])
        for relative_train, relative_valid in splitter.split(
            train_indices, groups[train_indices]
        )
    ]
    prepared = {
        preprocessing: prepare_svr_folds(
            arrays[preprocessing], y, groups, inner_splits
        )
        for preprocessing in PREPROCESSING
    }
    if search_profile == "legacy":
        screening_configs = [
            {"C": 10.0, "gamma_factor": 1.0, "epsilon_z": 0.08}
        ]
    elif search_profile == "v25_staged":
        screening_configs = [
            {"C": c, "gamma_factor": gamma, "epsilon_z": 0.05}
            for c in (3.0, 30.0)
            for gamma in (0.05, 0.25)
        ]
    else:
        raise ValueError(f"Unknown SVR search profile: {search_profile}")
    screening_rows: list[dict[str, Any]] = []
    for preprocessing in PREPROCESSING:
        for screening_config in screening_configs:
            global_rmse, domain_rmse = score_svr_config(
                prepared[preprocessing], screening_config
            )
            screening_rows.append(
                {
                    "stage": "preprocessing_screen",
                    "preprocessing": preprocessing,
                    **screening_config,
                    "global_cv_rmse": global_rmse,
                    "domain_cv_rmse": domain_rmse,
                }
            )
    preprocessing_best = {
        preprocessing: min(
            (
                row
                for row in screening_rows
                if row["preprocessing"] == preprocessing
            ),
            key=lambda row: (
                row["domain_cv_rmse"],
                row["C"],
                row["gamma_factor"],
            ),
        )
        for preprocessing in PREPROCESSING
    }
    selected_preprocessing = [
        preprocessing
        for preprocessing in sorted(
            PREPROCESSING,
            key=lambda preprocessing: (
                preprocessing_best[preprocessing]["domain_cv_rmse"],
                PREPROCESSING.index(preprocessing),
            ),
        )[:2]
    ]
    tuning_rows: list[dict[str, Any]] = []
    if search_profile == "legacy":
        configs_by_preprocessing = {
            preprocessing: [("parameter_tuning", config) for config in SVR_CONFIGS]
            for preprocessing in selected_preprocessing
        }
    else:
        configs_by_preprocessing: dict[str, list[tuple[str, dict[str, float]]]] = {}
        for preprocessing in selected_preprocessing:
            coarse = [
                {
                    "C": c,
                    "gamma_factor": gamma,
                    "epsilon_z": 0.05,
                }
                for c in V25_SVR_C_VALUES
                for gamma in V25_SVR_GAMMA_FACTORS
            ]
            coarse_scored: list[tuple[float, dict[str, float]]] = []
            for config in coarse:
                global_rmse, domain_rmse = score_svr_config(
                    prepared[preprocessing], config
                )
                tuning_rows.append(
                    {
                        "stage": "coarse_c_gamma",
                        "preprocessing": preprocessing,
                        **config,
                        "global_cv_rmse": global_rmse,
                        "domain_cv_rmse": domain_rmse,
                    }
                )
                coarse_scored.append((domain_rmse, config))
            top_pairs = [
                config
                for _, config in sorted(
                    coarse_scored,
                    key=lambda item: (
                        item[0],
                        item[1]["C"],
                        item[1]["gamma_factor"],
                    ),
                )[:2]
            ]
            refinement: list[tuple[str, dict[str, float]]] = []
            seen = set()
            for pair in top_pairs:
                for epsilon in V25_SVR_EPSILON_VALUES:
                    config = {
                        "C": float(pair["C"]),
                        "gamma_factor": float(pair["gamma_factor"]),
                        "epsilon_z": float(epsilon),
                    }
                    key = (config["C"], config["gamma_factor"], config["epsilon_z"])
                    if key not in seen and epsilon != 0.05:
                        refinement.append(("epsilon_refinement", config))
                        seen.add(key)
            configs_by_preprocessing[preprocessing] = refinement
    for preprocessing in selected_preprocessing:
        for stage, config in configs_by_preprocessing[preprocessing]:
            global_rmse, domain_rmse = score_svr_config(
                prepared[preprocessing], config
            )
            tuning_rows.append(
                {
                    "stage": stage,
                    "preprocessing": preprocessing,
                    **config,
                    "global_cv_rmse": global_rmse,
                    "domain_cv_rmse": domain_rmse,
                }
            )
    if search_profile == "v25_staged":
        # A boundary-aware extension prevents the corrected baseline from
        # repeating the legacy failure mode. Extension remains entirely inside
        # outer-training CV and varies one axis at a time around the current
        # winner, so it is much cheaper than an enormous Cartesian grid.
        hard_limits = {
            "C": (0.0003, 3000.0),
            "gamma_factor": (0.0001, 20.0),
            # A standardized epsilon of 1.5 was still selected at the upper
            # limit for all five pH folds in the first V25 correction run.
            # That is evidence that 1.5 was a search boundary, not evidence
            # that it was optimal.  Ten target SD is deliberately wider than
            # any practically useful epsilon and lets inner CV converge to a
            # near-constant residual model without clipping the search.
            "epsilon_z": (0.001, 10.0),
        }
        evaluated = {
            (
                str(row["preprocessing"]),
                float(row["C"]),
                float(row["gamma_factor"]),
                float(row["epsilon_z"]),
            )
            for row in tuning_rows
        }
        for _ in range(8):
            current = min(
                tuning_rows,
                key=lambda row: (
                    row["domain_cv_rmse"],
                    row["C"],
                    row["gamma_factor"],
                    row["epsilon_z"],
                    PREPROCESSING.index(row["preprocessing"]),
                ),
            )
            preprocessing = str(current["preprocessing"])
            same_preprocessing = [
                row for row in tuning_rows if row["preprocessing"] == preprocessing
            ]
            new_configs: list[dict[str, float]] = []
            for parameter in ("C", "gamma_factor", "epsilon_z"):
                values = sorted({float(row[parameter]) for row in same_preprocessing})
                value = float(current[parameter])
                lower, upper = hard_limits[parameter]
                proposals: list[float] = []
                if np.isclose(value, values[0]) and value > lower:
                    proposals.extend([max(lower, value / 3.0), max(lower, value / 10.0)])
                if np.isclose(value, values[-1]) and value < upper:
                    proposals.extend([min(upper, value * 3.0), min(upper, value * 10.0)])
                for proposal in proposals:
                    config = {
                        "C": float(current["C"]),
                        "gamma_factor": float(current["gamma_factor"]),
                        "epsilon_z": float(current["epsilon_z"]),
                    }
                    config[parameter] = float(proposal)
                    key = (
                        preprocessing,
                        config["C"],
                        config["gamma_factor"],
                        config["epsilon_z"],
                    )
                    if key not in evaluated:
                        new_configs.append(config)
                        evaluated.add(key)
            if not new_configs:
                break
            for config in new_configs:
                global_rmse, domain_rmse = score_svr_config(
                    prepared[preprocessing], config
                )
                tuning_rows.append(
                    {
                        "stage": "adaptive_boundary_extension",
                        "preprocessing": preprocessing,
                        **config,
                        "global_cv_rmse": global_rmse,
                        "domain_cv_rmse": domain_rmse,
                    }
                )
    choice = min(
        tuning_rows,
        key=lambda row: (
            row["domain_cv_rmse"],
            row["C"],
            row["gamma_factor"],
            row["epsilon_z"],
            PREPROCESSING.index(row["preprocessing"]),
        ),
    )
    return choice, [*screening_rows, *tuning_rows]


def fit_svr_prediction(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    choice: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    x = arrays[str(choice["preprocessing"])]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_indices])
    x_test = scaler.transform(x[test_indices])
    y_mean = float(np.mean(y[train_indices]))
    y_sd = max(float(np.std(y[train_indices], ddof=1)), 1e-12)
    model = SVR(
        kernel="rbf",
        C=float(choice["C"]),
        gamma=float(choice["gamma_factor"]) / x_train.shape[1],
        epsilon=float(choice["epsilon_z"]),
        cache_size=1600,
    )
    model.fit(x_train, (y[train_indices] - y_mean) / y_sd)
    fit_prediction = model.predict(x_train) * y_sd + y_mean
    global_prediction = model.predict(x_test) * y_sd + y_mean
    offsets = cultivar_offsets(
        y[train_indices], fit_prediction, groups[train_indices]
    )
    domain_prediction = apply_offsets(
        global_prediction, groups[test_indices], offsets
    )
    return global_prediction, domain_prediction


def evaluate_job(
    target: str,
    outer_fold: int,
    arrays: dict[str, np.ndarray],
    aligned: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: Path,
    pls_components: list[int],
    svr_search_profile: str,
) -> dict[str, Any]:
    trait = abbreviated_trait(target)
    run_dir = output_dir / trait / f"fold_{outer_fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    completed_path = run_dir / "predictions.parquet"
    if completed_path.exists() and (run_dir / "metadata.json").exists():
        return {"target": target, "trait": trait, "outer_fold": outer_fold, "status": "skipped"}

    sample_ids = aligned["sample_id"].astype(str).to_numpy()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    y = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
    sample_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    manifest_indices = np.asarray(
        [sample_to_index[sample_id] for sample_id in manifest["sample_id"]], dtype=int
    )
    fold_values = manifest["outer_fold"].to_numpy(int)
    train_indices = manifest_indices[fold_values != outer_fold]
    test_indices = manifest_indices[fold_values == outer_fold]
    seed = 20260820 + outer_fold

    global_pls_choice, domain_pls_choice, pls_cv = choose_pls(
        arrays, y, groups, train_indices, seed + 10_000, pls_components
    )
    global_pls = fit_pls_prediction(
        arrays, y, groups, train_indices, test_indices, global_pls_choice, False
    )
    domain_pls = fit_pls_prediction(
        arrays, y, groups, train_indices, test_indices, domain_pls_choice, True
    )
    svr_choice, svr_cv = choose_svr(
        arrays, y, groups, train_indices, seed + 20_000, svr_search_profile
    )
    global_svr, domain_svr = fit_svr_prediction(
        arrays, y, groups, train_indices, test_indices, svr_choice
    )

    prediction = pd.DataFrame(
        {
            "sample_id": sample_ids[test_indices],
            "cultivar_ascii": groups[test_indices],
            "target": target,
            "trait": trait,
            "outer_fold": outer_fold,
            "y_true": y[test_indices],
            "y_global_pls": global_pls,
            "y_domain_pls": domain_pls,
            "y_global_svr": global_svr,
            "y_domain_svr": domain_svr,
        }
    )
    prediction = add_cultivar_code(prediction)
    prediction.to_parquet(completed_path, index=False, compression="zstd")
    pd.DataFrame(pls_cv).to_csv(run_dir / "inner_pls_cv.csv", index=False)
    pd.DataFrame(svr_cv).to_csv(run_dir / "inner_svr_cv.csv", index=False)
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
        "protocol": (
            "V25 corrected-cohort non-overlapping five-fold nested baseline audit"
            if svr_search_profile == "v25_staged"
            else "V20 frozen non-overlapping five-fold nested baseline audit"
        ),
        "target": target,
        "trait": trait,
        "outer_fold": outer_fold,
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "global_pls_choice": global_pls_choice,
        "domain_pls_choice": domain_pls_choice,
        "domain_svr_choice": svr_choice,
        "pls_component_grid": pls_components,
        "svr_search_profile": svr_search_profile,
        "svr_search_space": (
            {
                "C": V25_SVR_C_VALUES,
                "gamma_factor": V25_SVR_GAMMA_FACTORS,
                "epsilon_z": V25_SVR_EPSILON_VALUES,
                "strategy": "broad C-gamma grid at epsilon=0.05 followed by epsilon refinement for the two best C-gamma pairs per preprocessing",
                "boundary_extension": "up to eight train-internal one-axis extensions around any boundary winner",
                "hard_limits": {"C": [0.0003, 3000.0], "gamma_factor": [0.0001, 20.0], "epsilon_z": [0.001, 10.0]},
            }
            if svr_search_profile == "v25_staged"
            else {"configs": SVR_CONFIGS}
        ),
        "metrics": metrics,
        "test_labels_used_for_selection": False,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"target": target, "trait": trait, "outer_fold": outer_fold, "status": "completed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--pls-components",
        default=",".join(str(value) for value in COMPONENTS),
        help="Comma-separated PLS component grid. Use 1,2,3,4,5,6,7,8,12,16,24 for the V25 correction.",
    )
    parser.add_argument(
        "--svr-search-profile",
        choices=("legacy", "v25_staged"),
        default="legacy",
    )
    args = parser.parse_args()

    multimodal_dir = args.multimodal_dir.resolve()
    raw = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    aligned["sample_id"] = aligned["sample_id"].astype(str)
    arrays = preprocess_all(raw, wavelength)
    manifest_path = args.fold_manifest.resolve()
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str, "cultivar_ascii": str})
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Fold manifest contains duplicate sample IDs")
    if set(manifest["sample_id"]) - set(aligned["sample_id"]):
        raise ValueError("Fold manifest contains samples absent from aligned NIR data")

    targets = (
        DEFAULT_TARGETS
        if args.targets == "all"
        else [value.strip() for value in args.targets.split(",") if value.strip()]
    )
    unknown_targets = sorted(set(targets) - set(SUPPORTED_TARGETS))
    if unknown_targets:
        raise ValueError(f"Unknown targets: {unknown_targets}")
    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    pls_components = sorted(
        {int(value.strip()) for value in args.pls_components.split(",") if value.strip()}
    )
    if not pls_components or min(pls_components) < 1:
        raise ValueError("PLS components must be positive integers")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(target, fold) for target in targets for fold in folds]
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_job,
                target,
                fold,
                arrays,
                aligned,
                manifest,
                output_dir,
                pls_components,
                args.svr_search_profile,
            ): (target, fold)
            for target, fold in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps(result), flush=True)

    prediction_paths = sorted(output_dir.glob("*/fold_*/predictions.parquet"))
    all_predictions = pd.concat(
        [pd.read_parquet(path) for path in prediction_paths], ignore_index=True
    )
    all_predictions.to_parquet(
        output_dir / "predictions.parquet", index=False, compression="zstd"
    )
    metric_rows: list[dict[str, Any]] = []
    for (target, trait), frame in all_predictions.groupby(["target", "trait"], observed=True):
        for model, column in {
            "global_pls": "y_global_pls",
            "domain_pls": "y_domain_pls",
            "global_svr_selected_for_domain": "y_global_svr",
            "domain_svr": "y_domain_svr",
        }.items():
            metric_rows.append(
                {"target": target, "trait": trait, "model": model, **regression_metrics(frame["y_true"], frame[column])}
            )
    pd.DataFrame(metric_rows).to_csv(output_dir / "pooled_metrics.csv", index=False)
    summary = {
        "protocol": "V20 nested strong baselines",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "targets_completed": int(all_predictions["target"].nunique()),
        "folds_completed": sorted(all_predictions["outer_fold"].unique().tolist()),
        "prediction_rows": int(len(all_predictions)),
        "test_labels_used_for_selection": False,
        "pls_component_grid": pls_components,
        "svr_search_profile": args.svr_search_profile,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
