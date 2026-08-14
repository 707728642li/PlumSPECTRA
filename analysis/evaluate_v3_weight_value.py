from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_texture_pls_loco import preprocess_all
from v2_registry import cultivar_registry, trait_registry


ALPHAS = [0.1, 1.0, 10.0, 100.0, 1_000.0]
PREPROCESSING = ["raw", "snv", "sg1", "snv_sg1"]
DEFAULT_DEVELOPMENT_CODES = ["L313", "CHL", "KLD", "WW", "WX"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def feature_matrix(arrays: dict[str, np.ndarray], weight: np.ndarray, preprocessing: str, mode: str) -> np.ndarray:
    if mode == "weight_only":
        return weight[:, None]
    if mode == "nir_weight":
        return np.column_stack([arrays[preprocessing], weight])
    raise ValueError(mode)


def choose_config(
    arrays: dict[str, np.ndarray],
    weight: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    mode: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    splitter = GroupKFold(n_splits=4)
    rows: list[dict[str, object]] = []
    preprocessing_values = ["weight_only"] if mode == "weight_only" else PREPROCESSING
    for preprocessing in preprocessing_values:
        key = "raw" if preprocessing == "weight_only" else preprocessing
        x = feature_matrix(arrays, weight, key, mode)
        for alpha in ALPHAS:
            cultivar_rmse: list[float] = []
            for relative_fit, relative_valid in splitter.split(train_indices, groups=groups[train_indices]):
                fit_indices = train_indices[relative_fit]
                valid_indices = train_indices[relative_valid]
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                estimator.fit(x[fit_indices], y[fit_indices])
                prediction = estimator.predict(x[valid_indices])
                for cultivar in np.unique(groups[valid_indices]):
                    mask = groups[valid_indices] == cultivar
                    cultivar_rmse.append(rmse(y[valid_indices][mask], prediction[mask]))
            rows.append(
                {
                    "mode": mode,
                    "preprocessing": preprocessing,
                    "alpha": alpha,
                    "macro_cultivar_rmse": float(np.mean(cultivar_rmse)),
                }
            )
    scores = pd.DataFrame(rows).sort_values(["macro_cultivar_rmse", "preprocessing", "alpha"])
    return scores.iloc[0].to_dict(), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--ridge-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traits", default="all")
    parser.add_argument("--development-codes", default=",".join(DEFAULT_DEVELOPMENT_CODES))
    args = parser.parse_args()

    registry = trait_registry().loc[lambda frame: frame["model_family"] == "endpoint"]
    target_map = registry.set_index("abbreviation")["target"].to_dict()
    traits = list(target_map) if args.traits == "all" else [value.strip().upper() for value in args.traits.split(",")]
    code_table = cultivar_registry().set_index("cultivar_code")
    development_codes = [value.strip().upper() for value in args.development_codes.split(",")]
    development_ascii = [str(code_table.loc[code, "cultivar_ascii"]) for code in development_codes]

    multimodal = args.multimodal_dir.resolve()
    raw = np.load(multimodal / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal / "wavelength_nm.npy")
    arrays = preprocess_all(raw, wavelength)
    row_index = pd.read_csv(multimodal / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    sample_ids = aligned["sample_id"].astype(str).to_numpy()
    weight = pd.to_numeric(aligned["fruit_weight_g"], errors="coerce").to_numpy(float)
    base_include = aligned["qc_analysis_include"].to_numpy(bool) & np.isfinite(weight)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[pd.DataFrame] = []
    score_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for trait in traits:
        target = str(target_map[trait])
        y = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
        eligible = base_include & np.isfinite(y)
        ridge_path = args.ridge_root.resolve() / f"ridge_{trait}" / "predictions.parquet"
        ridge_predictions = pd.read_parquet(ridge_path).set_index("sample_id")
        for heldout_code, heldout_ascii in zip(development_codes, development_ascii, strict=True):
            train_indices = np.flatnonzero(eligible & (groups != heldout_ascii))
            test_indices = np.flatnonzero(eligible & (groups == heldout_ascii))
            for mode in ["weight_only", "nir_weight"]:
                selected, scores = choose_config(arrays, weight, y, groups, train_indices, mode)
                key = "raw" if selected["preprocessing"] == "weight_only" else str(selected["preprocessing"])
                x = feature_matrix(arrays, weight, key, mode)
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=float(selected["alpha"])))
                estimator.fit(x[train_indices], y[train_indices])
                prediction = estimator.predict(x[test_indices])
                frame = pd.DataFrame(
                    {
                        "sample_id": sample_ids[test_indices],
                        "cultivar_ascii": heldout_ascii,
                        "cultivar_code": heldout_code,
                        "trait": trait,
                        "target": target,
                        "model": "Weight-only Ridge" if mode == "weight_only" else "NIR+weight Ridge",
                        "y_true": y[test_indices],
                        "y_pred": prediction,
                    }
                )
                prediction_rows.append(frame)
                score_rows.append(scores.assign(trait=trait, heldout_code=heldout_code))
                fold_rows.append(
                    {
                        "trait": trait,
                        "cultivar_code": heldout_code,
                        "model": frame["model"].iloc[0],
                        "n": int(len(frame)),
                        "rmse": rmse(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy()),
                        "r2": float(r2_score(frame["y_true"], frame["y_pred"])),
                        "selected_preprocessing": selected["preprocessing"],
                        "selected_alpha": selected["alpha"],
                    }
                )

            sample_order = sample_ids[test_indices]
            baseline = ridge_predictions.loc[sample_order]
            fold_rows.append(
                {
                    "trait": trait,
                    "cultivar_code": heldout_code,
                    "model": "NIR Ridge",
                    "n": int(len(baseline)),
                    "rmse": rmse(baseline["y_true"].to_numpy(float), baseline["y_pred"].to_numpy(float)),
                    "r2": float(r2_score(baseline["y_true"], baseline["y_pred"])),
                    "selected_preprocessing": "nested baseline",
                    "selected_alpha": np.nan,
                }
            )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.to_parquet(output / "predictions_development.parquet", index=False, compression="zstd")
    pd.concat(score_rows, ignore_index=True).to_parquet(output / "inner_scores.parquet", index=False, compression="zstd")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    summary = (
        folds.groupby(["trait", "model"], observed=True)
        .agg(folds=("cultivar_code", "nunique"), macro_rmse=("rmse", "mean"), median_rmse=("rmse", "median"))
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)
    report = {
        "status": "development-only diagnostic",
        "development_codes": development_codes,
        "traits": traits,
        "input_policy": "Fruit weight is nondestructive and available after NIR in the recorded workflow; SSC and pH are not used.",
        "fair_comparison": "NIR+weight must be compared with baselines receiving the same weight input before publication claims.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
