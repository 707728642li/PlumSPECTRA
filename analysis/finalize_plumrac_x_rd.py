from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_v3_confirmation import (
    cultivar_cluster_bootstrap,
    exact_sign_flip_test,
    fold_metrics,
    regression_metrics,
    rmse,
)


DEVELOPMENT_CODES = {"L313", "CHL", "KLD", "WW", "WX"}
CONFIRMATION_CODES = {"L611", "LA191", "FTL", "FWHH", "FRL", "L31", "NL", "QCL", "WD", "WJ", "ZSKLD"}
EXPECTED_SEEDS = {20260806, 20260807, 20260808, 20260809}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    folds = fold_metrics(frame, "y_pred")
    pls = float(folds["plsr_rmse"].mean())
    ai = float(folds["plumrac_x_rmse"].mean())
    return {
        "cultivars": int(len(folds)),
        "fold_wins": int(folds["plumrac_x_win"].sum()),
        "macro_plsr_rmse": pls,
        "macro_plumrac_x_rmse": ai,
        "macro_rmse_improvement_pct": 100.0 * (pls - ai) / pls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-seed06", type=Path, required=True)
    parser.add_argument("--development-extra-seeds", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--v2-rd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_paths = [args.development_seed06, args.development_extra_seeds, args.confirmation]
    predictions = pd.concat([pd.read_parquet(path) for path in source_paths], ignore_index=True)
    expected_codes = DEVELOPMENT_CODES | CONFIRMATION_CODES
    if set(predictions["cultivar_code"].astype(str).unique()) != expected_codes:
        raise ValueError("The combined RD data do not contain the frozen 16-cultivar set")
    if set(predictions["seed"].astype(int).unique()) != EXPECTED_SEEDS:
        raise ValueError("The combined RD data do not contain all four frozen seeds")
    duplicate_key = ["sample_id", "seed"]
    if predictions.duplicated(duplicate_key).any():
        raise ValueError("Duplicate sample/seed predictions found while combining RD sources")
    counts = predictions.groupby("sample_id", observed=True)["seed"].nunique()
    if not (counts == 4).all():
        raise ValueError("Every RD fruit must have exactly four seed predictions")

    ensemble = (
        predictions.groupby(["sample_id", "cultivar_ascii", "cultivar_code", "target"], observed=True, as_index=False)
        .agg(y_true=("y_true", "first"), y_pls_anchor=("y_pls_anchor", "first"), y_pred=("y_pred", "mean"))
    )
    if len(ensemble) != 5430:
        raise ValueError(f"Expected 5,430 RD fruits, found {len(ensemble)}")
    folds = fold_metrics(ensemble, "y_pred")
    bootstrap = cultivar_cluster_bootstrap(folds)
    sign_flip = exact_sign_flip_test(folds)
    truth = ensemble["y_true"].to_numpy(float)
    pls = ensemble["y_pls_anchor"].to_numpy(float)
    ai = ensemble["y_pred"].to_numpy(float)
    macro_pls = float(folds["plsr_rmse"].mean())
    macro_ai = float(folds["plumrac_x_rmse"].mean())

    seed_rows: list[dict[str, float | int]] = []
    for seed, group in predictions.groupby("seed", observed=True):
        seed_folds = fold_metrics(group, "y_pred")
        seed_pls = float(seed_folds["plsr_rmse"].mean())
        seed_ai = float(seed_folds["plumrac_x_rmse"].mean())
        seed_rows.append(
            {
                "seed": int(seed),
                "fold_wins": int(seed_folds["plumrac_x_win"].sum()),
                "macro_plsr_rmse": seed_pls,
                "macro_plumrac_x_rmse": seed_ai,
                "macro_rmse_improvement_pct": 100.0 * (seed_pls - seed_ai) / seed_pls,
                "pooled_plsr_rmse": rmse(group["y_true"].to_numpy(float), group["y_pls_anchor"].to_numpy(float)),
                "pooled_plumrac_x_rmse": rmse(group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float)),
            }
        )
    seed_table = pd.DataFrame(seed_rows).sort_values("seed")
    seed_table["pooled_rmse_improvement_pct"] = 100.0 * (
        seed_table["pooled_plsr_rmse"] - seed_table["pooled_plumrac_x_rmse"]
    ) / seed_table["pooled_plsr_rmse"]

    v2 = pd.read_parquet(args.v2_rd)
    v2_folds = fold_metrics(v2, "y_pred")
    v2_truth = v2["y_true"].to_numpy(float)
    v2_ai = v2["y_pred"].to_numpy(float)
    summary = {
        "model_family": "PLUMRAC-X",
        "scope": "Descriptive complete 16-cultivar RD result; five cultivars participated in V3 development",
        "n_unique_fruits": int(len(ensemble)),
        "cultivars": int(len(folds)),
        "seeds": sorted(EXPECTED_SEEDS),
        "fold_wins_vs_plsr": int(folds["plumrac_x_win"].sum()),
        "macro_plsr_rmse": macro_pls,
        "macro_plumrac_x_rmse": macro_ai,
        "macro_rmse_improvement_pct": 100.0 * (macro_pls - macro_ai) / macro_pls,
        "pooled_plsr": regression_metrics(truth, pls),
        "pooled_plumrac_x": regression_metrics(truth, ai),
        "pooled_rmse_improvement_pct": 100.0 * (rmse(truth, pls) - rmse(truth, ai)) / rmse(truth, pls),
        "development_subset": macro_summary(ensemble.loc[ensemble["cultivar_code"].isin(DEVELOPMENT_CODES)]),
        "sealed_confirmation_subset": macro_summary(ensemble.loc[ensemble["cultivar_code"].isin(CONFIRMATION_CODES)]),
        "cultivar_cluster_bootstrap": bootstrap,
        "exact_paired_sign_flip": sign_flip,
        "v2_comparator": {
            "fold_wins": int(v2_folds["plumrac_x_win"].sum()),
            "macro_rmse": float(v2_folds["plumrac_x_rmse"].mean()),
            "pooled_rmse": rmse(v2_truth, v2_ai),
            "pooled_r2": regression_metrics(v2_truth, v2_ai)["r2"],
        },
        "provenance_sha256": {str(path): sha256(path) for path in [*source_paths, args.v2_rd]},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_dir / "rd_predictions_by_seed.parquet", index=False)
    ensemble.to_parquet(args.output_dir / "rd_predictions_ensemble.parquet", index=False)
    folds.to_csv(args.output_dir / "rd_fold_metrics.csv", index=False)
    seed_table.to_csv(args.output_dir / "rd_seed_metrics.csv", index=False)
    (args.output_dir / "rd_full_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
