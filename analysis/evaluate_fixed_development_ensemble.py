from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    keys = ["sample_id", "cultivar_ascii", "cultivar_code", "target", "y_true", "y_pls_anchor"]
    left = pd.read_parquet(args.left).sort_values("sample_id").reset_index(drop=True)
    right = pd.read_parquet(args.right).sort_values("sample_id").reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError("Ensemble inputs are not aligned on identical samples, labels, and PLSR anchors")
    prediction = 0.5 * left["y_pred"].to_numpy(float) + 0.5 * right["y_pred"].to_numpy(float)
    output = left[keys].copy()
    output["y_pred_left"] = left["y_pred"].to_numpy(float)
    output["y_pred_right"] = right["y_pred"].to_numpy(float)
    output["y_pred"] = prediction
    output["ensemble_weight_left"] = 0.5
    output["ensemble_weight_right"] = 0.5

    fold_rows: list[dict[str, float | int | str | bool]] = []
    for cultivar, group in output.groupby("cultivar_ascii", sort=True):
        truth = group["y_true"].to_numpy(float)
        plsr = group["y_pls_anchor"].to_numpy(float)
        ensemble = group["y_pred"].to_numpy(float)
        plsr_rmse = rmse(truth, plsr)
        ensemble_rmse = rmse(truth, ensemble)
        fold_rows.append(
            {
                "cultivar_ascii": cultivar,
                "n": len(group),
                "plsr_rmse": plsr_rmse,
                "ensemble_rmse": ensemble_rmse,
                "rmse_improvement_pct": 100.0 * (plsr_rmse - ensemble_rmse) / plsr_rmse,
                "ensemble_win": ensemble_rmse < plsr_rmse,
            }
        )
    folds = pd.DataFrame(fold_rows)
    truth = output["y_true"].to_numpy(float)
    plsr = output["y_pls_anchor"].to_numpy(float)
    ensemble = output["y_pred"].to_numpy(float)
    pooled_plsr = rmse(truth, plsr)
    pooled_ensemble = rmse(truth, ensemble)
    macro_plsr = float(folds["plsr_rmse"].mean())
    macro_ensemble = float(folds["ensemble_rmse"].mean())
    summary = {
        "ensemble": "fixed equal-weight arithmetic mean",
        "left": str(args.left.resolve()),
        "right": str(args.right.resolve()),
        "weights": [0.5, 0.5],
        "fruits": int(len(output)),
        "cultivars": int(len(folds)),
        "fold_wins": int(folds["ensemble_win"].sum()),
        "pooled_plsr_rmse": pooled_plsr,
        "pooled_ensemble_rmse": pooled_ensemble,
        "pooled_rmse_improvement_pct": 100.0 * (pooled_plsr - pooled_ensemble) / pooled_plsr,
        "macro_plsr_rmse": macro_plsr,
        "macro_ensemble_rmse": macro_ensemble,
        "macro_rmse_improvement_pct": 100.0 * (macro_plsr - macro_ensemble) / macro_plsr,
        "selection_rule": "No ensemble-weight search; 0.5/0.5 specified before evaluation.",
        "claim_boundary": "Retrospective development diagnostic; requires frozen all-cultivar multiseed confirmation.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_dir / "predictions.parquet", index=False, compression="zstd")
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(folds.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
