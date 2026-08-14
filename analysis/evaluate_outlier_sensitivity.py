from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def concordance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    covariance = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    denominator = np.var(y_true) + np.var(y_pred) + (y_true.mean() - y_pred.mean()) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else np.nan


def score(group: pd.DataFrame) -> dict[str, float | int]:
    y_true = group["y_true"].to_numpy(float)
    y_pred = group["y_pred"].to_numpy(float)
    return {
        "n": len(group),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "ccc": concordance(y_true, y_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, help="name=path")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flags = pd.read_parquet(args.master, columns=["sample_id", "spectral_soft_outlier_flag"])
    rows: list[dict[str, object]] = []
    for specification in args.predictions:
        model, path = specification.split("=", 1)
        predictions = pd.read_parquet(path).merge(flags, on="sample_id", how="left", validate="many_to_one")
        if predictions["spectral_soft_outlier_flag"].isna().any():
            raise ValueError(f"Missing outlier flags for {model}")
        for target, group in predictions.groupby("target", sort=True):
            retained = group.loc[~group["spectral_soft_outlier_flag"]]
            all_metrics = score(group)
            retained_metrics = score(retained)
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "n_all": all_metrics["n"],
                    "n_soft_flagged": int(group["spectral_soft_outlier_flag"].sum()),
                    "n_without_soft_flagged": retained_metrics["n"],
                    **{f"all_{key}": value for key, value in all_metrics.items() if key != "n"},
                    **{f"without_soft_flagged_{key}": value for key, value in retained_metrics.items() if key != "n"},
                    "delta_rmse_after_exclusion": retained_metrics["rmse"] - all_metrics["rmse"],
                    "delta_r2_after_exclusion": retained_metrics["r2"] - all_metrics["r2"],
                    "delta_ccc_after_exclusion": retained_metrics["ccc"] - all_metrics["ccc"],
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "soft_spectral_outlier_sensitivity.csv", index=False)
    summary = {
        "policy": "Soft PCA flags were retained in all primary analyses; this table is a sensitivity analysis only.",
        "max_absolute_rmse_change": float(result["delta_rmse_after_exclusion"].abs().max()),
        "results": result.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
