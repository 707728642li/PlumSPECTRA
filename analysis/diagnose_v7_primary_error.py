from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def centered_rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_centered = truth - truth.mean()
    prediction_centered = prediction - prediction.mean()
    return float(np.sqrt(np.mean((prediction_centered - truth_centered) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_parquet(args.predictions)
    aliases = {"y_pls_anchor": "pls_anchor", "y_pred": "prediction"}
    predictions = predictions.rename(
        columns={source: target for source, target in aliases.items() if source in predictions.columns}
    )
    required = {"sample_id", "cultivar_ascii", "y_true", "pls_anchor", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    for cultivar, group in predictions.groupby("cultivar_ascii", sort=True):
        truth = group["y_true"].to_numpy(float)
        plsr = group["pls_anchor"].to_numpy(float)
        model = group["prediction"].to_numpy(float)
        rows.append(
            {
                "cultivar_ascii": cultivar,
                "n": len(group),
                "plsr_bias": float(np.mean(plsr - truth)),
                "model_bias": float(np.mean(model - truth)),
                "plsr_rmse": float(np.sqrt(np.mean((plsr - truth) ** 2))),
                "model_rmse": float(np.sqrt(np.mean((model - truth) ** 2))),
                "plsr_centered_rmse": centered_rmse(truth, plsr),
                "model_centered_rmse": centered_rmse(truth, model),
                "plsr_pearson_r": float(np.corrcoef(plsr, truth)[0, 1]),
                "model_pearson_r": float(np.corrcoef(model, truth)[0, 1]),
            }
        )
    folds = pd.DataFrame(rows)
    folds["absolute_bias_improvement"] = folds["plsr_bias"].abs() - folds["model_bias"].abs()
    folds["centered_rmse_improvement"] = folds["plsr_centered_rmse"] - folds["model_centered_rmse"]
    folds["correlation_improvement"] = folds["model_pearson_r"] - folds["plsr_pearson_r"]
    folds["rmse_improvement_pct"] = (
        100.0 * (folds["plsr_rmse"] - folds["model_rmse"]) / folds["plsr_rmse"]
    )

    truth = predictions["y_true"].to_numpy(float)
    plsr = predictions["pls_anchor"].to_numpy(float)
    model = predictions["prediction"].to_numpy(float)
    summary = {
        "cultivars": int(len(folds)),
        "fruits": int(len(predictions)),
        "pooled": {
            "plsr_bias": float(np.mean(plsr - truth)),
            "model_bias": float(np.mean(model - truth)),
            "plsr_centered_rmse": centered_rmse(truth, plsr),
            "model_centered_rmse": centered_rmse(truth, model),
        },
        "cultivar_macro": {
            "plsr_absolute_bias": float(folds["plsr_bias"].abs().mean()),
            "model_absolute_bias": float(folds["model_bias"].abs().mean()),
            "plsr_centered_rmse": float(folds["plsr_centered_rmse"].mean()),
            "model_centered_rmse": float(folds["model_centered_rmse"].mean()),
            "plsr_pearson_r": float(folds["plsr_pearson_r"].mean()),
            "model_pearson_r": float(folds["model_pearson_r"].mean()),
        },
        "cultivar_wins": {
            "absolute_bias": int((folds["absolute_bias_improvement"] > 0).sum()),
            "centered_rmse": int((folds["centered_rmse_improvement"] > 0).sum()),
            "pearson_r": int((folds["correlation_improvement"] > 0).sum()),
            "total": int(len(folds)),
        },
        "interpretation": (
            "Bias improvement indicates cultivar-level location transfer; centered RMSE and correlation "
            "indicate within-cultivar fruit ranking and shape transfer."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_dir / "cultivar_error_decomposition.csv", index=False)
    (args.output_dir / "error_decomposition_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(folds.sort_values("rmse_improvement_pct").to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
