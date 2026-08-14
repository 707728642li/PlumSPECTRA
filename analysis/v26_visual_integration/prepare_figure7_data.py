#!/usr/bin/env python3
"""Prepare evidence-locked deployment metrics for candidate Figure 7.

This is a presentation-only secondary analysis of the frozen outer-fold
predictions. No model is refitted and no manuscript file is changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TRAITS = ["FW", "SSC", "pH", "SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"]
GF_TO_N = 0.00980665


def fmt_mae(trait: str, value: float) -> tuple[float, str, str]:
    """Return publication-scale MAE, unit and formatted label."""
    if trait == "FW":
        return value, "g", f"{value:.2f} g"
    if trait == "SSC":
        return value, "percentage points", f"{value:.2f} pp"
    if trait == "pH":
        return value, "pH units", f"{value:.3f} pH"
    if trait in {"SRF", "PFD", "MFF", "F6", "AF"}:
        converted = value * GF_TO_N
        return converted, "N", f"{converted:.2f} N"
    if trait == "RD":
        return value, "APU", f"{value:.3f} APU"
    if trait == "LS":
        converted = value * GF_TO_N
        return converted, "N APU^-1", f"{converted:.3f} N/APU"
    if trait in {"LW", "PRW"}:
        converted = value * GF_TO_N
        return converted, "N APU", f"{converted:.2f} N·APU"
    raise KeyError(trait)


def main() -> int:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    pred_path = root / "evidence/final_analysis/v25_integrated_predictions.parquet"
    pred = pq.read_table(
        pred_path,
        columns=["sample_id", "cultivar_code", "trait", "outer_fold", "y_true", "y_final"],
    ).to_pandas()
    assert len(pred) == 58_206
    assert pred.duplicated(["sample_id", "trait"]).sum() == 0

    rows: list[dict[str, object]] = []
    for trait in TRAITS:
        group = pred.loc[pred["trait"].eq(trait)].copy()
        y = group["y_true"].to_numpy(float)
        yhat = group["y_final"].to_numpy(float)
        err = yhat - y
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        r2 = 1.0 - float(np.sum(err**2) / np.sum((y - y.mean()) ** 2))
        q25, q75 = np.quantile(y, [0.25, 0.75])
        rpiq = float((q75 - q25) / rmse)

        # Operational three-grade audit. Both observed and predicted grades use
        # thresholds computed from the other four outer folds only.
        observed_grade: list[np.ndarray] = []
        predicted_grade: list[np.ndarray] = []
        for fold in sorted(group["outer_fold"].unique()):
            train_y = group.loc[group["outer_fold"].ne(fold), "y_true"].to_numpy(float)
            test = group.loc[group["outer_fold"].eq(fold)]
            cut = np.quantile(train_y, [1 / 3, 2 / 3])
            observed_grade.append(np.digitize(test["y_true"].to_numpy(float), cut))
            predicted_grade.append(np.digitize(test["y_final"].to_numpy(float), cut))
        observed = np.concatenate(observed_grade)
        predicted = np.concatenate(predicted_grade)
        exact_grade = float(np.mean(observed == predicted))
        extreme_swap = float(np.mean(np.abs(observed - predicted) == 2))

        mae_display, mae_unit, mae_label = fmt_mae(trait, mae)
        rows.append(
            {
                "trait": trait,
                "domain": "Conventional quality" if trait in {"FW", "SSC", "pH"} else "Mechanical texture",
                "n": len(group),
                "r2": r2,
                "rmse_source": rmse,
                "mae_source": mae,
                "rpiq": rpiq,
                "mae_display": mae_display,
                "mae_unit": mae_unit,
                "mae_label": mae_label,
                "grade_accuracy": exact_grade,
                "extreme_swap": extreme_swap,
                "no_extreme_swap": 1.0 - extreme_swap,
            }
        )

    metrics = pd.DataFrame(rows)
    assert metrics["r2"].between(0.50, 0.83).all()
    assert metrics["grade_accuracy"].between(0.60, 0.81).all()
    assert metrics["extreme_swap"].between(0.0, 0.033).all()
    metrics.to_csv(out / "fig7_trait_metrics.csv", index=False)

    print(
        "Figure 7: 12 traits; OOF R2 "
        f"{metrics.r2.min():.3f}-{metrics.r2.max():.3f}; exact three-grade accuracy "
        f"{metrics.grade_accuracy.min():.3f}-{metrics.grade_accuracy.max():.3f}; "
        f"extreme swaps {metrics.extreme_swap.min():.3f}-{metrics.extreme_swap.max():.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
