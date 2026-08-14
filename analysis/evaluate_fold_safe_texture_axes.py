from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_texture_pls_loco import regression_metrics


ENDPOINTS = [
    "skin_break_force_g_mean",
    "skin_break_displacement_raw_mean",
    "skin_break_drop_g_mean",
    "flesh_force_mean_g_mean",
    "force_at_6_rawpos_g_mean",
    "loading_stiffness_g_per_rawpos_mean",
    "loading_work_g_rawpos_mean",
    "post_break_work_g_rawpos_mean",
    "adhesive_force_g_mean",
]
AXES = {
    "flesh_resistance_energy": {
        "flesh_force_mean_g_mean": 1.0,
        "force_at_6_rawpos_g_mean": 1.0,
        "loading_work_g_rawpos_mean": 1.0,
        "post_break_work_g_rawpos_mean": 1.0,
        "adhesive_force_g_mean": 1.0,
    },
    "deformation_compliance": {
        "skin_break_displacement_raw_mean": 1.0,
        "loading_stiffness_g_per_rawpos_mean": -1.0,
    },
    "skin_rupture_resistance": {
        "skin_break_force_g_mean": 1.0,
        "skin_break_drop_g_mean": 1.0,
        "loading_stiffness_g_per_rawpos_mean": 1.0,
        "loading_work_g_rawpos_mean": 1.0,
    },
}


def axis_score(values: pd.DataFrame, median: pd.Series, iqr: pd.Series, weights: dict[str, float]) -> np.ndarray:
    columns = list(weights)
    standardized = (values[columns].to_numpy(float) - median[columns].to_numpy(float)) / iqr[columns].to_numpy(float)
    weight_array = np.asarray([weights[column] for column in columns], dtype=float)
    return standardized @ weight_array / np.sum(np.abs(weight_array))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True, help="MODEL=path/to/predictions.parquet")
    parser.add_argument("--cohort", choices=["analysis", "primary", "sensitivity"], default="analysis")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(args.ledger)
    cohort_column = {
        "analysis": "qc_analysis_include",
        "primary": "qc_primary_include",
        "sensitivity": "qc_sensitivity_include",
    }[args.cohort]
    primary = ledger.loc[ledger[cohort_column]].copy()
    prediction_rows: list[dict[str, object]] = []

    for specification in args.predictions:
        model, path = specification.split("=", 1)
        frame = pd.read_parquet(Path(path).resolve())
        pivot_true = frame.pivot(index=["sample_id", "cultivar_ascii"], columns="target", values="y_true")
        pivot_pred = frame.pivot(index=["sample_id", "cultivar_ascii"], columns="target", values="y_pred")
        missing = sorted(set(ENDPOINTS) - set(pivot_pred.columns))
        if missing:
            raise ValueError(f"Missing endpoint predictions for {model}: {missing}")
        for cultivar in sorted(pivot_pred.index.get_level_values("cultivar_ascii").unique()):
            training = primary.loc[primary["cultivar_ascii"].ne(cultivar), ENDPOINTS]
            median = training.median()
            iqr = (training.quantile(0.75) - training.quantile(0.25)).replace(0, 1.0)
            index_mask = pivot_pred.index.get_level_values("cultivar_ascii") == cultivar
            truth = pivot_true.loc[index_mask, ENDPOINTS]
            prediction = pivot_pred.loc[index_mask, ENDPOINTS]
            for axis, weights in AXES.items():
                y_true = axis_score(truth, median, iqr, weights)
                y_pred = axis_score(prediction, median, iqr, weights)
                for (sample_id, cultivar_ascii), observed, estimate in zip(truth.index, y_true, y_pred):
                    prediction_rows.append(
                        {
                            "model": model,
                            "sample_id": sample_id,
                            "cultivar_ascii": cultivar_ascii,
                            "target": axis,
                            "y_true": float(observed),
                            "y_pred": float(estimate),
                            "residual": float(estimate - observed),
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    fold_rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []
    for (model, target, cultivar), group in predictions.groupby(["model", "target", "cultivar_ascii"], observed=True):
        fold_rows.append(
            {
                "model": model,
                "target": target,
                "heldout_cultivar": cultivar,
                **regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    for (model, target), group in predictions.groupby(["model", "target"], observed=True):
        pooled_rows.append(
            {
                "model": model,
                "target": target,
                **regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    predictions.to_parquet(output_dir / "texture_axis_predictions.parquet", index=False, compression="zstd")
    for model, model_frame in predictions.groupby("model", observed=True):
        safe_model = str(model).lower().replace("-", "_").replace(" ", "_")
        model_frame.to_parquet(
            output_dir / f"texture_axis_predictions_{safe_model}.parquet",
            index=False,
            compression="zstd",
        )
    pd.DataFrame(fold_rows).to_csv(output_dir / "texture_axis_fold_metrics.csv", index=False)
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(output_dir / "texture_axis_pooled_metrics.csv", index=False)
    summary = {
        "axis_definitions": AXES,
        "scaling": "median and IQR fitted only on outer-training cultivars for each held-out cultivar",
        "cohort": args.cohort,
        "models": sorted(predictions["model"].unique()),
        "pooled_metrics": pooled.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
