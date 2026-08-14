from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_pls_loco import metrics


def stable_seed(*values: object) -> int:
    text = "|".join(map(str, values))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", required=True, help="MODEL=path/to/predictions.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", default="0,1,3,5,10,20,50")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shots = [int(value) for value in args.shots.split(",")]

    model_frames: dict[str, pd.DataFrame] = {}
    for specification in args.predictions:
        model, path = specification.split("=", 1)
        frame = pd.read_parquet(Path(path).resolve())
        required = {"sample_id", "cultivar_ascii", "target", "y_true", "y_pred"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns for {model}: {sorted(required - set(frame.columns))}")
        model_frames[model] = frame

    repeat_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    for model, frame in model_frames.items():
        for target, target_frame in frame.groupby("target"):
            oracle_parts: list[pd.DataFrame] = []
            for cultivar, cultivar_frame in target_frame.groupby("cultivar_ascii"):
                adjusted = cultivar_frame.copy()
                adjustment = float((adjusted["y_true"] - adjusted["y_pred"]).mean())
                adjusted["y_pred"] += adjustment
                oracle_parts.append(adjusted)
            oracle = pd.concat(oracle_parts, ignore_index=True)
            oracle_rows.append(
                {
                    "model": model,
                    "target": target,
                    "calibration": "oracle_full_cultivar_intercept",
                    **metrics(oracle["y_true"].to_numpy(), oracle["y_pred"].to_numpy()),
                }
            )
            for shot in shots:
                for repeat in range(1, args.repeats + 1):
                    prediction_parts: list[pd.DataFrame] = []
                    cultivar_metric_parts: list[dict[str, float]] = []
                    for cultivar, cultivar_frame in target_frame.groupby("cultivar_ascii"):
                        cultivar_frame = cultivar_frame.reset_index(drop=True)
                        if shot == 0:
                            evaluation = cultivar_frame.copy()
                            intercept = 0.0
                        else:
                            if shot >= len(cultivar_frame) - 2:
                                raise ValueError(f"Too many shots for {target}/{cultivar}: {shot}")
                            rng = np.random.default_rng(stable_seed(target, cultivar, shot, repeat, 20260806))
                            calibration_positions = rng.choice(len(cultivar_frame), size=shot, replace=False)
                            calibration_mask = np.zeros(len(cultivar_frame), dtype=bool)
                            calibration_mask[calibration_positions] = True
                            calibration = cultivar_frame.loc[calibration_mask]
                            evaluation = cultivar_frame.loc[~calibration_mask].copy()
                            intercept = float((calibration["y_true"] - calibration["y_pred"]).mean())
                        evaluation["y_pred"] += intercept
                        prediction_parts.append(evaluation)
                        cultivar_metrics = metrics(evaluation["y_true"].to_numpy(), evaluation["y_pred"].to_numpy())
                        fold_rows.append(
                            {
                                "model": model,
                                "target": target,
                                "shots": shot,
                                "repeat": repeat,
                                "cultivar_ascii": cultivar,
                                "estimated_intercept": intercept,
                                **cultivar_metrics,
                            }
                        )
                        cultivar_metric_parts.append(cultivar_metrics)
                    pooled = pd.concat(prediction_parts, ignore_index=True)
                    pooled_metrics = metrics(pooled["y_true"].to_numpy(), pooled["y_pred"].to_numpy())
                    repeat_rows.append(
                        {
                            "model": model,
                            "target": target,
                            "shots": shot,
                            "repeat": repeat,
                            **pooled_metrics,
                            "cultivar_macro_rmse": float(np.mean([row["rmse"] for row in cultivar_metric_parts])),
                            "cultivar_macro_mae": float(np.mean([row["mae"] for row in cultivar_metric_parts])),
                            "cultivar_macro_r2": float(np.mean([row["r2"] for row in cultivar_metric_parts])),
                        }
                    )

    repeat_metrics = pd.DataFrame(repeat_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    oracle_metrics = pd.DataFrame(oracle_rows)
    repeat_metrics.to_csv(output_dir / "fewshot_repeat_metrics.csv", index=False)
    fold_metrics.to_parquet(output_dir / "fewshot_fold_metrics.parquet", index=False, compression="zstd")
    oracle_metrics.to_csv(output_dir / "oracle_intercept_metrics.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    columns = ["rmse", "mae", "bias", "r2", "pearson_r", "ccc", "rpd", "rpiq", "cultivar_macro_rmse", "cultivar_macro_mae", "cultivar_macro_r2"]
    for (model, target, shot), group in repeat_metrics.groupby(["model", "target", "shots"]):
        row: dict[str, object] = {"model": model, "target": target, "shots": shot, "repeats": len(group)}
        for column in columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std(ddof=1))
            row[f"{column}_ci025"] = float(group[column].quantile(0.025))
            row[f"{column}_ci975"] = float(group[column].quantile(0.975))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "fewshot_summary.csv", index=False)
    compact = summary.loc[summary["shots"].isin([0, 5, 10, 20, 50]), ["model", "target", "shots", "rmse_mean", "r2_mean", "ccc_mean", "cultivar_macro_rmse_mean"]]
    report = {
        "models": sorted(model_frames),
        "shots": shots,
        "repeats": args.repeats,
        "calibration": "intercept-only, estimated from randomly sampled labeled fruit in each held-out cultivar; calibration fruit excluded from evaluation",
        "compact_results": compact.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
