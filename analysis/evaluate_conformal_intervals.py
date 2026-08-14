from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def stable_seed(*values: object) -> int:
    return int(hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()[:8], 16)


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    rank = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    return float(np.sort(scores)[rank - 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", required=True, help="MODEL=predictions.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-sizes", default="10,20,50")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_sizes = [int(value) for value in args.calibration_sizes.split(",")]

    frames: dict[str, pd.DataFrame] = {}
    for specification in args.predictions:
        model, path = specification.split("=", 1)
        frames[model] = pd.read_parquet(Path(path).resolve())

    fold_rows: list[dict[str, object]] = []
    repeat_rows: list[dict[str, object]] = []
    for model, frame in frames.items():
        for target, target_frame in frame.groupby("target"):
            for calibration_size in calibration_sizes:
                intercept_size = calibration_size // 2
                interval_size = calibration_size - intercept_size
                for repeat in range(1, args.repeats + 1):
                    evaluation_parts: list[pd.DataFrame] = []
                    for cultivar, cultivar_frame in target_frame.groupby("cultivar_ascii"):
                        cultivar_frame = cultivar_frame.reset_index(drop=True)
                        if calibration_size >= len(cultivar_frame) - 2:
                            raise ValueError(f"Too many calibration samples for {target}/{cultivar}")
                        rng = np.random.default_rng(
                            stable_seed(target, cultivar, calibration_size, repeat, 20260806)
                        )
                        selected = rng.choice(len(cultivar_frame), size=calibration_size, replace=False)
                        intercept_positions = selected[:intercept_size]
                        interval_positions = selected[intercept_size:]
                        heldout_mask = np.ones(len(cultivar_frame), dtype=bool)
                        heldout_mask[selected] = False
                        intercept = float(
                            (
                                cultivar_frame.loc[intercept_positions, "y_true"]
                                - cultivar_frame.loc[intercept_positions, "y_pred"]
                            ).mean()
                        )
                        interval_prediction = cultivar_frame.loc[interval_positions, "y_pred"] + intercept
                        nonconformity = np.abs(
                            cultivar_frame.loc[interval_positions, "y_true"] - interval_prediction
                        ).to_numpy()
                        half_width = finite_sample_quantile(nonconformity, args.coverage)
                        evaluation = cultivar_frame.loc[heldout_mask].copy()
                        evaluation["point_prediction"] = evaluation["y_pred"] + intercept
                        evaluation["lower"] = evaluation["point_prediction"] - half_width
                        evaluation["upper"] = evaluation["point_prediction"] + half_width
                        evaluation["covered"] = (
                            evaluation["y_true"].ge(evaluation["lower"])
                            & evaluation["y_true"].le(evaluation["upper"])
                        )
                        evaluation_parts.append(evaluation)
                        fold_rows.append(
                            {
                                "model": model,
                                "target": target,
                                "calibration_size": calibration_size,
                                "intercept_fit_size": intercept_size,
                                "conformal_size": interval_size,
                                "repeat": repeat,
                                "cultivar_ascii": cultivar,
                                "n_evaluation": int(len(evaluation)),
                                "coverage": float(evaluation["covered"].mean()),
                                "mean_interval_width": float(2 * half_width),
                                "intercept": intercept,
                            }
                        )
                    pooled = pd.concat(evaluation_parts, ignore_index=True)
                    repeat_rows.append(
                        {
                            "model": model,
                            "target": target,
                            "calibration_size": calibration_size,
                            "intercept_fit_size": intercept_size,
                            "conformal_size": interval_size,
                            "repeat": repeat,
                            "n_evaluation": int(len(pooled)),
                            "coverage": float(pooled["covered"].mean()),
                            "mean_interval_width": float((pooled["upper"] - pooled["lower"]).mean()),
                            "median_interval_width": float((pooled["upper"] - pooled["lower"]).median()),
                        }
                    )

    folds = pd.DataFrame(fold_rows)
    repeats = pd.DataFrame(repeat_rows)
    folds.to_parquet(output_dir / "conformal_fold_metrics.parquet", index=False, compression="zstd")
    repeats.to_csv(output_dir / "conformal_repeat_metrics.csv", index=False)
    summary = (
        repeats.groupby(["model", "target", "calibration_size", "intercept_fit_size", "conformal_size"], as_index=False)
        .agg(
            empirical_coverage_mean=("coverage", "mean"),
            empirical_coverage_sd=("coverage", "std"),
            empirical_coverage_ci025=("coverage", lambda values: values.quantile(0.025)),
            empirical_coverage_ci975=("coverage", lambda values: values.quantile(0.975)),
            mean_interval_width=("mean_interval_width", "mean"),
            interval_width_sd=("mean_interval_width", "std"),
        )
    )
    summary["nominal_coverage"] = args.coverage
    summary.to_csv(output_dir / "conformal_summary.csv", index=False)
    report = {
        "nominal_coverage": args.coverage,
        "method": "Within-held-out-cultivar split conformal after intercept calibration; calibration fruit excluded from evaluation.",
        "repeats": args.repeats,
        "results": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
