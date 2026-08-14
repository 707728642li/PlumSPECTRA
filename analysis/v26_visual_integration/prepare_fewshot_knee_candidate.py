"""Prepare matched-resampling uncertainty for the calibration-efficiency plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v26_claudecode_integration/figures_candidate/"
            "Fewshot_calibration_resampling_uncertainty.csv"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    source = (
        root
        / "results/v25_external_review_corrections/final_analysis/"
        "fewshot_repeat_metrics.parquet"
    )
    data = pd.read_parquet(source)
    keep = data[
        (data["model"] == "Deep-kernel ensemble")
        & (data["aggregation"] == "pooled")
        & (
            ((data["shots"] == 0) & (data["adapter"] == "none"))
            | ((data["shots"] > 0) & (data["adapter"] == "shrunken_affine"))
        )
    ].copy()
    by_resample = (
        keep.groupby(["shots", "repeat"], as_index=False)["rmse_gain_pct"]
        .median()
        .rename(columns={"rmse_gain_pct": "gain"})
    )
    summary = (
        by_resample.groupby("shots")["gain"]
        .agg(
            median_resample="median",
            q025=lambda x: x.quantile(0.025),
            q975=lambda x: x.quantile(0.975),
            repeats="size",
        )
        .reset_index()
    )
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
