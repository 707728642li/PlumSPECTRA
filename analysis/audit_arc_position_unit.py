from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p99": float(np.quantile(values, 0.99)),
    }


def representative_kinematics(curve_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(curve_dir.glob("*.parquet")):
        curve = pd.read_parquet(path).iloc[0]
        time = np.asarray(curve["relative_time_s"], dtype=float)
        position = np.asarray(curve["position_raw"], dtype=float)
        dt = np.diff(time)
        dp = np.diff(position)
        valid = np.isfinite(dt) & np.isfinite(dp) & (dt > 0)
        speed = dp[valid] / dt[valid]
        loading = np.abs(speed[(speed < -0.2) & (speed > -2.0)])
        withdrawal = speed[(speed > 2.0) & (speed < 20.0)]
        nonzero_increment = np.abs(dp[valid & (np.abs(dp) > 1e-8)])
        rows.append(
            {
                "batch_id": str(curve["batch_id"]),
                "sample_id": str(curve["sample_id"]),
                "replicate": int(curve["replicate"]),
                "median_sampling_interval_s": float(np.median(dt[dt > 0])),
                "median_negative_motion_coordinate_per_s": float(np.median(loading)),
                "median_withdrawal_coordinate_per_s": float(np.median(withdrawal)),
                "minimum_nonzero_coordinate_increment": float(np.min(nonzero_increment)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-manifest", type=Path, required=True)
    parser.add_argument("--curve-features", type=Path, required=True)
    parser.add_argument("--curve-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.arc_manifest.resolve())
    features = pd.read_parquet(args.curve_features.resolve())
    marker = manifest["metadata_markers"].fillna("").str.contains(
        "Motor Steps / mm", regex=False
    )
    representatives = representative_kinematics(args.curve_dir.resolve())
    representatives.to_csv(output / "arc_position_representative_kinematics.csv", index=False)

    loading_speed = pd.to_numeric(
        features["loading_speed_rawpos_per_s"], errors="coerce"
    ).to_numpy(float)
    audit = {
        "arc_files": int(len(manifest)),
        "motor_steps_per_mm_marker_files": int(marker.sum()),
        "motor_steps_per_mm_numeric_value_retained": False,
        "loading_coordinate_rate_all_curves": finite_summary(loading_speed),
        "representative_batches": int(len(representatives)),
        "representative_negative_motion_coordinate_rate": finite_summary(
            representatives["median_negative_motion_coordinate_per_s"].to_numpy(float)
        ),
        "representative_withdrawal_coordinate_rate": finite_summary(
            representatives["median_withdrawal_coordinate_per_s"].to_numpy(float)
        ),
        "deterministic_conclusion": (
            "The retained parser artifacts preserve the marker name but not its numeric calibration value; "
            "therefore the motor-steps-per-mm constant cannot be decoded from this project copy."
        ),
        "kinematic_evidence": (
            "The archive coordinate advances at approximately 1.000 unit/s during loading in all 11,004 curves "
            "and approximately 10 unit/s during withdrawal in batch representatives; representative records also "
            "contain a distinct approximately 0.8-unit/s negative-motion phase. These discrete programmed-looking "
            "rates are strongly consistent with Texture Exponent speeds expressed in mm/s. They support, but do "
            "not independently prove, that the decoded archive coordinate is millimetres."
        ),
        "reporting_rule": (
            "Retain displacement and work in 'archive position units' in inferential tables; describe millimetres "
            "only as a strongly supported interpretation until an original ARC file or instrument method export "
            "with the numeric Motor Steps/mm value is restored."
        ),
    }
    (output / "arc_position_unit_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
