from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repair descriptive V25 SVR-search metadata written by jobs launched before the "
            "source wording was corrected. Predictions and CV tables are never modified."
        )
    )
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for directory in args.directories:
        for path in sorted(directory.resolve().glob("*/fold_*/metadata.json")):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("svr_search_profile") != "v25_staged":
                raise RuntimeError(f"Not a V25 staged baseline metadata file: {path}")
            prediction_path = path.with_name("predictions.parquet")
            cv_path = path.with_name("inner_svr_cv.csv")
            if not prediction_path.is_file() or not cv_path.is_file():
                raise FileNotFoundError(f"Incomplete baseline result beside {path}")
            choice = metadata["domain_svr_choice"]
            cv = pd.read_csv(cv_path)
            selected = cv.loc[
                cv["stage"].ne("preprocessing_screen")
                & cv["preprocessing"].eq(choice["preprocessing"])
                & np.isclose(cv["C"].astype(float), float(choice["C"]))
                & np.isclose(cv["gamma_factor"].astype(float), float(choice["gamma_factor"]))
                & np.isclose(cv["epsilon_z"].astype(float), float(choice["epsilon_z"]))
            ]
            if selected.empty:
                raise RuntimeError(f"Recorded SVR choice is absent from its inner-CV table: {path}")
            prediction_sha = digest(prediction_path)
            before_sha = digest(path)
            metadata["protocol"] = "V25 corrected-cohort non-overlapping five-fold nested baseline audit"
            recorded_search = metadata.get("svr_search_space", {})
            recorded_hard_limits = recorded_search.get("hard_limits", {})
            is_v25_wide_epsilon_run = float(
                recorded_hard_limits.get("epsilon_z", [0.001, 0.0])[-1]
            ) >= 10.0
            corrected_search_space = {
                "C": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
                "gamma_factor": [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
                "epsilon_z": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.4, 0.6, 0.8],
                "strategy": (
                    "broad C-gamma grid at epsilon=0.05 followed by epsilon refinement for the "
                    "two best C-gamma pairs per selected preprocessing"
                ),
                "boundary_extension": (
                    "up to six train-internal one-axis extensions around any boundary winner"
                ),
                "hard_limits": {
                    "C": [0.0003, 3000.0],
                    "gamma_factor": [0.0001, 20.0],
                    "epsilon_z": [0.001, 1.5],
                },
            }
            if not is_v25_wide_epsilon_run:
                metadata["svr_search_space"] = corrected_search_space
                metadata["metadata_schema_version"] = "v25.1"
                metadata["metadata_only_correction"] = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": (
                        "The launched process correctly used six adaptive extensions and epsilon hard "
                        "limit 1.5, but its descriptive JSON literal still stated four and 0.4."
                    ),
                    "predictions_modified": False,
                    "cv_scores_modified": False,
                    "prediction_sha256_before_and_after": prediction_sha,
                }
                repair_action = "legacy_literal_repaired_to_six_extensions_epsilon_1.5"
            else:
                # Selectively rerun boundary folds were generated after the
                # epsilon search was widened to 10.  Their metadata is already
                # accurate and must never be downgraded by this repair pass.
                metadata["metadata_schema_version"] = "v25.2"
                repair_action = "wide_epsilon_metadata_preserved"
            path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            if digest(prediction_path) != prediction_sha:
                raise RuntimeError(f"Prediction changed during metadata repair: {prediction_path}")
            rows.append(
                {
                    "metadata_path": str(path),
                    "trait": metadata["trait"],
                    "outer_fold": int(metadata["outer_fold"]),
                    "metadata_sha256_before": before_sha,
                    "metadata_sha256_after": digest(path),
                    "prediction_sha256": prediction_sha,
                    "prediction_unchanged": True,
                    "selected_svr_choice_verified_in_inner_cv": True,
                    "repair_action": repair_action,
                }
            )

    expected = sum(len(list(directory.resolve().glob("*/fold_*/predictions.parquet"))) for directory in args.directories)
    if len(rows) != expected:
        raise RuntimeError(f"Repaired {len(rows)} metadata files but observed {expected} predictions")
    output = args.audit_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps({"status": "PASS", "metadata_files": len(rows), "audit": str(output)}, indent=2))


if __name__ == "__main__":
    main()
