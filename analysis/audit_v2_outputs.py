from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2_registry import cultivar_registry, trait_registry


def parse_specs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        abbreviation, path = value.split("=", 1)
        result[abbreviation.upper()] = Path(path).resolve()
    return result


def parse_trait_seeds(values: list[str] | None) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for value in values or []:
        abbreviation, seeds = value.split("=", 1)
        result[abbreviation.strip().upper()] = {int(seed) for seed in seeds.split(",")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="ABBR=PlumRAC output directory")
    parser.add_argument("--expected-seeds", default="20260806")
    parser.add_argument(
        "--trait-seeds",
        action="append",
        help="Optional per-trait override, for example RD=20260806,20260807",
    )
    parser.add_argument("--expected-samples", type=int, default=5430)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    specs = parse_specs(args.model)
    expected_seeds = {int(value) for value in args.expected_seeds.split(",")}
    trait_seed_overrides = parse_trait_seeds(args.trait_seeds)
    cultivar_table = cultivar_registry()
    trait_table = trait_registry().set_index("abbreviation")
    failures: list[str] = []
    checks: list[dict[str, object]] = []

    expected_traits = set(trait_table.loc[trait_table["model_family"] == "endpoint"].index)
    if set(specs) != expected_traits:
        failures.append(f"Model trait set mismatch: expected {sorted(expected_traits)}, observed {sorted(specs)}")
    if not cultivar_table["cultivar_code"].str.fullmatch(r"[A-Z]+[A-Z0-9]*").all():
        failures.append("Invalid publication cultivar code")
    if len(cultivar_table) != 16 or cultivar_table["cultivar_code"].nunique() != 16:
        failures.append("Cultivar registry is not one-to-one for 16 cultivars/selections")

    for abbreviation, output_dir in specs.items():
        trait_expected_seeds = trait_seed_overrides.get(abbreviation, expected_seeds)
        summary_path = output_dir / "summary.json"
        prediction_path = output_dir / "predictions_ensemble.parquet"
        if not summary_path.exists() or not prediction_path.exists():
            failures.append(f"Missing summary/predictions for {abbreviation}: {output_dir}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        predictions = pd.read_parquet(prediction_path)
        expected_target = str(trait_table.loc[abbreviation, "target"])
        if summary.get("target") != expected_target or set(predictions["target"]) != {expected_target}:
            failures.append(f"Target mismatch for {abbreviation}")
        if len(predictions) != args.expected_samples or predictions["sample_id"].nunique() != args.expected_samples:
            failures.append(f"Sample count mismatch for {abbreviation}: {len(predictions)}")
        if predictions["cultivar_ascii"].nunique() != 16 or predictions["cultivar_code"].nunique() != 16:
            failures.append(f"Cultivar fold count mismatch for {abbreviation}")
        if not predictions["cultivar_code"].astype(str).str.fullmatch(r"[A-Z]+[A-Z0-9]*").all():
            failures.append(f"Invalid cultivar code in predictions for {abbreviation}")
        numeric_predictions = predictions[["y_true", "y_pred", "y_pls_anchor"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric_predictions.to_numpy(float)).all():
            failures.append(f"Non-finite prediction values for {abbreviation}")
        if set(summary.get("seeds", [])) != trait_expected_seeds:
            failures.append(f"Seed mismatch for {abbreviation}: {summary.get('seeds')}")
        if not summary.get("provenance_sha256"):
            failures.append(f"Missing provenance hashes for {abbreviation}")
        config = summary.get("config", {})
        if (
            config.get("min_gate_win_fraction") != 1.0
            or config.get("max_gate_worst_degradation") != 0.0
            or config.get("max_residual_gate") != 0.5
        ):
            failures.append(f"Unsafe or unexpected gate configuration for {abbreviation}: {config}")
        if summary.get("trainable_parameters") != 72530:
            failures.append(f"Unexpected parameter count for {abbreviation}: {summary.get('trainable_parameters')}")

        packaging_path = output_dir / "inference_packaging_summary.json"
        if not packaging_path.exists():
            failures.append(f"Missing inference packaging audit for {abbreviation}")
        else:
            packaging = json.loads(packaging_path.read_text(encoding="utf-8"))
            if packaging.get("status") != "PASS" or packaging.get("runs_packaged") != 16 * len(trait_expected_seeds):
                failures.append(f"Inference packaging audit failed for {abbreviation}: {packaging}")
            max_abs = float(packaging.get("maximum_prediction_difference", float("inf")))
            max_standardized = float(packaging.get("maximum_standardized_prediction_difference", float("inf")))
            if max_abs > 1e-3 and max_standardized > 2e-5:
                failures.append(f"Inference reconstruction tolerance exceeded for {abbreviation}: {packaging}")

        metadata_paths = list((output_dir / "runs" / abbreviation).glob("*/seed_*/metadata.json"))
        expected_metadata = 16 * len(trait_expected_seeds)
        if len(metadata_paths) != expected_metadata:
            failures.append(f"Fold metadata count mismatch for {abbreviation}: {len(metadata_paths)}")
        for metadata_path in metadata_paths:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            heldout = metadata["heldout_cultivar"]
            if heldout in metadata["validation_cultivars"]:
                failures.append(f"Held-out cultivar entered validation: {abbreviation}/{heldout}")
            if metadata["train_samples_retrained"] + metadata["test_samples"] != args.expected_samples:
                failures.append(f"Outer refit/test sample partition mismatch: {abbreviation}/{heldout}")
            gate = float(metadata["selected_gate"])
            if gate not in {0.0, 0.25, 0.5}:
                failures.append(f"Unexpected gate for {abbreviation}/{heldout}: {gate}")
            selected_rows = [row for row in metadata.get("gate_scores", []) if row.get("selected")]
            if len(selected_rows) != 1:
                failures.append(f"Gate selection row mismatch for {abbreviation}/{heldout}")
            elif gate > 0:
                selected = selected_rows[0]
                if (
                    selected.get("relative_improvement_vs_zero", -1) < 0.01
                    or selected.get("group_win_fraction", -1) < 1.0
                    or selected.get("worst_group_improvement", -1) < 0.0
                ):
                    failures.append(f"Nonzero gate violates consensus rule for {abbreviation}/{heldout}: {selected}")
        checks.append(
            {
                "trait": abbreviation,
                "samples": int(len(predictions)),
                "cultivars": int(predictions["cultivar_code"].nunique()),
                "seeds": sorted(trait_expected_seeds),
                "fold_metadata": len(metadata_paths),
                "pooled_r2": summary.get("pooled_metrics", {}).get("r2"),
            }
        )

    project_root = Path(__file__).resolve().parents[1]
    required_files = [
        project_root / "environment-lock.txt",
        project_root / "results" / "v2" / "figures" / "study_design" / "fig01_study_design.pdf",
        project_root / "results" / "v2" / "figures" / "plumrac_architecture" / "fig_plumrac_architecture.pdf",
    ]
    for path in required_files:
        if not path.exists():
            failures.append(f"Missing required V2 artifact: {path}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "expected_traits": sorted(expected_traits),
        "expected_samples_per_trait": args.expected_samples,
        "expected_cultivars": 16,
        "default_expected_seeds": sorted(expected_seeds),
        "trait_seed_overrides": {key: sorted(value) for key, value in trait_seed_overrides.items()},
        "checks": checks,
        "failures": failures,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
