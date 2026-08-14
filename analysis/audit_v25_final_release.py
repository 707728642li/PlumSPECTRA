#!/usr/bin/env python3
"""Scientific and provenance audit for the frozen PlumSPECTRA V25 release.

This audit intentionally checks scientific invariants (OOF uniqueness, matched
truths, train-only tuning metadata, converged search grids and complete protocol
coverage) in addition to file existence.  It is run before the external-review
package is assembled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_v25_external_review_corrections import read_hyperparameters


TEXTURE_TRAITS = {"SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_files(directory: Path, filename: str) -> list[Path]:
    return sorted(directory.glob(f"*/fold_*/{filename}"))


def read_fold_predictions(directory: Path) -> pd.DataFrame:
    files = fold_files(directory, "predictions.parquet")
    if not files:
        raise FileNotFoundError(f"No per-fold predictions under {directory}")
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_parquet(path)
        # Baseline files persist outer_fold, whereas neural files identify the
        # fold through their registered fold_# directory.  Normalise both
        # representations before any uniqueness or truth-matching audit.
        path_fold = int(path.parent.name.removeprefix("fold_"))
        if "outer_fold" not in frame.columns:
            frame["outer_fold"] = path_fold
        elif not frame["outer_fold"].astype(int).eq(path_fold).all():
            raise ValueError(f"Stored outer_fold disagrees with path: {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/v25_external_review_corrections"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/v25_external_review_corrections/final_release_audit"),
    )
    args = parser.parse_args()

    root = args.results_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, expected: Any, detail: str = "") -> None:
        if isinstance(expected, float):
            passed = bool(np.isclose(float(observed), expected, rtol=0, atol=1e-10))
        else:
            passed = observed == expected
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    texture_manifest_path = root / "splits/v20_fivefold_manifest.csv"
    quality_manifest_path = root / "splits/v22_quality_fivefold_manifest.csv"
    texture_manifest = pd.read_csv(texture_manifest_path)
    quality_manifest = pd.read_csv(quality_manifest_path)
    check("texture manifest rows", len(texture_manifest), 4853)
    check("texture manifest cultivars", texture_manifest["cultivar_ascii"].nunique(), 15)
    check("texture manifest folds", texture_manifest["outer_fold"].nunique(), 5)
    check(
        "texture manifest sha256",
        sha256(texture_manifest_path),
        "363ad2174d53d7eb2dcbeb8f2cecfb3cb32da98db3b3ba6176da32f43bf29a69",
    )
    check("quality manifest rows", len(quality_manifest), 4843)
    check("quality manifest cultivars", quality_manifest["cultivar_ascii"].nunique(), 15)
    check("quality manifest folds", quality_manifest["outer_fold"].nunique(), 5)
    check(
        "quality manifest sha256",
        sha256(quality_manifest_path),
        "7f859700cf7386e571f48305b53b922922b7d07338d7e8051b281351b59ad155",
    )

    experiment_specs = [
        ("texture_baseline", root / "baselines_texture_final", 45, 43677),
        ("quality_baseline", root / "baselines_quality_final", 15, 14529),
        ("texture_ai", root / "ai_texture_domain_anchor_final", 45, 43677),
        ("quality_ai", root / "ai_quality_domain_anchor_final", 15, 14529),
        ("crossbatch_baseline", root / "crossbatch_baselines_final", 45, 11304),
        ("crossbatch_ai", root / "crossbatch_ai_final", 45, 11304),
    ]
    inventory: list[dict[str, Any]] = []
    for label, directory, expected_files, expected_rows in experiment_specs:
        predictions = read_fold_predictions(directory)
        files = fold_files(directory, "predictions.parquet")
        duplicate_keys = ["sample_id", "target"]
        if label.startswith("crossbatch"):
            duplicate_keys.append("outer_fold")
        duplicates = int(predictions.duplicated(duplicate_keys).sum())
        check(f"{label} prediction files", len(files), expected_files)
        check(f"{label} prediction rows", len(predictions), expected_rows)
        check(f"{label} duplicate prediction keys", duplicates, 0)
        inventory.append(
            {
                "experiment": label,
                "directory": str(directory),
                "prediction_files": len(files),
                "prediction_rows": len(predictions),
                "duplicate_keys": duplicates,
            }
        )

    texture_baseline = read_fold_predictions(root / "baselines_texture_final")
    texture_ai = read_fold_predictions(root / "ai_texture_domain_anchor_final")
    quality_baseline = read_fold_predictions(root / "baselines_quality_final")
    quality_ai = read_fold_predictions(root / "ai_quality_domain_anchor_final")
    for label, baseline, ai in (
        ("texture", texture_baseline, texture_ai),
        ("quality", quality_baseline, quality_ai),
    ):
        keys = ["sample_id", "target", "outer_fold"]
        matched = baseline[keys + ["y_true"]].merge(
            ai[keys + ["y_true"]], on=keys, how="outer", validate="one_to_one", suffixes=("_b", "_a")
        )
        check(f"{label} baseline-AI matched rows", len(matched), len(baseline))
        check(
            f"{label} baseline-AI truth mismatch",
            int((~np.isclose(matched["y_true_b"], matched["y_true_a"], rtol=0, atol=1e-8)).sum()),
            0,
        )

    baseline_metadata = [
        *fold_files(root / "baselines_texture_final", "metadata.json"),
        *fold_files(root / "baselines_quality_final", "metadata.json"),
    ]
    check("formal baseline metadata files", len(baseline_metadata), 60)
    baseline_test_label_violations = 0
    baseline_grid_violations = 0
    for path in baseline_metadata:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        baseline_test_label_violations += int(bool(metadata.get("test_labels_used_for_selection", True)))
        baseline_grid_violations += int(
            metadata.get("pls_component_grid") != [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24]
        )
    check("baseline test-label tuning violations", baseline_test_label_violations, 0)
    check("baseline PLS grid violations", baseline_grid_violations, 0)

    hyper = read_hyperparameters(
        root / "baselines_texture_final", root / "baselines_quality_final", 5.0
    )
    check("PLSR upper-component boundary hits", int(hyper["domain_pls_component_upper_boundary"].sum()), 0)
    check("SVR C boundary hits", int(hyper["svr_C_boundary"].sum()), 0)
    check("SVR gamma boundary hits", int(hyper["svr_gamma_boundary"].sum()), 0)
    check("SVR epsilon boundary hits", int(hyper["svr_epsilon_boundary"].sum()), 0)
    hyper.to_csv(output / "formal_fold_hyperparameters.csv", index=False)

    ai_metadata = [
        *fold_files(root / "ai_texture_domain_anchor_final", "metadata.json"),
        *fold_files(root / "ai_quality_domain_anchor_final", "metadata.json"),
    ]
    check("formal AI metadata files", len(ai_metadata), 60)
    protocol_violations: list[str] = []
    for path in ai_metadata:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        valid = (
            metadata.get("domain_aware_anchor_selection") is True
            and int(metadata.get("crossfit_anchor_folds", -1)) == 4
            and metadata.get("gate_selection_mode") == "training_internal_validation"
            and metadata.get("test_labels_used_for_selection") is False
            and metadata.get("fixed_gate_requested") is None
        )
        if not valid:
            protocol_violations.append(str(path))
    check("AI protocol violations", len(protocol_violations), 0, "; ".join(protocol_violations))

    cross_baseline_metadata = fold_files(root / "crossbatch_baselines_final", "metadata.json")
    check("crossbatch baseline metadata files", len(cross_baseline_metadata), 45)
    cross_baseline_test_label_violations = 0
    cross_pls_upper_boundary_hits = 0
    cross_svr_boundary_hits = {"C": 0, "gamma_factor": 0, "epsilon_z": 0}
    for path in cross_baseline_metadata:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        cross_baseline_test_label_violations += int(
            bool(metadata.get("test_labels_used_for_selection", True))
        )
        pls_grid = [int(value) for value in metadata["pls_component_grid"]]
        selected_components = int(metadata["domain_pls_choice"]["n_components"])
        cross_pls_upper_boundary_hits += int(selected_components == max(pls_grid))
        inner_svr = pd.read_csv(path.parent / "inner_svr_cv.csv")
        selection = metadata["domain_svr_choice"]
        for parameter in cross_svr_boundary_hits:
            evaluated = inner_svr[parameter].astype(float)
            selected = float(selection[parameter])
            cross_svr_boundary_hits[parameter] += int(
                np.isclose(selected, evaluated.min(), rtol=0, atol=1e-12)
                or np.isclose(selected, evaluated.max(), rtol=0, atol=1e-12)
            )
    check("crossbatch baseline test-label tuning violations", cross_baseline_test_label_violations, 0)
    check("crossbatch PLSR upper-component boundary hits", cross_pls_upper_boundary_hits, 0)
    check("crossbatch SVR C boundary hits", cross_svr_boundary_hits["C"], 0)
    check("crossbatch SVR gamma boundary hits", cross_svr_boundary_hits["gamma_factor"], 0)
    check("crossbatch SVR epsilon boundary hits", cross_svr_boundary_hits["epsilon_z"], 0)

    cross_ai_metadata = fold_files(root / "crossbatch_ai_final", "metadata.json")
    check("crossbatch AI metadata files", len(cross_ai_metadata), 45)
    cross_ai_protocol_violations: list[str] = []
    for path in cross_ai_metadata:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        valid = (
            metadata.get("domain_aware_anchor_selection") is True
            and int(metadata.get("crossfit_anchor_folds", -1)) == 4
            and metadata.get("gate_selection_mode") == "training_internal_validation"
            and metadata.get("test_labels_used_for_selection") is False
            and metadata.get("fixed_gate_requested") is None
        )
        if not valid:
            cross_ai_protocol_violations.append(str(path))
    check(
        "crossbatch AI protocol violations",
        len(cross_ai_protocol_violations),
        0,
        "; ".join(cross_ai_protocol_violations),
    )
    cross_ai_manifest = json.loads(
        (root / "crossbatch_ai_final/run_manifest.json").read_text(encoding="utf-8")
    )
    cross_ai_jobs = cross_ai_manifest.get("jobs", [])
    check("crossbatch AI launcher jobs", len(cross_ai_jobs), 45)
    check(
        "crossbatch AI incomplete launcher jobs",
        sum(job.get("status") != "completed" for job in cross_ai_jobs),
        0,
    )

    multiseed_files = sorted(
        (root / "multiseed_final/ai").glob("*/fold_*/seed_repeat_*/predictions.parquet")
    )
    check("additional complete-pipeline seed fits", len(multiseed_files), 120)
    multiseed_manifest = json.loads((root / "multiseed_final/run_manifest.json").read_text(encoding="utf-8"))
    failures = multiseed_manifest.get("failed_jobs", multiseed_manifest.get("failures", []))
    check("multiseed launcher failures", len(failures), 0)
    multiseed_summary = pd.read_csv(root / "multiseed_analysis/multiseed_summary.csv")
    check("multiseed summary trait-candidate rows", len(multiseed_summary), 24)
    check("multiseed summary traits", multiseed_summary["trait"].nunique(), 12)

    integrated = pd.read_parquet(root / "final_analysis/v25_integrated_predictions.parquet")
    check("integrated OOF rows", len(integrated), 58206)
    check("integrated OOF fruit-trait duplicates", int(integrated.duplicated(["sample_id", "trait"]).sum()), 0)
    check("integrated OOF traits", integrated["trait"].nunique(), 12)
    check("integrated OOF cultivars", integrated["cultivar_ascii"].nunique(), 15)
    check(
        "integrated trait-fold panels",
        len(integrated[["target", "outer_fold"]].drop_duplicates()),
        60,
    )
    check(
        "final ensemble formula max abs error",
        float(
            np.max(
                np.abs(
                    integrated["y_final"]
                    - (
                        (1.0 - integrated["kernel_weight"]) * integrated["y_deep"]
                        + integrated["kernel_weight"] * integrated["y_domain_svr"]
                    )
                )
            )
        ),
        0.0,
    )
    check(
        "no-neural B50 formula max abs error",
        float(
            np.max(
                np.abs(
                    integrated["y_b50"]
                    - 0.5 * (integrated["y_domain_pls"] + integrated["y_domain_svr"])
                )
            )
        ),
        0.0,
    )
    check(
        "integrated texture fruit count",
        integrated.loc[integrated["trait"].isin(TEXTURE_TRAITS), "sample_id"].nunique(),
        4853,
    )
    check(
        "integrated conventional fruit count",
        integrated.loc[~integrated["trait"].isin(TEXTURE_TRAITS), "sample_id"].nunique(),
        4843,
    )

    gate_audit = pd.read_csv(root / "final_analysis/residual_gate_audit.csv")
    check("residual gate audit rows", len(gate_audit), 60)
    check(
        "recorded production gate overrides",
        int(gate_audit["recorded_gate_overrode_internal"].sum()),
        0,
    )
    check(
        "cross-fitted anchor fold counts",
        sorted(gate_audit["crossfit_anchor_folds"].astype(int).unique().tolist()),
        [4],
    )

    pooled = pd.read_csv(root / "final_analysis/pooled_metrics.csv")
    check("pooled metric traits", pooled["trait"].nunique(), 12)
    check("pooled metric models", pooled["model"].nunique(), 8)
    strongest = pooled.loc[pooled["model"].isin(["cultivar_aware_pls", "nested_rbf_svr"])].copy()
    strongest = strongest.sort_values("rmse").groupby("trait", as_index=False).first()
    final = pooled.loc[pooled["model"].eq("plumspectra_corrected")].copy()
    comparison = final.merge(
        strongest[["trait", "model", "rmse", "r2"]], on="trait", suffixes=("_final", "_strongest")
    )
    comparison["relative_rmse_improvement_pct"] = 100.0 * (
        comparison["rmse_strongest"] - comparison["rmse_final"]
    ) / comparison["rmse_strongest"]
    check("traits with lower final RMSE than strongest single-model baseline", int((comparison["relative_rmse_improvement_pct"] > 0).sum()), 12)
    comparison.to_csv(output / "final_vs_strongest_baseline.csv", index=False)

    cross = pd.read_parquet(root / "crossbatch_final_analysis/v21_merged_predictions.parquet")
    check("crossbatch merged prediction rows", len(cross), 11304)
    check("crossbatch traits", cross["trait"].nunique(), 9)
    check("crossbatch batches", cross["batch_id"].nunique(), 5)
    check("crossbatch cultivars", cross["cultivar_ascii"].nunique(), 2)

    loco = pd.read_parquet(root / "loco_pls_corrected/predictions.parquet")
    loco_fold = pd.read_csv(root / "loco_pls_corrected/fold_metrics.csv")
    check("LOCO predictions", len(loco), 43677)
    check("LOCO trait-cultivar folds", len(loco_fold), 135)
    check("LOCO traits", loco["target"].nunique(), 9)
    check("LOCO cultivars", loco["cultivar_ascii"].nunique(), 15)

    pls2 = pd.read_parquet(root / "multitrait_pls2_final/predictions.parquet")
    check("PLS2 predictions", len(pls2), 58116)
    check("PLS2 fruit-trait duplicates", int(pls2.duplicated(["sample_id", "trait"]).sum()), 0)
    check("PLS2 traits", pls2["trait"].nunique(), 12)

    qc = json.loads((root / "qc_rebuild/texture_qc_summary.json").read_text(encoding="utf-8"))
    check("QC source fruit total", int(qc["fruit_total"]), 5502)
    check("QC strict primary included", int(qc["primary_included"]), 4967)
    check("QC analysis included", int(qc["analysis_included"]), 5430)
    check("QC high-confidence excluded", int(qc["high_confidence_excluded"]), 72)
    check("QC uses model residuals", bool(qc["model_residuals_used"]), False)

    pd.DataFrame(inventory).to_csv(output / "experiment_inventory.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "scientific_release_checks.csv", index=False)
    failed = [row for row in checks if row["status"] != "PASS"]
    report = {
        "release": "PlumSPECTRA V25 external-review correction",
        "status": "PASS" if not failed else "FAIL",
        "scientific_checks": len(checks),
        "failed_checks": len(failed),
        "test_labels_used_for_selection": False,
        "model_residuals_used_for_cohort_selection": False,
        "claim_boundary": (
            "Same-session interpolation among 15 registered cultivars; same-cultivar held-batch "
            "and unseen-cultivar LOCO are separate transfer analyses and are not external validation."
        ),
        "checks": checks,
    }
    (output / "v25_final_release_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failed:
        raise RuntimeError(json.dumps(failed, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
