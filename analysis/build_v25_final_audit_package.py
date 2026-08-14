#!/usr/bin/env python3
"""Assemble the frozen PlumSPECTRA V25 final external-review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "review_package/HR_EXTERNAL_AUDIT_PACKAGE_V25_FINAL_20260810"
ZIP_BASE = PACKAGE.parent / PACKAGE.name
SOURCE_DOCS = ROOT / "review_package/v25_final_audit_sources"
RESULTS = ROOT / "results/v25_external_review_corrections"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination_rel: str, records: list[dict[str, object]]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = PACKAGE / destination_rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append(
        {
            "package_path": destination_rel.replace("\\", "/"),
            "source_path": str(source.resolve()),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "source_sha256": sha256(source),
        }
    )


def copy_glob(
    source_root: Path,
    patterns: Iterable[str],
    destination_root: str,
    records: list[dict[str, object]],
) -> None:
    seen: set[Path] = set()
    for pattern in patterns:
        for source in sorted(source_root.glob(pattern)):
            if source.is_file() and source not in seen:
                seen.add(source)
                relative = source.relative_to(source_root).as_posix()
                copy_file(source, f"{destination_root}/{relative}", records)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assemble(refresh: bool = False) -> dict[str, object]:
    if PACKAGE.exists() and not refresh:
        raise FileExistsError(f"Refusing to overwrite frozen package: {PACKAGE}")
    for manuscript_source in (
        ROOT / "manuscript/manuscript_plumspectra_v25_final.md",
        ROOT / "manuscript/supplement_plumspectra_v25_final.md",
    ):
        source_text = manuscript_source.read_text(encoding="utf-8")
        if "{{" in source_text or "}}" in source_text:
            raise RuntimeError(f"Unresolved manuscript placeholder: {manuscript_source}")
    PACKAGE.mkdir(parents=True, exist_ok=refresh)
    records: list[dict[str, object]] = []

    root_docs = [
        "00_READ_ME_FIRST_ZH.md",
        "01_PROJECT_BACKGROUND_TARGET_AND_REQUIREMENTS_ZH.md",
        "02_DATA_QC_AND_PROVENANCE_ZH.md",
        "03_BASELINES_EXPERIMENTS_AND_RESULTS_ZH.md",
        "04_CLAIM_EVIDENCE_AND_RISK_MATRIX_ZH.md",
        "05_FILE_MAP_AND_REPRODUCIBILITY_ZH.md",
        "06_EXTERNAL_REVIEW_INSTRUCTION_ZH.md",
        "07_REVIEW_REPORT_TEMPLATE_ZH.md",
    ]
    for filename in root_docs:
        source_text = (SOURCE_DOCS / filename).read_text(encoding="utf-8")
        if "{{" in source_text or "}}" in source_text:
            raise RuntimeError(f"Unresolved audit-source placeholder: {SOURCE_DOCS / filename}")
        copy_file(SOURCE_DOCS / filename, filename, records)

    fixed_copies = [
        # Final manuscript layer.
        (ROOT / "manuscript/manuscript_plumspectra_v25_final.md", "documents/Manuscript_V25_source.md"),
        (ROOT / "manuscript/supplement_plumspectra_v25_final.md", "documents/Supplement_V25_source.md"),
        (ROOT / "manuscript/PlumSPECTRA_V25_manuscript_review.docx", "documents/Manuscript_V25_review.docx"),
        (ROOT / "manuscript/PlumSPECTRA_V25_supplement_review.docx", "documents/Supplement_V25_review.docx"),
        (ROOT / "manuscript/PlumSPECTRA_V25_manuscript_review.pdf", "documents/Manuscript_V25_review.pdf"),
        (ROOT / "manuscript/PlumSPECTRA_V25_supplement_review.pdf", "documents/Supplement_V25_review.pdf"),
        (ROOT / "review_package/17_V25_FINAL_PROJECT_REPORT_ZH.md", "documents/V25_FINAL_PROJECT_REPORT_ZH.md"),
        (ROOT / "review_package/18_V25_REFERENCE_DOI_AUDIT.md", "documents/V25_REFERENCE_DOI_AUDIT.md"),
        (ROOT / "review_package/11_CLAUDECODE_REVIEW_DISPOSITION_V25_ZH.md", "documents/V25_CLAUDECODE_REVIEW_DISPOSITION_ZH.md"),
        (
            ROOT / "review_package/HR_EXTERNAL_AUDIT_PACKAGE_V24_20260808/10_EXTERNAL_REVIEW_REPORT_HR_V24_claudecode.md",
            "documents/PRIOR_EXTERNAL_REVIEW_HR_V24_claudecode.md",
        ),
        (ROOT / "review_package/09_AUTHOR_CONFIRMATION_AND_FORCE_UNIT_EVIDENCE_ZH.md", "documents/AUTHOR_CONFIRMATION_AND_FORCE_UNIT_EVIDENCE_ZH.md"),
        (ROOT / "review_package/12_REPRODUCIBILITY_README.md", "documents/REPRODUCIBILITY_README.md"),
        # Cohort and provenance.
        (RESULTS / "qc_rebuild/texture_qc_summary.json", "evidence/qc/texture_qc_summary.json"),
        (RESULTS / "qc_rebuild/texture_qc_audit.csv", "evidence/qc/texture_qc_audit.csv"),
        (RESULTS / "qc_rebuild/texture_qc_ledger.parquet", "evidence/qc/texture_qc_ledger.parquet"),
        (RESULTS / "qc_rebuild/texture_endpoint_registry.csv", "evidence/qc/texture_endpoint_registry.csv"),
        (RESULTS / "qc_rebuild/texture_batch_qc_summary.csv", "evidence/qc/texture_batch_qc_summary.csv"),
        (RESULTS / "qc_cultivar_audit/cultivar_exclusion_decision.json", "evidence/qc/cultivar_exclusion_decision.json"),
        (RESULTS / "qc_cultivar_audit/cultivar_measurement_quality_audit.csv", "evidence/qc/cultivar_measurement_quality_audit.csv"),
        (RESULTS / "splits/v20_fivefold_manifest.csv", "evidence/manifests/texture_fivefold_manifest.csv"),
        (RESULTS / "splits/v22_quality_fivefold_manifest.csv", "evidence/manifests/quality_fivefold_manifest.csv"),
        (RESULTS / "crossbatch_splits/v21_crossbatch_manifest.csv", "evidence/manifests/crossbatch_manifest.csv"),
        # Formal integrated analyses.
        (RESULTS / "final_analysis/v25_integrated_predictions.parquet", "evidence/final_analysis/v25_integrated_predictions.parquet"),
        (RESULTS / "final_analysis/pooled_metrics.csv", "evidence/final_analysis/pooled_metrics.csv"),
        (RESULTS / "final_analysis/within_cultivar_centered_metrics.csv", "evidence/final_analysis/within_cultivar_centered_metrics.csv"),
        (RESULTS / "final_analysis/fold_metrics.csv", "evidence/final_analysis/fold_metrics.csv"),
        (RESULTS / "final_analysis/cultivar_metrics.csv", "evidence/final_analysis/cultivar_metrics.csv"),
        (RESULTS / "final_analysis/pooled_null_centered_r2.csv", "evidence/final_analysis/pooled_null_centered_r2.csv"),
        (RESULTS / "final_analysis/extended_cluster_comparisons.csv", "evidence/final_analysis/extended_cluster_comparisons.csv"),
        (RESULTS / "final_analysis/multiplicity_adjusted_contrasts.csv", "evidence/final_analysis/multiplicity_adjusted_contrasts.csv"),
        (RESULTS / "final_analysis/multiplicity_baseline_family_sensitivity.csv", "evidence/final_analysis/multiplicity_baseline_family_sensitivity.csv"),
        (RESULTS / "final_analysis/multiplicity_strongest_baseline_family.csv", "evidence/final_analysis/multiplicity_strongest_baseline_family.csv"),
        (RESULTS / "final_analysis/equal_information_pls2_comparison.csv", "evidence/final_analysis/equal_information_pls2_comparison.csv"),
        (RESULTS / "final_analysis/residual_gate_audit.csv", "evidence/final_analysis/residual_gate_audit.csv"),
        (RESULTS / "final_analysis/fold_hyperparameter_choices.csv", "evidence/final_analysis/fold_hyperparameter_choices.csv"),
        (RESULTS / "final_analysis/fewshot_summary.csv", "evidence/final_analysis/fewshot_summary.csv"),
        (RESULTS / "final_analysis/fewshot_minimum_shots.csv", "evidence/final_analysis/fewshot_minimum_shots.csv"),
        (RESULTS / "final_analysis/heldbatch_claim_audit.csv", "evidence/final_analysis/heldbatch_claim_audit.csv"),
        (RESULTS / "final_analysis/texture_reliability_modeling_cohort.csv", "evidence/final_analysis/texture_reliability_modeling_cohort.csv"),
        (RESULTS / "final_analysis/cultivar_batch_counts.csv", "evidence/final_analysis/cultivar_batch_counts.csv"),
        (RESULTS / "final_analysis/within_cultivar_batch_effects.csv", "evidence/final_analysis/within_cultivar_batch_effects.csv"),
        (RESULTS / "final_analysis/cultivar_exclusion_performance_sensitivity.csv", "evidence/final_analysis/cultivar_exclusion_performance_sensitivity.csv"),
        (RESULTS / "final_analysis/cultivar_611_repeatability_sensitivity.csv", "evidence/final_analysis/cultivar_611_repeatability_sensitivity.csv"),
        (RESULTS / "final_analysis/cultivar_611_spectral_domain_decomposition.csv", "evidence/final_analysis/cultivar_611_spectral_domain_decomposition.csv"),
        (RESULTS / "final_analysis/texture_endpoint_correlation.csv", "evidence/final_analysis/texture_endpoint_correlation.csv"),
        (RESULTS / "final_analysis/texture_endpoint_pca_variance.csv", "evidence/final_analysis/texture_endpoint_pca_variance.csv"),
        (RESULTS / "final_analysis/v25_correction_summary.json", "evidence/final_analysis/v25_correction_summary.json"),
        (RESULTS / "final_analysis/within_cultivar_signal_summary.json", "evidence/final_analysis/within_cultivar_signal_summary.json"),
        # Robustness and transfer.
        (RESULTS / "multiseed_analysis/multiseed_summary.csv", "evidence/multiseed/multiseed_summary.csv"),
        (RESULTS / "multiseed_analysis/seed_fold_metrics.csv", "evidence/multiseed/seed_fold_metrics.csv"),
        (RESULTS / "multiseed_analysis/multiseed_fold_metrics.csv", "evidence/multiseed/multiseed_fold_metrics.csv"),
        (RESULTS / "multiseed_analysis/multiseed_cluster_contrasts.csv", "evidence/multiseed/multiseed_cluster_contrasts.csv"),
        (RESULTS / "multiseed_analysis/seed_metadata.csv", "evidence/multiseed/seed_metadata.csv"),
        (RESULTS / "multiseed_analysis/multiseed_summary.json", "evidence/multiseed/multiseed_summary.json"),
        (RESULTS / "multiseed_final/run_manifest.json", "evidence/multiseed/run_manifest.json"),
        (RESULTS / "crossbatch_final_analysis/v21_merged_predictions.parquet", "evidence/crossbatch/v21_merged_predictions.parquet"),
        (RESULTS / "crossbatch_final_analysis/pooled_and_batch_macro_metrics.csv", "evidence/crossbatch/pooled_and_batch_macro_metrics.csv"),
        (RESULTS / "crossbatch_final_analysis/per_batch_metrics.csv", "evidence/crossbatch/per_batch_metrics.csv"),
        (RESULTS / "crossbatch_final_analysis/descriptive_batch_bootstrap_comparisons.csv", "evidence/crossbatch/descriptive_batch_bootstrap_comparisons.csv"),
        (RESULTS / "crossbatch_final_analysis/train_internal_branch_selection.csv", "evidence/crossbatch/train_internal_branch_selection.csv"),
        (RESULTS / "crossbatch_final_analysis/audit_summary.json", "evidence/crossbatch/audit_summary.json"),
        (RESULTS / "crossbatch_ai_final/run_manifest.json", "evidence/crossbatch/ai_run_manifest.json"),
        (RESULTS / "loco_pls_corrected/predictions.parquet", "evidence/loco/loco_predictions.parquet"),
        (RESULTS / "loco_pls_corrected/fold_metrics.csv", "evidence/loco/loco_fold_metrics.csv"),
        (RESULTS / "loco_pls_corrected/selected_hyperparameters.csv", "evidence/loco/loco_selected_hyperparameters.csv"),
        (RESULTS / "multitrait_pls2_final/predictions.parquet", "evidence/pls2/multitrait_pls2_predictions.parquet"),
        (RESULTS / "multitrait_pls2_final/pooled_metrics.csv", "evidence/pls2/multitrait_pls2_pooled_metrics.csv"),
        (RESULTS / "multitrait_pls2_final/summary.json", "evidence/pls2/multitrait_pls2_summary.json"),
        # Measurement-unit and method-evolution audit.
        (RESULTS / "arc_position_audit/arc_position_unit_audit.json", "evidence/arc/arc_position_unit_audit.json"),
        (RESULTS / "arc_position_audit/arc_position_representative_kinematics.csv", "evidence/arc/arc_position_representative_kinematics.csv"),
        (RESULTS / "baseline_wide_epsilon_replacement_audit.csv", "evidence/method_audit/baseline_wide_epsilon_replacement_audit.csv"),
        (RESULTS / "baseline_metadata_repair_audit.csv", "evidence/method_audit/baseline_metadata_repair_audit.csv"),
        (RESULTS / "final_release_audit/v25_final_release_audit.json", "evidence/release_audit/v25_final_release_audit.json"),
        (RESULTS / "final_release_audit/scientific_release_checks.csv", "evidence/release_audit/scientific_release_checks.csv"),
        (RESULTS / "final_release_audit/experiment_inventory.csv", "evidence/release_audit/experiment_inventory.csv"),
        (RESULTS / "final_release_audit/formal_fold_hyperparameters.csv", "evidence/release_audit/formal_fold_hyperparameters.csv"),
        (RESULTS / "final_release_audit/final_vs_strongest_baseline.csv", "evidence/release_audit/final_vs_strongest_baseline.csv"),
        # Configuration.
        (ROOT / "configs/study.yaml", "evidence/config/study.yaml"),
        (ROOT / "configs/v2_nomenclature.csv", "evidence/config/v2_nomenclature.csv"),
        (ROOT / "configs/v2_trait_registry.csv", "evidence/config/v2_trait_registry.csv"),
        (ROOT / "configs/phenotype_corrections.csv", "evidence/config/phenotype_corrections.csv"),
        (ROOT / "AGENTS.md", "evidence/config/PROJECT_AGENTS.md"),
        (ROOT / "environment-lock.txt", "evidence/config/environment-lock.txt"),
    ]
    for source, destination in fixed_copies:
        if source.suffix.lower() == ".md":
            source_text = source.read_text(encoding="utf-8")
            if "{{" in source_text or "}}" in source_text:
                raise RuntimeError(f"Unresolved document placeholder: {source}")
        copy_file(source, destination, records)

    # Per-fold metadata and train-internal tuning tables are compact enough to
    # ship, unlike model weights.  They allow leakage and search-boundary audits.
    for source_name, destination in (
        ("baselines_texture_final", "evidence/fold_protocol/baselines_texture"),
        ("baselines_quality_final", "evidence/fold_protocol/baselines_quality"),
        ("ai_texture_domain_anchor_final", "evidence/fold_protocol/ai_texture"),
        ("ai_quality_domain_anchor_final", "evidence/fold_protocol/ai_quality"),
        ("crossbatch_baselines_final", "evidence/fold_protocol/crossbatch_baselines"),
        ("crossbatch_ai_final", "evidence/fold_protocol/crossbatch_ai"),
    ):
        copy_glob(
            RESULTS / source_name,
            [
                "*/fold_*/metadata.json",
                "*/fold_*/inner_pls_cv.csv",
                "*/fold_*/final_pls_cv.csv",
                "*/fold_*/inner_svr_cv.csv",
                "*/fold_*/selection_history.csv",
                "*/fold_*/retrain_history.csv",
            ],
            destination,
            records,
        )

    scripts = [
        "build_texture_qc_cohort.py",
        "audit_cultivar_measurement_quality.py",
        "run_v20_nested_baselines.py",
        "repair_v25_baseline_metadata.py",
        "extend_v25_svr_epsilon_search.py",
        "finalize_v25_wide_epsilon_replacements.py",
        "run_v20_fivefold_ai.py",
        "run_v22_quality_ai.py",
        "train_texture_pls_random.py",
        "train_plumrac_v5_stratified.py",
        "run_v25_multitrait_pls2_baseline.py",
        "run_v25_multiseed_robustness.py",
        "analyze_v25_multiseed_robustness.py",
        "train_texture_pls_loco.py",
        "create_v21_crossbatch_manifest.py",
        "analyze_v21_crossbatch.py",
        "analyze_v25_external_review_corrections.py",
        "analyze_v24_hr_strengthening.py",
        "prepare_v25_figure_data.py",
        "prepare_v22_figure_data.py",
        "render_v22_integrated_figures.R",
        "build_v22_integrated_docx.py",
        "audit_arc_position_unit.py",
        "v2_registry.py",
        "audit_v25_final_release.py",
        "build_v25_final_audit_package.py",
        "verify_v25_final_audit_package.py",
    ]
    for filename in scripts:
        copy_file(ROOT / "src" / filename, f"audit_scripts/{filename}", records)

    figures = [
        (ROOT / "results/v22_integrated/imagegen/fig01a_candidates/fig01a_candidate_05_papercut_25d.png", "main_figures/Figure_1A_unmerged.png"),
        (RESULTS / "figures_r/fig01b_cohort_depth.pdf", "main_figures/Figure_1B_unmerged.pdf"),
        (RESULTS / "figures_r/fig02_integrated_phenotype_atlas.pdf", "main_figures/Figure_2.pdf"),
        (RESULTS / "figures_r/fig03_plumspectra_architecture_performance.pdf", "main_figures/Figure_3.pdf"),
        (RESULTS / "figures_r/fig04_all12_observed_predicted.pdf", "main_figures/Figure_4.pdf"),
        (RESULTS / "figures_r/fig05_within_cultivar_heterogeneity.pdf", "main_figures/Figure_5.pdf"),
        (RESULTS / "figures_r/fig06_crossbatch_boundary.pdf", "main_figures/Figure_6.pdf"),
    ]
    for source, destination in figures:
        copy_file(source, destination, records)

    write_csv(
        PACKAGE / "SOURCE_MAP.csv",
        records,
        ["package_path", "source_path", "bytes", "sha256", "source_sha256"],
    )

    integrated = pd.read_parquet(PACKAGE / "evidence/final_analysis/v25_integrated_predictions.parquet")
    multiseed = pd.read_csv(PACKAGE / "evidence/multiseed/multiseed_summary.csv")
    cross = pd.read_parquet(PACKAGE / "evidence/crossbatch/v21_merged_predictions.parquet")
    loco = pd.read_parquet(PACKAGE / "evidence/loco/loco_predictions.parquet")
    release = json.loads((PACKAGE / "evidence/release_audit/v25_final_release_audit.json").read_text(encoding="utf-8"))
    instruction = (PACKAGE / "06_EXTERNAL_REVIEW_INSTRUCTION_ZH.md").read_text(encoding="utf-8")
    checks = [
        ("root audit markdown files", len(list(PACKAGE.glob("0[0-7]_*.md"))), 8),
        ("scientific release audit", release["status"], "PASS"),
        ("integrated predictions", len(integrated), 58206),
        ("integrated traits", integrated["trait"].nunique(), 12),
        ("integrated cultivars", integrated["cultivar_ascii"].nunique(), 15),
        ("integrated unique fruit-trait pairs", len(integrated[["sample_id", "trait"]].drop_duplicates()), 58206),
        ("multiseed summary rows", len(multiseed), 24),
        ("multiseed traits", multiseed["trait"].nunique(), 12),
        ("crossbatch prediction rows", len(cross), 11304),
        ("crossbatch test batches", cross["batch_id"].nunique(), 5),
        ("LOCO prediction rows", len(loco), 43677),
        ("main production files", len(list((PACKAGE / "main_figures").glob("*"))), 7),
        ("review suffix _codex.md", "_codex.md" in instruction, True),
        ("review suffix _claudecode.md", "_claudecode.md" in instruction, True),
        ("FINAL path in instruction", "V25_FINAL_20260810" in instruction, True),
    ]
    self_check = {
        "package": PACKAGE.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(observed == expected for _, observed, expected in checks) else "FAIL",
        "final_v25_performance_release": True,
        "review_report_files_are_allowed_outside_manifest": True,
        "checks": [
            {
                "name": name,
                "status": "PASS" if observed == expected else "FAIL",
                "observed": observed,
                "expected": expected,
            }
            for name, observed, expected in checks
        ],
    }
    (PACKAGE / "AUDIT_PACKAGE_SELF_CHECK.json").write_text(
        json.dumps(self_check, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if self_check["status"] != "PASS":
        raise RuntimeError(json.dumps(self_check, ensure_ascii=False, indent=2))

    manifest_rows: list[dict[str, object]] = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.name != "MANIFEST_SHA256.csv":
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(PACKAGE).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(PACKAGE / "MANIFEST_SHA256.csv", manifest_rows, ["relative_path", "bytes", "sha256"])
    zip_path = Path(shutil.make_archive(str(ZIP_BASE), "zip", root_dir=PACKAGE.parent, base_dir=PACKAGE.name))
    return {
        "status": "PASS",
        "package": str(PACKAGE),
        "tracked_files": len(manifest_rows),
        "package_bytes": sum(int(row["bytes"]) for row in manifest_rows),
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256(zip_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh an existing package in place and regenerate its manifests and ZIP.",
    )
    args = parser.parse_args()
    print(json.dumps(assemble(refresh=args.refresh), ensure_ascii=False, indent=2))
