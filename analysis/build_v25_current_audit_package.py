#!/usr/bin/env python3
"""Build a frozen, explicitly versioned audit snapshot for PlumSPECTRA V25."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "review_package" / "HR_EXTERNAL_AUDIT_PACKAGE_V25_CURRENT_20260810"
ZIP_BASE = ROOT / "review_package" / PACKAGE.name
SOURCE_DOCS = ROOT / "review_package" / "v25_audit_sources"


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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_prediction_files(path: Path) -> int:
    return len(list(path.rglob("predictions.parquet"))) if path.is_dir() else 0


def build_compute_snapshot() -> dict[str, object]:
    base = ROOT / "results" / "v25_external_review_corrections"
    jobs = [
        ("diagnostic_texture_ai_old_anchor", "ai_texture_final", 45, "diagnostic_only"),
        ("formal_texture_ai_domain_anchor", "ai_texture_domain_anchor_final", 45, "formal_pending"),
        ("formal_quality_ai_domain_anchor", "ai_quality_domain_anchor_final", 15, "formal_pending"),
        ("formal_texture_baselines", "baselines_texture_final", 45, "formal_pending"),
        ("formal_quality_baselines", "baselines_quality_final", 15, "formal_pending"),
        ("formal_crossbatch_baselines", "crossbatch_baselines_final", 45, "formal_pending"),
        ("formal_crossbatch_ai", "crossbatch_ai_final", 45, "formal_pending"),
        ("corrected_loco_pls", "loco_pls_corrected", 135, "formal_pending"),
    ]
    items = []
    for name, rel, expected, role in jobs:
        directory = base / rel
        completed = count_prediction_files(directory)
        items.append(
            {
                "name": name,
                "relative_directory": f"results/v25_external_review_corrections/{rel}",
                "role": role,
                "expected_prediction_files": expected,
                "observed_prediction_files": completed,
                "complete": completed == expected,
                "may_be_used_as_final_performance_evidence": role.startswith("formal") and completed == expected,
            }
        )
    return {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Counts record files visible when the package was frozen. Partial or diagnostic outputs are not "
            "valid final performance evidence. Process liveness is intentionally not inferred from file count."
        ),
        "jobs": items,
        "completed_v25_evidence": {
            "qc_v0_3": (base / "qc_rebuild" / "texture_qc_summary.json").is_file(),
            "new_texture_manifest": (base / "splits" / "v20_fivefold_manifest.csv").is_file(),
            "new_complete_case_manifest": (base / "splits" / "v22_quality_fivefold_manifest.csv").is_file(),
            "multitrait_pls2": (base / "multitrait_pls2_final" / "predictions.parquet").is_file(),
            "arc_position_audit": (base / "arc_position_audit" / "arc_position_unit_audit.json").is_file(),
        },
    }


def assemble() -> dict[str, object]:
    if PACKAGE.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing frozen package: {PACKAGE}. "
            "Rename/remove it deliberately or change the package version."
        )
    PACKAGE.mkdir(parents=True)
    records: list[dict[str, object]] = []

    for name in [
        "00_READ_ME_FIRST_ZH.md",
        "01_PROJECT_BACKGROUND_TARGET_AND_REQUIREMENTS_ZH.md",
        "02_DATA_QC_AND_PROVENANCE_ZH.md",
        "03_BASELINES_EXPERIMENTS_AND_RESULTS_ZH.md",
        "04_CLAIM_EVIDENCE_AND_RISK_MATRIX_ZH.md",
        "05_FILE_MAP_AND_REPRODUCIBILITY_ZH.md",
        "06_EXTERNAL_REVIEW_INSTRUCTION_ZH.md",
        "07_REVIEW_REPORT_TEMPLATE_ZH.md",
    ]:
        copy_file(SOURCE_DOCS / name, name, records)

    copies = [
        # Manuscript layer (current V24-facing draft; explicitly labelled in the package documents).
        (ROOT / "manuscript" / "PlumSPECTRA_integrated_manuscript_review.pdf", "documents/Manuscript_V24_current_review.pdf"),
        (ROOT / "manuscript" / "PlumSPECTRA_integrated_supplement_review.pdf", "documents/Supplement_V24_current_review.pdf"),
        (ROOT / "manuscript" / "manuscript_plumspectra_v22_integrated.md", "documents/Manuscript_V24_source.md"),
        (ROOT / "manuscript" / "supplement_plumspectra_v22.md", "documents/Supplement_V24_source.md"),
        (ROOT / "review_package" / "16_V24_HR_STRENGTHENING_REPORT_ZH.md", "documents/V24_HR_STRENGTHENING_REPORT_ZH.md"),
        (ROOT / "review_package" / "11_CLAUDECODE_REVIEW_DISPOSITION_V25_ZH.md", "documents/V25_REVIEW_DISPOSITION_ZH.md"),
        (ROOT / "review_package" / "12_REPRODUCIBILITY_README.md", "documents/REPRODUCIBILITY_README_V24.md"),
        (ROOT / "review_package" / "HR_EXTERNAL_AUDIT_PACKAGE_V24_20260808" / "10_EXTERNAL_REVIEW_REPORT_HR_V24_claudecode.md", "documents/PRIOR_EXTERNAL_REVIEW_HR_V24_claudecode.md"),
        (ROOT / "review_package" / "09_AUTHOR_CONFIRMATION_AND_FORCE_UNIT_EVIDENCE_ZH.md", "documents/AUTHOR_CONFIRMATION_AND_FORCE_UNIT_EVIDENCE_ZH.md"),
        # V24 frozen evidence.
        (ROOT / "results" / "v22_integrated" / "figure_data" / "predictions_all12.csv", "evidence/v24_frozen/predictions_all12.csv"),
        (ROOT / "results" / "v22_integrated" / "figure_data" / "pooled_metrics.csv", "evidence/v24_frozen/pooled_metrics.csv"),
        (ROOT / "results" / "v22_integrated" / "figure_data" / "within_cultivar_centered_metrics.csv", "evidence/v24_frozen/within_cultivar_centered_metrics.csv"),
        (ROOT / "results" / "v22_integrated" / "figure_data" / "final_model_cluster_comparisons.csv", "evidence/v24_frozen/final_model_cluster_comparisons.csv"),
        (ROOT / "results" / "v24_hr_strengthening" / "analysis" / "multiplicity_adjusted_contrasts.csv", "evidence/v24_frozen/multiplicity_adjusted_contrasts.csv"),
        (ROOT / "results" / "v24_hr_strengthening" / "analysis" / "fewshot_summary.csv", "evidence/v24_frozen/fewshot_summary.csv"),
        (ROOT / "results" / "v24_hr_strengthening" / "analysis" / "v24_hr_strengthening_summary.json", "evidence/v24_frozen/v24_hr_strengthening_summary.json"),
        (ROOT / "results" / "v24_hr_strengthening" / "multiseed_analysis" / "all12_multiseed_summary.csv", "evidence/v24_frozen/all12_multiseed_summary.csv"),
        (ROOT / "results" / "v24_hr_strengthening" / "multiseed_analysis" / "alltrait_seed_fold_metrics.csv", "evidence/v24_frozen/alltrait_seed_fold_metrics.csv"),
        (ROOT / "results" / "v24_hr_strengthening" / "multiseed_analysis" / "alltrait_seed_metadata.csv", "evidence/v24_frozen/alltrait_seed_metadata.csv"),
        (ROOT / "results" / "v22_integrated" / "supplement" / "tables" / "v21_pooled_batch_metrics.csv", "evidence/v24_frozen/v21_pooled_batch_metrics.csv"),
        (ROOT / "results" / "v22_integrated" / "supplement" / "tables" / "v21_per_batch_metrics.csv", "evidence/v24_frozen/v21_per_batch_metrics.csv"),
        (ROOT / "results" / "v22_integrated" / "supplement" / "tables" / "v21_batch_comparisons.csv", "evidence/v24_frozen/v21_batch_comparisons.csv"),
        (ROOT / "results" / "texture_prediction" / "texture_prediction_validation_matrix.csv", "evidence/v24_frozen/legacy_loco_validation_matrix.csv"),
        (ROOT / "results" / "pls_loco" / "aggregate_metrics.csv", "evidence/v24_frozen/legacy_loco_pls_metrics.csv"),
        # Completed V25 evidence.
        (ROOT / "results" / "v25_external_review_corrections" / "qc_rebuild" / "texture_qc_summary.json", "evidence/v25_completed/texture_qc_summary.json"),
        (ROOT / "results" / "v25_external_review_corrections" / "qc_rebuild" / "texture_qc_audit.csv", "evidence/v25_completed/texture_qc_audit.csv"),
        (ROOT / "results" / "v25_external_review_corrections" / "qc_rebuild" / "texture_qc_ledger.parquet", "evidence/v25_completed/texture_qc_ledger.parquet"),
        (ROOT / "results" / "v25_external_review_corrections" / "qc_rebuild" / "texture_endpoint_registry.csv", "evidence/v25_completed/texture_endpoint_registry.csv"),
        (ROOT / "results" / "v25_external_review_corrections" / "qc_rebuild" / "texture_batch_qc_summary.csv", "evidence/v25_completed/texture_batch_qc_summary.csv"),
        (ROOT / "results" / "v25_external_review_corrections" / "qc_cultivar_audit" / "cultivar_exclusion_decision.json", "evidence/v25_completed/cultivar_exclusion_decision.json"),
        (ROOT / "results" / "v25_external_review_corrections" / "qc_cultivar_audit" / "cultivar_measurement_quality_audit.csv", "evidence/v25_completed/cultivar_measurement_quality_audit.csv"),
        (ROOT / "results" / "v25_external_review_corrections" / "splits" / "v20_fivefold_manifest.csv", "evidence/v25_completed/v25_texture_fivefold_manifest.csv"),
        (ROOT / "results" / "v25_external_review_corrections" / "splits" / "v22_quality_fivefold_manifest.csv", "evidence/v25_completed/v25_complete_case_fivefold_manifest.csv"),
        (ROOT / "results" / "v25_external_review_corrections" / "crossbatch_splits" / "v21_crossbatch_manifest.csv", "evidence/v25_completed/v25_crossbatch_manifest.csv"),
        (ROOT / "results" / "v25_external_review_corrections" / "multitrait_pls2_final" / "predictions.parquet", "evidence/v25_completed/multitrait_pls2_predictions.parquet"),
        (ROOT / "results" / "v25_external_review_corrections" / "multitrait_pls2_final" / "pooled_metrics.csv", "evidence/v25_completed/multitrait_pls2_pooled_metrics.csv"),
        (ROOT / "results" / "v25_external_review_corrections" / "multitrait_pls2_final" / "summary.json", "evidence/v25_completed/multitrait_pls2_summary.json"),
        (ROOT / "results" / "v25_external_review_corrections" / "arc_position_audit" / "arc_position_unit_audit.json", "evidence/v25_completed/arc_position_unit_audit.json"),
        (ROOT / "results" / "v25_external_review_corrections" / "arc_position_audit" / "arc_position_representative_kinematics.csv", "evidence/v25_completed/arc_position_representative_kinematics.csv"),
        # Configuration/provenance.
        (ROOT / "configs" / "study.yaml", "evidence/config/study.yaml"),
        (ROOT / "configs" / "v2_nomenclature.csv", "evidence/config/v2_nomenclature.csv"),
        (ROOT / "configs" / "v2_trait_registry.csv", "evidence/config/v2_trait_registry.csv"),
        (ROOT / "configs" / "phenotype_corrections.csv", "evidence/config/phenotype_corrections.csv"),
        (ROOT / "AGENTS.md", "evidence/config/PROJECT_AGENTS.md"),
        (ROOT / "environment-lock.txt", "evidence/config/environment-lock.txt"),
        # Audit and reproduction code.
        (ROOT / "src" / "build_texture_qc_cohort.py", "audit_scripts/build_texture_qc_cohort.py"),
        (ROOT / "src" / "audit_cultivar_measurement_quality.py", "audit_scripts/audit_cultivar_measurement_quality.py"),
        (ROOT / "src" / "run_v20_nested_baselines.py", "audit_scripts/run_v20_nested_baselines.py"),
        (ROOT / "src" / "train_texture_pls_random.py", "audit_scripts/train_texture_pls_random.py"),
        (ROOT / "src" / "train_plumrac_v5_stratified.py", "audit_scripts/train_plumrac_v5_stratified.py"),
        (ROOT / "src" / "run_v20_fivefold_ai.py", "audit_scripts/run_v20_fivefold_ai.py"),
        (ROOT / "src" / "run_v22_quality_ai.py", "audit_scripts/run_v22_quality_ai.py"),
        (ROOT / "src" / "run_v25_multitrait_pls2_baseline.py", "audit_scripts/run_v25_multitrait_pls2_baseline.py"),
        (ROOT / "src" / "train_texture_pls_loco.py", "audit_scripts/train_texture_pls_loco.py"),
        (ROOT / "src" / "analyze_v21_crossbatch.py", "audit_scripts/analyze_v21_crossbatch.py"),
        (ROOT / "src" / "analyze_v25_external_review_corrections.py", "audit_scripts/analyze_v25_external_review_corrections.py"),
        (ROOT / "src" / "prepare_v25_figure_data.py", "audit_scripts/prepare_v25_figure_data.py"),
        (ROOT / "src" / "prepare_v22_figure_data.py", "audit_scripts/prepare_v22_figure_data.py"),
        (ROOT / "src" / "render_v22_integrated_figures.R", "audit_scripts/render_v22_integrated_figures.R"),
        (ROOT / "src" / "audit_arc_position_unit.py", "audit_scripts/audit_arc_position_unit.py"),
        (ROOT / "src" / "build_v25_current_audit_package.py", "audit_scripts/build_v25_current_audit_package.py"),
        (ROOT / "src" / "verify_v25_current_audit_package.py", "audit_scripts/verify_v25_current_audit_package.py"),
    ]

    figure_dir = ROOT / "manuscript" / "HR_submission_package_v24" / "main_figures"
    for figure in [
        "Figure_1A_unmerged.png",
        "Figure_1B_unmerged.pdf",
        "Figure_2.pdf",
        "Figure_3.pdf",
        "Figure_4.pdf",
        "Figure_5.pdf",
        "Figure_6.pdf",
    ]:
        copies.append((figure_dir / figure, f"main_figures_v24/{figure}"))

    for source, destination in copies:
        copy_file(source, destination, records)

    compute_snapshot = build_compute_snapshot()
    compute_path = PACKAGE / "evidence" / "v25_in_progress" / "current_compute_snapshot.json"
    compute_path.parent.mkdir(parents=True, exist_ok=True)
    compute_path.write_text(json.dumps(compute_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    records.append(
        {
            "package_path": "evidence/v25_in_progress/current_compute_snapshot.json",
            "source_path": "generated at package freeze from results/v25_external_review_corrections directory counts",
            "bytes": compute_path.stat().st_size,
            "sha256": sha256(compute_path),
            "source_sha256": "generated",
        }
    )

    for rel in [
        "ai_texture_final/launcher.stdout.log",
        "ai_texture_final/launcher.stderr.log",
        "baselines_texture_final/stdout.log",
        "baselines_texture_final/stderr.log",
        "baselines_quality_final/stdout.log",
        "baselines_quality_final/stderr.log",
    ]:
        source = ROOT / "results" / "v25_external_review_corrections" / rel
        if source.is_file():
            copy_file(source, f"evidence/v25_in_progress/logs/{rel.replace('/', '_')}", records)

    write_csv(
        PACKAGE / "SOURCE_MAP.csv",
        records,
        ["package_path", "source_path", "bytes", "sha256", "source_sha256"],
    )

    v24 = pd.read_csv(PACKAGE / "evidence" / "v24_frozen" / "predictions_all12.csv")
    pls2 = pd.read_parquet(PACKAGE / "evidence" / "v25_completed" / "multitrait_pls2_predictions.parquet")
    texture_manifest = pd.read_csv(PACKAGE / "evidence" / "v25_completed" / "v25_texture_fivefold_manifest.csv")
    complete_manifest = pd.read_csv(PACKAGE / "evidence" / "v25_completed" / "v25_complete_case_fivefold_manifest.csv")
    qc = json.loads((PACKAGE / "evidence" / "v25_completed" / "texture_qc_summary.json").read_text(encoding="utf-8"))
    instruction = (PACKAGE / "06_EXTERNAL_REVIEW_INSTRUCTION_ZH.md").read_text(encoding="utf-8")

    checks = [
        ("root audit markdown files", len(list(PACKAGE.glob("0[0-7]_*.md"))), 8),
        ("V24 prediction rows", len(v24), 58035),
        ("V24 unique fruit-trait pairs", len(v24[["sample_id", "trait"]].drop_duplicates()), 58035),
        ("V24 traits", v24["trait"].nunique(), 12),
        ("V24 cultivars", v24["cultivar_code"].nunique(), 15),
        ("V24 outer folds", v24["outer_fold"].nunique(), 5),
        ("V25 texture manifest rows", len(texture_manifest), 4853),
        ("V25 complete-case manifest rows", len(complete_manifest), 4843),
        ("V25 texture manifest SHA256", sha256(PACKAGE / "evidence" / "v25_completed" / "v25_texture_fivefold_manifest.csv"), "363ad2174d53d7eb2dcbeb8f2cecfb3cb32da98db3b3ba6176da32f43bf29a69"),
        ("V25 complete-case manifest SHA256", sha256(PACKAGE / "evidence" / "v25_completed" / "v25_complete_case_fivefold_manifest.csv"), "7f859700cf7386e571f48305b53b922922b7d07338d7e8051b281351b59ad155"),
        ("V25 PLS2 prediction rows", len(pls2), 58116),
        ("V25 PLS2 fruit-trait pairs", len(pls2[["sample_id", "trait"]].drop_duplicates()), 58116),
        ("V25 PLS2 traits", pls2["trait"].nunique(), 12),
        ("V25 QC total", qc["fruit_total"], 5502),
        ("V25 QC primary included", qc["primary_included"], 4967),
        ("V25 main figures", len(list((PACKAGE / "main_figures_v24").glob("*"))), 7),
        ("review suffix _codex.md", "_codex.md" in instruction, True),
        ("review suffix _claudecode.md", "_claudecode.md" in instruction, True),
    ]
    self_check = {
        "package": PACKAGE.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(observed == expected for _, observed, expected in checks) else "FAIL",
        "snapshot_is_not_a_final_v25_performance_release": True,
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
        if not path.is_file() or path.name == "MANIFEST_SHA256.csv":
            continue
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
        "compute_snapshot": compute_snapshot,
    }


if __name__ == "__main__":
    print(json.dumps(assemble(), ensure_ascii=False, indent=2))
