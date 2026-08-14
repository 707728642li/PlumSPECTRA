#!/usr/bin/env python3
"""Assemble and self-check the independent PlumSPECTRA V24 review package."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "review_package" / "HR_EXTERNAL_AUDIT_PACKAGE_V24_20260808"
ZIP_BASE = ROOT / "review_package" / "HR_EXTERNAL_AUDIT_PACKAGE_V24_20260808"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(source: Path, relative_destination: str, records: list[dict[str, object]]) -> None:
    destination = PACKAGE / relative_destination
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append(
        {
            "package_path": relative_destination.replace("\\", "/"),
            "source_path": str(source),
            "bytes": destination.stat().st_size,
            "sha256": digest(destination),
            "source_sha256": digest(source),
        }
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def assemble() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    manuscript = ROOT / "manuscript"
    v22 = ROOT / "results" / "v22_integrated"
    v24 = ROOT / "results" / "v24_hr_strengthening"

    copies = [
        (manuscript / "PlumSPECTRA_integrated_manuscript_review.pdf", "documents/Manuscript_review.pdf"),
        (manuscript / "PlumSPECTRA_integrated_supplement_review.pdf", "documents/Supplementary_information.pdf"),
        (manuscript / "manuscript_plumspectra_v22_integrated.md", "documents/Manuscript_source.md"),
        (manuscript / "supplement_plumspectra_v22.md", "documents/Supplementary_source.md"),
        (manuscript / "Horticulture_Research_submission_information_form.docx", "documents/Author_information_form.docx"),
        (ROOT / "review_package" / "16_V24_HR_STRENGTHENING_REPORT_ZH.md", "documents/V24_HR_STRENGTHENING_REPORT_ZH.md"),
        (ROOT / "review_package" / "12_REPRODUCIBILITY_README.md", "documents/REPRODUCIBILITY_README.md"),
        (ROOT / "review_package" / "final_release_audit" / "v24_hr_release_audit.json", "evidence/final_release_audit.json"),
        (v24 / "analysis" / "v24_hr_strengthening_summary.json", "evidence/v24_hr_strengthening_summary.json"),
        (v24 / "multiseed_analysis" / "all12_multiseed_summary.csv", "evidence/all12_multiseed_summary.csv"),
        (v24 / "multiseed_analysis" / "all12_multiseed_summary.json", "evidence/all12_multiseed_summary.json"),
        (v24 / "multiseed_analysis" / "alltrait_multiseed_cluster_bootstrap.csv", "evidence/alltrait_multiseed_cluster_bootstrap.csv"),
        (v24 / "multiseed_analysis" / "alltrait_multiseed_fold_metrics.csv", "evidence/alltrait_multiseed_fold_metrics.csv"),
        (v24 / "multiseed_analysis" / "alltrait_seed_fold_metrics.csv", "evidence/alltrait_seed_fold_metrics.csv"),
        (v24 / "multiseed_analysis" / "alltrait_seed_metadata.csv", "evidence/alltrait_seed_metadata.csv"),
        (v22 / "figure_data" / "predictions_all12.csv", "evidence/predictions_all12.csv"),
        (v22 / "figure_data" / "pooled_metrics.csv", "evidence/pooled_metrics.csv"),
        (v22 / "figure_data" / "final_model_cluster_comparisons.csv", "evidence/final_model_cluster_comparisons.csv"),
        (v22 / "figure_data" / "cohort_counts.csv", "evidence/cohort_counts.csv"),
        (v22 / "figure_data" / "within_cultivar_centered_metrics.csv", "evidence/within_cultivar_centered_metrics.csv"),
        (v22 / "figure_data" / "training_dynamics.csv", "evidence/training_dynamics.csv"),
        (v22 / "supplement" / "tables" / "texture_reliability.csv", "evidence/texture_reliability.csv"),
        (v22 / "supplement" / "tables" / "quality_invalid_exclusions.csv", "evidence/quality_invalid_exclusions.csv"),
        (v22 / "supplement" / "tables" / "quality_branch_selection.csv", "evidence/quality_branch_selection.csv"),
        (v22 / "supplement" / "tables" / "v21_pooled_batch_metrics.csv", "evidence/v21_pooled_batch_metrics.csv"),
        (v22 / "supplement" / "tables" / "v21_batch_comparisons.csv", "evidence/v21_batch_comparisons.csv"),
        (v22 / "supplement" / "tables" / "v21_per_batch_metrics.csv", "evidence/v21_per_batch_metrics.csv"),
        (v24 / "analysis" / "multiplicity_adjusted_contrasts.csv", "evidence/multiplicity_adjusted_contrasts.csv"),
        (v24 / "analysis" / "cultivar_texture_diversity.csv", "evidence/cultivar_texture_diversity.csv"),
        (v24 / "analysis" / "cultivar_texture_clusters.csv", "evidence/cultivar_texture_clusters.csv"),
        (v24 / "analysis" / "cultivar_texture_cluster_selection.csv", "evidence/cultivar_texture_cluster_selection.csv"),
        (v24 / "analysis" / "cultivar_texture_profiles.csv", "evidence/cultivar_texture_profiles.csv"),
        (v24 / "analysis" / "cultivar_qc_decisions_condensed.csv", "evidence/cultivar_qc_decisions_condensed.csv"),
        (v24 / "analysis" / "fewshot_minimum_shots.csv", "evidence/fewshot_minimum_shots.csv"),
        (v22 / "splits" / "v22_quality_fivefold_manifest.csv", "evidence/v22_quality_fivefold_manifest.csv"),
        (v22 / "figures_r" / "figure_manifest.csv", "evidence/figure_manifest.csv"),
        (ROOT / "src" / "audit_v22_integrated_release.py", "audit_scripts/audit_v22_integrated_release.py"),
        (ROOT / "src" / "analyze_v24_hr_strengthening.py", "audit_scripts/analyze_v24_hr_strengthening.py"),
        (ROOT / "src" / "analyze_v24_multiseed_alltraits.py", "audit_scripts/analyze_v24_multiseed_alltraits.py"),
        (ROOT / "src" / "run_v23_multiseed_robustness.py", "audit_scripts/run_v23_multiseed_robustness.py"),
        (ROOT / "src" / "render_v22_integrated_figures.R", "audit_scripts/render_v22_integrated_figures.R"),
        (ROOT / "src" / "build_v24_external_audit_package.py", "audit_scripts/build_v24_external_audit_package.py"),
        (ROOT / "src" / "verify_v24_external_audit_package.py", "audit_scripts/verify_v24_external_audit_package.py"),
    ]

    submission_figures = manuscript / "HR_submission_package_v24" / "main_figures"
    for name in [
        "Figure_1A_unmerged.png",
        "Figure_1B_unmerged.pdf",
        "Figure_2.pdf",
        "Figure_3.pdf",
        "Figure_4.pdf",
        "Figure_5.pdf",
        "Figure_6.pdf",
    ]:
        copies.append((submission_figures / name, f"main_figures/{name}"))

    for source, destination in copies:
        copy_file(source, destination, records)

    fewshot_source = v24 / "analysis" / "fewshot_summary.csv"
    fewshot_selected = [
        row for row in csv_rows(fewshot_source)
        if row["model"] == "Deep-kernel ensemble"
        and row["shots"] in {"0", "40"}
        and row["aggregation"] in {"pooled", "batch_macro"}
        and row["adapter"] in {"none", "intercept", "shrunken_affine"}
    ]
    fewshot_destination = PACKAGE / "evidence" / "fewshot_summary_selected.csv"
    if not fewshot_selected:
        raise RuntimeError("Few-shot selection yielded no rows")
    write_csv(fewshot_destination, fewshot_selected, list(fewshot_selected[0]))
    records.append(
        {
            "package_path": "evidence/fewshot_summary_selected.csv",
            "source_path": str(fewshot_source) + " [filtered: Deep-kernel ensemble; shots 0/40]",
            "bytes": fewshot_destination.stat().st_size,
            "sha256": digest(fewshot_destination),
            "source_sha256": digest(fewshot_source),
        }
    )

    write_csv(
        PACKAGE / "SOURCE_MAP.csv",
        records,
        ["package_path", "source_path", "bytes", "sha256", "source_sha256"],
    )

    predictions = csv_rows(PACKAGE / "evidence" / "predictions_all12.csv")
    all12 = csv_rows(PACKAGE / "evidence" / "all12_multiseed_summary.csv")
    release = json.loads((PACKAGE / "evidence" / "final_release_audit.json").read_text(encoding="utf-8"))
    strengthening = json.loads(
        (PACKAGE / "evidence" / "v24_hr_strengthening_summary.json").read_text(encoding="utf-8")
    )

    pair_count = len({(row["sample_id"], row["trait"]) for row in predictions})
    checks = [
        ("required static Markdown files", all((PACKAGE / f"0{i}_{name}").is_file() for i, name in [
            (0, "READ_ME_FIRST_ZH.md"),
            (1, "PROJECT_AUDIT_BRIEF_ZH.md"),
            (2, "CLAIM_EVIDENCE_MATRIX_ZH.md"),
            (3, "REVIEW_RISK_AND_CHECKLIST_ZH.md"),
            (4, "FILE_MAP_AND_REPRODUCTION_ZH.md"),
            (5, "EXTERNAL_REVIEW_INSTRUCTION_ZH.md"),
            (6, "REPORT_TEMPLATE_ZH.md"),
        ]), True),
        ("prediction rows", len(predictions), 58035),
        ("unique sample-trait pairs", pair_count, 58035),
        ("traits", len({row["trait"] for row in predictions}), 12),
        ("cultivars", len({row["cultivar_code"] for row in predictions}), 15),
        ("outer folds", len({row["outer_fold"] for row in predictions}), 5),
        ("multiseed traits", len(all12), 12),
        ("multiseed supported", sum(row["cluster_supported"].lower() == "true" for row in all12), 11),
        ("FW unsupported", next(row["cluster_supported"].lower() for row in all12 if row["trait"] == "FW"), "false"),
        ("release audit status", release["status"], "PASS"),
        ("release audit checks", release["summary"]["checks"], 118),
        ("multiplicity contrasts", strengthening["multiplicity"]["contrasts"], 36),
        ("multiplicity simultaneous", strengthening["multiplicity"]["supported_simultaneous"], 24),
        ("few-shot fruits", strengthening["fewshot"]["fruits"], 1236),
        ("few-shot repeats", strengthening["fewshot"]["repeats"], 500),
        ("main production files", len(list((PACKAGE / "main_figures").glob("*"))), 7),
    ]
    self_check = {
        "package": PACKAGE.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(observed == expected for _, observed, expected in checks) else "FAIL",
        "checks": [
            {
                "name": name,
                "status": "PASS" if observed == expected else "FAIL",
                "observed": observed,
                "expected": expected,
            }
            for name, observed, expected in checks
        ],
        "review_report_files_are_not_part_of_the_frozen_manifest": True,
    }
    (PACKAGE / "AUDIT_PACKAGE_SELF_CHECK.json").write_text(
        json.dumps(self_check, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if self_check["status"] != "PASS":
        raise RuntimeError(json.dumps(self_check, ensure_ascii=False))

    manifest_rows: list[dict[str, object]] = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST_SHA256.csv":
            continue
        manifest_rows.append(
            {
                "relative_path": path.relative_to(PACKAGE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    write_csv(PACKAGE / "MANIFEST_SHA256.csv", manifest_rows, ["relative_path", "bytes", "sha256"])

    zip_path = Path(shutil.make_archive(str(ZIP_BASE), "zip", root_dir=PACKAGE.parent, base_dir=PACKAGE.name))
    print(json.dumps({
        "status": self_check["status"],
        "package": str(PACKAGE),
        "manifest_files": len(manifest_rows),
        "package_bytes": sum(int(row["bytes"]) for row in manifest_rows),
        "zip": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": digest(zip_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    assemble()
