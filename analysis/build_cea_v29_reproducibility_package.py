from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = f"CEA_REPRODUCIBILITY_PACKAGE_FINAL_REVISION_{date.today():%Y%m%d}"
OUTPUT_ROOT = ROOT / "review_package" / PACKAGE_NAME
ARCHIVE = OUTPUT_ROOT.with_suffix(".zip")


FILES: list[tuple[str, str]] = [
    ("manuscript/manuscript_plumspectra_cea_v29.md", "manuscript/manuscript_plumspectra_cea_v29.md"),
    ("manuscript/supplement_plumspectra_cea_v29.md", "manuscript/supplement_plumspectra_cea_v29.md"),
    ("manuscript/cea_v29_submission/PlumSPECTRA_CEA_manuscript_revised.docx", "submission/PlumSPECTRA_CEA_manuscript_revised.docx"),
    ("manuscript/cea_v29_submission/PlumSPECTRA_CEA_supplement_revised.docx", "submission/PlumSPECTRA_CEA_supplement_revised.docx"),
    ("manuscript/cea_v29_submission/PlumSPECTRA_CEA_V29_highlights.md", "submission/PlumSPECTRA_CEA_V29_highlights.md"),
    ("manuscript/cea_v29_submission/PlumSPECTRA_CEA_V29_cover_letter.md", "submission/PlumSPECTRA_CEA_V29_cover_letter.md"),
    ("manuscript/cea_v29_submission/PlumSPECTRA_CEA_V29_submission_checklist.md", "submission/PlumSPECTRA_CEA_V29_submission_checklist.md"),
    ("review_package/CEA_REPRODUCIBILITY_PACKAGE_V29_FINAL_20260813/30_CEA_V29_TEXT_AND_LOGIC_REVIEW_claudecode.md", "audit/30_CEA_V29_TEXT_AND_LOGIC_REVIEW_claudecode.md"),
    ("review_package/31_CEA_FINAL_REVISION_AUDIT_ZH.md", "audit/31_CEA_FINAL_REVISION_AUDIT_ZH.md"),
    ("results/v25_external_review_corrections/final_analysis/v25_integrated_predictions.parquet", "evidence/final_analysis/v25_integrated_predictions.parquet"),
    ("results/v25_external_review_corrections/final_analysis/v25_correction_summary.json", "evidence/final_analysis/v25_correction_summary.json"),
    ("results/v29_cea_submission/deployment_model_card.csv", "evidence/deployment_model_card.csv"),
    ("results/v29_cea_submission/cea_v29_automated_audit.json", "audit/cea_v29_automated_audit.json"),
    ("results/v29_cea_submission/cea_v29_automated_audit.md", "audit/cea_v29_automated_audit.md"),
    ("results/cea_final_revision/docx_main_layout_final.json", "audit/docx_main_layout_final.json"),
    ("results/cea_final_revision/docx_supplement_layout_final.json", "audit/docx_supplement_layout_final.json"),
    ("results/cea_final_revision/prose_audit.json", "audit/prose_audit.json"),
    ("results/cea_final_revision/a11y_main.json", "audit/a11y_main.json"),
    ("results/cea_final_revision/a11y_supplement.json", "audit/a11y_supplement.json"),
    ("results/v29_cea_submission/frozen_main_figure_sha256.json", "audit/frozen_main_figure_sha256.json"),
    ("results/cea_final_revision/docx_qa3/manuscript/PlumSPECTRA_CEA_manuscript_revised.pdf", "submission/PlumSPECTRA_CEA_manuscript_revised_rendered_QA.pdf"),
    ("results/cea_final_revision/docx_qa4/supplement/PlumSPECTRA_CEA_supplement_revised.pdf", "submission/PlumSPECTRA_CEA_supplement_revised_rendered_QA.pdf"),
    ("results/cea_final_revision/cea_final_revision_summary.json", "evidence/final_revision/cea_final_revision_summary.json"),
    ("results/cea_final_revision/global_pls_vs_cultivar_mean_null.csv", "evidence/final_revision/global_pls_vs_cultivar_mean_null.csv"),
    ("results/cea_final_revision/multiplicity_independent_strongest_family.csv", "evidence/final_revision/multiplicity_branch_excluded_strongest_family.csv"),
    ("results/cea_final_revision/supplement_key_tables.md", "evidence/final_revision/supplement_key_tables.md"),
    ("environment-lock.txt", "config/environment-lock.txt"),
]


DIRECTORIES: list[tuple[str, str]] = [
    ("results/v25_external_review_corrections/final_analysis", "evidence/final_analysis"),
    ("results/v26_claudecode_integration/figure_data", "evidence/figure_data"),
    ("results/v28_submission_strengthening", "evidence/wavelength_and_practical_context"),
    ("review_package/HR_EXTERNAL_AUDIT_PACKAGE_V26_INTEGRATED_20260812/evidence/manifests", "evidence/manifests"),
    ("review_package/HR_EXTERNAL_AUDIT_PACKAGE_V26_INTEGRATED_20260812/evidence/crossbatch", "evidence/crossbatch"),
    ("review_package/HR_EXTERNAL_AUDIT_PACKAGE_V26_INTEGRATED_20260812/evidence/loco", "evidence/loco"),
    ("review_package/HR_EXTERNAL_AUDIT_PACKAGE_V26_INTEGRATED_20260812/evidence/multiseed", "evidence/multiseed"),
    ("review_package/HR_EXTERNAL_AUDIT_PACKAGE_V26_INTEGRATED_20260812/evidence/pls2", "evidence/pls2"),
    ("results/v26_claudecode_integration/figures_integrated", "figures/main"),
    ("results/v26_claudecode_integration/supplementary_figures", "figures/supplementary_S1_S24"),
    ("src", "code/src"),
]


EXCLUDE_NAMES = {"__pycache__", ".pytest_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


README = """# PlumSPECTRA reproducibility package

This review package supports the manuscript targeted to *Computers and Electronics in Agriculture*.

## What is included

- Anonymous manuscript, supplement, highlights, cover letter and submission checklist.
- Frozen out-of-fold prediction table (58,206 unique fruit-trait rows).
- Frozen cultivar-stratified split manifests and held-batch/LOCO manifests.
- Pooled, foldwise, cultivar-centred, multiseed, held-batch and LOCO evidence.
- Figure-data tables for all six main figures and current-model wavelength evidence.
- Machine-readable deployment model card and automated, prose, accessibility and layout audit reports.
- Analysis and document-build source code, excluding caches and trained model binaries.

## Reproduction boundary

The package permits independent recomputation of the reported metrics, contrasts and most figures from frozen predictions and evidence tables. It does not contain the 11,004 proprietary ARC archives, source spectral exports or trained neural weights. Those files are governed separately because of size and institutional data controls. The package must not be interpreted as external validation: same-session interpolation is primary, the held-batch audit covers five batches from two cultivars, and zero-shot unseen-cultivar transfer failed.

## Integrity

`MANIFEST_SHA256.json` records the SHA-256 hash and byte size of every packaged file except itself, including `PACKAGE_SUMMARY.json`. The summary records package counts and key scientific invariants.
"""


def rewrite_packaged_markdown_paths() -> None:
    manuscript = OUTPUT_ROOT / "manuscript/manuscript_plumspectra_cea_v29.md"
    text = manuscript.read_text(encoding="utf-8")
    text = text.replace(
        "../results/v26_claudecode_integration/figures_integrated/",
        "../figures/main/",
    )
    manuscript.write_text(text, encoding="utf-8")

    supplement = OUTPUT_ROOT / "manuscript/supplement_plumspectra_cea_v29.md"
    text = supplement.read_text(encoding="utf-8")
    text = text.replace(
        "../results/v26_claudecode_integration/supplementary_figures/",
        "../figures/supplementary_S1_S24/",
    ).replace(
        "../results/v28_submission_strengthening/figures/",
        "../figures/supplementary_S25/",
    )
    supplement.write_text(text, encoding="utf-8")

    source_hashes = ROOT / "results/v29_cea_submission/frozen_main_figure_sha256.json"
    payload = json.loads(source_hashes.read_text(encoding="utf-8"))
    payload["files"] = {
        key.replace("results/v26_claudecode_integration/figures_integrated/", "figures/main/"): value
        for key, value in payload["files"].items()
    }
    (OUTPUT_ROOT / "audit/frozen_main_figure_sha256.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_directory(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in EXCLUDE_NAMES for part in relative.parts) or path.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        copy_file(path, destination / relative)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    seen: set[Path] = set()
    for source_text, destination_text in FILES:
        source = ROOT / source_text
        destination = OUTPUT_ROOT / destination_text
        if destination in seen:
            continue
        copy_file(source, destination)
        seen.add(destination)
    for source_text, destination_text in DIRECTORIES:
        copy_directory(ROOT / source_text, OUTPUT_ROOT / destination_text)

    s25 = ROOT / "results/v28_submission_strengthening/figures/Figure_S25_v28_wavelength_evidence.pdf"
    if s25.exists():
        copy_file(s25, OUTPUT_ROOT / "figures/supplementary_S25/Figure_S25_v28_wavelength_evidence.pdf")
    s25_png = s25.with_suffix(".png")
    if s25_png.exists():
        copy_file(s25_png, OUTPUT_ROOT / "figures/supplementary_S25/Figure_S25_v28_wavelength_evidence.png")

    rewrite_packaged_markdown_paths()

    (OUTPUT_ROOT / "README.md").write_text(README, encoding="utf-8")
    packaged_before_summary = sorted(path for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    summary = {
        "package": PACKAGE_NAME,
        "generated": date.today().isoformat(),
        "target_journal": "Computers and Electronics in Agriculture",
        "files_hashed": len(packaged_before_summary) + 1,
        "zip_members": len(packaged_before_summary) + 2,
        "payload_bytes_excluding_summary_and_manifest": sum(path.stat().st_size for path in packaged_before_summary),
        "primary_oof_rows": 58206,
        "retained_cultivars": 15,
        "traits": 12,
        "outer_folds": 5,
        "main_figures_integrity_checked": 6,
        "supplementary_figures": 25,
        "supplementary_tables_described": 42,
    }
    (OUTPUT_ROOT / "PACKAGE_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    packaged = sorted(path for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    manifest = {
        path.relative_to(OUTPUT_ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in packaged
    }
    (OUTPUT_ROOT / "MANIFEST_SHA256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, Path(PACKAGE_NAME) / path.relative_to(OUTPUT_ROOT))
    print(json.dumps({**summary, "directory": str(OUTPUT_ROOT), "archive": str(ARCHIVE), "archive_sha256": sha256(ARCHIVE)}, indent=2))


if __name__ == "__main__":
    main()
