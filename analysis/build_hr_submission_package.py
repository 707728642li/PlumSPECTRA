#!/usr/bin/env python3
"""Assemble the non-destructive V24 Horticulture Research hand-off package."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
RESULTS = ROOT / "results" / "v22_integrated"
PACKAGE = MANUSCRIPT / "HR_submission_package_v24"
MAIN_RENDER = ROOT / "review_package" / "main_v24_render"
SUPP_RENDER = ROOT / "review_package" / "supplement_v24_render"


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def supplementary_table_registry() -> list[tuple[int, Path]]:
    text = (MANUSCRIPT / "supplement_plumspectra_v22.md").read_text(encoding="utf-8")
    matches = re.findall(
        r"\[Download Supplementary Table S(\d+)\]\(([^)]+\.csv)\)", text
    )
    registry: list[tuple[int, Path]] = []
    for index_text, relative in matches:
        source = (MANUSCRIPT / relative).resolve()
        registry.append((int(index_text), source))
    registry.sort()
    if [index for index, _ in registry] != list(range(1, 33)):
        raise RuntimeError("Supplementary-table registry is not exactly S1-S32")
    return registry


def build() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)

    main_pdf = MAIN_RENDER / "PlumSPECTRA_integrated_manuscript_review.pdf"
    supp_pdf = SUPP_RENDER / "PlumSPECTRA_integrated_supplement_review.pdf"
    copy(main_pdf, MANUSCRIPT / "PlumSPECTRA_integrated_manuscript_review.pdf")
    copy(supp_pdf, MANUSCRIPT / "PlumSPECTRA_integrated_supplement_review.pdf")

    core_files = [
        (MANUSCRIPT / "PlumSPECTRA_integrated_manuscript_review.docx", PACKAGE / "Manuscript_review.docx"),
        (main_pdf, PACKAGE / "Manuscript_review.pdf"),
        (MANUSCRIPT / "PlumSPECTRA_integrated_supplement_review.docx", PACKAGE / "Supplementary_information.docx"),
        (supp_pdf, PACKAGE / "Supplementary_information.pdf"),
        (MANUSCRIPT / "Horticulture_Research_cover_letter_draft.docx", PACKAGE / "Cover_letter_draft.docx"),
        (MANUSCRIPT / "Horticulture_Research_submission_information_form.docx", PACKAGE / "Author_information_form.docx"),
    ]
    for source, destination in core_files:
        copy(source, destination)

    copy(
        RESULTS / "imagegen" / "fig01a_candidates" / "fig01a_candidate_05_papercut_25d.png",
        PACKAGE / "main_figures" / "Figure_1A_unmerged.png",
    )
    copy(
        RESULTS / "figures_r" / "fig01b_cohort_depth.pdf",
        PACKAGE / "main_figures" / "Figure_1B_unmerged.pdf",
    )
    for index, stem in {
        2: "fig02_integrated_phenotype_atlas",
        3: "fig03_plumspectra_architecture_performance",
        4: "fig04_all12_observed_predicted",
        5: "fig05_within_cultivar_heterogeneity",
        6: "fig06_crossbatch_boundary",
    }.items():
        copy(RESULTS / "figures_r" / f"{stem}.pdf", PACKAGE / "main_figures" / f"Figure_{index}.pdf")

    for index in range(1, 26):
        candidates = sorted((RESULTS / "supplement" / "figures").glob(f"figS{index:02d}_*.pdf"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one source for Supplementary Figure S{index}: {candidates}")
        copy(candidates[0], PACKAGE / "supplementary_vector_figures" / f"Figure_S{index:02d}.pdf")

    for index, source in supplementary_table_registry():
        safe_stem = re.sub(r"[^A-Za-z0-9_]+", "_", source.stem)
        copy(source, PACKAGE / "supplementary_tables" / f"Table_S{index:02d}_{safe_stem}.csv")

    readme = """# PlumSPECTRA Horticulture Research V24 submission hand-off

This directory is a review-ready hand-off, not a one-click final submission.

## Scientifically complete

- Anonymous manuscript with 6 main figures, 3 editable tables and alt text.
- Supplementary information with 25 figures and 32 machine-readable tables.
- Separate vector PDFs for Figures 2-6 and all supplementary figures.
- Complete-pipeline multiseed audit: 190 pipeline instances across 12 traits.
- Multiplicity, cultivar mechanical-phenotype and held-batch few-shot audits.

## Author actions required before upload

1. Complete every yellow field in `Author_information_form.docx`.
2. Restore author names, affiliations, CRediT, funding and acknowledgements.
3. Create a data/code DOI or anonymous reviewer link and update Data availability.
4. Merge `Figure_1A_unmerged.png` and `Figure_1B_unmerged.pdf` into one Figure 1 file; verify A/B labels and legend.
5. Replace cover-letter placeholders and confirm the conflict-of-interest statement.
6. Do not claim zero-calibration transfer across years, orchards or instruments.

`SHA256SUMS.csv` records every payload file. The manuscript and supplement PDFs are visual-review copies; use the journal's current upload workflow for the editable manuscript and production figure files.
"""
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")

    payload = sorted(
        path for path in PACKAGE.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.csv"
    )
    with (PACKAGE / "SHA256SUMS.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in payload:
            writer.writerow([path.relative_to(PACKAGE).as_posix(), path.stat().st_size, sha256(path)])

    print(f"package={PACKAGE}")
    print(f"files={len(payload)}")


if __name__ == "__main__":
    build()
