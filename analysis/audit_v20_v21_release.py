from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


V20_MANIFEST_HASH = "f44902e12579b033c354c41a0c00f681801218b3bca5ddc52d7e9cee7dba4105"
V21_MANIFEST_HASH = "a73408ae0f53438ef378d5a77df69a62d65ad0767665f86958963d90bd93104e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_run_tree(root: Path, expected_manifest_hash: str) -> dict[str, Any]:
    metadata_paths = sorted(root.glob("*/fold_*/metadata.json"))
    prediction_paths = sorted(root.glob("*/fold_*/predictions.parquet"))
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
    stderr_paths = sorted(root.glob("*/fold_*/run.stderr.log"))
    return {
        "metadata_files": len(metadata_paths),
        "prediction_files": len(prediction_paths),
        "all_test_labels_excluded_from_selection": bool(
            metadata and all(not item["test_labels_used_for_selection"] for item in metadata)
        ),
        "all_manifest_hashes_match": bool(
            metadata
            and all(item.get("outer_fold_manifest_sha256") == expected_manifest_hash for item in metadata)
        ),
        "nonempty_stderr_logs": [str(path) for path in stderr_paths if path.stat().st_size > 0],
    }


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w–—-]+\b", text, flags=re.UNICODE))


def inspect_manuscript(project: Path) -> dict[str, Any]:
    manuscript_path = project / "manuscript/manuscript_v20_hr.md"
    text = manuscript_path.read_text(encoding="utf-8")
    abstract_match = re.search(r"^## Abstract\s*$([\s\S]*?)(?=^##\s)", text, flags=re.MULTILINE)
    abstract_words = word_count(abstract_match.group(1)) if abstract_match else 0
    references_match = re.search(r"^## References\s*$([\s\S]*?)(?=^##\s|\Z)", text, flags=re.MULTILINE)
    reference_count = (
        len(re.findall(r"^\d+\.\s", references_match.group(1), flags=re.MULTILINE))
        if references_match
        else 0
    )
    required_sections = [
        "## Abstract",
        "## Introduction",
        "## Results",
        "## Discussion",
        "## Materials and methods",
        "## Data availability",
        "## Conflict of interest",
        "## References",
        "## Figure legends",
    ]
    placeholder_patterns = [r"\[(?:AUTHOR|AFFILIATION|FUNDING|DOI|REPOSITORY|INSERT|TODO)[^\]]*\]"]
    unresolved_placeholders = [
        match.group(0)
        for pattern in placeholder_patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]
    figure_headings = len(re.findall(r"^### Figure\s+\d+\.", text, flags=re.MULTILINE))
    alt_texts = len(re.findall(r"^\*\*Alt text:\*\*", text, flags=re.MULTILINE))
    figure_dir = project / "results/v20/figures_hr"
    figure_pdfs = sorted(figure_dir.glob("fig[0-9][0-9]_*.pdf"))
    figure_pngs = sorted(figure_dir.glob("fig[0-9][0-9]_*.png"))
    figure_manifest = json.loads((figure_dir / "figure_manifest.json").read_text(encoding="utf-8"))
    main_tables = [project / f"manuscript/v20_tables/Table_{number}_" for number in range(1, 5)]
    main_table_hits = [
        any(path.parent.glob(path.name + "*.csv"))
        for path in main_tables
    ]
    docx_path = project / "manuscript/Cultivar_aware_deep_kernel_plum_texture_HR_review.docx"
    docx_images = 0
    docx_missing_alt = None
    if docx_path.exists():
        with zipfile.ZipFile(docx_path) as archive:
            docx_images = len(
                [name for name in archive.namelist() if name.startswith("word/media/")]
            )
            document_xml = archive.read("word/document.xml").decode("utf-8")
            drawing_props = re.findall(r"<wp:docPr\b[^>]*>", document_xml)
            docx_missing_alt = sum(
                1 for tag in drawing_props if not re.search(r'\bdescr="[^"]+"', tag)
            )
    a11y_path = project / "review_package/final_release_audit/docx_a11y_audit.json"
    a11y_counts = None
    if a11y_path.exists():
        a11y_counts = json.loads(a11y_path.read_text(encoding="utf-8")).get("counts")
    compliance = {
        "abstract_words": abstract_words,
        "abstract_limit": 250,
        "manuscript_words": word_count(text),
        "manuscript_limit": 6000,
        "reference_count": reference_count,
        "reference_limit": 50,
        "figure_headings": figure_headings,
        "alt_texts": alt_texts,
        "main_figure_pdfs": len(figure_pdfs),
        "main_figure_pngs": len(figure_pngs),
        "figure_manifest_entries": len(figure_manifest.get("figures", [])),
        "main_tables_present": sum(main_table_hits),
        "required_sections_present": {
            heading: heading in text for heading in required_sections
        },
        "unresolved_placeholders": unresolved_placeholders,
        "v21_boundary_language_present": (
            "limited batch transfer" in text.lower()
            and "prospective" in text.lower()
            and "negative" in text.lower()
        ),
        "docx_exists": docx_path.exists(),
        "docx_embedded_images": docx_images,
        "docx_images_missing_alt": docx_missing_alt,
        "docx_a11y_counts": a11y_counts,
    }
    compliance["pass"] = bool(
        0 < abstract_words <= 250
        and compliance["manuscript_words"] <= 6000
        and 0 < reference_count <= 50
        and figure_headings == 6
        and alt_texts == 6
        and len(figure_pdfs) == 6
        and len(figure_pngs) == 6
        and len(figure_manifest.get("figures", [])) == 6
        and sum(main_table_hits) == 4
        and all(compliance["required_sections_present"].values())
        and not unresolved_placeholders
        and compliance["v21_boundary_language_present"]
        and compliance["docx_exists"]
        and docx_images == 6
        and docx_missing_alt == 0
        and a11y_counts == {"high": 0, "medium": 0, "low": 0}
    )
    return compliance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v20_manifest = project / "results/v20/splits/v20_fivefold_manifest.csv"
    v21_manifest = project / "results/v21_crossbatch/splits/v21_crossbatch_manifest.csv"
    checks: dict[str, Any] = {
        "v20_manifest_hash": sha256(v20_manifest),
        "v20_manifest_hash_expected": V20_MANIFEST_HASH,
        "v21_manifest_hash": sha256(v21_manifest),
        "v21_manifest_hash_expected": V21_MANIFEST_HASH,
    }
    checks["v20_manifest_hash_pass"] = checks["v20_manifest_hash"] == V20_MANIFEST_HASH
    checks["v21_manifest_hash_pass"] = checks["v21_manifest_hash"] == V21_MANIFEST_HASH
    checks["v20_ai"] = inspect_run_tree(project / "results/v20/ai", V20_MANIFEST_HASH)
    checks["v21_ai"] = inspect_run_tree(project / "results/v21_crossbatch/ai", V21_MANIFEST_HASH)

    for label, relative, expected_rows, expected_unique, expected_folds in [
        (
            "v20",
            Path("results/v20/final_analysis/v20_merged_predictions.parquet"),
            43_551,
            4_839,
            5,
        ),
        (
            "v21",
            Path("results/v21_crossbatch/final_analysis/v21_merged_predictions.parquet"),
            11_124,
            1_236,
            5,
        ),
    ]:
        frame = pd.read_parquet(project / relative)
        coverage = frame.groupby("trait", observed=True).agg(
            rows=("sample_id", "size"),
            unique_samples=("sample_id", "nunique"),
            folds=("outer_fold", "nunique"),
        )
        checks[f"{label}_prediction_coverage"] = {
            "rows": int(len(frame)),
            "expected_rows": expected_rows,
            "traits": int(frame["trait"].nunique()),
            "samples_per_trait": sorted(coverage["rows"].unique().tolist()),
            "unique_samples_per_trait": sorted(coverage["unique_samples"].unique().tolist()),
            "folds_per_trait": sorted(coverage["folds"].unique().tolist()),
            "pass": bool(
                len(frame) == expected_rows
                and frame["trait"].nunique() == 9
                and (coverage["rows"] == expected_unique).all()
                and (coverage["unique_samples"] == expected_unique).all()
                and (coverage["folds"] == expected_folds).all()
            ),
        }

    v20_gates = json.loads(
        (project / "results/v20/final_analysis/success_gates.json").read_text(encoding="utf-8")
    )
    checks["v20_success_gates"] = v20_gates
    checks["v20_primary_success_pass"] = bool(
        v20_gates["all_nine_hybrid_numerically_better_than_global_pls"]
        and v20_gates["hybrid_vs_global_pls_ci_supported_traits"] == 9
        and v20_gates["all_nine_hybrid_numerically_better_than_domain_pls"]
        and v20_gates["hybrid_vs_domain_pls_ci_supported_traits"] == 9
        and v20_gates["hybrid_vs_domain_svr_traits_won"] == 9
    )
    checks["manuscript_compliance"] = inspect_manuscript(project)

    hash_candidates = [
        project / "results/v20/PROTOCOL.md",
        v20_manifest,
        project / "results/v20/final_analysis/audit_summary.json",
        project / "results/v20/final_analysis/paired_cluster_bootstrap_comparisons.csv",
        project / "results/v21_crossbatch/PROTOCOL.md",
        v21_manifest,
        project / "results/v21_crossbatch/final_analysis/audit_summary.json",
        project / "manuscript/manuscript_v20_hr.md",
        project / "manuscript/Cultivar_aware_deep_kernel_plum_texture_HR_review.docx",
        project / "manuscript/docx_qa_v20_final7/Cultivar_aware_deep_kernel_plum_texture_HR_review.pdf",
        project / "REVIEW_START_HERE.md",
        project / "review_package/00_READ_ME_FIRST.md",
        project / "review_package/04_EVIDENCE_INDEX.csv",
        project / "review_package/06_PACKAGE_MANIFEST.json",
        project / "review_package/11_V20_V21_FINAL_PROJECT_REPORT_ZH.md",
        project / "review_package/12_REPRODUCIBILITY_README.md",
        project / "review_package/final_release_audit/docx_a11y_audit.json",
        project / "results/v20/figures_hr/figure_manifest.json",
        project / "manuscript/v20_tables/table_manifest.json",
        project / "src/audit_v20_v21_release.py",
    ]
    hash_candidates.extend(sorted((project / "results/v20/figures_hr").glob("*.pdf")))
    hash_candidates.extend(sorted((project / "manuscript/v20_tables").glob("*.csv")))
    hash_rows = [
        {
            "relative_path": path.relative_to(project).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in hash_candidates
        if path.exists()
    ]
    pd.DataFrame(hash_rows).to_csv(output_dir / "release_sha256.csv", index=False)

    checks["all_required_checks_pass"] = bool(
        checks["v20_manifest_hash_pass"]
        and checks["v21_manifest_hash_pass"]
        and checks["v20_ai"]["metadata_files"] == 45
        and checks["v20_ai"]["prediction_files"] == 45
        and checks["v20_ai"]["all_test_labels_excluded_from_selection"]
        and checks["v20_ai"]["all_manifest_hashes_match"]
        and not checks["v20_ai"]["nonempty_stderr_logs"]
        and checks["v21_ai"]["metadata_files"] == 45
        and checks["v21_ai"]["prediction_files"] == 45
        and checks["v21_ai"]["all_test_labels_excluded_from_selection"]
        and checks["v21_ai"]["all_manifest_hashes_match"]
        and not checks["v21_ai"]["nonempty_stderr_logs"]
        and checks["v20_prediction_coverage"]["pass"]
        and checks["v21_prediction_coverage"]["pass"]
        and checks["v20_primary_success_pass"]
        and checks["manuscript_compliance"]["pass"]
    )
    (output_dir / "release_audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
