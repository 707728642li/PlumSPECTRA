from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
PACKAGE = MANUSCRIPT / "submission_package"


FILES = {
    "Manuscript.docx": MANUSCRIPT / "Cultivar_shift_plum_NIR_Horticulture_Research.docx",
    "Supplementary_material.docx": MANUSCRIPT / "Supplementary_material_plum_NIR.docx",
    "Submission_required_information.md": MANUSCRIPT / "submission_required_information.md",
    "Figure_1.pdf": ROOT / "results/figures_main/fig01_study_design.pdf",
    "Figure_2.pdf": ROOT / "results/eda/figures/fig02_phenotypic_spectral_diversity.pdf",
    "Figure_3.pdf": ROOT / "results/eda/figures/fig03_texture_reliability_biology.pdf",
    "Figure_4.pdf": ROOT / "results/model_comparison/figures/fig04_zero_shot_model_comparison.pdf",
    "Figure_5.pdf": ROOT / "results/model_comparison/figures/fig05_fewshot_calibration_curves.pdf",
    "Figure_6.pdf": ROOT / "results/model_comparison/figures/fig06_hierarchical_10shot_predictions.pdf",
    "Figure_S1.pdf": ROOT / "results/spectral_interpretation/figures/figS_pls_vip_stability.pdf",
    "Table_1.csv": MANUSCRIPT / "tables/table1_cohort.csv",
    "Table_2.csv": MANUSCRIPT / "tables/table2_model_performance.csv",
    "Table_3.csv": MANUSCRIPT / "tables/table3_fewshot_hierarchical.csv",
    "Table_4.csv": MANUSCRIPT / "tables/table4_conformal_intervals.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    missing = [str(path) for path in FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing submission assets: {missing}")
    PACKAGE.mkdir(exist_ok=True)
    records = []
    for name, source in FILES.items():
        destination = PACKAGE / name
        shutil.copy2(source, destination)
        records.append({"file": name, "bytes": destination.stat().st_size, "sha256": sha256(destination)})
    supplementary_size = (PACKAGE / "Supplementary_material.docx").stat().st_size
    if supplementary_size > 2 * 1024 * 1024:
        raise RuntimeError(f"Supplementary file exceeds the journal's 2 MB limit: {supplementary_size} bytes")
    readme = (
        "# Horticulture Research submission package\n\n"
        "The manuscript contains editable tables at first mention, six embedded main figures with legends and alt text, "
        "and explicit placeholders for author- and instrument-supplied metadata. The separate PDF figures are vector submission assets.\n\n"
        "Before submission, complete every item in `Submission_required_information.md`, replace manuscript placeholders, "
        "deposit data/code and add their permanent citations.\n"
    )
    (PACKAGE / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "journal": "Horticulture Research",
        "article_type": "Article",
        "supplementary_size_limit_bytes": 2 * 1024 * 1024,
        "supplementary_size_bytes": supplementary_size,
        "files": records,
    }
    (PACKAGE / "submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checksum_lines = [f"{record['sha256']}  {record['file']}" for record in records]
    (PACKAGE / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="ascii")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
