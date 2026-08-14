"""Automated scientific and submission audit for the CEA V29 release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "manuscript" / "manuscript_plumspectra_cea_v29.md"
SUPP = ROOT / "manuscript" / "supplement_plumspectra_cea_v29.md"
HIGHLIGHTS = ROOT / "manuscript" / "cea_v29_submission" / "PlumSPECTRA_CEA_V29_highlights.md"
OUT = ROOT / "results" / "v29_cea_submission"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def words(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manuscript = MAIN.read_text(encoding="utf-8-sig")
    supplement = SUPP.read_text(encoding="utf-8-sig")
    highlights = HIGHLIGHTS.read_text(encoding="utf-8-sig")
    checks: list[dict[str, object]] = []

    def check(name: str, condition: bool, evidence: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "evidence": evidence})

    abstract = between(manuscript, "## Abstract", "**Keywords:**")
    keyword_line = manuscript.split("**Keywords:**", 1)[1].splitlines()[0].strip()
    keyword_count = len([x for x in keyword_line.split(";") if x.strip()])
    highlight_lines = [line[2:].strip() for line in highlights.splitlines() if line.startswith("- ")]
    check("abstract_at_most_250_words", words(abstract) <= 250, words(abstract))
    check("six_keywords", keyword_count == 6, keyword_count)
    check("three_to_five_highlights", 3 <= len(highlight_lines) <= 5, len(highlight_lines))
    check("highlights_at_most_85_characters", all(len(x) <= 85 for x in highlight_lines), [len(x) for x in highlight_lines])

    required_order = [
        "## Introduction", "## Materials and methods", "## Results", "## Discussion",
        "## Conclusions", "## Data availability", "## Figure legends", "## References",
    ]
    positions = [manuscript.index(token) for token in required_order]
    check("imrad_and_endmatter_order", positions == sorted(positions), positions)
    check("no_numeric_bracket_citations", re.search(r"\[[0-9]", manuscript) is None, "author-year citations")
    check("no_old_target_journal_name", "Horticulture Research" not in manuscript + supplement, "absent")
    old_release_phrases = ["V27 systematic manuscript revision", "This supplement accompanies the V27 manuscript"]
    check("no_old_release_labels", not any(x in supplement for x in old_release_phrases), "clean title block")
    revision_leaks = ["Submission audit draft", "Manuscript version:", "obsolete V20", "superseded V20"]
    check("no_revision_history_leaks", not any(x.lower() in (manuscript + supplement).lower() for x in revision_leaks), "absent")
    check("branch_excluded_claim", "0.9–4.0%" in manuscript and "12 simultaneous intervals" in manuscript, "present")
    check("global_pls_null_context", "less accurate than the spectrum-free cultivar-mean predictor" in manuscript, "present")
    check("key_tables_rendered", supplement.count("| Trait |") >= 7, supplement.count("| Trait |"))
    check("no_arbitrary_five_percent_claim", "±5%" not in manuscript and "+/-5%" not in manuscript, "absent")

    for number in range(1, 7):
        check(f"figure_{number}_cited", bool(re.search(rf"\bFig(?:ure)?\.?\s*{number}\b", manuscript)), f"Fig. {number}")
        check(f"figure_{number}_legend", manuscript.count(f"### Figure {number}.") == 1, manuscript.count(f"### Figure {number}."))
    check("six_alt_text_blocks", manuscript.count("Alt text:") == 6, manuscript.count("Alt text:"))
    check("supplement_s25_declared", "Supplementary Figure S25" in supplement, "present")
    check("supplement_s42_declared", "Table S42" in supplement, "present")
    check("s25_png_exists", (ROOT / "results/v28_submission_strengthening/figures/Figure_S25_v28_wavelength_evidence.png").exists(), "PNG")
    check("s25_pdf_exists", (ROOT / "results/v28_submission_strengthening/figures/Figure_S25_v28_wavelength_evidence.pdf").exists(), "PDF")

    frozen = json.loads((OUT / "frozen_main_figure_sha256.json").read_text(encoding="utf-8"))["files"]
    for relative, expected in frozen.items():
        path = ROOT / relative
        actual = sha256(path)
        check(f"main_figure_unchanged_{path.stem}", actual == expected, actual)

    predictions = pd.read_parquet(ROOT / "results/v25_external_review_corrections/final_analysis/v25_integrated_predictions.parquet")
    check("oof_row_count", len(predictions) == 58_206, len(predictions))
    check("oof_unique_fruit_trait", predictions[["sample_id", "trait"]].drop_duplicates().shape[0] == 58_206,
          predictions[["sample_id", "trait"]].drop_duplicates().shape[0])
    check("twelve_traits", predictions["trait"].nunique() == 12, sorted(predictions["trait"].unique().tolist()))
    check("five_outer_folds", sorted(predictions["outer_fold"].unique().tolist()) == [1, 2, 3, 4, 5],
          sorted(predictions["outer_fold"].unique().tolist()))
    check("no_missing_final_predictions", predictions["y_final"].notna().all(), int(predictions["y_final"].isna().sum()))

    pooled = pd.read_csv(ROOT / "results/v25_external_review_corrections/final_analysis/pooled_metrics.csv")
    final = pooled[pooled["model"].eq("plumspectra_corrected")].set_index("trait")
    check("final_metric_rows", len(final) == 12, len(final))
    check("fruit_weight_r2", abs(float(final.loc["FW", "r2"]) - 0.827457) < 5e-6, float(final.loc["FW", "r2"]))
    check("ssc_r2", abs(float(final.loc["SSC", "r2"]) - 0.628828) < 5e-6, float(final.loc["SSC", "r2"]))
    check("ph_r2", abs(float(final.loc["pH", "r2"]) - 0.543648) < 5e-6, float(final.loc["pH", "r2"]))
    texture = final.loc[["SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"], "r2"]
    check("texture_r2_range", abs(texture.min() - 0.501714) < 5e-6 and abs(texture.max() - 0.718817) < 5e-6,
          [float(texture.min()), float(texture.max())])

    independent = pd.read_csv(ROOT / "results/v25_external_review_corrections/final_analysis/multiplicity_independent_strongest_family.csv")
    check("branch_excluded_family_twelve_traits", len(independent) == 12, len(independent))
    check("branch_excluded_gain_range", independent["relative_rmse_improvement_pct"].between(0.86, 4.02).all(),
          independent["relative_rmse_improvement_pct"].agg(["min", "max"]).to_dict())
    check("branch_excluded_simultaneous_support", independent["supported_simultaneous_0_05"].all(),
          int(independent["supported_simultaneous_0_05"].sum()))

    practical = pd.read_csv(ROOT / "results/v28_submission_strengthening/practical_accuracy_context.csv")
    check("practical_context_twelve_traits", len(practical) == 12, len(practical))
    check("texture_within_half_iqr_range", practical[practical["trait"].isin(texture.index)]["within_half_iqr_pct"].between(70.7, 83.9).all(),
          practical[practical["trait"].isin(texture.index)]["within_half_iqr_pct"].agg(["min", "max"]).to_dict())
    check("deployment_model_card_exists", (OUT / "deployment_model_card.csv").exists(), "Table S42")

    loco = pd.read_parquet(ROOT / "results/v25_external_review_corrections/loco_pls_corrected/predictions.parquet")
    check("loco_prediction_count", len(loco) == 43_677, len(loco))
    check("held_batch_summary_exists", (ROOT / "results/v25_external_review_corrections/crossbatch_final_analysis/pooled_and_batch_macro_metrics.csv").exists(), "present")
    check("multiseed_summary_exists", (ROOT / "results/v25_external_review_corrections/multiseed_analysis/multiseed_summary.csv").exists(), "present")

    reference_block = manuscript.split("## References", 1)[1]
    dois = re.findall(r"https://doi\.org/([^\s.]+(?:\.[^\s.]+)*)\.?", reference_block)
    reference_entries = [line for line in reference_block.splitlines() if line.strip() and not line.startswith("##")]
    check("reference_count_30", len(reference_entries) == 30, len(reference_entries))
    check("unique_reference_dois", len(dois) == len(set(dois)) == 30, [len(dois), len(set(dois))])
    check("cea_references_present", reference_block.count("*Computers and Electronics in Agriculture*") == 3,
          reference_block.count("*Computers and Electronics in Agriculture*"))

    body_before_refs = manuscript.split("## References", 1)[0]
    body_words = words(body_before_refs)
    check("manuscript_body_reported", body_words > 0, body_words)

    passed = sum(item["passed"] for item in checks)
    audit = {"release": "CEA submission revision", "passed": passed, "total": len(checks), "all_passed": passed == len(checks), "checks": checks}
    (OUT / "cea_v29_automated_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = ["# CEA submission automated audit", "", f"Result: **{passed}/{len(checks)} checks passed**.", ""]
    rows += [f"- {'PASS' if item['passed'] else 'FAIL'} — {item['name']}: `{item['evidence']}`" for item in checks]
    (OUT / "cea_v29_automated_audit.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "total": len(checks), "all_passed": audit["all_passed"]}, indent=2))
    if not audit["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
