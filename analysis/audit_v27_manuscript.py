from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "manuscript_plumspectra_v27.md"
EVIDENCE = (
    ROOT
    / "review_package"
    / "HR_EXTERNAL_AUDIT_PACKAGE_V26_INTEGRATED_20260812"
    / "evidence"
    / "final_analysis"
)
OUTPUT = ROOT / "manuscript" / "PlumSPECTRA_V27_manuscript_audit.json"


def words(text: str) -> int:
    return len(re.findall(r"\b[\w²×–-]+\b", text))


checks: list[dict[str, object]] = []


def check(name: str, observed: object, expected: object) -> None:
    passed = observed == expected
    checks.append(
        {"name": name, "pass": passed, "observed": observed, "expected": expected}
    )
    if not passed:
        raise AssertionError(f"{name}: observed={observed!r}; expected={expected!r}")


text = MANUSCRIPT.read_text(encoding="utf-8-sig")
abstract = re.search(r"## Abstract\s+(.*?)\s+\*\*Keywords:", text, re.S)
assert abstract
main_text = re.search(
    r"## Abstract\s+(.*?)\s+## Figure legends", text, re.S
)
assert main_text

check("abstract within 250 words", words(abstract.group(1)) <= 250, True)
check("article body within 6,000 words", words(main_text.group(1)) <= 6000, True)

required_headings = [
    "## Abstract",
    "## Introduction",
    "## Results",
    "## Discussion",
    "## Materials and methods",
    "## Acknowledgments",
    "## Contributions",
    "## Data availability",
    "## Conflict of interests",
    "## Supplementary information",
    "## References",
    "## Figure legends",
]
for heading in required_headings:
    check(f"required heading {heading}", heading in text, True)

check(
    "six main figure legends",
    len(re.findall(r"^### Figure [1-6]\.", text, re.M)),
    6,
)
check("six main-figure alt texts", len(re.findall(r"^Alt text:", text, re.M)), 6)

reference_block = re.search(r"## References\s+(.*)", text, re.S)
assert reference_block
reference_count = len(re.findall(r"^\d+\.", reference_block.group(1), re.M))
check("reference count no more than 50", reference_count <= 50, True)

for stale in (
    "Fig. 6c,d",
    "Bars, descriptive 95%",
    "GPU training used one",
    "does not claim two-GPU execution",
    "V26 figure-data",
):
    check(f"no stale text: {stale}", stale in text, False)

for required in (
    "12.8–51.5%",
    "8 of 12 simultaneous intervals",
    "58,206 out-of-fold predictions",
    "10 fruit as a descriptive early-return point",
    "20–40 fruit as a practical cost–stability window",
    "0.48 percentage points",
    "two local NVIDIA RTX 3090 graphics processors",
):
    check(f"required claim present: {required}", required in text, True)

pooled = pd.read_csv(EVIDENCE / "pooled_metrics.csv")
final = pooled[pooled.model.eq("plumspectra_corrected")].set_index("trait")
check("formal texture fruit count", int(final.loc["SRF", "n"]), 4853)
check("conventional complete-case count", int(final.loc["FW", "n"]), 4843)
check("fruit-weight R2", round(float(final.loc["FW", "r2"]), 3), 0.827)
check("SSC R2", round(float(final.loc["SSC", "r2"]), 3), 0.629)
check("pH R2", round(float(final.loc["pH", "r2"]), 3), 0.544)

baseline = pd.read_csv(EVIDENCE / "multiplicity_baseline_family_sensitivity.csv")
global_rows = baseline[baseline.baseline.eq("Global PLSR")]
check("global-PLSR family has 12 traits", len(global_rows), 12)
check(
    "global-PLSR all individual intervals positive",
    bool(global_rows.bootstrap_ci95_low_pct.gt(0).all()),
    True,
)
check(
    "global-PLSR simultaneous support within 12-trait family",
    int(global_rows.simultaneous_ci95_low_within_12_contrast_family.gt(0).sum()),
    12,
)

strongest = pd.read_csv(EVIDENCE / "multiplicity_strongest_baseline_family.csv")
check("strongest-baseline simultaneous support", int(strongest.supported_simultaneous_0_05.sum()), 8)

few = pd.read_csv(EVIDENCE / "fig6_fewshot_resampling_uncertainty.csv")
expected_gain = {5: 9.7, 10: 13.5, 20: 15.4, 40: 16.5, 80: 17.0}
observed_gain = {
    int(row.shots): round(float(row.median_resample), 1)
    for row in few.itertuples()
    if row.shots > 0
}
check("few-shot median gain curve", observed_gain, expected_gain)
check(
    "40-to-80 marginal gain",
    round(
        float(few.loc[few.shots.eq(80), "median_resample"].iloc[0])
        - float(few.loc[few.shots.eq(40), "median_resample"].iloc[0]),
        2,
    ),
    0.48,
)

report = {
    "release": "V27",
    "manuscript": str(MANUSCRIPT.relative_to(ROOT)),
    "abstract_words": words(abstract.group(1)),
    "article_words_before_figure_legends": words(main_text.group(1)),
    "checks_passed": sum(item["pass"] for item in checks),
    "checks_total": len(checks),
    "all_passed": all(item["pass"] for item in checks),
    "checks": checks,
    "unresolved_author_actions": [
        "Replace anonymous author, affiliation and corresponding-author placeholders.",
        "Insert funding, acknowledgments and CRediT contributions.",
        "Insert public repository DOI and reviewer-access link.",
    ],
}
OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"PASS {report['checks_passed']}/{report['checks_total']} -> {OUTPUT}")
