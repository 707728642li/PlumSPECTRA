"""Text and logic audit for the final CEA manuscript sources."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "manuscript/manuscript_plumspectra_cea_v29.md"
SUPP = ROOT / "manuscript/supplement_plumspectra_cea_v29.md"
OUT = ROOT / "results/cea_final_revision/prose_audit.json"


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, re.UNICODE))


def main() -> None:
    main = MAIN.read_text(encoding="utf-8-sig")
    supp = SUPP.read_text(encoding="utf-8-sig")
    body = main.split("## References", 1)[0]
    abstract = body.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", body)) if s.strip()]
    negative_tail = re.compile(r"\b(?:not|cannot|does not|did not|rather than|neither.+nor)\b[^.!?]*[.!?]$", re.I)
    negative_tail_count = sum(bool(negative_tail.search(s)) for s in sentences)
    tokens = re.findall(r"[a-z0-9]+", body.lower())
    grams = Counter(tuple(tokens[i:i + 8]) for i in range(max(0, len(tokens) - 7)))
    duplicated_8grams = sum(1 for count in grams.values() if count > 1)
    revision_leaks = re.findall(
        r"\b(?:corrected|superseded|obsolete|submission audit draft|version 0\.3|V20|V29)\b",
        main + "\n" + supp,
        re.I,
    )
    results_headings = re.findall(r"^### (.+)$", body.split("## Results", 1)[1].split("## Discussion", 1)[0], re.M)
    expected_headings = {
        "Cohort assembly and identity reconciliation",
        "Texture endpoint reliability and correlation structure",
        "Model configuration and training-internal selection",
        "Baseline search coverage",
        "Out-of-fold accuracy and comparison with baselines",
        "Optimisation stability across complete-pipeline seeds",
        "Cultivar-centred performance and the cultivar-mean null",
        "Wavelength association analysis",
        "Held-batch and unseen-cultivar transfer",
    }
    required_claims = [
        "weaker than a spectrum-free cultivar-mean predictor for 11 of 12 traits",
        "0.9–4.0%",
        "FW, F6, LS and LW",
        "95.4%",
        "0.14–0.28",
        "OpenAI Codex and Anthropic Claude Code",
    ]
    checks = {
        "abstract_words": word_count(abstract),
        "sentence_count": len(sentences),
        "negative_tail_count": negative_tail_count,
        "negative_tail_fraction": negative_tail_count / len(sentences),
        "therefore_count": len(re.findall(r"\btherefore\b", body, re.I)),
        "duplicated_internal_8grams": duplicated_8grams,
        "revision_leaks": revision_leaks,
        "results_headings": results_headings,
        "required_claims_present": {claim: claim in main for claim in required_claims},
        "supplement_rendered_table_count": len(re.findall(r"^\|.+\|$", supp, re.M)),
    }
    passed = (
        checks["abstract_words"] <= 250
        and checks["negative_tail_fraction"] <= 0.07
        and checks["therefore_count"] <= 8
        and not revision_leaks
        and set(results_headings) == expected_headings
        and all(checks["required_claims_present"].values())
        and checks["supplement_rendered_table_count"] >= 80
    )
    payload = {"passed": passed, **checks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
