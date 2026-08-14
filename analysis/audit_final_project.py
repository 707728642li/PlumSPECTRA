from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent / "NIRs_plums_research_ready_en_v1.1.0"


def citations(text: str) -> set[int]:
    found: set[int] = set()
    for match in re.finditer(r"\[([0-9,;–\-\s]+)\]", text):
        for part in re.split(r"[,;]", match.group(1)):
            part = part.strip()
            if not part:
                continue
            if "–" in part or "-" in part:
                start, end = re.split(r"[–-]", part)
                found.update(range(int(start), int(end) + 1))
            else:
                found.add(int(part))
    return found


def citation_first_appearance(text: str) -> list[int]:
    order: list[int] = []
    for match in re.finditer(r"\[([0-9,;–\-\s]+)\]", text):
        values: list[int] = []
        for part in re.split(r"[,;]", match.group(1)):
            part = part.strip()
            if "–" in part or "-" in part:
                start, end = re.split(r"[–-]", part)
                values.extend(range(int(start), int(end) + 1))
            elif part:
                values.append(int(part))
        for value in values:
            if value not in order:
                order.append(value)
    return order


def metric_check(path: Path, summary_path: Path, ensemble: bool) -> list[dict]:
    predictions = pd.read_parquet(path)
    record = json.loads(summary_path.read_text(encoding="utf-8"))
    saved = record["pooled_ensemble_metrics" if ensemble else "pooled_metrics"]
    rows = []
    for target, group in predictions.groupby("target"):
        rmse = float(np.sqrt(mean_squared_error(group["y_true"], group["y_pred"])))
        r2 = float(r2_score(group["y_true"], group["y_pred"]))
        rows.append(
            {
                "target": target,
                "n": len(group),
                "unique_samples": group["sample_id"].nunique(),
                "rmse_recomputed": rmse,
                "rmse_saved": saved[target]["rmse"],
                "r2_recomputed": r2,
                "r2_saved": saved[target]["r2"],
                "pass": (
                    len(group) == group["sample_id"].nunique()
                    and abs(rmse - saved[target]["rmse"]) < 1e-10
                    and abs(r2 - saved[target]["r2"]) < 1e-10
                ),
            }
        )
    return rows


def main() -> None:
    checks: list[dict] = []
    release_audit = json.loads(
        (RELEASE / "quality_control/independent_audit_report.json").read_text(encoding="utf-8")
    )
    checks.append(
        {
            "check": "research_ready_release",
            "pass": release_audit["status"] == "PASS"
            and release_audit["samples"] == 5487
            and not release_audit["checksum_failures"],
            "details": release_audit,
        }
    )

    models = {
        "pls": (ROOT / "results/pls_loco/predictions.parquet", ROOT / "results/pls_loco/summary.json", False),
        "hierarchical": (
            ROOT / "results/hierarchical_pls_loco/predictions.parquet",
            ROOT / "results/hierarchical_pls_loco/summary.json",
            False,
        ),
        "cnn": (
            ROOT / "results/cnn_multitask_loco/predictions_ensemble.parquet",
            ROOT / "results/cnn_multitask_loco/summary.json",
            True,
        ),
        "transformer": (
            ROOT / "results/transformer_multitask_loco/predictions_ensemble.parquet",
            ROOT / "results/transformer_multitask_loco/summary.json",
            True,
        ),
        "cnn_texture_aux": (
            ROOT / "results/cnn_texture_aux_loco/predictions_ensemble.parquet",
            ROOT / "results/cnn_texture_aux_loco/summary.json",
            True,
        ),
    }
    for name, (predictions, summary, ensemble) in models.items():
        rows = metric_check(predictions, summary, ensemble)
        seed_pass = True
        if ensemble:
            seed_pass = json.loads(summary.read_text(encoding="utf-8"))["seeds"] == [
                20260806,
                20260807,
                20260808,
            ]
        checks.append(
            {"check": f"saved_predictions_{name}", "pass": all(row["pass"] for row in rows) and seed_pass, "details": rows}
        )

    fewshot = pd.read_csv(ROOT / "results/fewshot_calibration_all/fewshot_summary.csv")
    expected_models = {"pls_direct", "hierarchical", "cnn", "transformer", "cnn_texture_aux"}
    checks.append(
        {
            "check": "fewshot_complete",
            "pass": set(fewshot["model"]) == expected_models
            and set(fewshot["shots"]) == {0, 1, 3, 5, 10, 20, 50}
            and set(fewshot["repeats"]) == {100},
            "details": {"rows": len(fewshot), "models": sorted(fewshot["model"].unique())},
        }
    )

    manuscript_path = ROOT / "manuscript/manuscript_final.md"
    manuscript = manuscript_path.read_text(encoding="utf-8")
    abstract = manuscript.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0]
    main_text = manuscript.split("## Abstract", 1)[1].split("## References", 1)[0]
    abstract_words = len(re.findall(r"\b[\w%²⁻]+\b", abstract))
    main_words = len(re.findall(r"\b[\w%²⁻]+\b", main_text))
    cited = citations(main_text)
    citation_order = citation_first_appearance(main_text)
    required_sections = {
        "## Abstract",
        "## Introduction",
        "## Results",
        "## Discussion",
        "## Materials and methods",
        "## Acknowledgments",
        "## Funding",
        "## Contributions",
        "## Data availability statement",
        "## Conflict of interests",
        "## Supplementary information",
        "## References",
        "## Figure legends",
    }
    checks.append(
        {
            "check": "manuscript_limits_and_citations",
            "pass": abstract_words <= 250
            and main_words <= 6000
            and cited == set(range(1, 31))
            and citation_order == list(range(1, 31))
            and not re.findall(r"\b[A-Z]{2,}\b", abstract)
            and all(section in manuscript for section in required_sections)
            and "{{" not in manuscript,
            "details": {
                "abstract_words": abstract_words,
                "main_text_words": main_words,
                "citations": sorted(cited),
                "citation_first_appearance": citation_order,
                "abstract_uppercase_tokens": re.findall(r"\b[A-Z]{2,}\b", abstract),
                "missing_sections": sorted(section for section in required_sections if section not in manuscript),
                "unresolved_tokens": "{{" in manuscript,
            },
        }
    )

    required_files = [
        ROOT / "manuscript/Cultivar_shift_plum_NIR_Horticulture_Research.docx",
        ROOT / "manuscript/Supplementary_material_plum_NIR.docx",
        ROOT / "results/figures_main/fig01_study_design.png",
        ROOT / "results/eda/figures/fig02_phenotypic_spectral_diversity.png",
        ROOT / "results/eda/figures/fig03_texture_reliability_biology.png",
        ROOT / "results/model_comparison/figures/fig04_zero_shot_model_comparison.png",
        ROOT / "results/model_comparison/figures/fig05_fewshot_calibration_curves.png",
        ROOT / "results/model_comparison/figures/fig06_hierarchical_10shot_predictions.png",
        *(ROOT / f"manuscript/tables/table{number}_{name}.csv" for number, name in [
            (1, "cohort"),
            (2, "model_performance"),
            (3, "fewshot_hierarchical"),
            (4, "conformal_intervals"),
        ]),
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    checks.append({"check": "required_artifacts", "pass": not missing, "details": {"missing": missing}})

    report = {"status": "PASS" if all(check["pass"] for check in checks) else "FAIL", "checks": checks}
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "final_project_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
