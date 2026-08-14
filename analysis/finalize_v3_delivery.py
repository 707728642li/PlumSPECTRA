from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative_path: str) -> dict:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    evidence = load_json("results/v3/final_evidence/final_evidence_audit.json")
    preset = load_json("reports/V3_docx_preset_audit.json")
    a11y = load_json("reports/V3_docx_a11y_audit.json")
    decision = load_json(
        "results/v3/plumrac_x_cross_trait_screen_analysis/seed_expansion_decision.json"
    )

    metrics_path = ROOT / "results/v3/final_evidence/all_model_pooled_metrics.csv"
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    traits = sorted({row["trait"] for row in metrics})
    models_by_trait = {
        trait: sorted(row["model"] for row in metrics if row["trait"] == trait)
        for trait in traits
    }

    report_md = ROOT / "reports/NIRs_plum_V3_final_model_report_zh.md"
    report_docx = ROOT / "reports/NIRs_plum_V3_final_model_report_zh.docx"
    rendered_pdf = (
        ROOT
        / "reports/_qa_v3_final_report/NIRs_plum_V3_final_model_report_zh_rev4.pdf"
    )
    final_pages = sorted(
        (ROOT / "reports/_qa_v3_final_report").glob("final-page-*.png")
    )
    figure_png = ROOT / "results/v3/final_evidence/fig_v3_final_model_evidence.png"
    strategy_log = ROOT / "reports/V3_model_strategy_log.md"
    artifacts = [
        report_md,
        report_docx,
        rendered_pdf,
        figure_png,
        metrics_path,
        ROOT / "results/v3/final_evidence/model_selection_reading.csv",
        ROOT / "results/v3/final_evidence/plumrac_x_transfer_summary.csv",
        ROOT / "results/v3/plumrac_x_cross_trait_screen_analysis/cross_trait_screen_summary.csv",
        ROOT / "results/v3/plumrac_x_rd_confirmation_analysis/confirmation_summary.json",
        strategy_log,
    ]

    with report_md.open("r", encoding="utf-8") as handle:
        report_text = handle.read()

    checks = {
        "final_evidence_audit_pass": evidence.get("status") == "PASS",
        "docx_preset_audit_pass": preset.get("status") == "PASS",
        "docx_a11y_clean": a11y.get("counts") == {"high": 0, "medium": 0, "low": 0},
        "nine_traits_present": len(traits) == 9,
        "four_models_per_trait": all(len(models) == 4 for models in models_by_trait.values()),
        "all_rows_use_5430_fruits": all(int(row["n"]) == 5430 for row in metrics),
        "no_cross_trait_seed_expansion": decision.get("additional_seed_traits") == [],
        "all_delivery_artifacts_exist": all(path.is_file() for path in artifacts),
        "ten_rendered_pages_present": len(final_pages) == 10,
        "wording_correction_present": "品种平衡采样" in report_text
        and "培养品种平衡采样" not in report_text,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    manifest = {
        "status": status,
        "scope": "Frozen V3 zero-shot LOCO model development and final technical report",
        "checks": checks,
        "traits": traits,
        "models_by_trait": models_by_trait,
        "visual_qa": {
            "status": "PASS" if len(final_pages) == 10 else "FAIL",
            "rendered_pages_inspected": len(final_pages),
            "rendered_pdf": str(rendered_pdf.relative_to(ROOT)),
        },
        "claim_boundary": evidence["claim_boundary"],
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in artifacts
            if path.is_file()
        },
    }
    output = ROOT / "reports/V3_final_delivery_audit.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(output)
    print(status)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
