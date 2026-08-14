from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from v2_registry import trait_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v2" / "plumrac_production_safe",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--fewshot-repeats", type=int, default=200)
    args = parser.parse_args()

    model_root = args.model_root.resolve()
    registry = trait_registry().loc[lambda frame: frame["model_family"] == "endpoint"].copy()
    traits = registry["abbreviation"].tolist()
    missing = [trait for trait in traits if not (model_root / trait / "summary.json").exists()]
    if missing:
        raise RuntimeError(f"Cannot finalize V2; incomplete traits: {missing}")

    python = sys.executable
    multimodal = PROJECT_ROOT / "data" / "processed" / "multimodal"
    ledger = PROJECT_ROOT / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"
    for trait in traits:
        run(
            [
                python,
                str(PROJECT_ROOT / "src" / "package_plumrac_models.py"),
                "--model-dir",
                str(model_root / trait),
                "--multimodal-dir",
                str(multimodal),
                "--qc-ledger",
                str(ledger),
                "--tolerance",
                "0.001",
                "--relative-tolerance",
                "0.00002",
            ]
        )

    comparison = PROJECT_ROOT / "results" / "v2" / "model_comparison_final"
    comparison_command = [
        python,
        str(PROJECT_ROOT / "src" / "build_v2_model_assets.py"),
        "--pls-predictions",
        str(PROJECT_ROOT / "results" / "models" / "texture_pls_loco_analysis" / "predictions.parquet"),
    ]
    for trait in traits:
        comparison_command.extend(
            ["--plumrac", f"{trait}={model_root / trait / 'predictions_ensemble.parquet'}"]
        )
        comparison_command.extend(
            ["--ridge", f"{trait}={PROJECT_ROOT / 'results' / 'v2' / f'ridge_{trait.lower()}' / 'predictions.parquet'}"]
        )
    comparison_command.extend(
        ["--output-dir", str(comparison), "--bootstrap-repeats", str(args.bootstrap_repeats)]
    )
    run(comparison_command)

    gate_audit = PROJECT_ROOT / "results" / "v2" / "gate_audit_final"
    gate_command = [python, str(PROJECT_ROOT / "src" / "build_v2_gate_audit.py")]
    for trait in traits:
        gate_command.extend(["--model", f"{trait}={model_root / trait}"])
    gate_command.extend(["--output-dir", str(gate_audit)])
    run(gate_command)

    seed_decision = PROJECT_ROOT / "results" / "v2" / "seed_expansion_final"
    run(
        [
            python,
            str(PROJECT_ROOT / "src" / "evaluate_v2_seed_expansion.py"),
            "--model-comparison",
            str(comparison),
            "--gate-audit",
            str(gate_audit),
            "--output-dir",
            str(seed_decision),
        ]
    )

    combined_path = model_root / "predictions_all_traits.parquet"
    combined = pd.concat(
        [pd.read_parquet(model_root / trait / "predictions_ensemble.parquet") for trait in traits],
        ignore_index=True,
    )
    if combined.duplicated(["sample_id", "target"]).any():
        raise RuntimeError("Duplicate sample/target rows in combined PlumRAC predictions")
    combined.to_parquet(combined_path, index=False, compression="zstd")

    fewshot = PROJECT_ROOT / "results" / "v2" / "fewshot_final"
    run(
        [
            python,
            str(PROJECT_ROOT / "src" / "evaluate_v2_fewshot.py"),
            "--predictions",
            f"PLSR={PROJECT_ROOT / 'results' / 'models' / 'texture_pls_loco_analysis' / 'predictions.parquet'}",
            "--predictions",
            f"PlumRAC-Net={combined_path}",
            "--output-dir",
            str(fewshot),
            "--shots",
            "0,5,10,20,40",
            "--repeats",
            str(args.fewshot_repeats),
        ]
    )

    run(
        [
            python,
            str(PROJECT_ROOT / "src" / "build_plumrac_attention_figure.py"),
            "--model-dir",
            str(model_root / "RD"),
            "--multimodal-dir",
            str(multimodal),
            "--qc-ledger",
            str(ledger),
            "--pls-vip",
            str(PROJECT_ROOT / "results" / "v2" / "texture_pls_vip" / "pls_vip_wavelength_summary.csv"),
            "--output-dir",
            str(PROJECT_ROOT / "results" / "v2" / "figures" / "rd_attention_final"),
        ]
    )

    report_path = PROJECT_ROOT / "reports" / "V2_autogenerated_results.md"
    run(
        [
            python,
            str(PROJECT_ROOT / "src" / "build_v2_results_report.py"),
            "--model-comparison",
            str(comparison),
            "--gate-audit",
            str(gate_audit),
            "--fewshot",
            str(fewshot),
            "--output",
            str(report_path),
        ]
    )

    audit_path = PROJECT_ROOT / "reports" / "V2_final_audit.json"
    audit_command = [python, str(PROJECT_ROOT / "src" / "audit_v2_outputs.py")]
    for trait in traits:
        audit_command.extend(["--model", f"{trait}={model_root / trait}"])
    audit_command.extend(["--expected-seeds", "20260806", "--output", str(audit_path)])
    run(audit_command)

    key_outputs = [
        comparison / "model_summary.csv",
        comparison / "plumrac_vs_plsr.csv",
        gate_audit / "gate_selection_summary.csv",
        seed_decision / "seed_expansion_decision.csv",
        fewshot / "fewshot_summary.csv",
        report_path,
        audit_path,
        combined_path,
    ]
    manifest = {
        "status": "complete",
        "completed_at": datetime.now().isoformat(),
        "traits": traits,
        "bootstrap_repeats": args.bootstrap_repeats,
        "fewshot_repeats": args.fewshot_repeats,
        "outputs_sha256": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in key_outputs},
    }
    (model_root / "finalization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
