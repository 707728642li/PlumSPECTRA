from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_v5_all15_candidate_screen import gpu_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    prior_root = project / "results" / "v5" / "all15_candidate_screen"
    prior_selection = pd.read_csv(prior_root / "selection" / "all15_single_seed_screen.csv")
    candidates = prior_selection.loc[prior_selection["eligible_for_multiseed"]].copy()
    gate_evidence = project / "results" / "v5" / "trait_screen" / "selection" / "trait_screen_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cycle": "V5.1 all-15-cultivar three-seed stability confirmation",
        "candidates": candidates[["trait_abbreviation", "target", "fixed_gate"]].to_dict("records"),
        "existing_seed": 20260806,
        "additional_seeds": [20260807, 20260808],
        "excluded_cultivars": ["6.11"],
        "one_final_model_per_trait": True,
        "claim_boundary": "Retrospective nested-LOCO stability analysis, not untouched external validation.",
    }
    (output_root / "confirmation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for row in candidates.itertuples(index=False):
        abbreviation = str(row.trait_abbreviation)
        target = str(row.target)
        gate = float(row.fixed_gate)
        seed06_dir = prior_root / "source" / abbreviation
        additional_dir = output_root / "additional_seeds" / abbreviation
        merged_dir = output_root / "merged" / abbreviation
        fixed_dir = output_root / "fixed" / abbreviation
        train_command = [
            sys.executable, str(project / "src" / "train_plumrac_v5_auxpretrain.py"),
            "--auxiliary-set", "quality12",
            "--pretrain-epochs", "12",
            "--pretrain-learning-rate", "0.001",
            "--pretrain-weight-decay", "0.001",
            "--multimodal-dir", str(project / "data" / "processed" / "multimodal"),
            "--qc-ledger", str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
            "--output-dir", str(additional_dir),
            "--target", target,
            "--cohort", "analysis",
            "--exclude-cultivars", "6.11",
            "--seeds", "20260807,20260808",
            "--pls-results", str(project / "results" / "v5" / "plsr_all_traits_excluding_611"),
            "--device", "cuda:0",
            "--profiles", "balanced",
            "--width", "48",
            "--blocks", "4",
            "--dropout", "0.12",
            "--batch-size", "256",
            "--max-epochs", "32",
            "--min-epochs", "6",
            "--patience", "6",
            "--learning-rate", "0.0005",
            "--weight-decay", "0.002",
            "--sampler-power", "1.0",
            "--validation-cultivars", "5",
            "--min-gate-improvement", "0.0025",
            "--min-gate-win-fraction", "0.6",
            "--max-gate-worst-degradation", "0.03",
            "--max-residual-gate", "1.0",
        ]
        merge_command = [
            sys.executable, str(project / "src" / "merge_v5_seed_runs.py"),
            "--run-dir", str(seed06_dir),
            "--run-dir", str(additional_dir),
            "--output-dir", str(merged_dir),
        ]
        apply_command = [
            sys.executable, str(project / "src" / "apply_fixed_residual_gate.py"),
            "--run-dir", str(merged_dir),
            "--output-dir", str(fixed_dir),
            "--gate", str(gate),
            "--selection-evidence", str(gate_evidence),
        ]
        analyze_command = [
            sys.executable, str(project / "src" / "analyze_v4_confirmation.py"),
            "--run-dir", str(fixed_dir),
            "--metadata-run-dir", str(seed06_dir),
            "--metadata-run-dir", str(additional_dir),
            "--output-dir", str(fixed_dir / "statistics"),
            "--bootstrap-repetitions", "5000",
            "--bootstrap-seed", "20260809",
        ]
        commands = [train_command, merge_command, apply_command, analyze_command]
        if args.dry_run:
            for command in commands:
                print(subprocess.list2cmdline(command), flush=True)
            continue
        snapshot = gpu_snapshot()
        if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
            raise RuntimeError(f"GPU0 is not free enough to start {abbreviation}: {snapshot}")
        additional_dir.mkdir(parents=True, exist_ok=True)
        (additional_dir / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"starting additional seeds for {abbreviation} with GPU snapshot {snapshot}", flush=True)
        for command in commands:
            subprocess.run(command, cwd=project, check=True)

    if not args.dry_run:
        subprocess.run(
            [
                sys.executable, str(project / "src" / "summarize_v5_all15_candidate_screen.py"),
                "--candidate-root", str(output_root),
                "--output-dir", str(output_root / "selection"),
            ],
            cwd=project,
            check=True,
        )


if __name__ == "__main__":
    main()
