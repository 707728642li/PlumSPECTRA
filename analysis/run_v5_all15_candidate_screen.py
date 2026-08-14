from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def gpu_snapshot() -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    processes = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "index": int(values[0]),
        "name": values[1],
        "memory_used_mib": int(values[2]),
        "memory_total_mib": int(values[3]),
        "utilization_pct": int(values[4]),
        "compute_processes": [line for line in processes.stdout.splitlines() if line.strip()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    source_root = output_root / "source"
    fixed_root = output_root / "fixed"
    screen_path = project / "results" / "v5" / "trait_screen" / "selection" / "trait_screen_summary.csv"
    screen = pd.read_csv(screen_path)
    candidates = screen.loc[
        screen["eligible_for_full_confirmation"] & screen["trait_abbreviation"].ne("RD")
    ].copy()
    if candidates.empty:
        raise ValueError("No non-RD candidates passed the development screen")
    manifest = {
        "cycle": "V5.1 all-15-cultivar single-seed candidate screen",
        "candidates": candidates[["trait_abbreviation", "target", "fixed_gate"]].to_dict("records"),
        "excluded_cultivars": ["6.11"],
        "seed": 20260806,
        "one_final_model_per_trait": True,
        "gate_source": str(screen_path),
        "multiseed_rule": "at least 8/15 wins and positive pooled and macro RMSE improvements",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "screen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for row in candidates.itertuples(index=False):
        abbreviation = str(row.trait_abbreviation)
        target = str(row.target)
        gate = float(row.fixed_gate)
        source_dir = source_root / abbreviation
        fixed_dir = fixed_root / abbreviation
        train_command = [
            sys.executable,
            str(project / "src" / "train_plumrac_v5_auxpretrain.py"),
            "--auxiliary-set", "quality12",
            "--pretrain-epochs", "12",
            "--pretrain-learning-rate", "0.001",
            "--pretrain-weight-decay", "0.001",
            "--multimodal-dir", str(project / "data" / "processed" / "multimodal"),
            "--qc-ledger", str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
            "--output-dir", str(source_dir),
            "--target", target,
            "--cohort", "analysis",
            "--exclude-cultivars", "6.11",
            "--seeds", "20260806",
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
        apply_command = [
            sys.executable,
            str(project / "src" / "apply_fixed_residual_gate.py"),
            "--run-dir", str(source_dir),
            "--output-dir", str(fixed_dir),
            "--gate", str(gate),
            "--selection-evidence", str(screen_path),
        ]
        analyze_command = [
            sys.executable,
            str(project / "src" / "analyze_v4_confirmation.py"),
            "--run-dir", str(fixed_dir),
            "--metadata-run-dir", str(source_dir),
            "--output-dir", str(fixed_dir / "statistics"),
            "--bootstrap-repetitions", "2000",
            "--bootstrap-seed", "20260807",
        ]
        if args.dry_run:
            print(subprocess.list2cmdline(train_command), flush=True)
            print(subprocess.list2cmdline(apply_command), flush=True)
            print(subprocess.list2cmdline(analyze_command), flush=True)
            continue
        snapshot = gpu_snapshot()
        if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
            raise RuntimeError(f"GPU0 is not free enough to start {abbreviation}: {snapshot}")
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"starting all15 candidate {abbreviation} with GPU snapshot {snapshot}", flush=True)
        subprocess.run(train_command, cwd=project, check=True)
        subprocess.run(apply_command, cwd=project, check=True)
        subprocess.run(analyze_command, cwd=project, check=True)

    if not args.dry_run:
        subprocess.run(
            [
                sys.executable,
                str(project / "src" / "summarize_v5_all15_candidate_screen.py"),
                "--candidate-root", str(output_root),
                "--output-dir", str(output_root / "selection"),
            ],
            cwd=project,
            check=True,
        )


if __name__ == "__main__":
    main()
