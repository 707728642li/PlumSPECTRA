from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_v5_all15_candidate_screen import gpu_snapshot


DEVELOPMENT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"
TARGET = "skin_break_displacement_raw_mean"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    variants = ["transformer", "dual_attention"]
    commands: dict[str, list[str]] = {}
    for variant in variants:
        output_dir = output_root / variant
        commands[variant] = [
            sys.executable,
            str(project / "src" / "train_plumrac_v6_global.py"),
            "--global-context", variant,
            "--auxiliary-set", "quality12",
            "--pretrain-epochs", "12",
            "--pretrain-learning-rate", "0.001",
            "--pretrain-weight-decay", "0.001",
            "--multimodal-dir", str(project / "data" / "processed" / "multimodal"),
            "--qc-ledger", str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
            "--output-dir", str(output_dir),
            "--target", TARGET,
            "--cohort", "analysis",
            "--exclude-cultivars", "6.11",
            "--heldout", DEVELOPMENT_HELDOUT,
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
    manifest = {
        "cycle": "V6 evidence-based global-context development suite",
        "variants": variants,
        "parallel_gpu_processes": 2,
        "heldout_cultivars": DEVELOPMENT_HELDOUT.split(","),
        "target": TARGET,
        "baseline": "V5 quality12 e12",
        "selection_primary": "macro RMSE improvement versus same-fold PLSR",
    }
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        for command in commands.values():
            print(subprocess.list2cmdline(command), flush=True)
        return

    snapshot = gpu_snapshot()
    if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
        raise RuntimeError(f"GPU0 is not free enough to start V6 parallel suite: {snapshot}")
    processes: dict[str, subprocess.Popen] = {}
    log_handles = []
    try:
        for variant, command in commands.items():
            output_dir = output_root / variant
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout = (output_dir / "run.stdout.log").open("w", encoding="utf-8")
            stderr = (output_dir / "run.stderr.log").open("w", encoding="utf-8")
            log_handles.extend([stdout, stderr])
            processes[variant] = subprocess.Popen(command, cwd=project, stdout=stdout, stderr=stderr)
        failures = {variant: process.wait() for variant, process in processes.items()}
    finally:
        for handle in log_handles:
            handle.close()
    failed = {variant: code for variant, code in failures.items() if code != 0}
    if failed:
        raise RuntimeError(f"V6 parallel candidates failed: {failed}")

    summarize = [
        sys.executable,
        str(project / "src" / "summarize_v5_trait_screen.py"),
        "--run", f"V5={project / 'results' / 'v5' / 'auxpretrain_suite' / 'quality12_e12'}",
        "--run", f"TF={output_root / 'transformer'}",
        "--run", f"DA={output_root / 'dual_attention'}",
        "--output-dir", str(output_root / "selection"),
    ]
    subprocess.run(summarize, cwd=project, check=True)


if __name__ == "__main__":
    main()
