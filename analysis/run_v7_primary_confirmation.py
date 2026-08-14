from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_v7_primary_suite import gpu_snapshot


TARGET = "skin_break_displacement_raw_mean"
SEEDS = [20260806, 20260807, 20260808]
FIXED_GATE = 0.75
MAX_PARALLEL_GPU_JOBS = 2


def train_command(project: Path, output_dir: Path, seed: int) -> list[str]:
    return [
        sys.executable,
        str(project / "src" / "train_plumrac_v5_auxpretrain.py"),
        "--auxiliary-set",
        "quality12",
        "--pretrain-epochs",
        "12",
        "--pretrain-learning-rate",
        "0.001",
        "--pretrain-weight-decay",
        "0.001",
        "--multimodal-dir",
        str(project / "data" / "processed" / "multimodal"),
        "--qc-ledger",
        str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
        "--output-dir",
        str(output_dir),
        "--target",
        TARGET,
        "--cohort",
        "primary",
        "--exclude-cultivars",
        "6.11",
        "--seeds",
        str(seed),
        "--pls-results",
        str(project / "results" / "v7" / "plsr_rd_primary_excluding_611"),
        "--device",
        "cuda:0",
        "--profiles",
        "balanced",
        "--width",
        "48",
        "--blocks",
        "4",
        "--dropout",
        "0.12",
        "--batch-size",
        "256",
        "--max-epochs",
        "32",
        "--min-epochs",
        "6",
        "--patience",
        "6",
        "--learning-rate",
        "0.0005",
        "--weight-decay",
        "0.002",
        "--sampler-power",
        "1.0",
        "--validation-cultivars",
        "5",
        "--min-gate-improvement",
        "0.0025",
        "--min-gate-win-fraction",
        "0.6",
        "--max-gate-worst-degradation",
        "0.03",
        "--max-residual-gate",
        "1.0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    source_root = output_root / "source"
    merged_dir = output_root / "merged"
    fixed_dir = output_root / "fixed_gate075"
    development_evidence = project / "results" / "v7" / "primary_suite" / "selection" / "trait_screen_summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cycle": "Frozen V7 primary-cohort all-cultivar multiseed confirmation",
        "candidate": "PLUMRAC-MT V5.1 on model-independent primary QC cohort",
        "target": TARGET,
        "cohort": "primary",
        "expected_fruits": 4839,
        "expected_cultivars": 15,
        "excluded_cultivars": ["6.11"],
        "seeds": SEEDS,
        "fixed_gate": FIXED_GATE,
        "gate_selection_evidence": str(development_evidence),
        "max_parallel_gpu_jobs": MAX_PARALLEL_GPU_JOBS,
        "one_final_model_per_trait": True,
        "claim_boundary": "Retrospective nested-LOCO multiseed confirmation; not untouched external validation.",
    }
    (output_root / "confirmation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    jobs = [
        (seed, source_root / f"seed_{seed}", train_command(project, source_root / f"seed_{seed}", seed))
        for seed in SEEDS
    ]
    if args.dry_run:
        for seed, _, command in jobs:
            print(seed, subprocess.list2cmdline(command), flush=True)
        return

    snapshot = gpu_snapshot()
    if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
        raise RuntimeError(f"GPU0 is not free enough to start confirmation: {snapshot}")
    (output_root / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    pending = list(jobs)
    running: dict[int, tuple[subprocess.Popen[str], object, object, Path]] = {}
    failures: list[int] = []
    while pending or running:
        while pending and len(running) < MAX_PARALLEL_GPU_JOBS:
            seed, output_dir, command = pending.pop(0)
            output_dir.mkdir(parents=True, exist_ok=True)
            stdout_handle = (output_dir / "run.stdout.log").open("w", encoding="utf-8")
            stderr_handle = (output_dir / "run.stderr.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=project,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            running[seed] = (process, stdout_handle, stderr_handle, output_dir)
            print(f"started seed={seed} pid={process.pid}", flush=True)
        for seed in list(running):
            process, stdout_handle, stderr_handle, _ = running[seed]
            return_code = process.poll()
            if return_code is None:
                continue
            stdout_handle.close()
            stderr_handle.close()
            del running[seed]
            print(f"finished seed={seed} exit={return_code}", flush=True)
            if return_code != 0:
                failures.append(seed)
        if pending or running:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"Seed jobs failed: {failures}")

    merge = [
        sys.executable,
        str(project / "src" / "merge_v5_seed_runs.py"),
    ]
    for seed in SEEDS:
        merge.extend(["--run-dir", str(source_root / f"seed_{seed}")])
    merge.extend(["--output-dir", str(merged_dir)])
    apply_gate = [
        sys.executable,
        str(project / "src" / "apply_fixed_residual_gate.py"),
        "--run-dir",
        str(merged_dir),
        "--output-dir",
        str(fixed_dir),
        "--gate",
        str(FIXED_GATE),
        "--selection-evidence",
        str(development_evidence),
    ]
    analyze = [
        sys.executable,
        str(project / "src" / "analyze_v4_confirmation.py"),
        "--run-dir",
        str(fixed_dir),
    ]
    for seed in SEEDS:
        analyze.extend(["--metadata-run-dir", str(source_root / f"seed_{seed}")])
    analyze.extend(
        [
            "--output-dir",
            str(fixed_dir / "statistics"),
            "--bootstrap-repetitions",
            "5000",
            "--bootstrap-seed",
            "20260810",
        ]
    )
    for command in [merge, apply_gate, analyze]:
        subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
