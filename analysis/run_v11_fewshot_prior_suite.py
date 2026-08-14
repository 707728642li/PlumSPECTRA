from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_v7_primary_suite import gpu_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    priors = [10.0, 100.0]
    manifest = {
        "cycle": "V11 neural few-shot last-layer L2-SP prior development screen",
        "heldout_cultivars": ["3.13", "Cuihongli", "Konglongdan", "Weiwang", "Weixin"],
        "shots": [5, 10, 20, 40],
        "repeats": 50,
        "ridge_priors": priors,
        "same_calibration_fruits_for_all_strategies": True,
        "calibration_fruits_excluded_from_evaluation": True,
        "max_parallel_gpu_jobs": 2,
        "selection_rule": "Largest mean improvement versus the better of PLSR intercept and affine adapters, requiring positive improvement at every shot count.",
        "claim_boundary": "Development-only prior selection; selected prior requires all-cultivar confirmation.",
    }
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    commands: dict[str, list[str]] = {}
    for prior in priors:
        name = f"prior_{int(prior)}"
        commands[name] = [
            sys.executable,
            str(project / "src" / "evaluate_plumrac_lastlayer_fewshot.py"),
            "--multimodal-dir",
            str(project / "data" / "processed" / "multimodal"),
            "--qc-ledger",
            str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
            "--run-dir",
            str(project / "results" / "v7" / "primary_confirmation" / "source" / "seed_20260806"),
            "--output-dir",
            str(output_root / name),
            "--shots",
            "5,10,20,40",
            "--repeats",
            "50",
            "--fixed-gate",
            "0.75",
            "--ridge-prior",
            str(prior),
            "--augmentations",
            "8",
            "--device",
            "cuda:0",
        ]
    if args.dry_run:
        for name, command in commands.items():
            print(name, subprocess.list2cmdline(command), flush=True)
        return

    snapshot = gpu_snapshot()
    if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
        raise RuntimeError(f"GPU0 is not free enough to start few-shot prior screen: {snapshot}")
    (output_root / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    running: dict[str, tuple[subprocess.Popen[str], object, object]] = {}
    for name, command in commands.items():
        output_dir = output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        stdout_handle = (output_dir / "run.stdout.log").open("w", encoding="utf-8")
        stderr_handle = (output_dir / "run.stderr.log").open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=project, stdout=stdout_handle, stderr=stderr_handle, text=True)
        running[name] = (process, stdout_handle, stderr_handle)
        print(f"started {name} pid={process.pid}", flush=True)
    failures: list[str] = []
    while running:
        for name in list(running):
            process, stdout_handle, stderr_handle = running[name]
            return_code = process.poll()
            if return_code is None:
                continue
            stdout_handle.close()
            stderr_handle.close()
            del running[name]
            print(f"finished {name} exit={return_code}", flush=True)
            if return_code != 0:
                failures.append(name)
        if running:
            time.sleep(1)
    if failures:
        raise RuntimeError(f"Few-shot prior jobs failed: {failures}")

    rows = []
    for prior in priors:
        name = f"prior_{int(prior)}"
        table = pd.read_csv(output_root / name / "neural_vs_plsr.csv")
        table.insert(0, "ridge_prior", prior)
        rows.append(table)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(output_root / "prior_comparison.csv", index=False)
    print(combined.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
