from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_v7_primary_suite import gpu_snapshot


DEVELOPMENT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"
TARGET = "skin_break_displacement_raw_mean"
MAX_PARALLEL_GPU_JOBS = 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    variants = {
        "GROUP_DRO": [
            "--domain-objective",
            "group_dro",
            "--robust-weight",
            "0.35",
            "--group-dro-temperature",
            "0.20",
        ],
        "VREX": [
            "--domain-objective",
            "vrex",
            "--vrex-weight",
            "0.25",
        ],
    }
    manifest = {
        "cycle": "V8 source-cultivar domain-generalization development screen",
        "target": TARGET,
        "heldout_cultivars": DEVELOPMENT_HELDOUT.split(","),
        "cohort": "primary",
        "excluded_cultivars": ["6.11"],
        "variants": variants,
        "max_parallel_gpu_jobs": MAX_PARALLEL_GPU_JOBS,
        "one_final_model_per_trait": True,
        "selection_rule": "Must exceed primary V5.1 on both pooled and macro RMSE improvement.",
        "claim_boundary": "Retrospective development screen; any winner requires frozen all-cultivar multiseed confirmation.",
    }
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    commands: dict[str, list[str]] = {}
    for name, variant_args in variants.items():
        output_dir = output_root / name.lower()
        commands[name] = [
            sys.executable,
            str(project / "src" / "train_plumrac_v8_domainrobust.py"),
            *variant_args,
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
            "--heldout",
            DEVELOPMENT_HELDOUT,
            "--seeds",
            "20260806",
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

    if args.dry_run:
        for name, command in commands.items():
            print(name, subprocess.list2cmdline(command), flush=True)
        return

    snapshot = gpu_snapshot()
    if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
        raise RuntimeError(f"GPU0 is not free enough to start the parallel suite: {snapshot}")
    (output_root / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    running: dict[str, tuple[subprocess.Popen[str], object, object]] = {}
    for name, command in commands.items():
        output_dir = output_root / name.lower()
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
            time.sleep(2)
    if failures:
        raise RuntimeError(f"Candidates failed: {failures}")

    summarize = [
        sys.executable,
        str(project / "src" / "summarize_v5_trait_screen.py"),
        "--run",
        f"PRIMARY_V5={project / 'results' / 'v7' / 'primary_suite' / 'primary_v5'}",
        "--run",
        f"GROUP_DRO={output_root / 'group_dro'}",
        "--run",
        f"VREX={output_root / 'vrex'}",
        "--output-dir",
        str(output_root / "selection"),
    ]
    subprocess.run(summarize, cwd=project, check=True)


if __name__ == "__main__":
    main()
