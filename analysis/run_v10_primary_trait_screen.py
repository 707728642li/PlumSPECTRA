from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from train_texture_pls_loco import DEFAULT_TARGETS
from v2_registry import abbreviated_trait
from run_v7_primary_suite import gpu_snapshot


DEVELOPMENT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"
RD_TARGET = "skin_break_displacement_raw_mean"
MAX_PARALLEL_GPU_JOBS = 2


def train_command(project: Path, output_dir: Path, target: str) -> list[str]:
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
        target,
        "--cohort",
        "primary",
        "--exclude-cultivars",
        "6.11",
        "--heldout",
        DEVELOPMENT_HELDOUT,
        "--seeds",
        "20260806",
        "--pls-results",
        str(project / "results" / "v10" / "plsr_all_traits_primary_excluding_611"),
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
    output_root.mkdir(parents=True, exist_ok=True)
    target_paths: dict[str, Path] = {
        "RD": project / "results" / "v7" / "primary_suite" / "primary_v5"
    }
    jobs: list[tuple[str, str, Path, list[str]]] = []
    for target in DEFAULT_TARGETS:
        if target == RD_TARGET:
            continue
        abbreviation = abbreviated_trait(target)
        output_dir = output_root / abbreviation
        target_paths[abbreviation] = output_dir
        jobs.append((abbreviation, target, output_dir, train_command(project, output_dir, target)))

    manifest = {
        "cycle": "V10 independent texture-trait primary-QC development screen",
        "targets": DEFAULT_TARGETS,
        "reused_target": {"RD": str(target_paths["RD"])},
        "jobs": [{"trait": abbreviation, "target": target} for abbreviation, target, _, _ in jobs],
        "heldout_cultivars": DEVELOPMENT_HELDOUT.split(","),
        "cohort": "primary",
        "excluded_cultivars": ["6.11"],
        "seed": 20260806,
        "max_parallel_gpu_jobs": MAX_PARALLEL_GPU_JOBS,
        "one_final_model_per_trait": True,
        "frozen_architecture": "quality12 source-only auxiliary pretraining e12 + basic multiscale encoder",
        "strong_confirmation_rule": (
            "At least 4/5 development cultivar wins, positive pooled and macro improvement, and at least "
            "one of pooled or macro RMSE improvement >=10%; then all-15-cultivar multiseed confirmation."
        ),
        "claim_boundary": "Retrospective development screen; no result is an untouched external confirmation.",
    }
    (output_root / "screen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        for abbreviation, _, _, command in jobs:
            print(abbreviation, subprocess.list2cmdline(command), flush=True)
        return

    pls_summary = project / "results" / "v10" / "plsr_all_traits_primary_excluding_611" / "summary.json"
    if not pls_summary.exists():
        raise FileNotFoundError(f"Primary all-trait PLSR baseline is incomplete: {pls_summary}")
    snapshot = gpu_snapshot()
    if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
        raise RuntimeError(f"GPU0 is not free enough to start trait screen: {snapshot}")
    (output_root / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    pending = list(jobs)
    running: dict[str, tuple[subprocess.Popen[str], object, object]] = {}
    failures: list[str] = []
    while pending or running:
        while pending and len(running) < MAX_PARALLEL_GPU_JOBS:
            abbreviation, _, output_dir, command = pending.pop(0)
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
            running[abbreviation] = (process, stdout_handle, stderr_handle)
            print(f"started {abbreviation} pid={process.pid}", flush=True)
        for abbreviation in list(running):
            process, stdout_handle, stderr_handle = running[abbreviation]
            return_code = process.poll()
            if return_code is None:
                continue
            stdout_handle.close()
            stderr_handle.close()
            del running[abbreviation]
            print(f"finished {abbreviation} exit={return_code}", flush=True)
            if return_code != 0:
                failures.append(abbreviation)
        if pending or running:
            time.sleep(2)
    if failures:
        raise RuntimeError(f"Trait jobs failed: {failures}")

    summarize = [sys.executable, str(project / "src" / "summarize_v5_trait_screen.py")]
    for abbreviation, path in sorted(target_paths.items()):
        summarize.extend(["--run", f"{abbreviation}={path}"])
    summarize.extend(["--output-dir", str(output_root / "selection")])
    subprocess.run(summarize, cwd=project, check=True)


if __name__ == "__main__":
    main()
