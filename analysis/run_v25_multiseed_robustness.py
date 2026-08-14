from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TRAITS = {
    "FW": ("fruit_weight_g", "quality"),
    "SSC": ("soluble_solids_pct", "quality"),
    "pH": ("ph", "quality"),
    "SRF": ("skin_break_force_g_mean", "texture"),
    "RD": ("skin_break_displacement_raw_mean", "texture"),
    "PFD": ("skin_break_drop_g_mean", "texture"),
    "MFF": ("flesh_force_mean_g_mean", "texture"),
    "F6": ("force_at_6_rawpos_g_mean", "texture"),
    "LS": ("loading_stiffness_g_per_rawpos_mean", "texture"),
    "LW": ("loading_work_g_rawpos_mean", "texture"),
    "PRW": ("post_break_work_g_rawpos_mean", "texture"),
    "AF": ("adhesive_force_g_mean", "texture"),
}


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer")
    return parsed


def gpu_state() -> dict[int, dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    state: dict[int, dict[str, int]] = {}
    for line in result.stdout.strip().splitlines():
        index, used, total, utilization = [int(value.strip()) for value in line.split(",")]
        state[index] = {
            "memory_used_mib": used,
            "memory_total_mib": total,
            "utilization_pct": utilization,
        }
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run two independent V25 complete-pipeline robustness seeds per frozen outer fold. "
            "All anchor, epoch and residual-gate selection remains inside outer training data."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--texture-manifest", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True)
    parser.add_argument("--traits", default="all")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--seed-repeats", default="101,102")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--jobs-per-device", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    trainer = Path(__file__).with_name("train_plumrac_v5_stratified.py")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_traits = (
        list(TRAITS)
        if args.traits == "all"
        else [item.strip() for item in args.traits.split(",") if item.strip()]
    )
    unknown = sorted(set(selected_traits) - set(TRAITS))
    if unknown:
        raise ValueError(f"Unknown traits: {unknown}")
    folds = parse_int_list(args.folds)
    if any(fold not in range(1, 6) for fold in folds):
        raise ValueError("Folds must be selected from 1,2,3,4,5")
    repeats = parse_int_list(args.seed_repeats)
    if len(set(repeats)) != len(repeats) or any(repeat < 6 for repeat in repeats):
        raise ValueError("Seed repeats must be unique integers >= 6")
    devices = parse_int_list(args.devices)
    initial_gpu_state = gpu_state()
    missing = sorted(set(devices) - set(initial_gpu_state))
    if missing:
        raise RuntimeError(
            f"Requested GPUs are not visible: {missing}; visible={sorted(initial_gpu_state)}"
        )

    pending: list[dict[str, Any]] = []
    for trait in selected_traits:
        target, family = TRAITS[trait]
        manifest = args.quality_manifest.resolve() if family == "quality" else args.texture_manifest.resolve()
        for fold in folds:
            for repeat in repeats:
                run_dir = output_dir / "ai" / trait / f"fold_{fold}" / f"seed_repeat_{repeat}"
                run_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(trainer),
                    "--multimodal-dir",
                    str(project / "data" / "processed" / "multimodal"),
                    "--qc-ledger",
                    str(args.qc_ledger.resolve()),
                    "--output-dir",
                    str(run_dir),
                    "--target",
                    target,
                    "--cohort",
                    "primary",
                    "--exclude-cultivars",
                    "6.11",
                    "--repeat",
                    str(repeat),
                    "--outer-fold-manifest",
                    str(manifest),
                    "--outer-fold",
                    str(fold),
                    "--cultivar-anchor-calibration",
                    "--validation-residual-target-mode",
                    "observed",
                    "--crossfit-anchor-folds",
                    "4",
                    "--domain-aware-anchor-selection",
                    "--device",
                    "cuda:0",
                ]
                if family == "quality":
                    command.append("--allow-manifest-subset")
                pending.append(
                    {
                        "trait": trait,
                        "family": family,
                        "fold": fold,
                        "seed_repeat": repeat,
                        "run_dir": run_dir,
                        "command": command,
                    }
                )

    slots = [device for device in devices for _ in range(args.jobs_per_device)]
    running: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_at = time.time()
    while pending or running:
        available_slots = list(slots)
        for task in running:
            available_slots.remove(int(task["physical_device"]))
        while pending and available_slots:
            physical_device = available_slots.pop(0)
            task = pending.pop(0)
            run_dir = Path(task["run_dir"])
            if (run_dir / "metadata.json").is_file() and (run_dir / "predictions.parquet").is_file():
                completed.append(
                    {
                        "trait": task["trait"],
                        "fold": task["fold"],
                        "seed_repeat": task["seed_repeat"],
                        "status": "skipped_complete",
                    }
                )
                continue
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(physical_device)
            environment.setdefault("OMP_NUM_THREADS", "2")
            environment.setdefault("MKL_NUM_THREADS", "2")
            environment.setdefault("OPENBLAS_NUM_THREADS", "2")
            stdout_handle = (run_dir / "run.stdout.log").open("wb")
            stderr_handle = (run_dir / "run.stderr.log").open("wb")
            process = subprocess.Popen(
                task["command"],
                cwd=project,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            task.update(
                process=process,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                physical_device=physical_device,
                started_at=time.time(),
            )
            running.append(task)
            print(
                f"launched {task['trait']} fold {task['fold']} seed-repeat "
                f"{task['seed_repeat']} on physical GPU {physical_device} as PID {process.pid}",
                flush=True,
            )
        time.sleep(2)
        survivors: list[dict[str, Any]] = []
        for task in running:
            return_code = task["process"].poll()
            if return_code is None:
                survivors.append(task)
                continue
            task["stdout_handle"].close()
            task["stderr_handle"].close()
            row = {
                "trait": task["trait"],
                "family": task["family"],
                "fold": task["fold"],
                "seed_repeat": task["seed_repeat"],
                "physical_device": task["physical_device"],
                "return_code": int(return_code),
                "elapsed_seconds": time.time() - float(task["started_at"]),
                "run_dir": str(task["run_dir"]),
            }
            if return_code == 0:
                row["status"] = "completed"
                completed.append(row)
                print(
                    f"completed {task['trait']} fold {task['fold']} seed-repeat "
                    f"{task['seed_repeat']} in {row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            else:
                row["status"] = "failed"
                failures.append(row)
                print(f"FAILED: {json.dumps(row)}", flush=True)
        running = survivors

    manifest = {
        "protocol": (
            "V25 post hoc optimization-stability audit: two additional complete-pipeline seeds "
            "per frozen outer fold, with four-fold cross-fitted cultivar-aware PLS neural targets, "
            "domain-aware anchor hyperparameter selection and train-internal residual-gate selection."
        ),
        "traits": selected_traits,
        "folds": folds,
        "seed_repeats": repeats,
        "physical_devices": devices,
        "jobs_per_device": args.jobs_per_device,
        "initial_gpu_state": initial_gpu_state,
        "elapsed_seconds": time.time() - started_at,
        "completed_jobs": completed,
        "failed_jobs": failures,
        "test_labels_used_for_selection": False,
        "biological_replication_claimed": False,
        "production_results_overwritten": False,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

