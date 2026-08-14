from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


TRAITS = {
    "FW": "fruit_weight_g",
    "SSC": "soluble_solids_pct",
    "pH": "ph",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument(
        "--qc-ledger",
        type=Path,
        default=Path("data/processed/texture_qc/texture_qc_ledger.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traits", default="all")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--jobs-per-device", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--domain-aware-anchor-selection",
        dest="domain_aware_anchor_selection",
        action="store_true",
        help="Select the PLS anchor using cultivar-calibrated inner-CV loss (V25 default).",
    )
    parser.add_argument(
        "--legacy-global-anchor-selection",
        dest="domain_aware_anchor_selection",
        action="store_false",
        help="Reproduce the historical global-loss anchor selection for an explicit diagnostic only.",
    )
    parser.set_defaults(domain_aware_anchor_selection=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    trainer = Path(__file__).with_name("train_plumrac_v5_stratified.py")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_traits = (
        list(TRAITS)
        if args.traits == "all"
        else [value.strip() for value in args.traits.split(",") if value.strip()]
    )
    unknown = sorted(set(selected_traits) - set(TRAITS))
    if unknown:
        raise ValueError(f"Unknown traits: {unknown}")
    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    if not folds or any(fold not in range(1, 6) for fold in folds):
        raise ValueError("Folds must be selected from 1,2,3,4,5")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("At least one CUDA device must be supplied")

    pending: list[dict[str, object]] = []
    for trait in selected_traits:
        target = TRAITS[trait]
        for fold in folds:
            run_dir = output_dir / trait / f"fold_{fold}"
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
                str(fold),
                "--outer-fold-manifest",
                str(args.fold_manifest.resolve()),
                "--outer-fold",
                str(fold),
                "--allow-manifest-subset",
                "--cultivar-anchor-calibration",
                "--validation-residual-target-mode",
                "observed",
                "--crossfit-anchor-folds",
                "4",
                "--device",
                "cuda:0",
            ]
            if args.domain_aware_anchor_selection:
                command.append("--domain-aware-anchor-selection")
            pending.append({"trait": trait, "fold": fold, "command": command, "run_dir": run_dir})

    slots = [device for device in devices for _ in range(args.jobs_per_device)]
    running: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    while pending or running:
        used_slots = [str(task["physical_device"]) for task in running]
        available_slots = list(slots)
        for used in used_slots:
            available_slots.remove(used)
        while pending and available_slots:
            physical_device = available_slots.pop(0)
            task = pending.pop(0)
            run_dir = Path(task["run_dir"])
            if (run_dir / "metadata.json").exists() and (run_dir / "predictions.parquet").exists():
                completed.append({"trait": task["trait"], "fold": task["fold"], "status": "skipped"})
                print(f"skipped {task['trait']} fold {task['fold']}", flush=True)
                continue
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = physical_device
            environment.setdefault("OMP_NUM_THREADS", "2")
            environment.setdefault("MKL_NUM_THREADS", "2")
            environment.setdefault("OPENBLAS_NUM_THREADS", "2")
            stdout_handle = (run_dir / "run.stdout.log").open("wb")
            stderr_handle = (run_dir / "run.stderr.log").open("wb")
            process = subprocess.Popen(
                task["command"], cwd=project, env=environment,
                stdout=stdout_handle, stderr=stderr_handle,
            )
            task.update(
                process=process,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                started_at=time.time(),
                physical_device=physical_device,
            )
            running.append(task)
            print(
                f"launched {task['trait']} fold {task['fold']} on physical GPU {physical_device} as PID {process.pid}",
                flush=True,
            )
        time.sleep(2)
        survivors: list[dict[str, object]] = []
        for task in running:
            process = task["process"]
            return_code = process.poll()
            if return_code is None:
                survivors.append(task)
                continue
            task["stdout_handle"].close()
            task["stderr_handle"].close()
            elapsed = time.time() - float(task["started_at"])
            if return_code != 0:
                raise RuntimeError(
                    f"{task['trait']} fold {task['fold']} failed with code {return_code}; "
                    f"see {Path(task['run_dir']) / 'run.stderr.log'}"
                )
            completed.append(
                {
                    "trait": task["trait"], "fold": task["fold"],
                    "physical_device": task["physical_device"],
                    "status": "completed", "elapsed_seconds": elapsed,
                }
            )
            print(f"completed {task['trait']} fold {task['fold']} in {elapsed:.1f}s", flush=True)
        running = survivors

    manifest = {
        "protocol": (
            "V25 corrected-cohort cross-fitted domain-aware-anchor conventional-quality audit"
            if args.domain_aware_anchor_selection
            else "V22-compatible frozen non-overlapping conventional-quality audit"
        ),
        "fold_manifest": str(args.fold_manifest.resolve()),
        "qc_ledger": str(args.qc_ledger.resolve()),
        "physical_cuda_devices": devices,
        "jobs_per_device": args.jobs_per_device,
        "domain_aware_anchor_selection": bool(args.domain_aware_anchor_selection),
        "jobs": completed,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
