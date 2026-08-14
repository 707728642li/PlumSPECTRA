from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def query_free_gpus(requested: list[int]) -> list[int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    states = {}
    for line in result.stdout.strip().splitlines():
        index, memory_used, utilization = [int(value.strip()) for value in line.split(",")]
        states[index] = (memory_used, utilization)
    missing = sorted(set(requested) - set(states))
    if missing:
        raise RuntimeError(f"Requested GPUs not reported by nvidia-smi: {missing}")
    busy = {
        index: states[index]
        for index in requested
        if states[index][0] > 512 or states[index][1] > 10
    }
    if busy:
        raise RuntimeError(f"Requested GPUs are already occupied: {busy}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpus", default="0,1")
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs_per_gpu <= 2:
        raise ValueError("Production scheduler permits one or two jobs per GPU")
    requested_gpus = [int(value.strip()) for value in args.physical_gpus.split(",") if value.strip()]
    free_gpus = query_free_gpus(requested_gpus)
    plan = pd.read_csv(args.plan)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_plumrac_v5_full.py")
    pending = []
    for row in plan.itertuples(index=False):
        trait_dir = output_dir / str(row.trait)
        trait_dir.mkdir(parents=True, exist_ok=True)
        pending.append((row, trait_dir))

    capacity = {gpu: args.jobs_per_gpu for gpu in free_gpus}
    running = []
    while pending or running:
        for gpu in free_gpus:
            used = sum(1 for item in running if item[0] == gpu)
            while pending and used < capacity[gpu]:
                row, trait_dir = pending.pop(0)
                command = [
                    sys.executable,
                    str(trainer),
                    "--multimodal-dir",
                    "data/processed/multimodal",
                    "--qc-ledger",
                    "data/processed/texture_qc/texture_qc_ledger.parquet",
                    "--output-dir",
                    str(trait_dir),
                    "--target",
                    str(row.target),
                    "--cohort",
                    "primary",
                    "--exclude-cultivars",
                    "6.11",
                    "--fixed-epochs",
                    str(int(row.fixed_epochs)),
                    "--fixed-gate",
                    str(float(row.fixed_gate)),
                    "--seed",
                    str(int(row.production_seed)),
                    "--device",
                    "cuda:0",
                    "--physical-gpu-index",
                    str(gpu),
                ]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                stdout_handle = (trait_dir / "run.stdout.log").open("wb")
                stderr_handle = (trait_dir / "run.stderr.log").open("wb")
                process = subprocess.Popen(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=environment,
                )
                running.append(
                    (gpu, str(row.trait), process, stdout_handle, stderr_handle, time.time())
                )
                used += 1
                print(
                    f"launched {row.trait} on physical GPU{gpu} as PID {process.pid}",
                    flush=True,
                )
        time.sleep(2)
        survivors = []
        for gpu, trait, process, stdout_handle, stderr_handle, started in running:
            code = process.poll()
            if code is None:
                survivors.append((gpu, trait, process, stdout_handle, stderr_handle, started))
                continue
            stdout_handle.close()
            stderr_handle.close()
            if code != 0:
                raise RuntimeError(f"Production model {trait} failed on GPU{gpu} with exit code {code}")
            print(
                f"completed {trait} on physical GPU{gpu} in {time.time() - started:.1f}s",
                flush=True,
            )
        running = survivors

    rows = []
    for row in plan.itertuples(index=False):
        bundle_path = output_dir / str(row.trait) / "deployment_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "trait": row.trait,
                "target": row.target,
                "training_samples": bundle["training_samples"],
                "cultivars": len(bundle["training_cultivars"]),
                "fixed_epochs": bundle["fixed_epochs"],
                "fixed_gate": bundle["fixed_gate"],
                "pls_preprocessing": bundle["pls_anchor"]["preprocessing"],
                "pls_components": bundle["pls_anchor"]["n_components"],
                "trainable_parameters": bundle["trainable_parameters"],
                "physical_gpu_index": bundle["physical_gpu_index"],
                "bundle": str(bundle_path),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "production_models.csv", index=False)
    manifest = {
        "models": int(len(table)),
        "one_model_per_trait": True,
        "training_samples_per_model": sorted(table["training_samples"].unique().tolist()),
        "cultivars_per_model": sorted(table["cultivars"].unique().tolist()),
        "physical_gpus": free_gpus,
        "jobs_per_gpu": args.jobs_per_gpu,
        "scheduler": "CUDA_VISIBLE_DEVICES isolates each native-Windows PyTorch worker to one physical GPU",
        "claim_boundary": "Production fits use all eligible labels; validation claims come from archived holdouts.",
    }
    (output_dir / "production_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
