from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONFIRMATION_HELDOUT = (
    "6.11,A181,Fengtangli,Fengwei Huanghou,Furongli,L31,Naili,"
    "Qingcuili,Weidi,Weijin,Zaoshu Konglongdan"
)
SEEDS = "20260806,20260807,20260808,20260809"


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    frozen_config = project / "configs" / "v3_plumrac_x_frozen.json"
    config = json.loads(frozen_config.read_text(encoding="utf-8"))
    if config["claim_boundary"] != (
        "No architecture, training-hyperparameter, or RD routing change is permitted after confirmation evaluation begins."
    ):
        raise RuntimeError("Frozen confirmation boundary is missing or changed")

    command = [
        sys.executable,
        str(project / "src" / "train_plumrac_v3_variant.py"),
        "--activation", "gelu",
        "--normalization", "group",
        "--channel-set", "basic",
        "--anchor-policy", "plsr",
        "--multimodal-dir", str(project / "data" / "processed" / "multimodal"),
        "--qc-ledger", str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
        "--output-dir", str(output_dir),
        "--target", "skin_break_displacement_raw_mean",
        "--cohort", "analysis",
        "--heldout", CONFIRMATION_HELDOUT,
        "--seeds", SEEDS,
        "--pls-results", str(project / "results" / "models" / "texture_pls_loco_analysis"),
        "--device", "cuda:0",
        "--profiles", "absolute,balanced,ranking",
        "--width", "64",
        "--blocks", "4",
        "--dropout", "0.12",
        "--batch-size", "256",
        "--max-epochs", "40",
        "--min-epochs", "6",
        "--patience", "8",
        "--learning-rate", "0.0005",
        "--weight-decay", "0.002",
        "--sampler-power", "1.0",
        "--validation-cultivars", "5",
        "--min-gate-improvement", "0.01",
        "--min-gate-win-fraction", "1.00",
        "--max-gate-worst-degradation", "0.00",
        "--max-residual-gate", "0.50",
    ]
    if args.dry_run:
        print(subprocess.list2cmdline(command))
        return
    snapshot = gpu_snapshot()
    if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
        raise RuntimeError(f"GPU0 is not free enough to start frozen confirmation: {snapshot}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    (output_dir / "frozen_config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"starting frozen RD confirmation with GPU snapshot {snapshot}", flush=True)
    subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
