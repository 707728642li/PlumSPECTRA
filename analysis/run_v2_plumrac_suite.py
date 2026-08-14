from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from v2_registry import trait_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def gpu_compute_processes() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed before GPU task:\n{result.stdout}\n{result.stderr}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "v2" / "plumrac_production")
    parser.add_argument("--traits", default="all", help="Comma-separated abbreviations; default is all nine endpoints.")
    parser.add_argument("--seeds", default="20260806")
    parser.add_argument("--profiles", default="absolute,balanced,ranking")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    args = parser.parse_args()

    registry = trait_registry()
    registry = registry.loc[registry["model_family"] == "endpoint"].copy()
    if args.traits != "all":
        requested = [value.strip().upper() for value in args.traits.split(",")]
        missing = sorted(set(requested) - set(registry["abbreviation"]))
        if missing:
            raise ValueError(f"Unknown endpoint abbreviations: {missing}")
        registry = registry.set_index("abbreviation").loc[requested].reset_index()
    else:
        priority_order = {"primary": 0, "secondary": 1}
        registry["priority_order"] = registry["priority"].map(priority_order)
        registry = registry.sort_values(["priority_order", "abbreviation"])

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "suite_progress.json"
    progress: dict[str, object] = {
        "started_at": datetime.now().isoformat(),
        "status": "running",
        "seeds": args.seeds,
        "profiles": args.profiles,
        "traits": registry["abbreviation"].tolist(),
        "completed": [],
    }
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for row in registry.itertuples(index=False):
        abbreviation = str(row.abbreviation)
        target = str(row.target)
        target_output = output_root / abbreviation
        summary_path = target_output / "summary.json"
        if summary_path.exists():
            progress["completed"].append(abbreviation)
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            continue

        occupied = gpu_compute_processes()
        if occupied:
            progress["status"] = "stopped_gpu_occupied"
            progress["blocked_before"] = abbreviation
            progress["gpu_processes"] = occupied
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            raise RuntimeError(f"GPU0 compute process detected before {abbreviation}; refusing to start: {occupied}")

        command = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "train_plumrac_loco.py"),
            "--multimodal-dir",
            str(PROJECT_ROOT / "data" / "processed" / "multimodal"),
            "--qc-ledger",
            str(PROJECT_ROOT / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
            "--output-dir",
            str(target_output),
            "--target",
            target,
            "--cohort",
            "analysis",
            "--heldout",
            "all",
            "--seeds",
            args.seeds,
            "--pls-results",
            str(PROJECT_ROOT / "results" / "models" / "texture_pls_loco_analysis"),
            "--device",
            "cuda:0",
            "--profiles",
            args.profiles,
            "--batch-size",
            str(args.batch_size),
            "--max-epochs",
            str(args.max_epochs),
            "--min-epochs",
            str(args.min_epochs),
            "--patience",
            str(args.patience),
            "--learning-rate",
            str(args.learning_rate),
            "--validation-cultivars",
            "5",
            "--min-gate-improvement",
            "0.01",
            "--min-gate-win-fraction",
            "1.00",
            "--max-gate-worst-degradation",
            "0.00",
            "--max-residual-gate",
            "0.50",
        ]
        target_output.mkdir(parents=True, exist_ok=True)
        with (target_output / "stdout.log").open("a", encoding="utf-8") as stdout, (
            target_output / "stderr.log"
        ).open("a", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        if completed.returncode != 0 or not summary_path.exists():
            progress["status"] = "failed"
            progress["failed_trait"] = abbreviation
            progress["returncode"] = completed.returncode
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            raise RuntimeError(f"PlumRAC-Net production task failed for {abbreviation}; inspect {target_output}")
        progress["completed"].append(abbreviation)
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    progress["status"] = "complete"
    progress["completed_at"] = datetime.now().isoformat()
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
