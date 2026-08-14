from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from train_texture_pls_loco import DEFAULT_TARGETS
from v2_registry import abbreviated_trait


DEVELOPMENT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"
RD_TARGET = "skin_break_displacement_raw_mean"


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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target_paths: dict[str, Path] = {
        "RD": project / "results" / "v5" / "auxpretrain_suite" / "quality12_e12"
    }
    targets_to_run = [target for target in DEFAULT_TARGETS if target != RD_TARGET]
    manifest = {
        "cycle": "V5.1 independent texture-trait development screen",
        "targets": DEFAULT_TARGETS,
        "targets_reused": {"RD": str(target_paths["RD"])},
        "targets_to_run": targets_to_run,
        "heldout_cultivars": DEVELOPMENT_HELDOUT.split(","),
        "excluded_cultivars": ["6.11"],
        "one_final_model_per_trait": True,
        "seed": 20260806,
        "frozen_architecture": "quality12 auxiliary pretraining e12 + basic multiscale encoder",
    }
    (output_root / "screen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for target in targets_to_run:
        abbreviation = abbreviated_trait(target)
        target_dir = output_root / abbreviation
        target_paths[abbreviation] = target_dir
        command = [
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
            str(target_dir),
            "--target",
            target,
            "--cohort",
            "analysis",
            "--exclude-cultivars",
            "6.11",
            "--heldout",
            DEVELOPMENT_HELDOUT,
            "--seeds",
            "20260806",
            "--pls-results",
            str(project / "results" / "v5" / "plsr_all_traits_excluding_611"),
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
            print(subprocess.list2cmdline(command), flush=True)
            continue
        snapshot = gpu_snapshot()
        if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
            raise RuntimeError(f"GPU0 is not free enough to start {abbreviation}: {snapshot}")
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"starting trait {abbreviation} with GPU snapshot {snapshot}", flush=True)
        subprocess.run(command, cwd=project, check=True)

    if not args.dry_run:
        command = [sys.executable, str(project / "src" / "summarize_v5_trait_screen.py")]
        for abbreviation, path in sorted(target_paths.items()):
            command.extend(["--run", f"{abbreviation}={path}"])
        command.extend(["--output-dir", str(output_root / "selection")])
        subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
