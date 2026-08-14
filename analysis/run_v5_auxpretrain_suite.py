from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEVELOPMENT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"
CANDIDATES = [
    {
        "name": "control_no_pretrain",
        "trainer": "train_plumrac_v4_phy.py",
        "auxiliary_set": None,
        "pretrain_epochs": 0,
    },
    {
        "name": "texture9_e6",
        "trainer": "train_plumrac_v5_auxpretrain.py",
        "auxiliary_set": "texture9",
        "pretrain_epochs": 6,
    },
    {
        "name": "quality12_e6",
        "trainer": "train_plumrac_v5_auxpretrain.py",
        "auxiliary_set": "quality12",
        "pretrain_epochs": 6,
    },
    {
        "name": "quality12_e12",
        "trainer": "train_plumrac_v5_auxpretrain.py",
        "auxiliary_set": "quality12",
        "pretrain_epochs": 12,
    },
]


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
    manifest = {
        "cycle": "V5 source-only auxiliary-supervision ablation",
        "trait": "skin_break_displacement_raw_mean",
        "heldout_cultivars": DEVELOPMENT_HELDOUT.split(","),
        "excluded_cultivars": ["6.11"],
        "seed": 20260806,
        "candidates": CANDIDATES,
        "fixed_final_architecture": "basic three-view multiscale width48 blocks4",
        "selection_boundary": (
            "Historical development folds only. Auxiliary labels and normalization statistics are restricted to "
            "source training cultivars in every nested stage; every final model has one RD output."
        ),
    }
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    score_args = [str(project / "src" / "summarize_v4_retrospective.py")]
    for candidate in CANDIDATES:
        candidate_dir = output_root / str(candidate["name"])
        score_args.extend(["--variant", f"{candidate['name']}={candidate_dir}"])
        command = [sys.executable, str(project / "src" / str(candidate["trainer"]))]
        if candidate["auxiliary_set"] is None:
            command.extend(
                [
                    "--channel-set",
                    "basic",
                    "--architecture",
                    "multiscale",
                    "--mixstyle-p",
                    "0.0",
                ]
            )
        else:
            command.extend(
                [
                    "--auxiliary-set",
                    str(candidate["auxiliary_set"]),
                    "--pretrain-epochs",
                    str(candidate["pretrain_epochs"]),
                    "--pretrain-learning-rate",
                    "0.001",
                    "--pretrain-weight-decay",
                    "0.001",
                ]
            )
        command.extend(
            [
                "--multimodal-dir",
                str(project / "data" / "processed" / "multimodal"),
                "--qc-ledger",
                str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
                "--output-dir",
                str(candidate_dir),
                "--target",
                "skin_break_displacement_raw_mean",
                "--cohort",
                "analysis",
                "--exclude-cultivars",
                "6.11",
                "--heldout",
                DEVELOPMENT_HELDOUT,
                "--seeds",
                "20260806",
                "--pls-results",
                str(project / "results" / "v4" / "plsr_rd_excluding_611"),
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
        )
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        snapshot = gpu_snapshot()
        if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
            raise RuntimeError(f"GPU0 is not free enough to start {candidate['name']}: {snapshot}")
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"starting {candidate['name']} with GPU snapshot {snapshot}", flush=True)
        subprocess.run(command, cwd=project, check=True)

    if not args.dry_run:
        score_args.extend(["--output-dir", str(output_root / "selection")])
        subprocess.run([sys.executable, *score_args], cwd=project, check=True)


if __name__ == "__main__":
    main()
