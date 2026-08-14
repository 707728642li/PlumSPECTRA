from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEVELOPMENT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"

# This is a deliberately small, prespecified neighbourhood around the winning
# V3 operator/capacity candidate.  It is not an open-ended test-set search.
CANDIDATES = [
    {
        "name": "silu_group_32x3_currenthash",
        "activation": "silu",
        "width": 32,
        "blocks": 3,
        "dropout": 0.12,
        "learning_rate": 5e-4,
        "weight_decay": 2e-3,
        "sampler_power": 0.5,
        "augmentation": True,
        "purpose": "provenance reproduction after wrapper hash changed",
    },
    {
        "name": "gelu_group_64x4_lr025",
        "activation": "gelu",
        "width": 64,
        "blocks": 4,
        "dropout": 0.12,
        "learning_rate": 2.5e-4,
        "weight_decay": 2e-3,
        "sampler_power": 0.5,
        "augmentation": True,
        "purpose": "lower learning-rate neighbour",
    },
    {
        "name": "gelu_group_64x4_lr100",
        "activation": "gelu",
        "width": 64,
        "blocks": 4,
        "dropout": 0.12,
        "learning_rate": 1e-3,
        "weight_decay": 2e-3,
        "sampler_power": 0.5,
        "augmentation": True,
        "purpose": "higher learning-rate neighbour",
    },
    {
        "name": "gelu_group_64x4_drop005",
        "activation": "gelu",
        "width": 64,
        "blocks": 4,
        "dropout": 0.05,
        "learning_rate": 5e-4,
        "weight_decay": 2e-3,
        "sampler_power": 0.5,
        "augmentation": True,
        "purpose": "lower dropout neighbour",
    },
    {
        "name": "gelu_group_64x4_noaug",
        "activation": "gelu",
        "width": 64,
        "blocks": 4,
        "dropout": 0.12,
        "learning_rate": 5e-4,
        "weight_decay": 2e-3,
        "sampler_power": 0.5,
        "augmentation": False,
        "purpose": "physical spectral augmentation ablation",
    },
    {
        "name": "gelu_group_64x4_domainbalanced",
        "activation": "gelu",
        "width": 64,
        "blocks": 4,
        "dropout": 0.12,
        "learning_rate": 5e-4,
        "weight_decay": 2e-3,
        "sampler_power": 1.0,
        "augmentation": True,
        "purpose": "capacity-by-domain-balancing interaction",
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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--operator-winner", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cycle": "V3 bounded training-hyperparameter optimization",
        "trait": "RD",
        "heldout_cultivars": DEVELOPMENT_HELDOUT.split(","),
        "baseline": str(args.baseline.resolve()),
        "operator_winner": str(args.operator_winner.resolve()),
        "candidates": CANDIDATES,
        "selection_boundary": "Five prespecified development cultivars only; eleven confirmation cultivars remain sealed.",
    }
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    score_args = [
        str(project / "src" / "summarize_v3_development.py"),
        "--variant",
        f"v2_safe={args.baseline.resolve()}",
        "--variant",
        f"operator_winner={args.operator_winner.resolve()}",
    ]
    for candidate in CANDIDATES:
        candidate_dir = output_root / str(candidate["name"])
        score_args.extend(["--variant", f"{candidate['name']}={candidate_dir}"])
        command = [
            sys.executable,
            str(project / "src" / "train_plumrac_v3_variant.py"),
            "--activation",
            str(candidate["activation"]),
            "--normalization",
            "group",
            "--channel-set",
            "basic",
            "--anchor-policy",
            "plsr",
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
            "--heldout",
            DEVELOPMENT_HELDOUT,
            "--seeds",
            "20260806",
            "--pls-results",
            str(project / "results" / "models" / "texture_pls_loco_analysis"),
            "--device",
            "cuda:0",
            "--profiles",
            "absolute,balanced,ranking",
            "--width",
            str(candidate["width"]),
            "--blocks",
            str(candidate["blocks"]),
            "--dropout",
            str(candidate["dropout"]),
            "--batch-size",
            "256",
            "--max-epochs",
            "40",
            "--min-epochs",
            "6",
            "--patience",
            "8",
            "--learning-rate",
            str(candidate["learning_rate"]),
            "--weight-decay",
            str(candidate["weight_decay"]),
            "--sampler-power",
            str(candidate["sampler_power"]),
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
        if not candidate["augmentation"]:
            command.append("--no-augmentation")
        if args.dry_run:
            print(subprocess.list2cmdline(command))
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
