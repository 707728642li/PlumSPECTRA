from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEVELOPMENT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"

# Prespecified factorial ablation.  Only the final row combines every proposed
# V4 ingredient; the preceding rows identify where any gain comes from.
CANDIDATES = [
    {
        "name": "basic_dilated_nomix",
        "channel_set": "basic",
        "architecture": "dilated",
        "mixstyle_p": 0.0,
        "purpose": "expanded-gate V3-like control",
    },
    {
        "name": "physics_dilated_nomix",
        "channel_set": "physics",
        "architecture": "dilated",
        "mixstyle_p": 0.0,
        "purpose": "physics-view front-end effect",
    },
    {
        "name": "basic_multiscale_nomix",
        "channel_set": "basic",
        "architecture": "multiscale",
        "mixstyle_p": 0.0,
        "purpose": "multiscale depthwise architecture effect",
    },
    {
        "name": "physics_multiscale_nomix",
        "channel_set": "physics",
        "architecture": "multiscale",
        "mixstyle_p": 0.0,
        "purpose": "physics-by-multiscale interaction",
    },
    {
        "name": "physics_multiscale_mixstyle",
        "channel_set": "physics",
        "architecture": "multiscale",
        "mixstyle_p": 0.5,
        "purpose": "complete PLUMRAC-PHY V4 candidate",
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
    common = {
        "trait": "skin_break_displacement_raw_mean",
        "cohort": "analysis",
        "heldout_cultivars": DEVELOPMENT_HELDOUT.split(","),
        "seed": 20260806,
        "width": 48,
        "blocks": 4,
        "dropout": 0.12,
        "batch_size": 256,
        "max_epochs": 32,
        "min_epochs": 6,
        "patience": 6,
        "learning_rate": 5e-4,
        "weight_decay": 2e-3,
        "sampler_power": 1.0,
        "profiles": "absolute,balanced",
        "gate": {
            "minimum_internal_improvement": 0.0025,
            "minimum_validation_cultivar_win_fraction": 0.60,
            "maximum_worst_validation_cultivar_degradation": 0.03,
            "maximum_residual_gate": 1.0,
        },
    }
    manifest = {
        "cycle": "PLUMRAC-PHY V4 bounded factorial ablation",
        "common": common,
        "candidates": CANDIDATES,
        "selection_boundary": (
            "Retrospective development only. Each target cultivar is excluded from all training, preprocessing, "
            "epoch/profile selection, and residual-gate selection in its LOCO fold."
        ),
    }
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    score_args = [
        str(project / "src" / "summarize_v4_retrospective.py"),
        "--variant",
        f"v2_safe={project / 'results' / 'v2' / 'plumrac_dev_rd_safe'}",
        "--variant",
        f"v3_frozen={project / 'results' / 'v3' / 'rd_optimization_suite' / 'gelu_group_64x4_domainbalanced'}",
    ]
    for candidate in CANDIDATES:
        candidate_dir = output_root / str(candidate["name"])
        score_args.extend(["--variant", f"{candidate['name']}={candidate_dir}"])
        command = [
            sys.executable,
            str(project / "src" / "train_plumrac_v4_phy.py"),
            "--channel-set",
            str(candidate["channel_set"]),
            "--architecture",
            str(candidate["architecture"]),
            "--mixstyle-p",
            str(candidate["mixstyle_p"]),
            "--mixstyle-alpha",
            "0.30",
            "--curvature-sd",
            "0.003",
            "--low-frequency-sd",
            "0.002",
            "--multimodal-dir",
            str(project / "data" / "processed" / "multimodal"),
            "--qc-ledger",
            str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
            "--output-dir",
            str(candidate_dir),
            "--target",
            str(common["trait"]),
            "--cohort",
            str(common["cohort"]),
            "--heldout",
            DEVELOPMENT_HELDOUT,
            "--seeds",
            str(common["seed"]),
            "--pls-results",
            str(project / "results" / "models" / "texture_pls_loco_analysis"),
            "--device",
            "cuda:0",
            "--profiles",
            str(common["profiles"]),
            "--width",
            str(common["width"]),
            "--blocks",
            str(common["blocks"]),
            "--dropout",
            str(common["dropout"]),
            "--batch-size",
            str(common["batch_size"]),
            "--max-epochs",
            str(common["max_epochs"]),
            "--min-epochs",
            str(common["min_epochs"]),
            "--patience",
            str(common["patience"]),
            "--learning-rate",
            str(common["learning_rate"]),
            "--weight-decay",
            str(common["weight_decay"]),
            "--sampler-power",
            str(common["sampler_power"]),
            "--validation-cultivars",
            "5",
            "--min-gate-improvement",
            str(common["gate"]["minimum_internal_improvement"]),
            "--min-gate-win-fraction",
            str(common["gate"]["minimum_validation_cultivar_win_fraction"]),
            "--max-gate-worst-degradation",
            str(common["gate"]["maximum_worst_validation_cultivar_degradation"]),
            "--max-residual-gate",
            str(common["gate"]["maximum_residual_gate"]),
        ]
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
