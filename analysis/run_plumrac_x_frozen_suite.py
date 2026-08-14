from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from v2_registry import trait_abbreviation_map


DEVELOPMENT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"


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
    trait_map = {abbreviation: target for target, abbreviation in trait_abbreviation_map().items()}
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--traits", required=True, help="Comma-separated frozen trait abbreviations")
    parser.add_argument("--heldout", default="all", help="all, development, or trainer-compatible cultivar list")
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    frozen_config = project / "configs" / "v3_plumrac_x_frozen.json"
    config = json.loads(frozen_config.read_text(encoding="utf-8"))
    if config["freeze_version"] != "3.0.0" or config["model_family"] != "PLUMRAC-X":
        raise RuntimeError("Unexpected frozen model configuration")
    traits = [item.strip().upper() for item in args.traits.split(",") if item.strip()]
    unknown = sorted(set(traits) - set(trait_map))
    if unknown:
        raise ValueError(f"Unknown trait abbreviations: {unknown}")
    heldout = DEVELOPMENT_HELDOUT if args.heldout.lower() == "development" else args.heldout
    output_root = args.output_root.resolve()
    manifest = {
        "model_family": "PLUMRAC-X",
        "frozen_config": str(frozen_config.resolve()),
        "traits": traits,
        "targets": {trait: trait_map[trait] for trait in traits},
        "heldout": heldout,
        "seeds": [int(item) for item in args.seeds.split(",")],
        "purpose": "Frozen cross-trait transfer; no trait-specific architecture search",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for trait in traits:
        output_dir = output_root / trait
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
            "--target", trait_map[trait],
            "--cohort", "analysis",
            "--heldout", heldout,
            "--seeds", args.seeds,
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
            continue
        snapshot = gpu_snapshot()
        if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
            raise RuntimeError(f"GPU0 is not free enough to start {trait}: {snapshot}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"starting PLUMRAC-X / {trait} with GPU snapshot {snapshot}", flush=True)
        subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
