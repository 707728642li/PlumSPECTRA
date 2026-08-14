from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from v2_registry import trait_abbreviation_map


DEVELOPMENT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"

VARIANTS = [
    {"name": "silu_group_32x3", "activation": "silu", "normalization": "group", "channel_set": "basic", "width": 32, "blocks": 3, "sampler_power": 0.5},
    {"name": "relu_group_32x3", "activation": "relu", "normalization": "group", "channel_set": "basic", "width": 32, "blocks": 3, "sampler_power": 0.5},
    {"name": "gelu_layer_32x3", "activation": "gelu", "normalization": "layer", "channel_set": "basic", "width": 32, "blocks": 3, "sampler_power": 0.5},
    {"name": "silu_layer_32x3", "activation": "silu", "normalization": "layer", "channel_set": "basic", "width": 32, "blocks": 3, "sampler_power": 0.5},
    {"name": "gelu_group_16x2", "activation": "gelu", "normalization": "group", "channel_set": "basic", "width": 16, "blocks": 2, "sampler_power": 0.5},
    {"name": "gelu_group_64x4", "activation": "gelu", "normalization": "group", "channel_set": "basic", "width": 64, "blocks": 4, "sampler_power": 0.5},
    {"name": "gelu_group_32x3_domainbalanced", "activation": "gelu", "normalization": "group", "channel_set": "basic", "width": 32, "blocks": 3, "sampler_power": 1.0},
    {"name": "gelu_group_32x3_multiview", "activation": "gelu", "normalization": "group", "channel_set": "multiview", "width": 32, "blocks": 3, "sampler_power": 0.5},
]


def gpu_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "-i",
        "0",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 5:
        raise RuntimeError(f"Unexpected nvidia-smi output: {result.stdout!r}")
    process_result = subprocess.run(
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
    processes = [line.strip() for line in process_result.stdout.splitlines() if line.strip()]
    return {
        "index": int(values[0]),
        "name": values[1],
        "memory_used_mib": int(values[2]),
        "memory_total_mib": int(values[3]),
        "utilization_pct": int(values[4]),
        "compute_processes": processes,
    }


def main() -> None:
    trait_by_abbreviation = {abbreviation: target for target, abbreviation in trait_abbreviation_map().items()}
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--trait", default="RD", choices=sorted(trait_by_abbreviation))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = trait_by_abbreviation[args.trait]
    manifest = {
        "cycle": "V3 operator and capacity development",
        "trait": args.trait,
        "target": target,
        "heldout_cultivars": DEVELOPMENT_HELDOUT.split(","),
        "baseline": str(args.baseline.resolve()),
        "variants": VARIANTS,
        "selection_boundary": "Five prespecified development cultivars; remaining eleven cultivars not evaluated here.",
    }
    (output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    score_args = [
        str(project / "src" / "summarize_v3_development.py"),
        "--variant",
        f"v2_safe={args.baseline.resolve()}",
    ]
    for variant in VARIANTS:
        variant_dir = output_root / variant["name"]
        score_args.extend(["--variant", f"{variant['name']}={variant_dir}"])
        command = [
            sys.executable,
            str(project / "src" / "train_plumrac_v3_variant.py"),
            "--activation",
            str(variant["activation"]),
            "--normalization",
            str(variant["normalization"]),
            "--channel-set",
            str(variant["channel_set"]),
            "--multimodal-dir",
            str(project / "data" / "processed" / "multimodal"),
            "--qc-ledger",
            str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
            "--output-dir",
            str(variant_dir),
            "--target",
            target,
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
            str(variant["width"]),
            "--blocks",
            str(variant["blocks"]),
            "--dropout",
            "0.12",
            "--batch-size",
            "256",
            "--max-epochs",
            "40",
            "--min-epochs",
            "6",
            "--patience",
            "8",
            "--learning-rate",
            "0.0005",
            "--sampler-power",
            str(variant["sampler_power"]),
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
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        snapshot = gpu_snapshot()
        if int(snapshot["memory_used_mib"]) > 2_000 or snapshot["compute_processes"]:
            raise RuntimeError(f"GPU0 is not free enough to start {variant['name']}: {snapshot}")
        (variant_dir / "gpu_before_start.json").parent.mkdir(parents=True, exist_ok=True)
        (variant_dir / "gpu_before_start.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"starting {variant['name']} with GPU snapshot {snapshot}", flush=True)
        subprocess.run(command, cwd=project, check=True)

    if not args.dry_run:
        score_args.extend(["--output-dir", str(output_root / "selection")])
        subprocess.run([sys.executable, *score_args], cwd=project, check=True)


if __name__ == "__main__":
    main()
