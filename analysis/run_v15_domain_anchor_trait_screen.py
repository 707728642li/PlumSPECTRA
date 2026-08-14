from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import train_plumrac_loco as v2
from v2_registry import abbreviated_trait


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--targets",
        default=",".join(v2.DEFAULT_TARGETS),
        help="Comma-separated texture targets; every target is trained independently.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_parallel <= 2:
        raise ValueError("The single-GPU queue is capped at two concurrent models")
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    unknown = sorted(set(targets) - set(v2.DEFAULT_TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_plumrac_v5_stratified.py")
    pending = []
    for target in targets:
        trait = abbreviated_trait(target)
        trait_dir = output_dir / trait
        trait_dir.mkdir(parents=True, exist_ok=True)
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
            target,
            "--cohort",
            "primary",
            "--exclude-cultivars",
            "6.11",
            "--repeat",
            str(args.repeat),
            "--cultivar-anchor-calibration",
            "--validation-residual-target-mode",
            "legacy_zero",
        ]
        pending.append((trait, target, command, trait_dir / "run.stdout.log", trait_dir / "run.stderr.log"))

    running = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            trait, target, command, stdout_path, stderr_path = pending.pop(0)
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
            running.append((trait, target, process, stdout_handle, stderr_handle))
            print(f"launched {trait} as PID {process.pid}", flush=True)
        time.sleep(2)
        survivors = []
        for trait, target, process, stdout_handle, stderr_handle in running:
            code = process.poll()
            if code is None:
                survivors.append((trait, target, process, stdout_handle, stderr_handle))
                continue
            stdout_handle.close()
            stderr_handle.close()
            if code != 0:
                raise RuntimeError(f"{trait} failed with exit code {code}")
            print(f"completed {trait}", flush=True)
        running = survivors

    rows = []
    for target in targets:
        trait = abbreviated_trait(target)
        metadata = json.loads((output_dir / trait / "metadata.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "target": target,
                "trait": trait,
                "repeat": int(metadata["repeat"]),
                "selected_epoch": int(metadata["selected_epoch"]),
                "selected_gate": float(metadata["selected_gate"]),
                "ai_rmse": float(metadata["ai_metrics"]["rmse"]),
                "domain_pls_rmse": float(metadata["pls_anchor_metrics"]["rmse"]),
                "global_pls_rmse": float(metadata["global_pls_anchor_metrics"]["rmse"]),
                "ai_vs_domain_pls_pct": float(metadata["relative_rmse_improvement_pct"]),
                "ai_vs_global_pls_pct": float(
                    metadata["relative_rmse_improvement_vs_global_pls_pct"]
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("ai_vs_global_pls_pct", ascending=False)
    table.to_csv(output_dir / "trait_screen.csv", index=False)
    report = {
        "scope": "independent single-output domain-anchored V5 model per texture trait",
        "validation": f"development screen on stratified holdout repeat {args.repeat}",
        "cohort": "primary",
        "model_independent_excluded_cultivars": ["6.11"],
        "max_parallel_gpu_jobs": args.max_parallel,
        "targets": targets,
        "selection_rule_for_confirmation": (
            "positive improvement over domain PLS and at least 10% improvement over ordinary global PLS"
        ),
        "confirmation_candidates": table.loc[
            (table["ai_vs_domain_pls_pct"] > 0) & (table["ai_vs_global_pls_pct"] >= 10),
            "trait",
        ].tolist(),
    }
    (output_dir / "screen_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
