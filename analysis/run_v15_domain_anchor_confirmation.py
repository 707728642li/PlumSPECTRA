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
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--confirmation-repeats", default="2,3,4,5")
    args = parser.parse_args()
    if not 1 <= args.max_parallel <= 2:
        raise ValueError("The single-GPU queue is capped at two concurrent models")
    screen_dir = args.screen_dir.resolve()
    screen = pd.read_csv(screen_dir / "trait_screen.csv")
    selected = screen.loc[
        (screen["ai_vs_domain_pls_pct"] > 0) & (screen["ai_vs_global_pls_pct"] >= 10)
    ].copy()
    targets = selected["target"].tolist()
    repeats = [int(value.strip()) for value in args.confirmation_repeats.split(",") if value.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_plumrac_v5_stratified.py")

    pending = []
    for target in targets:
        trait = abbreviated_trait(target)
        for repeat in repeats:
            run_dir = output_dir / trait / f"repeat_{repeat}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(trainer),
                "--multimodal-dir",
                "data/processed/multimodal",
                "--qc-ledger",
                "data/processed/texture_qc/texture_qc_ledger.parquet",
                "--output-dir",
                str(run_dir),
                "--target",
                target,
                "--cohort",
                "primary",
                "--exclude-cultivars",
                "6.11",
                "--repeat",
                str(repeat),
                "--cultivar-anchor-calibration",
                "--validation-residual-target-mode",
                "legacy_zero",
            ]
            pending.append(
                (trait, repeat, command, run_dir / "run.stdout.log", run_dir / "run.stderr.log")
            )

    running = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            trait, repeat, command, stdout_path, stderr_path = pending.pop(0)
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
            running.append((trait, repeat, process, stdout_handle, stderr_handle))
            print(f"launched {trait} repeat {repeat} as PID {process.pid}", flush=True)
        time.sleep(2)
        survivors = []
        for trait, repeat, process, stdout_handle, stderr_handle in running:
            code = process.poll()
            if code is None:
                survivors.append((trait, repeat, process, stdout_handle, stderr_handle))
                continue
            stdout_handle.close()
            stderr_handle.close()
            if code != 0:
                raise RuntimeError(f"{trait} repeat {repeat} failed with exit code {code}")
            print(f"completed {trait} repeat {repeat}", flush=True)
        running = survivors

    summary_rows = []
    for target in targets:
        trait = abbreviated_trait(target)
        frames = [pd.read_parquet(screen_dir / trait / "predictions.parquet")]
        frames.extend(
            pd.read_parquet(output_dir / trait / f"repeat_{repeat}" / "predictions.parquet")
            for repeat in repeats
        )
        predictions = pd.concat(frames, ignore_index=True)
        trait_dir = output_dir / trait
        predictions.to_parquet(trait_dir / "predictions.parquet", index=False, compression="zstd")
        repeat_rows = []
        for repeat, group in predictions.groupby("repeat", observed=True):
            ai = v2.regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
            domain = v2.regression_metrics(
                group["y_true"].to_numpy(), group["y_pls_anchor"].to_numpy()
            )
            global_pls = v2.regression_metrics(
                group["y_true"].to_numpy(), group["y_global_pls_anchor"].to_numpy()
            )
            repeat_rows.append(
                {
                    "repeat": int(repeat),
                    "ai_rmse": ai["rmse"],
                    "ai_r2": ai["r2"],
                    "domain_pls_rmse": domain["rmse"],
                    "domain_pls_r2": domain["r2"],
                    "global_pls_rmse": global_pls["rmse"],
                    "global_pls_r2": global_pls["r2"],
                    "ai_vs_domain_pls_pct": 100.0 * (1.0 - ai["rmse"] / domain["rmse"]),
                    "ai_vs_global_pls_pct": 100.0 * (1.0 - ai["rmse"] / global_pls["rmse"]),
                }
            )
        repeat_table = pd.DataFrame(repeat_rows).sort_values("repeat")
        repeat_table.to_csv(trait_dir / "repeat_metrics.csv", index=False)
        summary_rows.append(
            {
                "target": target,
                "trait": trait,
                "repeats": int(len(repeat_table)),
                "ai_rmse_mean": float(repeat_table["ai_rmse"].mean()),
                "ai_r2_mean": float(repeat_table["ai_r2"].mean()),
                "domain_pls_rmse_mean": float(repeat_table["domain_pls_rmse"].mean()),
                "global_pls_rmse_mean": float(repeat_table["global_pls_rmse"].mean()),
                "ai_vs_domain_pls_pct_mean": float(repeat_table["ai_vs_domain_pls_pct"].mean()),
                "ai_vs_global_pls_pct_mean": float(repeat_table["ai_vs_global_pls_pct"].mean()),
                "wins_vs_domain_pls": int((repeat_table["ai_vs_domain_pls_pct"] > 0).sum()),
                "wins_vs_global_pls": int((repeat_table["ai_vs_global_pls_pct"] > 0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("ai_vs_global_pls_pct_mean", ascending=False)
    summary.to_csv(output_dir / "confirmation_summary.csv", index=False)
    manifest = {
        "model": "independent single-output domain-anchored PLUMRAC-MT V5 per trait",
        "development_repeat": 1,
        "confirmation_repeats": repeats,
        "reported_repeats": [1, *repeats],
        "candidate_rule": "development repeat: AI > domain PLS and AI >=10% better than global PLS",
        "candidates": [abbreviated_trait(target) for target in targets],
        "max_parallel_gpu_jobs": args.max_parallel,
        "cohort": "primary",
        "model_independent_excluded_cultivars": ["6.11"],
        "claim_boundary": (
            "Retrospective repeated known-cultivar validation; repeat 1 selected candidates and repeats 2-5 "
            "assess stability. An external year/orchard dataset remains required for independent confirmation."
        ),
    }
    (output_dir / "confirmation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
