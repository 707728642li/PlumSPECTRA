from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import train_plumrac_loco as v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--repeats", default="1,2,3,4,5")
    parser.add_argument("--cultivar-anchor-calibration", action="store_true")
    parser.add_argument(
        "--target",
        default="skin_break_displacement_raw_mean",
        choices=v2.DEFAULT_TARGETS,
    )
    parser.add_argument("--fixed-gate", type=float)
    args = parser.parse_args()
    if not 1 <= args.max_parallel <= 2:
        raise ValueError("This single-GPU queue is intentionally capped at one or two concurrent jobs")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repeats = [int(value.strip()) for value in args.repeats.split(",") if value.strip()]
    jobs: list[tuple[int, list[str], Path, Path]] = []
    trainer = Path(__file__).with_name("train_plumrac_v5_stratified.py")
    for repeat in repeats:
        repeat_dir = output_dir / f"repeat_{repeat}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(trainer),
            "--multimodal-dir",
            "data/processed/multimodal",
            "--qc-ledger",
            "data/processed/texture_qc/texture_qc_ledger.parquet",
            "--output-dir",
            str(repeat_dir),
            "--target",
            args.target,
            "--cohort",
            "primary",
            "--exclude-cultivars",
            "6.11",
            "--repeat",
            str(repeat),
            "--validation-residual-target-mode",
            "legacy_zero",
        ]
        if args.cultivar_anchor_calibration:
            command.append("--cultivar-anchor-calibration")
        if args.fixed_gate is not None:
            command.extend(["--fixed-gate", str(args.fixed_gate)])
        jobs.append((repeat, command, repeat_dir / "run.stdout.log", repeat_dir / "run.stderr.log"))

    pending = list(jobs)
    running: list[tuple[int, subprocess.Popen[bytes], object, object]] = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            repeat, command, stdout_path, stderr_path = pending.pop(0)
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
            running.append((repeat, process, stdout_handle, stderr_handle))
            print(f"launched repeat {repeat} as PID {process.pid}", flush=True)
        time.sleep(2)
        survivors = []
        for repeat, process, stdout_handle, stderr_handle in running:
            return_code = process.poll()
            if return_code is None:
                survivors.append((repeat, process, stdout_handle, stderr_handle))
                continue
            stdout_handle.close()
            stderr_handle.close()
            if return_code != 0:
                raise RuntimeError(f"repeat {repeat} failed with exit code {return_code}")
            print(f"completed repeat {repeat}", flush=True)
        running = survivors

    frames = [pd.read_parquet(output_dir / f"repeat_{repeat}" / "predictions.parquet") for repeat in repeats]
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    metric_rows = []
    for repeat, frame in predictions.groupby("repeat", observed=True):
        ai = v2.regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        pls = v2.regression_metrics(frame["y_true"].to_numpy(), frame["y_pls_anchor"].to_numpy())
        global_pls = v2.regression_metrics(
            frame["y_true"].to_numpy(), frame["y_global_pls_anchor"].to_numpy()
        )
        metric_rows.append(
            {
                "repeat": int(repeat),
                **{f"ai_{key}": value for key, value in ai.items()},
                **{f"global_pls_{key}": value for key, value in global_pls.items()},
                **{f"pls_{key}": value for key, value in pls.items()},
                "relative_rmse_improvement_pct": 100.0 * (1.0 - ai["rmse"] / pls["rmse"]),
                "relative_rmse_improvement_vs_global_pls_pct": 100.0
                * (1.0 - ai["rmse"] / global_pls["rmse"]),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values("repeat")
    metrics.to_csv(output_dir / "repeat_metrics.csv", index=False)
    summary = {
        "model": "PLUMRAC-MT V5 known-cultivar single-trait model",
        "validation": "repeated cultivar-stratified fruit holdout",
        "target": args.target,
        "cohort": "primary",
        "model_independent_excluded_cultivars": ["6.11"],
        "repeats": repeats,
        "max_parallel_gpu_jobs": args.max_parallel,
        "cultivar_anchor_calibration": bool(args.cultivar_anchor_calibration),
        "fixed_gate": args.fixed_gate,
        "ai_metrics_mean": {
            key.removeprefix("ai_"): float(metrics[key].mean())
            for key in metrics.columns
            if key.startswith("ai_")
        },
        "ai_metrics_sd": {
            key.removeprefix("ai_"): float(metrics[key].std(ddof=1))
            for key in metrics.columns
            if key.startswith("ai_")
        },
        "pls_metrics_mean": {
            key.removeprefix("pls_"): float(metrics[key].mean())
            for key in metrics.columns
            if key.startswith("pls_")
        },
        "global_pls_metrics_mean": {
            key.removeprefix("global_pls_"): float(metrics[key].mean())
            for key in metrics.columns
            if key.startswith("global_pls_")
        },
        "relative_rmse_improvement_pct_mean": float(metrics["relative_rmse_improvement_pct"].mean()),
        "relative_rmse_improvement_pct_sd": float(metrics["relative_rmse_improvement_pct"].std(ddof=1)),
        "relative_rmse_improvement_vs_global_pls_pct_mean": float(
            metrics["relative_rmse_improvement_vs_global_pls_pct"].mean()
        ),
        "wins": int((metrics["relative_rmse_improvement_pct"] > 0).sum()),
        "claim_boundary": (
            "Known-cultivar new-fruit generalization. Test fruit labels are absent from preprocessing, "
            "hyperparameter selection, early stopping, residual-gate selection, and training. "
            "Unknown-cultivar performance must be reported separately using LOCO."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
