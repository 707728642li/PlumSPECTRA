from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from v2_registry import abbreviated_trait


TARGETS = {
    "LS": "loading_stiffness_g_per_rawpos_mean",
    "RD": "skin_break_displacement_raw_mean",
}

BASELINES = {
    ("LS", 1): Path("results/v15/domain_anchor_trait_screen/LS/metadata.json"),
    ("LS", 2): Path("results/v15/domain_anchor_confirmation/LS/repeat_2/metadata.json"),
    ("RD", 1): Path("results/v14/domain_anchor_rd_primary_stratified/repeat_1/metadata.json"),
    ("RD", 2): Path("results/v14/domain_anchor_rd_primary_stratified/repeat_2/metadata.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=2, choices=[1, 2])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_plumrac_v5_stratified.py")
    child_environment = os.environ.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = "0"

    pending: list[tuple[str, int, list[str], Path, Path]] = []
    for trait, target in TARGETS.items():
        if abbreviated_trait(target) != trait:
            raise RuntimeError(f"Registry mismatch for {target}")
        for repeat in [1, 2]:
            run_dir = output_dir / trait / f"repeat_{repeat}"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(trainer),
                "--multimodal-dir",
                str(project / "data" / "processed" / "multimodal"),
                "--qc-ledger",
                str(project / "data" / "processed" / "texture_qc" / "texture_qc_ledger.parquet"),
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
                "--crossfit-anchor-folds",
                str(args.folds),
                "--validation-residual-target-mode",
                "legacy_zero",
                "--device",
                "cuda:0",
            ]
            pending.append((trait, repeat, command, run_dir / "run.stdout.log", run_dir / "run.stderr.log"))

    running: list[tuple[str, int, subprocess.Popen[bytes], object, object]] = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            trait, repeat, command, stdout_path, stderr_path = pending.pop(0)
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            process = subprocess.Popen(
                command,
                cwd=project,
                env=child_environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
            running.append((trait, repeat, process, stdout_handle, stderr_handle))
            print(f"launched {trait} repeat {repeat} as PID {process.pid}", flush=True)
        time.sleep(2)
        survivors = []
        for trait, repeat, process, stdout_handle, stderr_handle in running:
            return_code = process.poll()
            if return_code is None:
                survivors.append((trait, repeat, process, stdout_handle, stderr_handle))
                continue
            stdout_handle.close()
            stderr_handle.close()
            if return_code != 0:
                raise RuntimeError(f"{trait} repeat {repeat} failed with exit code {return_code}")
            print(f"completed {trait} repeat {repeat}", flush=True)
        running = survivors

    rows: list[dict[str, object]] = []
    for trait in TARGETS:
        for repeat in [1, 2]:
            candidate = json.loads(
                (output_dir / trait / f"repeat_{repeat}" / "metadata.json").read_text(encoding="utf-8")
            )
            baseline = json.loads((project / BASELINES[(trait, repeat)]).read_text(encoding="utf-8"))
            candidate_rmse = float(candidate["ai_metrics"]["rmse"])
            baseline_rmse = float(baseline["ai_metrics"]["rmse"])
            rows.append(
                {
                    "trait": trait,
                    "repeat": repeat,
                    "baseline_v5_rmse": baseline_rmse,
                    "crossfit_v18_rmse": candidate_rmse,
                    "v18_vs_v5_improvement_pct": 100.0 * (1.0 - candidate_rmse / baseline_rmse),
                    "v18_gate": float(candidate["selected_gate"]),
                    "v5_gate": float(baseline["selected_gate"]),
                    "v18_epoch": int(candidate["selected_epoch"]),
                    "v5_epoch": int(baseline["selected_epoch"]),
                }
            )
    table = pd.DataFrame(rows)
    improvements = table["v18_vs_v5_improvement_pct"].to_numpy(float)
    decision = {
        "predeclared_rule": {
            "mean_improvement_pct_at_least": 2.0,
            "minimum_wins_out_of_four": 3,
            "worst_improvement_pct_at_least": -1.0,
        },
        "observed": {
            "mean_improvement_pct": float(improvements.mean()),
            "wins_out_of_four": int((improvements > 0).sum()),
            "worst_improvement_pct": float(improvements.min()),
        },
    }
    decision["passed"] = bool(
        decision["observed"]["mean_improvement_pct"] >= 2.0
        and decision["observed"]["wins_out_of_four"] >= 3
        and decision["observed"]["worst_improvement_pct"] >= -1.0
    )
    table.to_csv(output_dir / "comparison_with_v5.csv", index=False)
    (output_dir / "screen_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
