from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


TARGETS = {
    "LS": "loading_stiffness_g_per_rawpos_mean",
    "SRF": "skin_break_force_g_mean",
}


def legacy_metadata(project: Path, trait: str, repeat: int) -> Path:
    return project / "results" / "v15" / "domain_anchor_confirmation" / trait / f"repeat_{repeat}" / "metadata.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    screen_dir = args.screen_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_plumrac_v5_stratified.py")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"

    pending: list[tuple[str, int, list[str], Path, Path]] = []
    for trait, target in TARGETS.items():
        for repeat in [3, 4, 5]:
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
                "--validation-residual-target-mode",
                "observed",
                "--device",
                "cuda:0",
            ]
            pending.append((trait, repeat, command, run_dir / "run.stdout.log", run_dir / "run.stderr.log"))

    running: list[tuple[str, int, subprocess.Popen[bytes], object, object]] = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            trait, repeat, command, stdout_path, stderr_path = pending.pop(0)
            if (stdout_path.parent / "metadata.json").exists() and (
                stdout_path.parent / "predictions.parquet"
            ).exists():
                print(f"skipped completed {trait} repeat {repeat}", flush=True)
                continue
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            process = subprocess.Popen(
                command,
                cwd=project,
                env=environment,
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

    run_rows: list[dict[str, object]] = []
    trait_rows: list[dict[str, object]] = []
    for trait in TARGETS:
        for repeat in [1, 2, 3, 4, 5]:
            candidate_dir = (
                screen_dir / trait / f"repeat_{repeat}"
                if repeat in [1, 2]
                else output_dir / trait / f"repeat_{repeat}"
            )
            candidate = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
            legacy_path = (
                project / "results" / "v15" / "domain_anchor_trait_screen" / trait / "metadata.json"
                if repeat == 1
                else legacy_metadata(project, trait, repeat)
            )
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            new_rmse = float(candidate["ai_metrics"]["rmse"])
            old_rmse = float(legacy["ai_metrics"]["rmse"])
            run_rows.append(
                {
                    "trait": trait,
                    "repeat": repeat,
                    "subset": "development" if repeat <= 2 else "confirmation",
                    "legacy_v17_rmse": old_rmse,
                    "corrected_v19_rmse": new_rmse,
                    "v19_vs_v17_improvement_pct": 100.0 * (1.0 - new_rmse / old_rmse),
                    "v19_vs_domain_pls_improvement_pct": float(candidate["relative_rmse_improvement_pct"]),
                    "v19_vs_global_pls_improvement_pct": float(
                        candidate["relative_rmse_improvement_vs_global_pls_pct"]
                    ),
                    "v19_epoch": int(candidate["selected_epoch"]),
                    "v17_epoch": int(legacy["selected_epoch"]),
                    "v19_gate": float(candidate["selected_gate"]),
                    "v17_gate": float(legacy["selected_gate"]),
                }
            )
        trait_table = pd.DataFrame([row for row in run_rows if row["trait"] == trait])
        confirm = trait_table[trait_table["subset"] == "confirmation"]
        changes = confirm["v19_vs_v17_improvement_pct"].to_numpy(float)
        domain = confirm["v19_vs_domain_pls_improvement_pct"].to_numpy(float)
        passed = bool(
            changes.mean() >= 2.0
            and (changes > 0).sum() >= 2
            and changes.min() >= -1.0
            and domain.mean() >= 4.0
        )
        trait_rows.append(
            {
                "trait": trait,
                "confirmation_mean_v19_vs_v17_pct": float(changes.mean()),
                "confirmation_wins_out_of_three": int((changes > 0).sum()),
                "confirmation_worst_v19_vs_v17_pct": float(changes.min()),
                "confirmation_mean_v19_vs_domain_pls_pct": float(domain.mean()),
                "passed": passed,
                "production_decision": "replace V17" if passed else "retain V17",
            }
        )

    runs = pd.DataFrame(run_rows).sort_values(["trait", "repeat"])
    decisions = pd.DataFrame(trait_rows).sort_values("trait")
    runs.to_csv(output_dir / "all_run_comparison.csv", index=False)
    decisions.to_csv(output_dir / "trait_confirmation_decisions.csv", index=False)
    manifest = {
        "development_repeats": [1, 2],
        "sealed_confirmation_repeats": [3, 4, 5],
        "candidate_traits": list(TARGETS),
        "replacement_rule": {
            "mean_v19_vs_v17_pct_at_least": 2.0,
            "minimum_wins_out_of_three": 2,
            "worst_v19_vs_v17_pct_at_least": -1.0,
            "mean_v19_vs_domain_pls_pct_at_least": 4.0,
        },
        "decisions": trait_rows,
    }
    (output_dir / "confirmation_decision.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(runs.to_string(index=False), flush=True)
    print(decisions.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
