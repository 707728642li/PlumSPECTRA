from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--rd-dir", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    args = parser.parse_args()
    if not 1 <= args.max_parallel <= 8:
        raise ValueError("--max-parallel must be in [1, 8]")
    confirmation_dir = args.confirmation_dir.resolve()
    summary = pd.read_csv(confirmation_dir / "confirmation_summary.csv")
    analyzer = Path(__file__).with_name("analyze_v14_domain_anchor.py")
    pending = []
    for row in summary.itertuples(index=False):
        trait = str(row.trait)
        trait_dir = confirmation_dir / trait
        statistics_dir = trait_dir / "statistics"
        statistics_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(analyzer),
            "--predictions",
            str(trait_dir / "predictions.parquet"),
            "--output-dir",
            str(statistics_dir),
            "--bootstrap-iterations",
            str(args.bootstrap_iterations),
        ]
        pending.append((trait, command, statistics_dir / "run.stdout.log", statistics_dir / "run.stderr.log"))

    running = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            trait, command, stdout_path, stderr_path = pending.pop(0)
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
            running.append((trait, process, stdout_handle, stderr_handle))
            print(f"launched statistics {trait} as PID {process.pid}", flush=True)
        time.sleep(1)
        survivors = []
        for trait, process, stdout_handle, stderr_handle in running:
            code = process.poll()
            if code is None:
                survivors.append((trait, process, stdout_handle, stderr_handle))
                continue
            stdout_handle.close()
            stderr_handle.close()
            if code != 0:
                raise RuntimeError(f"statistics {trait} failed with exit code {code}")
            print(f"completed statistics {trait}", flush=True)
        running = survivors

    sources = {
        str(row.trait): confirmation_dir / str(row.trait) / "statistics" / "statistics.json"
        for row in summary.itertuples(index=False)
    }
    sources["RD"] = args.rd_dir.resolve() / "statistics" / "statistics.json"
    rows = []
    for trait, path in sources.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        global_pls = report["comparisons"]["global_pls"]
        domain_pls = report["comparisons"]["domain_pls"]
        rows.append(
            {
                "trait": trait,
                "prediction_records": report["prediction_records"],
                "unique_test_fruits": report["unique_fruits_in_any_test_split"],
                "cultivars": report["cultivars"],
                "ai_rmse": global_pls["ai"]["pooled_rmse"],
                "global_pls_rmse": global_pls["baseline"]["pooled_rmse"],
                "ai_vs_global_pls_pct": global_pls["pooled_relative_rmse_improvement_pct"],
                "global_ci95_low_pct": global_pls["cultivar_cluster_bootstrap_pooled_ci95_pct"][0],
                "global_ci95_high_pct": global_pls["cultivar_cluster_bootstrap_pooled_ci95_pct"][1],
                "global_cultivar_wins": global_pls["cultivar_wins"],
                "global_wilcoxon_p": global_pls["wilcoxon_greater_p"],
                "domain_pls_rmse": domain_pls["baseline"]["pooled_rmse"],
                "ai_vs_domain_pls_pct": domain_pls["pooled_relative_rmse_improvement_pct"],
                "domain_ci95_low_pct": domain_pls["cultivar_cluster_bootstrap_pooled_ci95_pct"][0],
                "domain_ci95_high_pct": domain_pls["cultivar_cluster_bootstrap_pooled_ci95_pct"][1],
                "domain_cultivar_wins": domain_pls["cultivar_wins"],
                "domain_wilcoxon_p": domain_pls["wilcoxon_greater_p"],
            }
        )
    table = pd.DataFrame(rows).sort_values("ai_vs_global_pls_pct", ascending=False)
    table.to_csv(confirmation_dir / "all_confirmed_trait_statistics.csv", index=False)
    manifest = {
        "scope": "eight independently trained single-output domain-anchored AI texture models",
        "traits": table["trait"].tolist(),
        "bootstrap_iterations_per_trait": args.bootstrap_iterations,
        "cluster_unit": "cultivar",
        "max_parallel_cpu_jobs": args.max_parallel,
        "all_traits_global_ci_excludes_zero": bool((table["global_ci95_low_pct"] > 0).all()),
        "all_traits_domain_ci_excludes_zero": bool((table["domain_ci95_low_pct"] > 0).all()),
        "claim_boundary": (
            "Retrospective repeated known-cultivar validation. Cultivar-cluster inference retains repeated "
            "predictions from the same fruit within the same independent cultivar cluster."
        ),
    }
    (confirmation_dir / "statistics_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
