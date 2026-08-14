from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = [path.resolve() for path in args.run_dir]
    predictions = []
    summaries = []
    for run_dir in run_dirs:
        predictions.append(pd.read_parquet(run_dir / "predictions_by_seed.parquet"))
        summaries.append(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
    table = pd.concat(predictions, ignore_index=True)
    identity = ["sample_id", "cultivar_ascii", "target", "seed"]
    if table.duplicated(identity).any():
        duplicates = table.loc[table.duplicated(identity, keep=False), identity]
        raise ValueError(f"Duplicate seed predictions across source runs:\n{duplicates.head()}")
    targets = table["target"].unique().tolist()
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target, found {targets}")
    seeds = sorted(int(value) for value in table["seed"].unique())
    sample_counts = table.groupby("seed")["sample_id"].nunique()
    if sample_counts.nunique() != 1:
        raise ValueError(f"Seed sample counts differ: {sample_counts.to_dict()}")

    summary = dict(summaries[0])
    summary["seeds"] = seeds
    summary["seed_run_merge"] = {
        "source_runs": [str(path) for path in run_dirs],
        "seed_count": len(seeds),
        "prediction_rows": int(len(table)),
        "samples_per_seed": int(sample_counts.iloc[0]),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_dir / "predictions_by_seed.parquet", index=False, compression="zstd")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["seed_run_merge"], indent=2), flush=True)


if __name__ == "__main__":
    main()
