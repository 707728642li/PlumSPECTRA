from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


QUALITY_TARGETS = ["fruit_weight_g", "soluble_solids_pct", "ph"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source_manifest.resolve()
    source = pd.read_csv(source_path, dtype={"sample_id": str, "cultivar_ascii": str})
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[source["sample_id"]].copy()

    validity_columns = [f"{target}_valid" for target in QUALITY_TARGETS]
    missing = sorted(set(validity_columns) - set(aligned.columns))
    if missing:
        raise ValueError(f"QC ledger is missing validity columns: {missing}")
    complete = aligned[validity_columns].astype(bool).all(axis=1)
    manifest = source.loc[complete.to_numpy()].copy().reset_index(drop=True)
    if manifest["sample_id"].duplicated().any():
        raise RuntimeError("Quality manifest contains duplicate sample IDs")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "v22_quality_fivefold_manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    counts = (
        manifest.groupby(["cultivar_ascii", "outer_fold"], observed=True)
        .size()
        .rename("samples")
        .reset_index()
    )
    counts.to_csv(output_dir / "v22_quality_fivefold_counts.csv", index=False)

    summary = {
        "protocol": "V22 common complete-case conventional-quality cohort inheriting V20 folds",
        "source_manifest": str(source_path),
        "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "targets": QUALITY_TARGETS,
        "validity_columns": validity_columns,
        "source_samples": int(len(source)),
        "samples": int(len(manifest)),
        "excluded_invalid_samples": int(len(source) - len(manifest)),
        "cultivars": int(manifest["cultivar_ascii"].nunique()),
        "folds": int(manifest["outer_fold"].nunique()),
        "fold_samples": {
            str(fold): int(count)
            for fold, count in manifest.groupby("outer_fold", observed=True).size().items()
        },
        "every_sample_tested_once": bool(
            manifest["sample_id"].nunique() == len(manifest)
            and set(manifest["outer_fold"]) == {1, 2, 3, 4, 5}
        ),
    }
    (output_dir / "v22_quality_fivefold_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
