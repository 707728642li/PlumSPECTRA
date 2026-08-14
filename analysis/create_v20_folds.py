from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


TARGETS = [
    "skin_break_force_g_mean",
    "skin_break_displacement_raw_mean",
    "skin_break_drop_g_mean",
    "flesh_force_mean_g_mean",
    "force_at_6_rawpos_g_mean",
    "loading_stiffness_g_per_rawpos_mean",
    "loading_work_g_rawpos_mean",
    "post_break_work_g_rawpos_mean",
    "adhesive_force_g_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--excluded-cultivar", default="6.11")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    row_index = pd.read_csv(args.multimodal_dir.resolve() / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    complete_targets = aligned[TARGETS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    eligible = (
        aligned["qc_primary_include"].astype(bool)
        & complete_targets
        & (aligned["cultivar_ascii"].astype(str) != args.excluded_cultivar)
    )
    cohort = aligned.loc[eligible, ["sample_id", "cultivar_ascii"]].copy().reset_index(drop=True)
    cohort["sample_id"] = cohort["sample_id"].astype(str)
    cohort["cultivar_ascii"] = cohort["cultivar_ascii"].astype(str)
    if cohort["sample_id"].duplicated().any():
        raise RuntimeError("Eligible primary cohort contains duplicate sample_id values")

    fold_assignment = np.zeros(len(cohort), dtype=np.int16)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (_, test_relative) in enumerate(
        splitter.split(np.zeros(len(cohort)), cohort["cultivar_ascii"]), start=1
    ):
        fold_assignment[test_relative] = fold
    if np.any(fold_assignment == 0):
        raise RuntimeError("At least one eligible fruit was not assigned to an outer fold")
    cohort["outer_fold"] = fold_assignment

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "v20_fivefold_manifest.csv"
    cohort.to_csv(manifest_path, index=False, lineterminator="\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    counts = (
        cohort.groupby(["cultivar_ascii", "outer_fold"], observed=True)
        .size()
        .rename("samples")
        .reset_index()
    )
    counts.to_csv(output_dir / "v20_fivefold_counts.csv", index=False, lineterminator="\n")
    summary = {
        "protocol": "V20 frozen non-overlapping cultivar-stratified five-fold audit",
        "seed": args.seed,
        "folds": args.folds,
        "cohort": "qc_primary_include with complete nine-trait texture labels",
        "excluded_cultivar": args.excluded_cultivar,
        "samples": int(len(cohort)),
        "cultivars": int(cohort["cultivar_ascii"].nunique()),
        "manifest": str(manifest_path),
        "manifest_sha256": digest,
        "fold_samples": {
            str(fold): int(count)
            for fold, count in cohort.groupby("outer_fold", observed=True).size().items()
        },
        "every_sample_tested_once": bool(
            len(cohort) == cohort["sample_id"].nunique()
            and set(cohort["outer_fold"]) == set(range(1, args.folds + 1))
        ),
    }
    (output_dir / "v20_fivefold_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
