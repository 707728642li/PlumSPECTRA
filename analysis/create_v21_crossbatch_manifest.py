from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


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

BATCH_FOLDS = {
    "KLD_B01": 1,
    "KLD_B02": 2,
    "KLD_B03": 3,
    "WW_B01": 4,
    "WW_B02": 5,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    row_index = pd.read_csv(args.multimodal_dir.resolve() / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    complete_targets = aligned[TARGETS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    eligible = (
        aligned["qc_primary_include"].astype(bool)
        & complete_targets
        & (aligned["cultivar_ascii"].astype(str) != "6.11")
    )
    cohort = aligned.loc[
        eligible, ["sample_id", "cultivar_ascii", "batch_id"]
    ].copy().reset_index(drop=True)
    cohort["sample_id"] = cohort["sample_id"].astype(str)
    cohort["cultivar_ascii"] = cohort["cultivar_ascii"].astype(str)
    cohort["batch_id"] = cohort["batch_id"].astype(str)
    cohort["outer_fold"] = cohort["batch_id"].map(BATCH_FOLDS).fillna(0).astype(int)
    observed = set(cohort.loc[cohort["outer_fold"] > 0, "batch_id"])
    if observed != set(BATCH_FOLDS):
        raise RuntimeError(
            f"Cross-batch manifest mismatch: missing={sorted(set(BATCH_FOLDS) - observed)}"
        )
    for batch_id, fold in BATCH_FOLDS.items():
        batch = cohort[cohort["batch_id"] == batch_id]
        cultivar = str(batch["cultivar_ascii"].iloc[0])
        training_same_cultivar = cohort[
            (cohort["cultivar_ascii"] == cultivar) & (cohort["batch_id"] != batch_id)
        ]
        if training_same_cultivar.empty:
            raise RuntimeError(f"Fold {fold} has no same-cultivar training batch")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "v21_crossbatch_manifest.csv"
    cohort.to_csv(manifest_path, index=False, lineterminator="\n")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    fold_rows = []
    for batch_id, fold in BATCH_FOLDS.items():
        test = cohort[cohort["outer_fold"] == fold]
        cultivar = str(test["cultivar_ascii"].iloc[0])
        fold_rows.append(
            {
                "outer_fold": fold,
                "heldout_batch": batch_id,
                "cultivar_ascii": cultivar,
                "test_samples": int(len(test)),
                "same_cultivar_training_samples": int(
                    ((cohort["cultivar_ascii"] == cultivar) & (cohort["outer_fold"] != fold)).sum()
                ),
                "all_training_samples": int((cohort["outer_fold"] != fold).sum()),
            }
        )
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "v21_crossbatch_fold_counts.csv", index=False, lineterminator="\n"
    )
    summary = {
        "protocol": "V21 limited same-cultivar leave-one-batch-out audit",
        "eligible_samples": int(len(cohort)),
        "testable_unique_samples": int((cohort["outer_fold"] > 0).sum()),
        "test_cultivars": sorted(cohort.loc[cohort["outer_fold"] > 0, "cultivar_ascii"].unique().tolist()),
        "test_batches": list(BATCH_FOLDS),
        "fold_mapping": BATCH_FOLDS,
        "non_testable_samples_outer_fold_zero": int((cohort["outer_fold"] == 0).sum()),
        "manifest": str(manifest_path),
        "manifest_sha256": digest,
    }
    (output_dir / "v21_crossbatch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
