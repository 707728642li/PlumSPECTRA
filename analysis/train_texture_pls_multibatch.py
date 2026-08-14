from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from train_texture_pls_loco import (
    DEFAULT_TARGETS,
    preprocess_all,
    regression_metrics,
    run_outer_fold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inner-splits", type=int, default=4)
    args = parser.parse_args()

    multimodal_dir = args.multimodal_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    absorbance = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    arrays = preprocess_all(absorbance, wavelength)
    sample_ids = aligned["sample_id"].to_numpy()
    batch_groups = aligned["batch_id"].to_numpy()
    multi_batch_cultivars = (
        aligned.groupby("cultivar_ascii", observed=True)["batch_id"].nunique().loc[lambda x: x > 1].index.tolist()
    )
    heldout_batches = sorted(aligned.loc[aligned["cultivar_ascii"].isin(multi_batch_cultivars), "batch_id"].unique())

    futures = []
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for target in DEFAULT_TARGETS:
            y = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
            eligible = aligned["qc_primary_include"].to_numpy(bool) & np.isfinite(y)
            for batch in heldout_batches:
                futures.append(
                    executor.submit(
                        run_outer_fold,
                        target,
                        batch,
                        arrays,
                        sample_ids,
                        batch_groups,
                        y,
                        eligible,
                        output_dir / "models",
                        args.inner_splits,
                    )
                )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"completed {result['target']} held-out batch {result['heldout_cultivar']} "
                f"with {result['selected']['preprocessing']}/{result['selected']['n_components']}"
            )

    prediction_rows = []
    fold_rows = []
    selection_rows = []
    for result in sorted(results, key=lambda row: (row["target"], row["heldout_cultivar"])):
        target = result["target"]
        indices = result["test_indices"]
        truth = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)[indices]
        prediction = result["prediction"]
        for index, y_true, y_pred in zip(indices, truth, prediction):
            prediction_rows.append(
                {
                    "sample_id": sample_ids[index],
                    "cultivar_ascii": aligned.iloc[index]["cultivar_ascii"],
                    "heldout_batch": batch_groups[index],
                    "target": target,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "residual": float(y_pred - y_true),
                }
            )
        fold_rows.append(
            {
                "target": target,
                "heldout_batch": result["heldout_cultivar"],
                "cultivar_ascii": aligned.loc[aligned["batch_id"].eq(result["heldout_cultivar"]), "cultivar_ascii"].iloc[0],
                **regression_metrics(truth, prediction),
            }
        )
        selection_rows.append(
            {
                "target": target,
                "heldout_batch": result["heldout_cultivar"],
                "preprocessing": result["selected"]["preprocessing"],
                "n_components": result["selected"]["n_components"],
                "inner_macro_normalized_rmse": result["selected"]["macro_normalized_rmse"],
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    folds.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / "selected_hyperparameters.csv", index=False)
    pooled = {
        target: regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        for target, group in predictions.groupby("target", observed=True)
    }
    summary = {
        "model": "PLSRegression",
        "validation": "leave-one-batch-out within cultivars represented by multiple batches",
        "heldout_batches": heldout_batches,
        "multi_batch_cultivars": multi_batch_cultivars,
        "pooled_metrics": pooled,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
