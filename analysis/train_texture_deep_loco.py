from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from train_deep_loco import (
    TrainingConfig,
    build_channels,
    regression_metrics,
    train_one,
)


DEFAULT_TARGETS = [
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
    parser.add_argument("--architecture", choices=["cnn", "transformer"], required=True)
    parser.add_argument("--targets", default="all")
    parser.add_argument("--cohort", choices=["analysis", "primary", "sensitivity"], default="analysis")
    parser.add_argument("--heldout", default="all")
    parser.add_argument("--seeds", default="20260806")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--validation-cultivars", type=int, default=3)
    parser.add_argument("--validation-mode", choices=["heldout_cultivars", "within_cultivar"], default="heldout_cultivars")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the deep texture LOCO experiment")

    multimodal_dir = args.multimodal_dir.resolve()
    absorbance = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    channels = build_channels(absorbance, wavelength)
    sample_ids = aligned["sample_id"].to_numpy()
    groups = aligned["cultivar_ascii"].to_numpy()
    all_groups = sorted(np.unique(groups).tolist())
    heldout_groups = all_groups if args.heldout == "all" else [item.strip() for item in args.heldout.split(",")]
    unknown_groups = sorted(set(heldout_groups) - set(all_groups))
    if unknown_groups:
        raise ValueError(f"Unknown held-out cultivars: {unknown_groups}")
    targets = DEFAULT_TARGETS if args.targets == "all" else [item.strip() for item in args.targets.split(",")]
    missing = [target for target in targets if target not in aligned.columns]
    if missing:
        raise ValueError(f"Missing targets: {missing}")
    seeds = [int(item.strip()) for item in args.seeds.split(",")]

    label_values = aligned[targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    label_mask = np.isfinite(label_values)
    cohort_column = {
        "analysis": "qc_analysis_include",
        "primary": "qc_primary_include",
        "sensitivity": "qc_sensitivity_include",
    }[args.cohort]
    label_mask &= aligned[cohort_column].to_numpy(bool)[:, None]
    config = TrainingConfig(
        architecture=args.architecture,
        primary_targets=targets,
        auxiliary_texture=False,
        augmentation=not args.no_augmentation,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        auxiliary_loss_weight=0.0,
        validation_cultivars=args.validation_cultivars,
        validation_mode=args.validation_mode,
        validation_fraction=args.validation_fraction,
        loss=args.loss,
    )

    prediction_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    for heldout in heldout_groups:
        for seed in seeds:
            predictions, _, metadata = train_one(
                config,
                heldout,
                seed,
                channels,
                label_values,
                label_mask,
                targets,
                sample_ids,
                groups,
                all_groups,
                output_dir,
                device,
            )
            prediction_frames.append(predictions)
            metadata_rows.append(metadata)
            print(
                f"completed {args.architecture} held-out {heldout} seed {seed}: "
                f"best epoch {metadata['best_epoch']}, val {metadata['best_validation_macro_normalized_rmse']:.4f}"
            )

    by_seed = pd.concat(prediction_frames, ignore_index=True)
    ensemble = (
        by_seed.groupby(["sample_id", "cultivar_ascii", "target"], as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            prediction_sd=("y_pred", "std"),
            seeds=("seed", "nunique"),
        )
    )
    ensemble["residual"] = ensemble["y_pred"] - ensemble["y_true"]
    seed_metrics: list[dict[str, Any]] = []
    for (target, cultivar, seed), group in by_seed.groupby(["target", "cultivar_ascii", "seed"], observed=True):
        seed_metrics.append(
            {
                "target": target,
                "heldout_cultivar": cultivar,
                "seed": seed,
                **regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    fold_metrics: list[dict[str, Any]] = []
    for (target, cultivar), group in ensemble.groupby(["target", "cultivar_ascii"], observed=True):
        fold_metrics.append(
            {
                "target": target,
                "heldout_cultivar": cultivar,
                **regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )
    pooled = {
        target: regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        for target, group in ensemble.groupby("target", observed=True)
    }
    by_seed.to_parquet(output_dir / "predictions_by_seed.parquet", index=False, compression="zstd")
    ensemble.to_parquet(output_dir / "predictions_ensemble.parquet", index=False, compression="zstd")
    pd.DataFrame(seed_metrics).to_csv(output_dir / "fold_metrics_by_seed.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics_ensemble.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(output_dir / "run_metadata.csv", index=False)
    summary = {
        "config": asdict(config),
        "cohort": {
            "analysis": "high-confidence-QC analysis cohort",
            "primary": "strict 10-percent consensus-QC cohort",
            "sensitivity": "hard-valid full sensitivity",
        }[args.cohort],
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "heldout_cultivars_completed": heldout_groups,
        "seeds": seeds,
        "pooled_ensemble_metrics": pooled,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
