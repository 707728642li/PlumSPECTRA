from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import train_plumrac_loco as v2
import train_plumrac_v4_phy as v4
import train_plumrac_v5_auxpretrain as v5
from train_plumrac_v5_stratified import prepare_anchor_targets, predict_indices
from train_texture_pls_random import select_hyperparameters
from v2_registry import abbreviated_trait, add_cultivar_code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", required=True, choices=v2.DEFAULT_TARGETS)
    parser.add_argument("--cohort", choices=["analysis", "primary", "sensitivity"], default="primary")
    parser.add_argument("--exclude-cultivars", default="6.11")
    parser.add_argument("--fixed-epochs", type=int, required=True)
    parser.add_argument("--fixed-gate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    args = parser.parse_args()
    if args.fixed_epochs < 1 or not 0.0 <= args.fixed_gate <= 1.5:
        raise ValueError("Invalid fixed epoch or gate")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "deployment_bundle.json"
    if bundle_path.exists():
        print(f"deployment model already complete: {output_dir}", flush=True)
        return
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Full PLUMRAC-MT training requires CUDA")
    config = v2.RACConfig(
        width=48,
        blocks=4,
        dropout=0.12,
        batch_size=256,
        max_epochs=args.fixed_epochs,
        min_epochs=args.fixed_epochs,
        patience=args.fixed_epochs,
        learning_rate=5e-4,
        weight_decay=2e-3,
        center_weight=0.20,
        rank_weight=0.08,
        sampler_power=0.50,
        validation_cultivars=0,
        min_gate_improvement=0.0,
        min_gate_win_fraction=0.50,
        max_gate_worst_degradation=0.05,
        max_residual_gate=1.0,
        structure_on_final=True,
        attention_tail=True,
    )
    multimodal_dir = args.multimodal_dir.resolve()
    raw = np.load(multimodal_dir / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy").astype(np.float32)
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    sample_ids = aligned["sample_id"].astype(str).to_numpy()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    y = pd.to_numeric(aligned[args.target], errors="coerce").to_numpy(float)
    cohort_column = {
        "analysis": "qc_analysis_include",
        "primary": "qc_primary_include",
        "sensitivity": "qc_sensitivity_include",
    }[args.cohort]
    eligible = aligned[cohort_column].to_numpy(bool) & np.isfinite(y)
    excluded = sorted({value.strip() for value in args.exclude_cultivars.split(",") if value.strip()})
    known = sorted(np.unique(groups).tolist())
    unknown = sorted(set(excluded) - set(known))
    if unknown:
        raise ValueError(f"Unknown cultivar exclusions: {unknown}")
    eligible &= ~np.isin(groups, excluded)
    train_indices = np.flatnonzero(eligible)
    retained_cultivars = sorted(np.unique(groups[train_indices]).tolist())
    group_map = {group: index for index, group in enumerate(known)}
    group_index = np.asarray([group_map[group] for group in groups], dtype=np.int64)
    arrays = v2.preprocess_all(raw, wavelength)
    clean_channels = v2.build_clean_channels(raw, wavelength)

    v4.CHANNEL_SET = "basic"
    v4.ARCHITECTURE = "multiscale"
    v4.MIXSTYLE_P = 0.0
    v5.GLOBAL_AUXILIARY_TARGETS = list(v5.QUALITY_AUXILIARY_TARGETS)
    v5.GLOBAL_AUXILIARY_Y = np.column_stack(
        [
            pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
            for target in v5.GLOBAL_AUXILIARY_TARGETS
        ]
    )
    v5.PRETRAIN_EPOCHS = args.pretrain_epochs
    v5.PRETRAIN_LEARNING_RATE = 1e-3
    v5.PRETRAIN_WEIGHT_DECAY = 1e-3

    preprocessing, n_components, cv_rows = select_hyperparameters(
        arrays, y, groups, train_indices, args.seed
    )
    (
        anchor_model,
        anchor_prediction,
        residual_target,
        anchor_standardized,
        target_mean,
        target_sd,
        cultivar_offsets,
    ) = prepare_anchor_targets(
        arrays,
        y,
        groups,
        train_indices,
        [train_indices],
        preprocessing,
        n_components,
        True,
    )
    model, channel_builder, history, _ = v5.train_auxiliary_initialized_residual_model(
        raw,
        wavelength,
        clean_channels,
        residual_target,
        anchor_standardized,
        group_index,
        groups,
        train_indices,
        None,
        args.seed + 1_000_003,
        config,
        device,
        fixed_epochs=args.fixed_epochs,
    )
    deep_residual_z = predict_indices(
        model,
        channel_builder,
        raw,
        residual_target,
        anchor_standardized,
        group_index,
        train_indices,
        config.batch_size,
        device,
    )
    prediction = anchor_prediction[train_indices] + args.fixed_gate * deep_residual_z * target_sd
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids[train_indices],
            "cultivar_ascii": groups[train_indices],
            "target": args.target,
            "y_true": y[train_indices],
            "y_pred": prediction,
            "y_domain_pls_anchor": anchor_prediction[train_indices],
            "deep_residual": deep_residual_z * target_sd,
        }
    )
    frame = add_cultivar_code(frame)
    frame.to_parquet(output_dir / "training_fit_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output_dir / "pls_cv_scores.csv", index=False)
    torch.save(model.state_dict(), output_dir / "plumrac_state.pt")
    torch.save(channel_builder.state_dict(), output_dir / "channel_builder_state.pt")
    joblib.dump(anchor_model, output_dir / "pls_anchor.joblib", compress=3)
    (output_dir / "cultivar_offsets.json").write_text(
        json.dumps(cultivar_offsets, indent=2), encoding="utf-8"
    )
    fit_metrics = v2.regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
    parameter_count = int(
        sum(
            parameter.numel()
            for parameter in v4.V4PlumRACNet(
                config.width, config.blocks, config.dropout, config.attention_tail
            ).parameters()
        )
    )
    bundle = {
        "model": "Domain-anchored PLUMRAC-MT V5 production model",
        "target": args.target,
        "trait": abbreviated_trait(args.target),
        "one_model_one_target": True,
        "cohort": args.cohort,
        "training_samples": int(len(train_indices)),
        "training_cultivars": retained_cultivars,
        "model_independent_excluded_cultivars": excluded,
        "known_cultivar_required": True,
        "fixed_epochs": args.fixed_epochs,
        "fixed_gate": args.fixed_gate,
        "seed": args.seed,
        "pls_anchor": {"preprocessing": preprocessing, "n_components": int(n_components)},
        "target_mean": target_mean,
        "target_sd": target_sd,
        "config": asdict(config),
        "pretraining": {
            "targets": v5.GLOBAL_AUXILIARY_TARGETS,
            "epochs": args.pretrain_epochs,
            "training_data_only": True,
        },
        "trainable_parameters": parameter_count,
        "spectral_points": int(raw.shape[1]),
        "wavelength_min_nm": float(wavelength.min()),
        "wavelength_max_nm": float(wavelength.max()),
        "physical_gpu_index": args.physical_gpu_index,
        "visible_device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "training_fit_metrics_not_validation": fit_metrics,
        "files": {
            "network_state": "plumrac_state.pt",
            "channel_builder_state": "channel_builder_state.pt",
            "pls_anchor": "pls_anchor.joblib",
            "cultivar_offsets": "cultivar_offsets.json",
            "training_history": "training_history.csv",
            "pls_cv_scores": "pls_cv_scores.csv",
        },
        "provenance_sha256": {
            "trainer": sha256_file(Path(__file__).resolve()),
            "qc_ledger": sha256_file(args.qc_ledger.resolve()),
            "nir_absorbance": sha256_file(multimodal_dir / "nir_c_absorbance.npy"),
            "wavelength": sha256_file(multimodal_dir / "wavelength_nm.npy"),
            "row_index": sha256_file(multimodal_dir / "nir_c_row_index.csv"),
        },
        "claim_boundary": (
            "Final full-cohort deployment fit; training-fit metrics are not validation evidence. "
            "Use the separately archived repeated holdout and LOCO results for performance claims."
        ),
    }
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(json.dumps(bundle, indent=2), flush=True)


if __name__ == "__main__":
    main()
