from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train_plumrac_loco import (
    PlumRACNet,
    RACConfig,
    SpectralChannelBuilder,
    build_clean_channels,
    fit_channel_scaler,
)
from train_texture_pls_loco import preprocess_all


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_residual_cpu(
    model: PlumRACNet,
    channel_builder: SpectralChannelBuilder,
    raw: np.ndarray,
    anchor_standardized: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    values = []
    model.eval()
    channel_builder.eval()
    with torch.no_grad():
        for start in range(0, len(raw), batch_size):
            stop = min(start + batch_size, len(raw))
            raw_batch = torch.from_numpy(raw[start:stop].astype(np.float32))
            anchor_batch = torch.from_numpy(anchor_standardized[start:stop].astype(np.float32))
            channels = channel_builder(raw_batch, augment=False)
            values.append(model(channels, anchor_batch).numpy())
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--relative-tolerance", type=float, default=2e-5)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
    target = str(summary["target"])
    abbreviation = str(summary["trait_abbreviation"])
    cohort = str(summary["cohort"])
    multimodal = args.multimodal_dir.resolve()
    raw = np.load(multimodal / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(multimodal / "wavelength_nm.npy").astype(np.float32)
    row_index = pd.read_csv(multimodal / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    y = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    cohort_column = {
        "analysis": "qc_analysis_include",
        "primary": "qc_primary_include",
        "sensitivity": "qc_sensitivity_include",
    }[cohort]
    eligible = aligned[cohort_column].to_numpy(bool) & np.isfinite(y)
    clean_channels = build_clean_channels(raw, wavelength)
    pls_arrays = preprocess_all(raw, wavelength)

    rows = []
    for metadata_path in sorted((model_dir / "runs" / abbreviation).glob("*/seed_*/metadata.json")):
        run_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        heldout = str(metadata["heldout_cultivar"])
        outer_train = np.flatnonzero(eligible & (groups != heldout))
        test_indices = np.flatnonzero(eligible & (groups == heldout))
        target_mean = float(np.mean(y[outer_train]))
        target_sd = max(float(np.std(y[outer_train], ddof=1)), 1e-6)
        channel_mean, channel_sd = fit_channel_scaler(clean_channels, outer_train)

        config_values = dict(metadata["config"])
        config = RACConfig(**config_values)
        state_path = run_dir / "inference_state.npz"
        np.savez_compressed(
            state_path,
            wavelength_nm=wavelength,
            channel_mean=channel_mean,
            channel_sd=channel_sd,
            target_mean=np.asarray([target_mean], dtype=np.float64),
            target_sd=np.asarray([target_sd], dtype=np.float64),
        )

        anchor = joblib.load(run_dir / "pls_anchor.joblib")
        preprocessing = str(metadata["pls_anchor"]["preprocessing"])
        anchor_test = anchor.predict(pls_arrays[preprocessing][test_indices]).ravel()
        anchor_standardized = ((anchor_test - target_mean) / target_sd).astype(np.float32)
        model = PlumRACNet(config.width, config.blocks, config.dropout, config.attention_tail)
        model.load_state_dict(torch.load(run_dir / "plumrac_state.pt", map_location="cpu", weights_only=True))
        builder = SpectralChannelBuilder(wavelength, channel_mean, channel_sd, config)
        residual_z = predict_residual_cpu(model, builder, raw[test_indices], anchor_standardized)
        prediction = anchor_test + float(metadata["selected_gate"]) * residual_z * target_sd
        saved = pd.read_parquet(run_dir / "predictions.parquet").set_index("sample_id")
        order = aligned.iloc[test_indices]["sample_id"].tolist()
        saved_values = saved.loc[order, "y_pred"].to_numpy(float)
        max_abs_difference = float(np.max(np.abs(prediction - saved_values)))
        max_standardized_difference = float(max_abs_difference / target_sd)
        if max_abs_difference > args.tolerance and max_standardized_difference > args.relative_tolerance:
            raise RuntimeError(
                f"Packaged inference mismatch for {abbreviation}/{heldout}/{run_dir.name}: "
                f"absolute={max_abs_difference}, standardized={max_standardized_difference}"
            )
        inference_manifest = {
            "model": "PlumRAC-Net",
            "target": target,
            "trait_abbreviation": abbreviation,
            "heldout_cultivar": heldout,
            "seed": metadata["seed"],
            "selected_gate": metadata["selected_gate"],
            "selected_objective_profile": metadata["selected_objective_profile"],
            "target_mean": target_mean,
            "target_sd": target_sd,
            "channel_state": state_path.name,
            "preprocessing": preprocessing,
            "n_components": metadata["pls_anchor"]["n_components"],
            "verification_max_abs_difference": max_abs_difference,
            "verification_max_standardized_difference": max_standardized_difference,
            "sha256": {
                "plumrac_state.pt": sha256_file(run_dir / "plumrac_state.pt"),
                "pls_anchor.joblib": sha256_file(run_dir / "pls_anchor.joblib"),
                "inference_state.npz": sha256_file(state_path),
            },
        }
        (run_dir / "inference_manifest.json").write_text(
            json.dumps(inference_manifest, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                "trait": abbreviation,
                "heldout_cultivar": heldout,
                "seed": metadata["seed"],
                "test_samples": len(test_indices),
                "max_abs_difference": max_abs_difference,
                "max_standardized_difference": max_standardized_difference,
                "status": "PASS",
            }
        )

    verification = pd.DataFrame(rows)
    verification.to_csv(model_dir / "inference_packaging_audit.csv", index=False)
    report = {
        "status": "PASS",
        "trait": abbreviation,
        "runs_packaged": int(len(verification)),
        "maximum_prediction_difference": float(verification["max_abs_difference"].max()),
        "maximum_standardized_prediction_difference": float(verification["max_standardized_difference"].max()),
        "absolute_tolerance": args.tolerance,
        "standardized_tolerance": args.relative_tolerance,
        "acceptance": "absolute difference <= absolute tolerance OR standardized difference <= standardized tolerance",
    }
    (model_dir / "inference_packaging_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
