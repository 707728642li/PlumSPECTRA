from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

import train_plumrac_loco as v2
import train_plumrac_v4_phy as v4
from train_plumrac_v5_stratified import predict_indices


def predict_bundle(
    model_dir: Path,
    raw: np.ndarray,
    wavelength: np.ndarray,
    sample_ids: np.ndarray,
    cultivars: np.ndarray,
    device_name: str = "cpu",
    cpu_threads: int = 8,
) -> pd.DataFrame:
    model_dir = model_dir.resolve()
    bundle = json.loads((model_dir / "deployment_bundle.json").read_text(encoding="utf-8"))
    if raw.ndim != 2 or raw.shape[1] != int(bundle["spectral_points"]):
        raise ValueError(
            f"Expected spectra [n,{bundle['spectral_points']}], received {raw.shape}"
        )
    if len(raw) != len(sample_ids) or len(raw) != len(cultivars):
        raise ValueError("Spectra, sample IDs and cultivar labels must have the same row count")
    if not np.isclose(wavelength.min(), bundle["wavelength_min_nm"], atol=1e-6) or not np.isclose(
        wavelength.max(), bundle["wavelength_max_nm"], atol=1e-6
    ):
        raise ValueError("Wavelength range does not match the deployment bundle")
    offsets = json.loads((model_dir / bundle["files"]["cultivar_offsets"]).read_text(encoding="utf-8"))
    unknown = sorted(set(map(str, cultivars)) - set(offsets))
    if unknown:
        raise ValueError(
            f"This known-cultivar model has no calibration offset for: {unknown}. "
            "Use the separately validated LOCO model for unknown cultivars."
        )
    arrays = v2.preprocess_all(np.asarray(raw, dtype=np.float32), np.asarray(wavelength, dtype=np.float32))
    preprocessing = str(bundle["pls_anchor"]["preprocessing"])
    anchor_model = joblib.load(model_dir / bundle["files"]["pls_anchor"])
    anchor = anchor_model.predict(arrays[preprocessing]).ravel()
    anchor = anchor + np.asarray([offsets[str(cultivar)] for cultivar in cultivars], dtype=float)
    anchor_standardized = ((anchor - bundle["target_mean"]) / bundle["target_sd"]).astype(np.float32)

    torch.set_num_threads(max(1, cpu_threads))
    device = torch.device(device_name)
    config = v2.RACConfig(**bundle["config"])
    v4.CHANNEL_SET = "basic"
    v4.ARCHITECTURE = "multiscale"
    v4.MIXSTYLE_P = 0.0
    model = v4.V4PlumRACNet(
        config.width, config.blocks, config.dropout, config.attention_tail
    ).to(device)
    state = torch.load(
        model_dir / bundle["files"]["network_state"], map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    points = raw.shape[1]
    dummy_mean = np.zeros((1, 3, points), dtype=np.float32)
    dummy_sd = np.ones((1, 3, points), dtype=np.float32)
    channel_builder = v2.SpectralChannelBuilder(
        wavelength, dummy_mean, dummy_sd, config
    ).to(device)
    channel_state = torch.load(
        model_dir / bundle["files"]["channel_builder_state"],
        map_location=device,
        weights_only=True,
    )
    channel_builder.load_state_dict(channel_state)
    indices = np.arange(len(raw), dtype=int)
    deep_z = predict_indices(
        model,
        channel_builder,
        np.asarray(raw, dtype=np.float32),
        np.zeros(len(raw), dtype=np.float32),
        anchor_standardized,
        np.zeros(len(raw), dtype=np.int64),
        indices,
        config.batch_size,
        device,
    )
    prediction = anchor + bundle["fixed_gate"] * deep_z * bundle["target_sd"]
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cultivar_ascii": cultivars,
            "target": bundle["target"],
            "trait": bundle["trait"],
            "y_pred": prediction,
            "y_domain_pls_anchor": anchor,
            "deep_residual": deep_z * bundle["target_sd"],
            "residual_gate": bundle["fixed_gate"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--absorbance", type=Path, required=True)
    parser.add_argument("--wavelength", type=Path, required=True)
    parser.add_argument("--sample-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    sample_table = pd.read_csv(args.sample_table)
    required = {"sample_id", "cultivar_ascii"}
    missing = sorted(required - set(sample_table.columns))
    if missing:
        raise ValueError(f"Sample table is missing columns: {missing}")
    raw = np.load(args.absorbance)
    wavelength = np.load(args.wavelength)
    predictions = predict_bundle(
        args.model_dir,
        raw,
        wavelength,
        sample_table["sample_id"].astype(str).to_numpy(),
        sample_table["cultivar_ascii"].astype(str).to_numpy(),
        args.device,
        args.cpu_threads,
    )
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        predictions.to_parquet(args.output, index=False, compression="zstd")
    else:
        predictions.to_csv(args.output, index=False)
    print(f"wrote {len(predictions)} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
