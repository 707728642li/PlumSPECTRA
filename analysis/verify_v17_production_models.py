from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from predict_v17_texture import predict_bundle


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for replay reports and freshly computed hashes. Defaults to a sibling "
            "verification_replay directory so frozen model bundles remain read-only."
        ),
    )
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument("--absolute-tolerance", type=float, default=2e-3)
    parser.add_argument("--relative-sd-tolerance", type=float, default=5e-5)
    args = parser.parse_args()
    models_dir = args.models_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else models_dir.parent / f"{models_dir.name}_verification_replay"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    multimodal_dir = args.multimodal_dir.resolve()
    raw = np.load(multimodal_dir / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy").astype(np.float32)
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    raw_lookup = pd.Series(row_index.index.to_numpy(), index=row_index["sample_id"].astype(str)).to_dict()
    rows = []
    model_hashes = {}
    for model_dir in sorted(
        path.parent for path in models_dir.glob("*/deployment_bundle.json")
    ):
        bundle_path = model_dir / "deployment_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        reference = pd.read_parquet(model_dir / "training_fit_predictions.parquet").head(
            args.sample_limit
        )
        indices = np.asarray([raw_lookup[str(sample_id)] for sample_id in reference["sample_id"]])
        replay = predict_bundle(
            model_dir,
            raw[indices],
            wavelength,
            reference["sample_id"].astype(str).to_numpy(),
            reference["cultivar_ascii"].astype(str).to_numpy(),
            device_name="cpu",
            cpu_threads=8,
        )
        difference = replay["y_pred"].to_numpy() - reference["y_pred"].to_numpy()
        max_abs = float(np.max(np.abs(difference)))
        replay_rmse = float(np.sqrt(np.mean(difference**2)))
        tolerance = max(
            float(args.absolute_tolerance),
            float(args.relative_sd_tolerance) * float(bundle["target_sd"]),
        )
        required_files = [
            bundle_path,
            model_dir / bundle["files"]["network_state"],
            model_dir / bundle["files"]["channel_builder_state"],
            model_dir / bundle["files"]["pls_anchor"],
            model_dir / bundle["files"]["cultivar_offsets"],
        ]
        if not all(path.exists() and path.stat().st_size > 0 for path in required_files):
            raise RuntimeError(f"Missing or empty deployment artifact in {model_dir}")
        if max_abs > tolerance:
            raise RuntimeError(
                f"Serialized replay exceeded tolerance for {bundle['trait']}: "
                f"observed={max_abs}, allowed={tolerance}"
            )
        hashes = {path.name: sha256_file(path) for path in required_files}
        model_hashes[bundle["trait"]] = hashes
        (output_dir / f"{bundle['trait']}_artifact_sha256.json").write_text(
            json.dumps(hashes, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                "trait": bundle["trait"],
                "target": bundle["target"],
                "training_samples": bundle["training_samples"],
                "cultivars": len(bundle["training_cultivars"]),
                "parameters": bundle["trainable_parameters"],
                "replay_samples": len(reference),
                "replay_max_abs_error": max_abs,
                "replay_rmse": replay_rmse,
                "allowed_error": tolerance,
                "passed": True,
            }
        )
    table = pd.DataFrame(rows).sort_values("trait")
    if len(table) != 9 or table["trait"].nunique() != 9:
        raise RuntimeError("Expected exactly nine verified production models")
    table.to_csv(output_dir / "production_verification.csv", index=False)
    report = {
        "verified_models": int(len(table)),
        "serialized_cpu_replay": True,
        "sample_limit_per_model": args.sample_limit,
        "absolute_tolerance_floor": args.absolute_tolerance,
        "relative_target_sd_tolerance": args.relative_sd_tolerance,
        "observed_max_absolute_error": float(table["replay_max_abs_error"].max()),
        "all_passed": bool(table["passed"].all()),
        "artifact_sha256": model_hashes,
    }
    (output_dir / "production_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
