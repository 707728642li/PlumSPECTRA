from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

import train_plumrac_loco as v2
import train_plumrac_v4_phy as v4
from train_texture_pls_loco import preprocess_all


TARGET = "skin_break_displacement_raw_mean"
DEFAULT_HELDOUT = "3.13,Cuihongli,Konglongdan,Weiwang,Weixin"


def stable_seed(*values: object) -> int:
    payload = "|".join(map(str, values)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_adapter(x: np.ndarray, y: np.ndarray, mode: str, slope_prior_strength: float) -> tuple[float, float]:
    if mode == "intercept":
        return 1.0, float(np.mean(y - x))
    if mode != "affine":
        raise ValueError(mode)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    ss_x = float(np.sum(x_centered**2))
    scale = max(ss_x / max(len(x) - 1, 1), np.finfo(float).eps)
    penalty = slope_prior_strength * scale
    slope = float((np.sum(x_centered * y_centered) + penalty) / (ss_x + penalty))
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def config_from_metadata(value: dict[str, object]) -> v2.RACConfig:
    allowed = {field.name for field in fields(v2.RACConfig)}
    return v2.RACConfig(**{key: item for key, item in value.items() if key in allowed})


@torch.no_grad()
def hidden_features(
    model: v4.V4PlumRACNet,
    channel_builder: torch.nn.Module,
    raw: torch.Tensor,
    anchor_z: torch.Tensor,
    augment: bool,
) -> torch.Tensor:
    channels = channel_builder(raw, augment=augment)
    encoded = model.encode(channels)
    pooled = model.pool_features(encoded)
    representation = torch.cat([pooled, anchor_z[:, None]], dim=1)
    hidden = model.trait_tail[1](model.trait_tail[0](representation))
    return hidden


def ridge_prior_update(features: np.ndarray, target: np.ndarray, prior: np.ndarray, penalty: float) -> np.ndarray:
    residual = target - features @ prior
    gram = features @ features.T
    dual = np.linalg.solve(gram + penalty * np.eye(len(features)), residual)
    return prior + features.T @ dual


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heldout", default=DEFAULT_HELDOUT)
    parser.add_argument("--shots", default="5,10,20,40")
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--fixed-gate", type=float, default=0.75)
    parser.add_argument("--ridge-prior", type=float, required=True)
    parser.add_argument("--augmentations", type=int, default=8)
    parser.add_argument("--slope-prior-strength", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.ridge_prior <= 0 or args.augmentations < 0 or args.repeats < 1:
        raise ValueError("Ridge prior and repeats must be positive; augmentations must be non-negative")
    heldout_cultivars = [value.strip() for value in args.heldout.split(",") if value.strip()]
    shots = [int(value) for value in args.shots.split(",")]
    device = torch.device(args.device)
    multimodal = args.multimodal_dir.resolve()
    raw = np.load(multimodal / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(multimodal / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    sample_ids = aligned["sample_id"].astype(str).to_numpy()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    y = pd.to_numeric(aligned[TARGET], errors="coerce").to_numpy(float)
    eligible = aligned["qc_primary_include"].to_numpy(bool) & np.isfinite(y) & (groups != "6.11")
    arrays = preprocess_all(raw, wavelength)
    clean_channels = v2.build_clean_channels(raw, wavelength)

    v4.CHANNEL_SET = "basic"
    v4.ARCHITECTURE = "multiscale"
    v4.MIXSTYLE_P = 0.0
    fold_cache: dict[str, dict[str, np.ndarray]] = {}
    validation_rows: list[dict[str, object]] = []
    for heldout in heldout_cultivars:
        fold_dir = args.run_dir.resolve() / "runs" / "RD" / heldout.replace(" ", "_") / "seed_20260806"
        metadata = json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))
        config = config_from_metadata(metadata["config"])
        source_indices = np.flatnonzero(eligible & (groups != heldout))
        target_indices = np.flatnonzero(eligible & (groups == heldout))
        target_mean = float(np.mean(y[source_indices]))
        target_sd = max(float(np.std(y[source_indices], ddof=1)), 1e-6)
        anchor_model = joblib.load(fold_dir / "pls_anchor.joblib")
        preprocessing = str(metadata["pls_anchor"]["preprocessing"])
        anchor = anchor_model.predict(arrays[preprocessing][target_indices]).ravel().astype(np.float32)
        anchor_z = ((anchor - target_mean) / target_sd).astype(np.float32)

        channel_mean, channel_sd = v2.fit_channel_scaler(clean_channels, source_indices)
        channel_builder = v2.SpectralChannelBuilder(wavelength, channel_mean, channel_sd, config).to(device)
        model = v4.V4PlumRACNet(config.width, config.blocks, config.dropout, config.attention_tail).to(device)
        state = torch.load(fold_dir / "plumrac_state.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        raw_tensor = torch.from_numpy(raw[target_indices]).to(device)
        anchor_tensor = torch.from_numpy(anchor_z).to(device)
        clean_hidden = hidden_features(model, channel_builder, raw_tensor, anchor_tensor, augment=False)
        hidden_views = [clean_hidden.cpu().numpy().astype(np.float64)]
        for augmentation in range(args.augmentations):
            torch.manual_seed(stable_seed(heldout, augmentation, "fewshot_hidden"))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(stable_seed(heldout, augmentation, "fewshot_hidden"))
            hidden_views.append(
                hidden_features(model, channel_builder, raw_tensor, anchor_tensor, augment=True)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        hidden_average = np.mean(np.stack(hidden_views, axis=0), axis=0)
        hidden_clean = hidden_views[0]
        final_layer = model.trait_tail[3]
        prior = np.concatenate(
            [
                final_layer.weight.detach().cpu().numpy().ravel().astype(np.float64),
                final_layer.bias.detach().cpu().numpy().ravel().astype(np.float64),
            ]
        )
        design_clean = np.column_stack([hidden_clean, np.ones(len(hidden_clean))])
        design_average = np.column_stack([hidden_average, np.ones(len(hidden_average))])
        deep_z = design_clean @ prior
        saved = pd.read_parquet(fold_dir / "predictions.parquet").set_index("sample_id")
        saved_deep = saved.loc[sample_ids[target_indices], "deep_residual"].to_numpy(float) / target_sd
        max_deviation = float(np.max(np.abs(deep_z - saved_deep)))
        if max_deviation > 5e-4:
            raise RuntimeError(f"Reconstructed model mismatch for {heldout}: {max_deviation}")
        validation_rows.append(
            {
                "cultivar_ascii": heldout,
                "n": len(target_indices),
                "source_samples": len(source_indices),
                "max_abs_deep_z_reconstruction_error": max_deviation,
                "target_mean_source_only": target_mean,
                "target_sd_source_only": target_sd,
            }
        )
        fold_cache[heldout] = {
            "sample_id": sample_ids[target_indices],
            "truth": y[target_indices].astype(np.float64),
            "anchor": anchor.astype(np.float64),
            "target_sd": np.asarray([target_sd], dtype=np.float64),
            "design_clean": design_clean,
            "design_average": design_average,
            "prior": prior,
            "source_ai": anchor.astype(np.float64) + args.fixed_gate * deep_z * target_sd,
        }

    repeat_rows: list[dict[str, object]] = []
    cultivar_rows: list[dict[str, object]] = []
    strategies = ["PLSR_intercept", "PLSR_affine", "AI_intercept", "AI_lastlayer_L2SP"]
    for shot in shots:
        for repeat in range(1, args.repeats + 1):
            pooled: dict[str, list[np.ndarray]] = {strategy: [] for strategy in strategies}
            pooled_truth: list[np.ndarray] = []
            for heldout in heldout_cultivars:
                cache = fold_cache[heldout]
                truth = cache["truth"]
                anchor = cache["anchor"]
                source_ai = cache["source_ai"]
                n = len(truth)
                if shot >= n - 2:
                    raise ValueError(f"Shot count {shot} leaves too few evaluation fruits for {heldout}")
                rng = np.random.default_rng(stable_seed(TARGET, heldout, shot, repeat, 20260806))
                calibration = np.sort(rng.choice(n, size=shot, replace=False))
                evaluation = np.setdiff1d(np.arange(n), calibration)
                plsr_intercept = fit_adapter(
                    anchor[calibration], truth[calibration], "intercept", args.slope_prior_strength
                )
                plsr_affine = fit_adapter(
                    anchor[calibration], truth[calibration], "affine", args.slope_prior_strength
                )
                ai_intercept = fit_adapter(
                    source_ai[calibration], truth[calibration], "intercept", args.slope_prior_strength
                )
                target_deep = (
                    (truth[calibration] - anchor[calibration])
                    / (args.fixed_gate * float(cache["target_sd"][0]))
                )
                adapted_prior = ridge_prior_update(
                    cache["design_average"][calibration],
                    target_deep,
                    cache["prior"],
                    args.ridge_prior,
                )
                predictions = {
                    "PLSR_intercept": plsr_intercept[0] * anchor[evaluation] + plsr_intercept[1],
                    "PLSR_affine": plsr_affine[0] * anchor[evaluation] + plsr_affine[1],
                    "AI_intercept": ai_intercept[0] * source_ai[evaluation] + ai_intercept[1],
                    "AI_lastlayer_L2SP": anchor[evaluation]
                    + args.fixed_gate
                    * float(cache["target_sd"][0])
                    * (cache["design_clean"][evaluation] @ adapted_prior),
                }
                pooled_truth.append(truth[evaluation])
                for strategy, prediction in predictions.items():
                    pooled[strategy].append(prediction)
                    cultivar_rows.append(
                        {
                            "cultivar_ascii": heldout,
                            "shots": shot,
                            "repeat": repeat,
                            "strategy": strategy,
                            "n_evaluation": len(evaluation),
                            "rmse": rmse(truth[evaluation], prediction),
                        }
                    )
            truth_all = np.concatenate(pooled_truth)
            for strategy, parts in pooled.items():
                prediction_all = np.concatenate(parts)
                repeat_rows.append(
                    {
                        "shots": shot,
                        "repeat": repeat,
                        "strategy": strategy,
                        "n_evaluation": len(truth_all),
                        "rmse": rmse(truth_all, prediction_all),
                        "r2": float(r2_score(truth_all, prediction_all)),
                    }
                )

    repeats = pd.DataFrame(repeat_rows)
    cultivar_metrics = pd.DataFrame(cultivar_rows)
    summary = (
        repeats.groupby(["shots", "strategy"], observed=True)
        .agg(
            repeats=("repeat", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            rmse_ci025=("rmse", lambda value: value.quantile(0.025)),
            rmse_ci975=("rmse", lambda value: value.quantile(0.975)),
            r2_mean=("r2", "mean"),
        )
        .reset_index()
    )
    comparison_rows = []
    for shot in shots:
        shot_summary = summary.loc[summary["shots"] == shot].set_index("strategy")
        neural = float(shot_summary.loc["AI_lastlayer_L2SP", "rmse_mean"])
        best_plsr = min(
            float(shot_summary.loc["PLSR_intercept", "rmse_mean"]),
            float(shot_summary.loc["PLSR_affine", "rmse_mean"]),
        )
        comparison_rows.append(
            {
                "shots": shot,
                "neural_rmse": neural,
                "best_plsr_adapter_rmse": best_plsr,
                "neural_improvement_vs_best_plsr_pct": 100.0 * (best_plsr - neural) / best_plsr,
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repeats.to_parquet(output / "fewshot_repeat_metrics.parquet", index=False, compression="zstd")
    cultivar_metrics.to_parquet(output / "fewshot_cultivar_metrics.parquet", index=False, compression="zstd")
    summary.to_csv(output / "fewshot_summary.csv", index=False)
    comparison.to_csv(output / "neural_vs_plsr.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(output / "reconstruction_validation.csv", index=False)
    report = {
        "target": TARGET,
        "heldout_cultivars": heldout_cultivars,
        "shots": shots,
        "repeats": args.repeats,
        "ridge_prior": args.ridge_prior,
        "augmentations_averaged_for_calibration": args.augmentations + 1,
        "fixed_gate": args.fixed_gate,
        "adapted_parameters": 145,
        "frozen_parameters": 142140,
        "identical_calibration_fruits_across_strategies": True,
        "calibration_fruits_excluded_from_evaluation": True,
        "heldout_labels_used_before_calibration": False,
        "comparison": comparison.to_dict("records"),
        "provenance_sha256": {
            "evaluator": sha256_file(Path(__file__).resolve()),
            "qc_ledger": sha256_file(args.qc_ledger.resolve()),
            "nir_absorbance": sha256_file(multimodal / "nir_c_absorbance.npy"),
        },
        "claim_boundary": "Development cultivars only; any selected adapter and prior require all-cultivar confirmation.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
