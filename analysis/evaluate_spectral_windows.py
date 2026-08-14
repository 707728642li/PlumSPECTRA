from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from train_pls_loco import preprocess_all


TARGET_LABELS = {
    "fruit_weight_g": "Fruit weight",
    "soluble_solids_pct": "Soluble solids",
    "ph": "pH",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-nm", type=float, default=50.0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    multimodal_dir = args.multimodal_dir.resolve()
    x = np.load(multimodal_dir / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy")
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    master = pd.read_parquet(multimodal_dir / "master_samples.parquet").set_index("sample_id")
    aligned = master.loc[row_index["sample_id"]].reset_index()
    arrays = preprocess_all(x, wavelength)
    sample_to_row = {sample_id: index for index, sample_id in enumerate(aligned["sample_id"])}
    saved_predictions = pd.read_parquet(args.predictions)

    start = np.floor(float(wavelength.min()) / args.window_nm) * args.window_nm
    edges = np.arange(start, float(wavelength.max()) + args.window_nm, args.window_nm)
    windows: list[tuple[float, float, np.ndarray]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (wavelength >= low) & ((wavelength < high) if high < edges[-1] else (wavelength <= high))
        if mask.any():
            windows.append((max(low, float(wavelength.min())), min(high, float(wavelength.max())), mask))

    detail_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    verification_error = 0.0
    for model_path in sorted(args.model_dir.rglob("*.joblib")):
        payload = joblib.load(model_path)
        target = payload["target"]
        cultivar = payload["heldout_cultivar"]
        preprocessing = payload["preprocessing"]
        estimator = payload["estimator"]
        fold_predictions = saved_predictions.loc[
            (saved_predictions["target"] == target)
            & (saved_predictions["cultivar_ascii"] == cultivar)
        ].copy()
        indices = np.asarray([sample_to_row[sample_id] for sample_id in fold_predictions["sample_id"]], dtype=int)
        x_test = arrays[preprocessing][indices]
        y_true = fold_predictions["y_true"].to_numpy(float)
        baseline = estimator.predict(x_test).ravel()
        verification_error = max(
            verification_error,
            float(np.max(np.abs(baseline - fold_predictions["y_pred"].to_numpy(float)))),
        )
        baseline_mse = float(mean_squared_error(y_true, baseline))
        for window_index, (low, high, mask) in enumerate(windows, start=1):
            ablated = x_test.copy()
            ablated[:, mask] = estimator._x_mean[mask]
            prediction = estimator.predict(ablated).ravel()
            ablated_mse = float(mean_squared_error(y_true, prediction))
            detail_rows.append(
                {
                    "target": target,
                    "heldout_cultivar": cultivar,
                    "preprocessing": preprocessing,
                    "n": len(y_true),
                    "window_index": window_index,
                    "window_low_nm": low,
                    "window_high_nm": high,
                    "window_center_nm": (low + high) / 2,
                    "baseline_mse": baseline_mse,
                    "ablated_mse": ablated_mse,
                    "delta_mse": ablated_mse - baseline_mse,
                    "mean_abs_prediction_shift": float(np.mean(np.abs(prediction - baseline))),
                }
            )
            sample_rows.extend(
                {
                    "sample_id": sample_id,
                    "target": target,
                    "heldout_cultivar": cultivar,
                    "window_index": window_index,
                    "y_true": truth,
                    "baseline_prediction": base,
                    "ablated_prediction": pred,
                }
                for sample_id, truth, base, pred in zip(
                    fold_predictions["sample_id"], y_true, baseline, prediction
                )
            )

    detail = pd.DataFrame(detail_rows)
    sample = pd.DataFrame(sample_rows)
    aggregate_rows: list[dict[str, object]] = []
    for (target, window_index), group in sample.groupby(["target", "window_index"], sort=True):
        y_true = group["y_true"].to_numpy(float)
        baseline = group["baseline_prediction"].to_numpy(float)
        ablated = group["ablated_prediction"].to_numpy(float)
        descriptor = detail.loc[
            (detail["target"] == target) & (detail["window_index"] == window_index)
        ].iloc[0]
        baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline)))
        ablated_rmse = float(np.sqrt(mean_squared_error(y_true, ablated)))
        fold_delta = detail.loc[
            (detail["target"] == target) & (detail["window_index"] == window_index), "delta_mse"
        ]
        aggregate_rows.append(
            {
                "target": target,
                "window_index": int(window_index),
                "window_low_nm": descriptor["window_low_nm"],
                "window_high_nm": descriptor["window_high_nm"],
                "window_center_nm": descriptor["window_center_nm"],
                "n": len(group),
                "baseline_rmse": baseline_rmse,
                "ablated_rmse": ablated_rmse,
                "delta_rmse": ablated_rmse - baseline_rmse,
                "relative_delta_rmse_pct": 100 * (ablated_rmse - baseline_rmse) / baseline_rmse,
                "median_fold_delta_mse": float(fold_delta.median()),
                "positive_fold_fraction": float((fold_delta > 0).mean()),
                "mean_abs_prediction_shift": float(np.mean(np.abs(ablated - baseline))),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["importance_rank"] = aggregate.groupby("target")["delta_rmse"].rank(
        ascending=False, method="min"
    ).astype(int)

    detail.to_parquet(output_dir / "window_ablation_fold_metrics.parquet", index=False)
    aggregate.to_csv(output_dir / "window_ablation_summary.csv", index=False)
    top = aggregate.sort_values(["target", "importance_rank"]).groupby("target").head(6)
    top.to_csv(output_dir / "top_spectral_windows.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.6), sharex=True, constrained_layout=True)
    for axis, target in zip(axes, TARGET_LABELS):
        frame = aggregate.loc[aggregate["target"] == target].sort_values("window_center_nm")
        colors = np.where(frame["delta_rmse"] >= 0, "#2f6f9f", "#bf5b5b")
        axis.bar(
            frame["window_center_nm"],
            frame["relative_delta_rmse_pct"],
            width=args.window_nm * 0.82,
            color=colors,
            edgecolor="white",
            linewidth=0.4,
        )
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_ylabel("$\\Delta$RMSE (%)")
        axis.set_title(TARGET_LABELS[target], loc="left", fontweight="bold")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[-1].set_xlabel("Wavelength window center (nm)")
    fig.suptitle("Held-out-cultivar PLS spectral-window ablation", fontsize=13, fontweight="bold")
    for suffix in ["png", "pdf"]:
        fig.savefig(figure_dir / f"figS_spectral_window_ablation.{suffix}", dpi=300)
    plt.close(fig)

    summary = {
        "method": "Each 50-nm window in the held-out cultivar was replaced by the corresponding outer-training PLS feature mean; no test labels were used.",
        "window_nm": args.window_nm,
        "model_prediction_verification_max_abs_error": verification_error,
        "top_windows": top.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
