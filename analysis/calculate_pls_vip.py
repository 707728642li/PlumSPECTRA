from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v2_registry import trait_registry


TARGET_LABELS = {
    "fruit_weight_g": "Fruit weight",
    "soluble_solids_pct": "Soluble solids",
    "ph": "pH",
}


def vip_scores(estimator: object) -> np.ndarray:
    """Return standard PLS variable-importance-in-projection scores."""
    scores = np.asarray(estimator.x_scores_, dtype=float)
    weights = np.asarray(estimator.x_weights_, dtype=float)
    loadings = np.asarray(estimator.y_loadings_, dtype=float).reshape(-1)
    explained_y = np.sum(scores**2, axis=0) * loadings**2
    weight_norm = np.sum(weights**2, axis=0)
    contribution = (weights**2 / np.maximum(weight_norm, 1e-15)) @ explained_y
    return np.sqrt(weights.shape[0] * contribution / np.maximum(explained_y.sum(), 1e-15))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wavelengths", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-nm", type=float, default=50.0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    wavelength = np.load(args.wavelengths)

    rows: list[dict[str, object]] = []
    for path in sorted(args.model_dir.rglob("*.joblib")):
        payload = joblib.load(path)
        vip = vip_scores(payload["estimator"])
        if len(vip) != len(wavelength):
            raise ValueError(f"VIP length mismatch in {path}")
        rows.extend(
            {
                "target": payload["target"],
                "heldout_cultivar": payload["heldout_cultivar"],
                "preprocessing": payload["preprocessing"],
                "n_components": payload["n_components"],
                "wavelength_nm": float(wl),
                "vip": float(value),
            }
            for wl, value in zip(wavelength, vip)
        )
    fold_scores = pd.DataFrame(rows)
    wavelength_summary = (
        fold_scores.groupby(["target", "wavelength_nm"], as_index=False)
        .agg(
            vip_median=("vip", "median"),
            vip_q25=("vip", lambda value: value.quantile(0.25)),
            vip_q75=("vip", lambda value: value.quantile(0.75)),
            vip_mean=("vip", "mean"),
            fold_fraction_vip_gt_1=("vip", lambda value: float((value > 1).mean())),
        )
    )

    start = np.floor(float(wavelength.min()) / args.window_nm) * args.window_nm
    edges = np.arange(start, float(wavelength.max()) + args.window_nm, args.window_nm)
    window_rows: list[dict[str, object]] = []
    for target, group in fold_scores.groupby("target", sort=True):
        for low, high in zip(edges[:-1], edges[1:]):
            selected = group.loc[
                (group["wavelength_nm"] >= low)
                & ((group["wavelength_nm"] < high) if high < edges[-1] else (group["wavelength_nm"] <= high))
            ]
            if selected.empty:
                continue
            per_fold = selected.groupby("heldout_cultivar")["vip"].mean()
            window_rows.append(
                {
                    "target": target,
                    "window_low_nm": max(low, float(wavelength.min())),
                    "window_high_nm": min(high, float(wavelength.max())),
                    "window_center_nm": (max(low, float(wavelength.min())) + min(high, float(wavelength.max()))) / 2,
                    "vip_median_across_folds": float(per_fold.median()),
                    "vip_q25_across_folds": float(per_fold.quantile(0.25)),
                    "vip_q75_across_folds": float(per_fold.quantile(0.75)),
                    "fold_fraction_mean_vip_gt_1": float((per_fold > 1).mean()),
                }
            )
    window_summary = pd.DataFrame(window_rows)
    window_summary["importance_rank"] = window_summary.groupby("target")["vip_median_across_folds"].rank(
        ascending=False, method="min"
    ).astype(int)
    top_windows = (
        window_summary.sort_values(["target", "importance_rank"])
        .groupby("target", as_index=False)
        .head(5)
    )

    fold_scores.to_parquet(output_dir / "pls_vip_by_fold.parquet", index=False)
    wavelength_summary.to_csv(output_dir / "pls_vip_wavelength_summary.csv", index=False)
    window_summary.to_csv(output_dir / "pls_vip_window_summary.csv", index=False)
    top_windows.to_csv(output_dir / "top_pls_vip_windows.csv", index=False)

    registry = trait_registry()
    registered_order = registry["target"].tolist()
    observed = set(fold_scores["target"].unique())
    targets = [target for target in registered_order if target in observed]
    targets.extend(sorted(observed - set(targets)))
    registered_labels = dict(zip(registry["target"], registry["abbreviation"], strict=True))
    labels = {**TARGET_LABELS, **registered_labels}
    fig, axes = plt.subplots(
        len(targets),
        1,
        figsize=(9.0, max(3.0, 2.35 * len(targets))),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, target in zip(axes.flat, targets):
        frame = wavelength_summary.loc[wavelength_summary["target"] == target].sort_values("wavelength_nm")
        x = frame["wavelength_nm"].to_numpy(float)
        median = frame["vip_median"].to_numpy(float)
        q25 = frame["vip_q25"].to_numpy(float)
        q75 = frame["vip_q75"].to_numpy(float)
        axis.fill_between(x, q25, q75, color="#80a8c2", alpha=0.28, linewidth=0)
        axis.plot(x, median, color="#1f5a7a", linewidth=1.4)
        axis.axhline(1.0, color="#9a4d4d", linewidth=0.9, linestyle="--")
        axis.set_ylabel("PLS-VIP")
        axis.set_title(labels.get(target, target), loc="left", fontweight="bold")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[-1, 0].set_xlabel("Wavelength (nm)")
    for suffix in ["png", "pdf"]:
        fig.savefig(figure_dir / f"figS_pls_vip_stability.{suffix}", dpi=300)
    plt.close(fig)

    summary = {
        "method": "Standard PLS VIP computed independently from each outer-training model; curves summarize the 16 held-out-cultivar folds.",
        "interpretation_limit": "VIP is model-based importance under collinearity and is not a causal or compound-specific assignment.",
        "top_windows": top_windows.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
