from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from v2_registry import cultivar_registry


DEFAULT_DEVELOPMENT_CODES = ["L313", "CHL", "KLD", "WW", "WX"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--fold-errors", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-codes", default=",".join(DEFAULT_DEVELOPMENT_CODES))
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--neighbors", type=int, default=10)
    args = parser.parse_args()

    multimodal = args.multimodal_dir.resolve()
    raw = np.load(multimodal / "nir_c_absorbance.npy").astype(np.float32)
    row_index = pd.read_csv(multimodal / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    eligible = aligned["qc_analysis_include"].to_numpy(bool)
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    sample_sd = raw.std(axis=1, ddof=1, keepdims=True)
    snv = (raw - raw.mean(axis=1, keepdims=True)) / np.where(sample_sd > 1e-8, sample_sd, 1.0)

    code_table = cultivar_registry().set_index("cultivar_code")
    development_codes = [value.strip().upper() for value in args.development_codes.split(",")]
    rows: list[dict[str, object]] = []
    for code in development_codes:
        heldout = str(code_table.loc[code, "cultivar_ascii"])
        source_indices = np.flatnonzero(eligible & (groups != heldout))
        target_indices = np.flatnonzero(eligible & (groups == heldout))
        pca = PCA(n_components=args.components, svd_solver="randomized", random_state=20260806)
        source_scores = pca.fit_transform(snv[source_indices])
        target_scores = pca.transform(snv[target_indices])
        scale = np.maximum(source_scores.std(axis=0, ddof=1), 1e-8)
        source_z = source_scores / scale
        target_z = target_scores / scale

        source_group_values = groups[source_indices]
        centroids = np.vstack(
            [source_z[source_group_values == cultivar].mean(axis=0) for cultivar in sorted(np.unique(source_group_values))]
        )
        target_centroid = target_z.mean(axis=0)
        centroid_distances = np.linalg.norm(centroids - target_centroid, axis=1) / np.sqrt(args.components)
        neighbors = NearestNeighbors(n_neighbors=args.neighbors, metric="euclidean", n_jobs=1).fit(source_z)
        neighbor_distances, _ = neighbors.kneighbors(target_z)

        source_raw_centroids = np.vstack(
            [snv[source_indices][source_group_values == cultivar].mean(axis=0) for cultivar in sorted(np.unique(source_group_values))]
        )
        target_raw_centroid = snv[target_indices].mean(axis=0)
        outside = (target_raw_centroid < source_raw_centroids.min(axis=0)) | (
            target_raw_centroid > source_raw_centroids.max(axis=0)
        )
        rows.append(
            {
                "cultivar_code": code,
                "cultivar_ascii": heldout,
                "target_samples": int(len(target_indices)),
                "pca_components": args.components,
                "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                "nearest_source_centroid_distance": float(centroid_distances.min()),
                "mean_source_centroid_distance": float(centroid_distances.mean()),
                "median_sample_knn_distance": float(np.median(neighbor_distances.mean(axis=1)) / np.sqrt(args.components)),
                "q90_sample_knn_distance": float(np.quantile(neighbor_distances.mean(axis=1), 0.90) / np.sqrt(args.components)),
                "spectral_centroid_extrapolation_fraction": float(np.mean(outside)),
            }
        )
    shift = pd.DataFrame(rows)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shift.to_csv(output / "spectral_shift_by_development_cultivar.csv", index=False)
    associations: list[dict[str, object]] = []
    joined = None
    if args.fold_errors:
        folds = pd.read_csv(args.fold_errors.resolve())
        folds = folds.loc[folds["scope"] == "development"].copy()
        joined = folds.merge(shift, on="cultivar_code", validate="many_to_one")
        joined.to_csv(output / "spectral_shift_with_fold_errors.csv", index=False)
        for (trait, model), group in joined.groupby(["trait", "model"], observed=True):
            for distance in [
                "nearest_source_centroid_distance",
                "median_sample_knn_distance",
                "spectral_centroid_extrapolation_fraction",
            ]:
                associations.append(
                    {
                        "trait": trait,
                        "model": model,
                        "distance": distance,
                        "folds": int(len(group)),
                        "spearman_with_rmse": float(spearmanr(group[distance], group["rmse"]).statistic),
                        "spearman_with_absolute_bias": float(
                            spearmanr(group[distance], group["bias"].abs()).statistic
                        ),
                    }
                )
        pd.DataFrame(associations).to_csv(output / "spectral_shift_error_associations.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    sns.barplot(data=shift, x="cultivar_code", y="nearest_source_centroid_distance", color="#3F7CA6", ax=axes[0])
    axes[0].set_title("A  Nearest source-domain centroid", loc="left", fontweight="bold")
    axes[0].set_ylabel("Standardized PCA distance")
    axes[0].set_xlabel("Development cultivar")
    sns.barplot(data=shift, x="cultivar_code", y="median_sample_knn_distance", color="#C58C2A", ax=axes[1])
    axes[1].set_title("B  Sample-level applicability distance", loc="left", fontweight="bold")
    axes[1].set_ylabel("Median standardized 10-NN distance")
    axes[1].set_xlabel("Development cultivar")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig_v3_spectral_shift.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "status": "development-fold spectral applicability diagnostic",
        "development_codes": development_codes,
        "preprocessing": "sample-wise SNV; PCA fitted separately on each outer source set",
        "target_label_use": "none for spectral distances; target labels only enter the post-hoc error association table",
        "interpretation_limit": "Five development domains are sufficient for diagnosis but not a precise correlation estimate.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
