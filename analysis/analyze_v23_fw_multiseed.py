from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_texture_pls_loco import regression_metrics


BASELINES = {
    "global_pls": "y_global_pls",
    "domain_pls": "y_domain_pls",
    "domain_svr": "y_domain_svr",
}


def cluster_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    groups = sorted(frame["cultivar_ascii"].astype(str).unique())
    counts = np.asarray([(frame["cultivar_ascii"].astype(str) == group).sum() for group in groups])
    candidate_sse = np.asarray(
        [
            np.square(
                frame.loc[frame["cultivar_ascii"].astype(str) == group, candidate]
                - frame.loc[frame["cultivar_ascii"].astype(str) == group, "y_true"]
            ).sum()
            for group in groups
        ]
    )
    baseline_sse = np.asarray(
        [
            np.square(
                frame.loc[frame["cultivar_ascii"].astype(str) == group, baseline]
                - frame.loc[frame["cultivar_ascii"].astype(str) == group, "y_true"]
            ).sum()
            for group in groups
        ]
    )
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(groups), size=(iterations, len(groups)))
    boot_n = counts[draw].sum(axis=1)
    candidate_rmse = np.sqrt(candidate_sse[draw].sum(axis=1) / boot_n)
    baseline_rmse = np.sqrt(baseline_sse[draw].sum(axis=1) / boot_n)
    improvement = 100 * (1 - candidate_rmse / baseline_rmse)
    point_candidate = float(np.sqrt(np.mean(np.square(frame[candidate] - frame["y_true"]))))
    point_baseline = float(np.sqrt(np.mean(np.square(frame[baseline] - frame["y_true"]))))
    return {
        "candidate_rmse": point_candidate,
        "baseline_rmse": point_baseline,
        "relative_improvement_pct": float(100 * (1 - point_candidate / point_baseline)),
        "ci95_low": float(np.quantile(improvement, 0.025)),
        "ci95_high": float(np.quantile(improvement, 0.975)),
        "probability_candidate_better": float(np.mean(improvement > 0)),
        "iterations": iterations,
        "cultivar_clusters": len(groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=50_000)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    multiseed_dir = args.multiseed_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_parquet(project / "results" / "v22_integrated" / "baselines" / "predictions.parquet")
    baseline = baseline.loc[baseline["trait"].eq("FW")].copy()

    original_root = project / "results" / "v22_integrated" / "ai" / "FW"
    seed_rows: list[pd.DataFrame] = []
    seed_metric_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        paths = [("original", original_root / f"fold_{fold}" / "predictions.parquet")]
        paths.extend(
            (path.parent.name, path)
            for path in sorted((multiseed_dir / "ai" / "FW" / f"fold_{fold}").glob("seed_repeat_*/predictions.parquet"))
        )
        if len(paths) != 5:
            raise RuntimeError(f"Expected five FW seeds in fold {fold}, found {len(paths)}")
        expected_ids: set[str] | None = None
        fold_baseline = baseline.loc[baseline["outer_fold"].eq(fold)].copy()
        for seed_label, path in paths:
            prediction = pd.read_parquet(path)
            prediction["sample_id"] = prediction["sample_id"].astype(str)
            ids = set(prediction["sample_id"])
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise RuntimeError(f"FW fold {fold} seed sample sets differ")
            prediction = prediction[["sample_id", "cultivar_ascii", "y_true", "y_pred"]].rename(
                columns={"y_pred": "y_deep"}
            )
            prediction["outer_fold"] = fold
            prediction["seed_label"] = seed_label
            seed_rows.append(prediction)
            merged = fold_baseline.merge(
                prediction[["sample_id", "y_deep"]], on="sample_id", how="inner", validate="one_to_one"
            )
            deep_rmse = regression_metrics(merged["y_true"], merged["y_deep"])["rmse"]
            row: dict[str, Any] = {
                "outer_fold": fold,
                "seed_label": seed_label,
                "deep_rmse": deep_rmse,
                "samples": len(merged),
            }
            for baseline_name, column in BASELINES.items():
                baseline_rmse = regression_metrics(merged["y_true"], merged[column])["rmse"]
                row[f"improvement_vs_{baseline_name}_pct"] = 100 * (1 - deep_rmse / baseline_rmse)
            seed_metric_rows.append(row)
            metadata_path = path.with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_rows.append(
                {
                    "outer_fold": fold,
                    "seed_label": seed_label,
                    "repeat": int(metadata["repeat"]),
                    "seed": int(metadata["seed"]),
                    "selected_epoch": int(metadata["selected_epoch"]),
                    "selected_gate": float(metadata["selected_gate"]),
                    "test_rmse": float(metadata["ai_metrics"]["rmse"]),
                }
            )

    all_seed_predictions = pd.concat(seed_rows, ignore_index=True)
    ensemble = (
        all_seed_predictions.groupby(
            ["sample_id", "cultivar_ascii", "outer_fold"], observed=True, as_index=False
        )
        .agg(y_true=("y_true", "first"), y_multiseed=("y_deep", "mean"), seeds=("seed_label", "nunique"))
    )
    if not ensemble["seeds"].eq(5).all():
        raise RuntimeError("Every FW test fruit must have five seed predictions")
    merged_ensemble = baseline.merge(
        ensemble.drop(columns="cultivar_ascii", errors="ignore"),
        on=["sample_id", "outer_fold"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ensemble"),
    )
    if not np.allclose(merged_ensemble["y_true"], merged_ensemble["y_true_ensemble"]):
        raise RuntimeError("Baseline and seed-ensemble truth differ")
    merged_ensemble = merged_ensemble.drop(columns="y_true_ensemble")

    fold_rows: list[dict[str, Any]] = []
    for fold, frame in merged_ensemble.groupby("outer_fold", observed=True):
        candidate_rmse = regression_metrics(frame["y_true"], frame["y_multiseed"])["rmse"]
        row = {"outer_fold": int(fold), "multiseed_rmse": candidate_rmse, "samples": len(frame)}
        for baseline_name, column in BASELINES.items():
            baseline_rmse = regression_metrics(frame["y_true"], frame[column])["rmse"]
            row[f"improvement_vs_{baseline_name}_pct"] = 100 * (1 - candidate_rmse / baseline_rmse)
        fold_rows.append(row)
    fold_metrics = pd.DataFrame(fold_rows)

    contrast_rows = []
    for baseline_name, column in BASELINES.items():
        stable = hashlib.sha256(f"V23|FW|multiseed|{baseline_name}".encode()).digest()
        contrast_rows.append(
            {
                "trait": "FW",
                "candidate": "five_seed_prediction_ensemble",
                "baseline": baseline_name,
                **cluster_bootstrap(
                    merged_ensemble,
                    "y_multiseed",
                    column,
                    args.bootstrap_iterations,
                    int.from_bytes(stable[:4], "little"),
                ),
            }
        )
    contrasts = pd.DataFrame(contrast_rows)

    seed_metrics = pd.DataFrame(seed_metric_rows)
    original_fold = seed_metrics.loc[seed_metrics["seed_label"].eq("original")]
    ensemble_domain = fold_metrics["improvement_vs_domain_pls_pct"]
    original_domain = original_fold["improvement_vs_domain_pls_pct"]
    original_se = float(original_domain.std(ddof=1) / np.sqrt(len(original_domain)))
    ensemble_se = float(ensemble_domain.std(ddof=1) / np.sqrt(len(ensemble_domain)))
    within_fold_seed_sd = float(
        seed_metrics.groupby("outer_fold", observed=True)["improvement_vs_domain_pls_pct"].std(ddof=1).mean()
    )
    summary = {
        "protocol": (
            "FW diagnostic multiseed audit: original plus four new full-pipeline random seeds "
            "per frozen outer fold; five predictions averaged per test fruit"
        ),
        "outer_folds": 5,
        "seeds_per_fold": 5,
        "total_fits": 25,
        "unique_test_fruits": int(merged_ensemble["sample_id"].nunique()),
        "original_fold_mean_improvement_vs_domain_pls_pct": float(original_domain.mean()),
        "original_fold_se_pct_points": original_se,
        "multiseed_fold_mean_improvement_vs_domain_pls_pct": float(ensemble_domain.mean()),
        "multiseed_fold_se_pct_points": ensemble_se,
        "se_reduction_pct": float(100 * (1 - ensemble_se / original_se)),
        "multiseed_fold_wins_vs_domain_pls": int((ensemble_domain > 0).sum()),
        "mean_within_fold_seed_sd_pct_points": within_fold_seed_sd,
        "cluster_bootstrap_domain_pls": contrasts.loc[
            contrasts["baseline"].eq("domain_pls")
        ].iloc[0].to_dict(),
        "interpretation_boundary": (
            "Seed ensembling can reduce optimization variance but cannot create new cultivar clusters; "
            "cultivar-cluster uncertainty remains the inferential boundary."
        ),
    }
    seed_metrics.to_csv(output_dir / "fw_seed_fold_metrics.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(output_dir / "fw_seed_metadata.csv", index=False)
    fold_metrics.to_csv(output_dir / "fw_multiseed_fold_metrics.csv", index=False)
    contrasts.to_csv(output_dir / "fw_multiseed_cluster_bootstrap.csv", index=False)
    merged_ensemble.to_parquet(
        output_dir / "fw_multiseed_predictions.parquet", index=False, compression="zstd"
    )
    (output_dir / "fw_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
