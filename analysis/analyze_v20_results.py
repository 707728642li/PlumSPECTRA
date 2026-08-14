from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_texture_pls_loco import regression_metrics


MODEL_COLUMNS = {
    "global_pls": "y_global_pls",
    "domain_pls": "y_domain_pls",
    "domain_svr": "y_domain_svr",
    "deep": "y_deep",
    "deep_kernel": "y_deep_kernel",
}


def cluster_bootstrap_contrast(
    frame: pd.DataFrame,
    candidate_column: str,
    baseline_column: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    clusters = sorted(frame["cultivar_ascii"].astype(str).unique())
    n = np.asarray([(frame["cultivar_ascii"].astype(str) == cluster).sum() for cluster in clusters])
    candidate_sse = np.asarray(
        [
            np.sum(
                (
                    frame.loc[frame["cultivar_ascii"].astype(str) == cluster, candidate_column]
                    - frame.loc[frame["cultivar_ascii"].astype(str) == cluster, "y_true"]
                )
                ** 2
            )
            for cluster in clusters
        ],
        dtype=float,
    )
    baseline_sse = np.asarray(
        [
            np.sum(
                (
                    frame.loc[frame["cultivar_ascii"].astype(str) == cluster, baseline_column]
                    - frame.loc[frame["cultivar_ascii"].astype(str) == cluster, "y_true"]
                )
                ** 2
            )
            for cluster in clusters
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(clusters), size=(iterations, len(clusters)))
    boot_n = n[draw].sum(axis=1)
    candidate_rmse = np.sqrt(candidate_sse[draw].sum(axis=1) / boot_n)
    baseline_rmse = np.sqrt(baseline_sse[draw].sum(axis=1) / boot_n)
    improvement = 100.0 * (1.0 - candidate_rmse / baseline_rmse)
    delta = baseline_rmse - candidate_rmse
    point_candidate = float(
        np.sqrt(np.mean((frame[candidate_column] - frame["y_true"]) ** 2))
    )
    point_baseline = float(
        np.sqrt(np.mean((frame[baseline_column] - frame["y_true"]) ** 2))
    )
    return {
        "candidate_rmse": point_candidate,
        "baseline_rmse": point_baseline,
        "rmse_delta_baseline_minus_candidate": point_baseline - point_candidate,
        "rmse_delta_ci_low": float(np.quantile(delta, 0.025)),
        "rmse_delta_ci_high": float(np.quantile(delta, 0.975)),
        "relative_rmse_improvement_pct": float(
            100.0 * (1.0 - point_candidate / point_baseline)
        ),
        "relative_improvement_ci_low": float(np.quantile(improvement, 0.025)),
        "relative_improvement_ci_high": float(np.quantile(improvement, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(delta > 0)),
        "cluster_bootstrap_iterations": int(iterations),
        "clusters": int(len(clusters)),
    }


def centered_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float | int]:
    centered_true = frame["y_true"] - frame.groupby("cultivar_ascii", observed=True)["y_true"].transform("mean")
    centered_prediction = frame[prediction_column] - frame.groupby("cultivar_ascii", observed=True)[prediction_column].transform("mean")
    return regression_metrics(centered_true.to_numpy(), centered_prediction.to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=50_000)
    args = parser.parse_args()

    baseline_path = args.baseline_dir.resolve() / "predictions.parquet"
    baseline = pd.read_parquet(baseline_path)
    ai_paths = sorted(args.ai_dir.resolve().glob("*/fold_*/predictions.parquet"))
    if len(ai_paths) != 45:
        raise RuntimeError(f"Expected 45 AI outer-fold prediction files, found {len(ai_paths)}")
    ai = pd.concat([pd.read_parquet(path) for path in ai_paths], ignore_index=True)
    ai = ai.rename(columns={"repeat": "outer_fold", "y_pred": "y_deep"})
    ai["outer_fold"] = ai["outer_fold"].astype(int)
    ai_subset = ai[
        [
            "sample_id",
            "cultivar_ascii",
            "target",
            "outer_fold",
            "y_true",
            "y_deep",
            "deep_residual",
            "residual_gate",
        ]
    ].copy()
    merged = baseline.merge(
        ai_subset,
        on=["sample_id", "cultivar_ascii", "target", "outer_fold"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ai"),
    )
    if not np.allclose(merged["y_true"], merged["y_true_ai"], rtol=0, atol=1e-10):
        raise RuntimeError("AI and baseline truth columns disagree")
    merged = merged.drop(columns="y_true_ai")
    merged["y_deep_kernel"] = 0.5 * merged["y_deep"] + 0.5 * merged["y_domain_svr"]
    merged["deep_svr_error_product"] = (
        (merged["y_deep"] - merged["y_true"])
        * (merged["y_domain_svr"] - merged["y_true"])
    )

    expected_rows = 4_839 * 9
    if len(merged) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} merged predictions, found {len(merged)}")
    counts = merged.groupby("target", observed=True).agg(
        samples=("sample_id", "size"),
        unique_samples=("sample_id", "nunique"),
        folds=("outer_fold", "nunique"),
        cultivars=("cultivar_ascii", "nunique"),
    )
    if not (
        (counts["samples"] == 4_839)
        & (counts["unique_samples"] == 4_839)
        & (counts["folds"] == 5)
        & (counts["cultivars"] == 15)
    ).all():
        raise RuntimeError(f"Outer-fold coverage audit failed:\n{counts}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_dir / "v20_merged_predictions.parquet", index=False, compression="zstd")

    pooled_rows: list[dict[str, Any]] = []
    centered_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    cultivar_rows: list[dict[str, Any]] = []
    complementarity_rows: list[dict[str, Any]] = []
    for (target, trait), frame in merged.groupby(["target", "trait"], observed=True):
        deep_error = frame["y_deep"] - frame["y_true"]
        svr_error = frame["y_domain_svr"] - frame["y_true"]
        complementarity_rows.append(
            {
                "target": target,
                "trait": trait,
                "deep_svr_error_correlation": float(np.corrcoef(deep_error, svr_error)[0, 1]),
                "opposite_error_sign_fraction": float(np.mean(deep_error * svr_error < 0)),
                "mean_error_product": float(np.mean(deep_error * svr_error)),
            }
        )
        for model, column in MODEL_COLUMNS.items():
            pooled_rows.append(
                {"target": target, "trait": trait, "model": model, **regression_metrics(frame["y_true"], frame[column])}
            )
            centered_rows.append(
                {"target": target, "trait": trait, "model": model, **centered_metrics(frame, column)}
            )
            for fold, fold_frame in frame.groupby("outer_fold", observed=True):
                fold_rows.append(
                    {"target": target, "trait": trait, "model": model, "outer_fold": int(fold), **regression_metrics(fold_frame["y_true"], fold_frame[column])}
                )
            for cultivar, cultivar_frame in frame.groupby("cultivar_ascii", observed=True):
                cultivar_rows.append(
                    {"target": target, "trait": trait, "model": model, "cultivar_ascii": str(cultivar), **regression_metrics(cultivar_frame["y_true"], cultivar_frame[column])}
                )
    pd.DataFrame(pooled_rows).to_csv(output_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(centered_rows).to_csv(output_dir / "within_cultivar_centered_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(cultivar_rows).to_csv(output_dir / "cultivar_metrics.csv", index=False)
    pd.DataFrame(complementarity_rows).to_csv(output_dir / "deep_svr_complementarity.csv", index=False)

    contrasts = [
        ("deep", "global_pls"),
        ("deep", "domain_pls"),
        ("deep_kernel", "global_pls"),
        ("deep_kernel", "domain_pls"),
        ("deep_kernel", "domain_svr"),
    ]
    comparison_rows: list[dict[str, Any]] = []
    for (target, trait), frame in merged.groupby(["target", "trait"], observed=True):
        for contrast_index, (candidate, baseline_name) in enumerate(contrasts):
            stable = hashlib.sha256(
                f"V20|{target}|{candidate}|{baseline_name}".encode("utf-8")
            ).digest()
            seed = int.from_bytes(stable[:4], "little")
            comparison_rows.append(
                {
                    "target": target,
                    "trait": trait,
                    "candidate": candidate,
                    "baseline": baseline_name,
                    **cluster_bootstrap_contrast(
                        frame,
                        MODEL_COLUMNS[candidate],
                        MODEL_COLUMNS[baseline_name],
                        args.bootstrap_iterations,
                        seed + contrast_index,
                    ),
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["claim_status"] = np.where(
        comparisons["relative_improvement_ci_low"] > 0,
        "supported_outperformance",
        np.where(
            comparisons["relative_rmse_improvement_pct"] > 0,
            "numerically_lower_not_ci_supported",
            "candidate_not_better",
        ),
    )
    comparisons.to_csv(output_dir / "paired_cluster_bootstrap_comparisons.csv", index=False)

    hybrid_global = comparisons[
        (comparisons["candidate"] == "deep_kernel")
        & (comparisons["baseline"] == "global_pls")
    ]
    hybrid_domain = comparisons[
        (comparisons["candidate"] == "deep_kernel")
        & (comparisons["baseline"] == "domain_pls")
    ]
    hybrid_svr = comparisons[
        (comparisons["candidate"] == "deep_kernel")
        & (comparisons["baseline"] == "domain_svr")
    ]
    gates = {
        "all_nine_hybrid_numerically_better_than_global_pls": bool(
            (hybrid_global["relative_rmse_improvement_pct"] > 0).all()
        ),
        "hybrid_vs_global_pls_ci_supported_traits": int(
            (hybrid_global["relative_improvement_ci_low"] > 0).sum()
        ),
        "all_nine_hybrid_numerically_better_than_domain_pls": bool(
            (hybrid_domain["relative_rmse_improvement_pct"] > 0).all()
        ),
        "hybrid_vs_domain_pls_ci_supported_traits": int(
            (hybrid_domain["relative_improvement_ci_low"] > 0).sum()
        ),
        "hybrid_vs_domain_svr_traits_won": int(
            (hybrid_svr["relative_rmse_improvement_pct"] > 0).sum()
        ),
        "mean_hybrid_improvement_vs_domain_svr_pct": float(
            hybrid_svr["relative_rmse_improvement_pct"].mean()
        ),
        "all_outer_test_predictions_unique": True,
        "external_validation_available": False,
    }
    (output_dir / "success_gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    audit = {
        "protocol": "V20 frozen non-overlapping five-fold final internal audit",
        "manifest_sha256": hashlib.sha256(args.fold_manifest.resolve().read_bytes()).hexdigest(),
        "prediction_rows": int(len(merged)),
        "samples_per_trait": 4_839,
        "traits": 9,
        "cultivars": 15,
        "outer_folds": 5,
        "each_fruit_tested_once_per_trait": True,
        "ensemble_weight_deep": 0.5,
        "ensemble_weight_domain_svr": 0.5,
        "test_labels_used_for_ensemble_weight": False,
        "claim_boundary": "same-session interpolation among 15 registered cultivars",
        "success_gates": gates,
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
