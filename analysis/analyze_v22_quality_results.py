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
    "plumspectra": "y_plumspectra",
}


def centered_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float | int]:
    truth = frame["y_true"] - frame.groupby("cultivar_ascii", observed=True)["y_true"].transform("mean")
    prediction = frame[prediction_column] - frame.groupby("cultivar_ascii", observed=True)[prediction_column].transform("mean")
    return regression_metrics(truth.to_numpy(), prediction.to_numpy())


def cluster_bootstrap_contrast(
    frame: pd.DataFrame,
    candidate_column: str,
    baseline_column: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    clusters = sorted(frame["cultivar_ascii"].astype(str).unique())
    n = np.asarray([(frame["cultivar_ascii"].astype(str) == cluster).sum() for cluster in clusters])
    candidate_sse = np.asarray([
        np.sum((frame.loc[frame["cultivar_ascii"].astype(str) == cluster, candidate_column]
                - frame.loc[frame["cultivar_ascii"].astype(str) == cluster, "y_true"]) ** 2)
        for cluster in clusters
    ])
    baseline_sse = np.asarray([
        np.sum((frame.loc[frame["cultivar_ascii"].astype(str) == cluster, baseline_column]
                - frame.loc[frame["cultivar_ascii"].astype(str) == cluster, "y_true"]) ** 2)
        for cluster in clusters
    ])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(clusters), size=(iterations, len(clusters)))
    boot_n = n[draw].sum(axis=1)
    candidate_rmse = np.sqrt(candidate_sse[draw].sum(axis=1) / boot_n)
    baseline_rmse = np.sqrt(baseline_sse[draw].sum(axis=1) / boot_n)
    improvement = 100.0 * (1.0 - candidate_rmse / baseline_rmse)
    delta = baseline_rmse - candidate_rmse
    point_candidate = float(np.sqrt(np.mean((frame[candidate_column] - frame["y_true"]) ** 2)))
    point_baseline = float(np.sqrt(np.mean((frame[baseline_column] - frame["y_true"]) ** 2)))
    return {
        "candidate_rmse": point_candidate,
        "baseline_rmse": point_baseline,
        "rmse_delta_baseline_minus_candidate": point_baseline - point_candidate,
        "rmse_delta_ci_low": float(np.quantile(delta, 0.025)),
        "rmse_delta_ci_high": float(np.quantile(delta, 0.975)),
        "relative_rmse_improvement_pct": float(100.0 * (1.0 - point_candidate / point_baseline)),
        "relative_improvement_ci_low": float(np.quantile(improvement, 0.025)),
        "relative_improvement_ci_high": float(np.quantile(improvement, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(delta > 0)),
        "cluster_bootstrap_iterations": int(iterations),
        "clusters": int(len(clusters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=50_000)
    parser.add_argument(
        "--kernel-eligibility-margin-pct",
        type=float,
        default=5.0,
        help=(
            "Use the fixed 0.5 deep/kernel blend only when training-internal domain-SVR "
            "CV RMSE is within this percentage of domain-PLSR; otherwise use deep only."
        ),
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.fold_manifest.resolve(), dtype={"sample_id": str, "cultivar_ascii": str})
    baseline = pd.read_parquet(args.baseline_dir.resolve() / "predictions.parquet")
    ai_paths = sorted(args.ai_dir.resolve().glob("*/fold_*/predictions.parquet"))
    if len(ai_paths) != 15:
        raise RuntimeError(f"Expected 15 AI prediction files, found {len(ai_paths)}")
    ai = pd.concat([pd.read_parquet(path) for path in ai_paths], ignore_index=True)
    ai = ai.rename(columns={"repeat": "outer_fold", "y_pred": "y_deep"})
    ai["outer_fold"] = ai["outer_fold"].astype(int)
    ai_subset = ai[[
        "sample_id", "cultivar_ascii", "target", "outer_fold", "y_true",
        "y_deep", "deep_residual", "residual_gate",
    ]].copy()
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

    selection_rows = []
    for metadata_path in sorted(args.baseline_dir.resolve().glob("*/fold_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pls_cv = float(metadata["domain_pls_choice"]["domain_cv_rmse"])
        svr_cv = float(metadata["domain_svr_choice"]["domain_cv_rmse"])
        degradation_pct = 100.0 * (svr_cv / pls_cv - 1.0)
        kernel_weight = 0.5 if degradation_pct <= args.kernel_eligibility_margin_pct else 0.0
        selection_rows.append({
            "target": metadata["target"],
            "trait": metadata["trait"],
            "outer_fold": int(metadata["outer_fold"]),
            "domain_pls_inner_cv_rmse": pls_cv,
            "domain_svr_inner_cv_rmse": svr_cv,
            "domain_svr_degradation_vs_pls_pct": degradation_pct,
            "kernel_eligibility_margin_pct": float(args.kernel_eligibility_margin_pct),
            "deep_weight": 1.0 - kernel_weight,
            "kernel_weight": kernel_weight,
            "selected_variant": "deep_kernel" if kernel_weight > 0 else "deep",
            "selection_uses_outer_test_labels": False,
        })
    selection = pd.DataFrame(selection_rows)
    if len(selection) != 15:
        raise RuntimeError(f"Expected 15 train-internal branch selections, found {len(selection)}")
    merged = merged.merge(
        selection[["target", "outer_fold", "deep_weight", "kernel_weight", "selected_variant"]],
        on=["target", "outer_fold"], how="left", validate="many_to_one",
    )
    merged["y_plumspectra"] = (
        merged["deep_weight"] * merged["y_deep"]
        + merged["kernel_weight"] * merged["y_domain_svr"]
    )

    expected_rows = len(manifest) * 3
    if len(merged) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} merged rows, found {len(merged)}")
    coverage = merged.groupby("target", observed=True).agg(
        samples=("sample_id", "size"),
        unique_samples=("sample_id", "nunique"),
        folds=("outer_fold", "nunique"),
        cultivars=("cultivar_ascii", "nunique"),
    )
    if not (
        (coverage["samples"] == len(manifest))
        & (coverage["unique_samples"] == len(manifest))
        & (coverage["folds"] == 5)
        & (coverage["cultivars"] == 15)
    ).all():
        raise RuntimeError(f"Coverage audit failed:\n{coverage}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_dir / "v22_quality_merged_predictions.parquet", index=False, compression="zstd")
    selection.to_csv(output_dir / "train_internal_branch_selection.csv", index=False)

    pooled_rows: list[dict[str, Any]] = []
    centered_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    cultivar_rows: list[dict[str, Any]] = []
    for (target, trait), frame in merged.groupby(["target", "trait"], observed=True):
        for model, column in MODEL_COLUMNS.items():
            pooled_rows.append({"target": target, "trait": trait, "model": model, **regression_metrics(frame["y_true"], frame[column])})
            centered_rows.append({"target": target, "trait": trait, "model": model, **centered_metrics(frame, column)})
            for fold, fold_frame in frame.groupby("outer_fold", observed=True):
                fold_rows.append({
                    "target": target, "trait": trait, "model": model,
                    "outer_fold": int(fold),
                    **regression_metrics(fold_frame["y_true"], fold_frame[column]),
                })
            for cultivar, cultivar_frame in frame.groupby("cultivar_ascii", observed=True):
                cultivar_rows.append({
                    "target": target, "trait": trait, "model": model,
                    "cultivar_ascii": str(cultivar),
                    **regression_metrics(cultivar_frame["y_true"], cultivar_frame[column]),
                })
    pd.DataFrame(pooled_rows).to_csv(output_dir / "pooled_metrics.csv", index=False)
    pd.DataFrame(centered_rows).to_csv(output_dir / "within_cultivar_centered_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(cultivar_rows).to_csv(output_dir / "cultivar_metrics.csv", index=False)

    comparisons = []
    contrasts = [
        ("deep", "global_pls"), ("deep", "domain_pls"),
        ("deep_kernel", "global_pls"), ("deep_kernel", "domain_pls"),
        ("deep_kernel", "domain_svr"),
        ("plumspectra", "global_pls"),
        ("plumspectra", "domain_pls"),
        ("plumspectra", "domain_svr"),
    ]
    for (target, trait), frame in merged.groupby(["target", "trait"], observed=True):
        for candidate, baseline_name in contrasts:
            stable = hashlib.sha256(f"V22|{target}|{candidate}|{baseline_name}".encode()).digest()
            comparisons.append({
                "target": target, "trait": trait,
                "candidate": candidate, "baseline": baseline_name,
                **cluster_bootstrap_contrast(
                    frame, MODEL_COLUMNS[candidate], MODEL_COLUMNS[baseline_name],
                    args.bootstrap_iterations, int.from_bytes(stable[:4], "little"),
                ),
            })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["claim_status"] = np.where(
        comparison_frame["relative_improvement_ci_low"] > 0,
        "supported_outperformance",
        np.where(
            comparison_frame["relative_rmse_improvement_pct"] > 0,
            "numerically_lower_not_ci_supported",
            "candidate_not_better",
        ),
    )
    comparison_frame.to_csv(output_dir / "paired_cluster_bootstrap_comparisons.csv", index=False)

    audit = {
        "protocol": "V22 conventional-quality frozen five-fold internal audit",
        "model_name": "PlumSPECTRA",
        "manifest_sha256": hashlib.sha256(args.fold_manifest.resolve().read_bytes()).hexdigest(),
        "prediction_rows": int(len(merged)),
        "samples_per_trait": int(len(manifest)),
        "traits": 3,
        "cultivars": 15,
        "outer_folds": 5,
        "each_fruit_tested_once_per_trait": True,
        "eligible_kernel_blend_weights": {"deep": 0.5, "domain_svr": 0.5},
        "final_variant_selection": "training-internal CV; deep-only when domain-SVR exceeds the eligibility margin",
        "test_labels_used_for_ensemble_weight": False,
        "kernel_eligibility_margin_pct": float(args.kernel_eligibility_margin_pct),
        "selected_variants_by_trait": {
            trait: sorted(group["selected_variant"].unique().tolist())
            for trait, group in selection.groupby("trait", observed=True)
        },
        "claim_boundary": "same-session interpolation among 15 registered cultivars",
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
