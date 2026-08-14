from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from train_texture_pls_loco import regression_metrics


MODELS = {
    "global_pls": "y_global_pls",
    "domain_pls": "y_domain_pls",
    "domain_svr": "y_domain_svr",
    "deep": "y_deep",
    "deep_kernel": "y_deep_kernel",
}


def batch_macro_rmse(frame: pd.DataFrame, prediction: str) -> float:
    values = []
    for _, batch in frame.groupby("batch_id", observed=True):
        values.append(float(np.sqrt(np.mean((batch[prediction] - batch["y_true"]) ** 2))))
    return float(np.mean(values))


def batch_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    batches = sorted(frame["batch_id"].unique())
    candidate_rmse = np.asarray(
        [
            np.sqrt(
                np.mean(
                    (
                        frame.loc[frame["batch_id"] == batch, candidate]
                        - frame.loc[frame["batch_id"] == batch, "y_true"]
                    )
                    ** 2
                )
            )
            for batch in batches
        ]
    )
    baseline_rmse = np.asarray(
        [
            np.sqrt(
                np.mean(
                    (
                        frame.loc[frame["batch_id"] == batch, baseline]
                        - frame.loc[frame["batch_id"] == batch, "y_true"]
                    )
                    ** 2
                )
            )
            for batch in batches
        ]
    )
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(batches), size=(iterations, len(batches)))
    candidate_boot = candidate_rmse[draw].mean(axis=1)
    baseline_boot = baseline_rmse[draw].mean(axis=1)
    improvement = 100.0 * (1.0 - candidate_boot / baseline_boot)
    point_candidate = float(candidate_rmse.mean())
    point_baseline = float(baseline_rmse.mean())
    return {
        "candidate_batch_macro_rmse": point_candidate,
        "baseline_batch_macro_rmse": point_baseline,
        "relative_batch_macro_improvement_pct": float(
            100.0 * (1.0 - point_candidate / point_baseline)
        ),
        "descriptive_ci_low": float(np.quantile(improvement, 0.025)),
        "descriptive_ci_high": float(np.quantile(improvement, 0.975)),
        "bootstrap_probability_candidate_better": float(
            np.mean(candidate_boot < baseline_boot)
        ),
        "batch_clusters": int(len(batches)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=50_000)
    args = parser.parse_args()

    baseline = pd.read_parquet(args.baseline_dir.resolve() / "predictions.parquet")
    branch_rows: list[dict[str, Any]] = []
    for metadata_path in sorted(args.baseline_dir.resolve().glob("*/fold_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pls_rmse = float(metadata["domain_pls_choice"]["domain_cv_rmse"])
        svr_rmse = float(metadata["domain_svr_choice"]["domain_cv_rmse"])
        degradation = 100.0 * (svr_rmse / pls_rmse - 1.0)
        branch_rows.append(
            {
                "target": metadata["target"],
                "outer_fold": int(metadata["outer_fold"]),
                "kernel_weight": 0.5 if degradation <= 5.0 else 0.0,
                "domain_svr_degradation_vs_pls_pct": degradation,
            }
        )
    branch = pd.DataFrame(branch_rows)
    if len(branch) != 45:
        raise RuntimeError(f"Expected 45 cross-batch branch-selection rows, found {len(branch)}")
    ai_paths = sorted(args.ai_dir.resolve().glob("*/fold_*/predictions.parquet"))
    if len(ai_paths) != 45:
        raise RuntimeError(f"Expected 45 AI prediction files, found {len(ai_paths)}")
    ai = pd.concat([pd.read_parquet(path) for path in ai_paths], ignore_index=True).rename(
        columns={"repeat": "outer_fold", "y_pred": "y_deep"}
    )
    ai["outer_fold"] = ai["outer_fold"].astype(int)
    merged = baseline.merge(
        ai[["sample_id", "cultivar_ascii", "target", "outer_fold", "y_true", "y_deep"]],
        on=["sample_id", "cultivar_ascii", "target", "outer_fold"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ai"),
    )
    if not np.allclose(merged["y_true"], merged["y_true_ai"], rtol=0, atol=1e-10):
        raise RuntimeError("Truth mismatch between baseline and AI predictions")
    merged = merged.drop(columns="y_true_ai")
    manifest_path = args.fold_manifest.resolve()
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": str, "batch_id": str})
    test_manifest = manifest.loc[manifest["outer_fold"] > 0, ["sample_id", "batch_id", "outer_fold"]]
    expected_test_fruits = int(test_manifest["sample_id"].nunique())
    merged = merged.merge(
        test_manifest,
        on=["sample_id", "outer_fold"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged.merge(
        branch,
        on=["target", "outer_fold"],
        how="left",
        validate="many_to_one",
    )
    merged["y_deep_kernel"] = (
        (1.0 - merged["kernel_weight"]) * merged["y_deep"]
        + merged["kernel_weight"] * merged["y_domain_svr"]
    )
    if len(merged) != expected_test_fruits * 9:
        raise RuntimeError(
            f"Expected {expected_test_fruits * 9} cross-batch rows, found {len(merged)}"
        )
    coverage = merged.groupby("target", observed=True).agg(
        samples=("sample_id", "size"),
        unique_samples=("sample_id", "nunique"),
        batches=("batch_id", "nunique"),
        cultivars=("cultivar_ascii", "nunique"),
    )
    if not (
        (coverage["samples"] == expected_test_fruits)
        & (coverage["unique_samples"] == expected_test_fruits)
        & (coverage["batches"] == 5)
        & (coverage["cultivars"] == 2)
    ).all():
        raise RuntimeError(f"V21 coverage audit failed:\n{coverage}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_dir / "v21_merged_predictions.parquet", index=False, compression="zstd")
    pooled_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    contrasts = [
        ("deep", "global_pls"),
        ("deep", "domain_pls"),
        ("deep_kernel", "global_pls"),
        ("deep_kernel", "domain_pls"),
        ("deep_kernel", "domain_svr"),
    ]
    for (target, trait), frame in merged.groupby(["target", "trait"], observed=True):
        for model, column in MODELS.items():
            pooled_rows.append(
                {
                    "target": target,
                    "trait": trait,
                    "model": model,
                    "batch_macro_rmse": batch_macro_rmse(frame, column),
                    **regression_metrics(frame["y_true"], frame[column]),
                }
            )
            for batch_id, batch in frame.groupby("batch_id", observed=True):
                batch_rows.append(
                    {
                        "target": target,
                        "trait": trait,
                        "model": model,
                        "batch_id": batch_id,
                        "cultivar_ascii": str(batch["cultivar_ascii"].iloc[0]),
                        **regression_metrics(batch["y_true"], batch[column]),
                    }
                )
        for candidate, baseline_name in contrasts:
            seed = int.from_bytes(
                hashlib.sha256(
                    f"V21|{target}|{candidate}|{baseline_name}".encode("utf-8")
                ).digest()[:4],
                "little",
            )
            comparison_rows.append(
                {
                    "target": target,
                    "trait": trait,
                    "candidate": candidate,
                    "baseline": baseline_name,
                    **batch_bootstrap(
                        frame,
                        MODELS[candidate],
                        MODELS[baseline_name],
                        args.bootstrap_iterations,
                        seed,
                    ),
                }
            )
    pooled = pd.DataFrame(pooled_rows)
    batches = pd.DataFrame(batch_rows)
    comparisons = pd.DataFrame(comparison_rows)
    pooled.to_csv(output_dir / "pooled_and_batch_macro_metrics.csv", index=False)
    batches.to_csv(output_dir / "per_batch_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "descriptive_batch_bootstrap_comparisons.csv", index=False)
    branch.to_csv(output_dir / "train_internal_branch_selection.csv", index=False)

    primary = comparisons[
        (comparisons["candidate"] == "deep_kernel")
        & (comparisons["baseline"] == "global_pls")
    ]
    strong = comparisons[
        (comparisons["candidate"] == "deep_kernel")
        & (comparisons["baseline"] == "domain_pls")
    ]
    svr = comparisons[
        (comparisons["candidate"] == "deep_kernel")
        & (comparisons["baseline"] == "domain_svr")
    ]
    summary = {
        "protocol": "V21 limited same-cultivar leave-one-batch-out audit",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "test_predictions": int(len(merged)),
        "unique_test_fruits_per_trait": expected_test_fruits,
        "test_batches": 5,
        "test_cultivars": 2,
        "hybrid_vs_global_pls_traits_won_batch_macro_rmse": int(
            (primary["relative_batch_macro_improvement_pct"] > 0).sum()
        ),
        "hybrid_vs_domain_pls_traits_won_batch_macro_rmse": int(
            (strong["relative_batch_macro_improvement_pct"] > 0).sum()
        ),
        "hybrid_vs_domain_svr_traits_won_batch_macro_rmse": int(
            (svr["relative_batch_macro_improvement_pct"] > 0).sum()
        ),
        "claim_boundary": "five batches from two multi-batch cultivars; descriptive internal batch-transfer evidence",
        "confidence_interval_warning": "Only five batch clusters; bootstrap intervals are descriptive and not broad external-validation evidence.",
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
