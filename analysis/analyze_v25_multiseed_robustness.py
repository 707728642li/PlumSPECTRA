from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASELINES = {
    "Global PLSR": "y_global_pls",
    "Cultivar-aware PLSR": "y_domain_pls",
    "Nested RBF-SVR": "y_domain_svr",
    "No-neural B50": "y_b50",
    "Cultivar-mean null": "y_cultivar_mean_null",
}


def rmse(y: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y, float) - np.asarray(prediction, float)))))


def cluster_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    cultivars = sorted(frame["cultivar_ascii"].astype(str).unique())
    sample_counts = []
    candidate_sse = []
    baseline_sse = []
    for cultivar in cultivars:
        part = frame.loc[frame["cultivar_ascii"].astype(str).eq(cultivar)]
        sample_counts.append(len(part))
        candidate_sse.append(float(np.square(part["y_true"] - part[candidate]).sum()))
        baseline_sse.append(float(np.square(part["y_true"] - part[baseline]).sum()))
    sample_counts = np.asarray(sample_counts, float)
    candidate_sse = np.asarray(candidate_sse, float)
    baseline_sse = np.asarray(baseline_sse, float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(cultivars), size=(iterations, len(cultivars)))
    boot_n = sample_counts[draw].sum(axis=1)
    candidate_rmse = np.sqrt(candidate_sse[draw].sum(axis=1) / boot_n)
    baseline_rmse = np.sqrt(baseline_sse[draw].sum(axis=1) / boot_n)
    improvement = 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse
    point_candidate = rmse(frame["y_true"], frame[candidate])
    point_baseline = rmse(frame["y_true"], frame[baseline])
    return {
        "candidate_rmse": point_candidate,
        "baseline_rmse": point_baseline,
        "relative_improvement_pct": 100.0 * (point_baseline - point_candidate) / point_baseline,
        "ci95_low_pct": float(np.quantile(improvement, 0.025)),
        "ci95_high_pct": float(np.quantile(improvement, 0.975)),
        "probability_candidate_better": float(np.mean(improvement > 0.0)),
        "iterations": iterations,
        "cultivar_clusters": len(cultivars),
    }


def metric_row(frame: pd.DataFrame, candidate: str) -> dict[str, float]:
    candidate_rmse = rmse(frame["y_true"], frame[candidate])
    row: dict[str, float] = {"rmse": candidate_rmse}
    for baseline_name, baseline in BASELINES.items():
        baseline_rmse = rmse(frame["y_true"], frame[baseline])
        key = baseline_name.lower().replace("-", "_").replace(" ", "_")
        row[f"improvement_vs_{key}_pct"] = 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-analysis-dir", type=Path, required=True)
    parser.add_argument("--multiseed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=200_000)
    parser.add_argument("--seed-repeats", default="101,102")
    args = parser.parse_args()

    final_dir = args.final_analysis_dir.resolve()
    multiseed_dir = args.multiseed_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repeats = [int(value.strip()) for value in args.seed_repeats.split(",") if value.strip()]
    if len(repeats) != 2 or len(set(repeats)) != 2:
        raise ValueError("The V25 audit expects exactly two distinct additional seed repeats")

    integrated = pd.read_parquet(final_dir / "v25_integrated_predictions.parquet")
    required = {
        "sample_id",
        "cultivar_ascii",
        "target",
        "trait",
        "outer_fold",
        "y_true",
        "y_deep",
        "y_final",
        "y_global_pls",
        "y_domain_pls",
        "y_domain_svr",
        "y_b50",
        "y_cultivar_mean_null",
        "kernel_weight",
    }
    missing = sorted(required - set(integrated.columns))
    if missing:
        raise ValueError(f"Integrated predictions lack required columns: {missing}")
    if integrated["trait"].nunique() != 12 or integrated["outer_fold"].nunique() != 5:
        raise RuntimeError("Expected 12 traits and five outer folds in V25 primary predictions")

    keys = ["sample_id", "cultivar_ascii", "target", "trait", "outer_fold"]
    seed_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    seed_fold_rows: list[dict[str, Any]] = []
    for (trait, fold), primary in integrated.groupby(["trait", "outer_fold"], observed=True):
        primary = primary.copy()
        primary["seed_label"] = "primary"
        primary["seed_repeat"] = int(fold)
        seed_frames.append(primary)
        for candidate in ("y_deep", "y_final"):
            seed_fold_rows.append(
                {
                    "trait": trait,
                    "outer_fold": int(fold),
                    "seed_label": "primary",
                    "candidate": "deep_only" if candidate == "y_deep" else "selected_final",
                    "samples": len(primary),
                    **metric_row(primary, candidate),
                }
            )

        expected_ids = set(primary["sample_id"].astype(str))
        for repeat in repeats:
            path = (
                multiseed_dir
                / "ai"
                / str(trait)
                / f"fold_{int(fold)}"
                / f"seed_repeat_{repeat}"
                / "predictions.parquet"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            prediction = pd.read_parquet(path)
            prediction["sample_id"] = prediction["sample_id"].astype(str)
            if set(prediction["sample_id"]) != expected_ids:
                raise RuntimeError(f"Sample mismatch: {path}")
            metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
            if not metadata.get("domain_aware_anchor_selection"):
                raise RuntimeError(f"Multiseed run did not use domain-aware anchor selection: {path}")
            if int(metadata.get("crossfit_anchor_folds", 0)) != 4:
                raise RuntimeError(f"Multiseed run did not use four cross-fit anchor folds: {path}")
            if metadata.get("gate_selection_mode") != "training_internal_validation":
                raise RuntimeError(f"Multiseed run did not use a training-internal gate: {path}")

            merged = primary.drop(columns=["y_deep", "y_final"]).merge(
                prediction[["sample_id", "y_true", "y_pred"]].rename(
                    columns={"y_true": "y_true_seed", "y_pred": "y_deep"}
                ),
                on="sample_id",
                how="inner",
                validate="one_to_one",
            )
            if not np.allclose(merged["y_true"], merged["y_true_seed"], rtol=0, atol=1e-8):
                raise ValueError(f"Truth mismatch: {path}")
            merged = merged.drop(columns="y_true_seed")
            merged["y_final"] = (
                (1.0 - merged["kernel_weight"]) * merged["y_deep"]
                + merged["kernel_weight"] * merged["y_domain_svr"]
            )
            merged["seed_label"] = f"seed_repeat_{repeat}"
            merged["seed_repeat"] = repeat
            seed_frames.append(merged)
            for candidate in ("y_deep", "y_final"):
                seed_fold_rows.append(
                    {
                        "trait": trait,
                        "outer_fold": int(fold),
                        "seed_label": f"seed_repeat_{repeat}",
                        "candidate": "deep_only" if candidate == "y_deep" else "selected_final",
                        "samples": len(merged),
                        **metric_row(merged, candidate),
                    }
                )
            metadata_rows.append(
                {
                    "trait": trait,
                    "outer_fold": int(fold),
                    "seed_repeat": repeat,
                    "seed": int(metadata["seed"]),
                    "selected_epoch": int(metadata["selected_epoch"]),
                    "selected_gate": float(metadata["selected_gate"]),
                    "anchor_preprocessing": metadata["final_pls"]["preprocessing"],
                    "anchor_components": int(metadata["final_pls"]["n_components"]),
                    "domain_aware_anchor_selection": True,
                    "crossfit_anchor_folds": 4,
                    "gate_selection_mode": metadata["gate_selection_mode"],
                    "device": metadata["device"],
                }
            )

    all_seeds = pd.concat(seed_frames, ignore_index=True)
    if all_seeds.duplicated(["sample_id", "trait", "seed_label"]).any():
        raise ValueError("Duplicate sample/trait/seed predictions")
    seed_counts = all_seeds.groupby(["sample_id", "trait"], observed=True)["seed_label"].nunique()
    if not seed_counts.eq(3).all():
        raise RuntimeError("Every fruit-trait pair must have three complete-pipeline predictions")
    all_seeds.to_parquet(output / "all_seed_predictions.parquet", index=False, compression="zstd")

    aggregation_columns = {
        "y_true": "first",
        "y_deep": "mean",
        "y_final": "mean",
        "y_global_pls": "first",
        "y_domain_pls": "first",
        "y_domain_svr": "first",
        "y_b50": "first",
        "y_cultivar_mean_null": "first",
        "kernel_weight": "first",
        "seed_label": "nunique",
    }
    ensemble = (
        all_seeds.groupby(keys, observed=True, as_index=False)
        .agg(aggregation_columns)
        .rename(
            columns={
                "y_deep": "y_deep_multiseed",
                "y_final": "y_final_multiseed",
                "seed_label": "seeds",
            }
        )
    )
    if not ensemble["seeds"].eq(3).all():
        raise RuntimeError("Multiseed aggregation did not retain three seeds")
    ensemble.to_parquet(output / "multiseed_mean_predictions.parquet", index=False, compression="zstd")

    seed_fold_metrics = pd.DataFrame(seed_fold_rows)
    seed_fold_metrics.to_csv(output / "seed_fold_metrics.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(output / "seed_metadata.csv", index=False)

    ensemble_fold_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for trait, trait_frame in ensemble.groupby("trait", observed=True):
        for fold, group in trait_frame.groupby("outer_fold", observed=True):
            for candidate, label in (
                ("y_deep_multiseed", "deep_only"),
                ("y_final_multiseed", "selected_final"),
            ):
                ensemble_fold_rows.append(
                    {
                        "trait": trait,
                        "outer_fold": int(fold),
                        "candidate": label,
                        "samples": len(group),
                        **metric_row(group, candidate),
                    }
                )
        for candidate, label in (
            ("y_deep_multiseed", "deep_only"),
            ("y_final_multiseed", "selected_final"),
        ):
            for baseline_name, baseline in BASELINES.items():
                stable = hashlib.sha256(
                    f"V25|{trait}|{label}|{baseline_name}".encode("utf-8")
                ).digest()
                contrast_rows.append(
                    {
                        "trait": trait,
                        "candidate": label,
                        "baseline": baseline_name,
                        **cluster_bootstrap(
                            trait_frame,
                            candidate,
                            baseline,
                            args.bootstrap_iterations,
                            int.from_bytes(stable[:4], "little"),
                        ),
                    }
                )

        for candidate_label in ("deep_only", "selected_final"):
            seed_part = seed_fold_metrics.loc[
                seed_fold_metrics["trait"].eq(trait)
                & seed_fold_metrics["candidate"].eq(candidate_label)
            ]
            ensemble_part = pd.DataFrame(ensemble_fold_rows)
            ensemble_part = ensemble_part.loc[
                ensemble_part["trait"].eq(trait)
                & ensemble_part["candidate"].eq(candidate_label)
            ].sort_values("outer_fold")
            effect_column = "improvement_vs_cultivar_aware_plsr_pct"
            primary_effect = (
                seed_part.loc[seed_part["seed_label"].eq("primary")]
                .sort_values("outer_fold")[effect_column]
                .to_numpy(float)
            )
            multiseed_effect = ensemble_part[effect_column].to_numpy(float)
            within_seed_sd = (
                seed_part.groupby("outer_fold", observed=True)[effect_column]
                .std(ddof=1)
                .mean()
            )
            cluster = next(
                row
                for row in contrast_rows
                if row["trait"] == trait
                and row["candidate"] == candidate_label
                and row["baseline"] == "Cultivar-aware PLSR"
            )
            summary_rows.append(
                {
                    "trait": trait,
                    "candidate": candidate_label,
                    "outer_folds": 5,
                    "seeds_per_fold": 3,
                    "pipeline_instances": 15,
                    "primary_fold_mean_improvement_pct": float(np.mean(primary_effect)),
                    "primary_fold_se_pct_points": float(np.std(primary_effect, ddof=1) / np.sqrt(5)),
                    "multiseed_fold_mean_improvement_pct": float(np.mean(multiseed_effect)),
                    "multiseed_fold_se_pct_points": float(np.std(multiseed_effect, ddof=1) / np.sqrt(5)),
                    "primary_fold_wins": int(np.sum(primary_effect > 0)),
                    "multiseed_fold_wins": int(np.sum(multiseed_effect > 0)),
                    "mean_within_fold_seed_sd_pct_points": float(within_seed_sd),
                    "cluster_relative_improvement_pct": float(cluster["relative_improvement_pct"]),
                    "cluster_ci95_low_pct": float(cluster["ci95_low_pct"]),
                    "cluster_ci95_high_pct": float(cluster["ci95_high_pct"]),
                    "cluster_supported": bool(cluster["ci95_low_pct"] > 0),
                }
            )

    ensemble_fold_metrics = pd.DataFrame(ensemble_fold_rows)
    contrasts = pd.DataFrame(contrast_rows)
    summary = pd.DataFrame(summary_rows)
    ensemble_fold_metrics.to_csv(output / "multiseed_fold_metrics.csv", index=False)
    contrasts.to_csv(output / "multiseed_cluster_contrasts.csv", index=False)
    summary.to_csv(output / "multiseed_summary.csv", index=False)
    report = {
        "protocol": (
            "V25 primary plus two independent post hoc complete-pipeline seeds per frozen fold. "
            "Predictions are averaged within fruit; seed variation is reported separately from "
            "outer-fold and cultivar-cluster uncertainty."
        ),
        "prediction_rows_all_seeds": int(len(all_seeds)),
        "fruit_trait_pairs": int(len(ensemble)),
        "traits": int(ensemble["trait"].nunique()),
        "outer_folds": int(ensemble["outer_fold"].nunique()),
        "additional_fits": int(len(metadata_rows)),
        "pipeline_instances_including_primary": int(12 * 5 * 3),
        "bootstrap_iterations": args.bootstrap_iterations,
        "test_labels_used_for_selection": False,
        "biological_replication_claimed": False,
        "summary": summary.to_dict(orient="records"),
    }
    (output / "multiseed_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

