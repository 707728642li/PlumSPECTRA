from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASELINES = {
    "Global PLSR": "y_global_pls",
    "Cultivar-aware PLSR": "y_domain_pls",
    "Nested RBF-SVR": "y_domain_svr",
}


def rmse(y: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y, float) - np.asarray(pred, float)))))


def cluster_bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    baseline: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    cultivars = sorted(frame["cultivar_ascii"].astype(str).unique())
    n = []
    candidate_sse = []
    baseline_sse = []
    for cultivar in cultivars:
        part = frame.loc[frame["cultivar_ascii"].astype(str).eq(cultivar)]
        n.append(len(part))
        candidate_sse.append(float(np.square(part["y_true"] - part[candidate]).sum()))
        baseline_sse.append(float(np.square(part["y_true"] - part[baseline]).sum()))
    n = np.asarray(n, float)
    candidate_sse = np.asarray(candidate_sse, float)
    baseline_sse = np.asarray(baseline_sse, float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(cultivars), size=(iterations, len(cultivars)))
    boot_n = n[draw].sum(axis=1)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed-dir", type=Path, default=Path("results/v24_hr_strengthening/multiseed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v24_hr_strengthening/multiseed_analysis"))
    parser.add_argument("--bootstrap-iterations", type=int, default=50_000)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    multiseed_dir = args.multiseed_dir.resolve() if args.multiseed_dir.is_absolute() else (project / args.multiseed_dir).resolve()
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (project / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    integrated = pd.read_csv(project / "results/v22_integrated/figure_data/predictions_all12.csv")
    traits = [trait for trait in integrated["trait"].drop_duplicates() if trait != "FW"]

    seed_metric_rows: list[dict[str, object]] = []
    ensemble_rows: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    for trait in traits:
        trait_frame = integrated.loc[integrated["trait"].eq(trait)].copy()
        for fold in range(1, 6):
            base = trait_frame.loc[trait_frame["outer_fold"].eq(fold)].copy()
            keys = ["sample_id", "cultivar_ascii", "outer_fold", "trait"]
            originals = base[keys + ["y_true", "y_deep", "y_final", *BASELINES.values()]].copy()
            originals["seed_label"] = "original"
            seed_frames = [originals]
            paths = sorted((multiseed_dir / "ai" / trait / f"fold_{fold}").glob("seed_repeat_*/predictions.parquet"))
            if len(paths) != 2:
                raise RuntimeError(f"Expected two new seeds for {trait} fold {fold}, found {len(paths)}")
            expected_ids = set(originals["sample_id"].astype(str))
            for path in paths:
                prediction = pd.read_parquet(path)
                prediction["sample_id"] = prediction["sample_id"].astype(str)
                if set(prediction["sample_id"]) != expected_ids:
                    raise RuntimeError(f"Sample mismatch for {trait} fold {fold}: {path}")
                seed = base[keys + ["y_true", *BASELINES.values()]].merge(
                    prediction[["sample_id", "y_pred"]].rename(columns={"y_pred": "y_deep"}),
                    on="sample_id", how="inner", validate="one_to_one",
                )
                if trait == "FW":
                    seed["y_final"] = seed["y_deep"]
                else:
                    seed["y_final"] = 0.5 * seed["y_deep"] + 0.5 * seed["y_domain_svr"]
                seed["seed_label"] = path.parent.name
                seed_frames.append(seed)
                metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
                metadata_rows.append({
                    "trait": trait,
                    "outer_fold": fold,
                    "seed_label": path.parent.name,
                    "repeat": int(metadata["repeat"]),
                    "seed": int(metadata["seed"]),
                    "selected_epoch": int(metadata["selected_epoch"]),
                    "selected_gate": float(metadata["selected_gate"]),
                    "device": metadata["device"],
                })

            for seed in seed_frames:
                final_rmse = rmse(seed["y_true"], seed["y_final"])
                row: dict[str, object] = {
                    "trait": trait,
                    "outer_fold": fold,
                    "seed_label": seed["seed_label"].iloc[0],
                    "samples": len(seed),
                    "final_rmse": final_rmse,
                }
                for baseline_name, column in BASELINES.items():
                    baseline_rmse = rmse(seed["y_true"], seed[column])
                    row[f"improvement_vs_{column}_pct"] = 100.0 * (baseline_rmse - final_rmse) / baseline_rmse
                seed_metric_rows.append(row)

            all_seeds = pd.concat(seed_frames, ignore_index=True)
            ensemble = (
                all_seeds.groupby(keys, observed=True, as_index=False)
                .agg(
                    y_true=("y_true", "first"),
                    y_multiseed=("y_final", "mean"),
                    y_global_pls=("y_global_pls", "first"),
                    y_domain_pls=("y_domain_pls", "first"),
                    y_domain_svr=("y_domain_svr", "first"),
                    seeds=("seed_label", "nunique"),
                )
            )
            if not ensemble["seeds"].eq(3).all():
                raise RuntimeError(f"Every {trait} fold {fold} fruit must have three seed predictions")
            ensemble_rows.append(ensemble)

    seed_metrics = pd.DataFrame(seed_metric_rows)
    ensembles = pd.concat(ensemble_rows, ignore_index=True)
    fold_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for trait, trait_ensemble in ensembles.groupby("trait", observed=True):
        for fold, group in trait_ensemble.groupby("outer_fold", observed=True):
            candidate_rmse = rmse(group["y_true"], group["y_multiseed"])
            row: dict[str, object] = {"trait": trait, "outer_fold": int(fold), "samples": len(group), "multiseed_rmse": candidate_rmse}
            for _, column in BASELINES.items():
                baseline_rmse = rmse(group["y_true"], group[column])
                row[f"improvement_vs_{column}_pct"] = 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse
            fold_rows.append(row)
        for baseline_name, column in BASELINES.items():
            stable = hashlib.sha256(f"V24|{trait}|multiseed|{baseline_name}".encode()).digest()
            contrast_rows.append({
                "trait": trait,
                "candidate": "three-seed prediction ensemble",
                "baseline": baseline_name,
                **cluster_bootstrap(trait_ensemble, "y_multiseed", column, args.bootstrap_iterations, int.from_bytes(stable[:4], "little")),
            })

    fold_metrics = pd.DataFrame(fold_rows)
    contrasts = pd.DataFrame(contrast_rows)
    for trait in traits:
        original = seed_metrics.loc[(seed_metrics["trait"].eq(trait)) & (seed_metrics["seed_label"].eq("original"))].sort_values("outer_fold")
        ensemble = fold_metrics.loc[fold_metrics["trait"].eq(trait)].sort_values("outer_fold")
        original_effect = original["improvement_vs_y_domain_pls_pct"].to_numpy(float)
        ensemble_effect = ensemble["improvement_vs_y_domain_pls_pct"].to_numpy(float)
        original_se = float(np.std(original_effect, ddof=1) / np.sqrt(5))
        ensemble_se = float(np.std(ensemble_effect, ddof=1) / np.sqrt(5))
        seed_sd = (
            seed_metrics.loc[seed_metrics["trait"].eq(trait)]
            .groupby("outer_fold", observed=True)["improvement_vs_y_domain_pls_pct"]
            .std(ddof=1)
            .mean()
        )
        cluster = contrasts.loc[(contrasts["trait"].eq(trait)) & (contrasts["baseline"].eq("Cultivar-aware PLSR"))].iloc[0]
        summary_rows.append({
            "trait": trait,
            "outer_folds": 5,
            "seeds_per_fold": 3,
            "total_pipeline_instances_including_original": 15,
            "original_fold_mean_improvement_vs_domain_pls_pct": float(np.mean(original_effect)),
            "original_fold_se_pct_points": original_se,
            "multiseed_fold_mean_improvement_vs_domain_pls_pct": float(np.mean(ensemble_effect)),
            "multiseed_fold_se_pct_points": ensemble_se,
            "se_change_pct": float(100.0 * (ensemble_se / original_se - 1.0)) if original_se else np.nan,
            "original_fold_wins": int(np.sum(original_effect > 0)),
            "multiseed_fold_wins": int(np.sum(ensemble_effect > 0)),
            "mean_within_fold_seed_sd_pct_points": float(seed_sd),
            "cluster_relative_improvement_pct": float(cluster["relative_improvement_pct"]),
            "cluster_ci95_low_pct": float(cluster["ci95_low_pct"]),
            "cluster_ci95_high_pct": float(cluster["ci95_high_pct"]),
            "cluster_supported": bool(cluster["ci95_low_pct"] > 0),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "alltrait_multiseed_summary.csv", index=False)
    seed_metrics.to_csv(output / "alltrait_seed_fold_metrics.csv", index=False)
    fold_metrics.to_csv(output / "alltrait_multiseed_fold_metrics.csv", index=False)
    contrasts.to_csv(output / "alltrait_multiseed_cluster_bootstrap.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(output / "alltrait_seed_metadata.csv", index=False)
    ensembles.to_parquet(output / "alltrait_multiseed_predictions.parquet", index=False, compression="zstd")

    # Join the earlier, deeper five-seed FW audit so the release has one
    # all-12-trait optimization-stability table without discarding the frozen
    # primary estimates or pretending seed repeats are biological replicates.
    fw_dir = project / "results/v23_multiseed/analysis"
    fw_report = json.loads((fw_dir / "fw_multiseed_summary.json").read_text(encoding="utf-8"))
    fw_seed_metrics = pd.read_csv(fw_dir / "fw_seed_fold_metrics.csv")
    fw_cluster = fw_report["cluster_bootstrap_domain_pls"]
    fw_row = {
        "trait": "FW",
        "outer_folds": int(fw_report["outer_folds"]),
        "seeds_per_fold": int(fw_report["seeds_per_fold"]),
        "total_pipeline_instances_including_original": int(fw_report["total_fits"]),
        "original_fold_mean_improvement_vs_domain_pls_pct": float(fw_report["original_fold_mean_improvement_vs_domain_pls_pct"]),
        "original_fold_se_pct_points": float(fw_report["original_fold_se_pct_points"]),
        "multiseed_fold_mean_improvement_vs_domain_pls_pct": float(fw_report["multiseed_fold_mean_improvement_vs_domain_pls_pct"]),
        "multiseed_fold_se_pct_points": float(fw_report["multiseed_fold_se_pct_points"]),
        "se_change_pct": -float(fw_report["se_reduction_pct"]),
        "original_fold_wins": int((fw_seed_metrics.loc[fw_seed_metrics["seed_label"].eq("original"), "improvement_vs_domain_pls_pct"] > 0).sum()),
        "multiseed_fold_wins": int(fw_report["multiseed_fold_wins_vs_domain_pls"]),
        "mean_within_fold_seed_sd_pct_points": float(fw_report["mean_within_fold_seed_sd_pct_points"]),
        "cluster_relative_improvement_pct": float(fw_cluster["relative_improvement_pct"]),
        "cluster_ci95_low_pct": float(fw_cluster["ci95_low"]),
        "cluster_ci95_high_pct": float(fw_cluster["ci95_high"]),
        "cluster_supported": bool(fw_cluster["ci95_low"] > 0),
    }
    all12_summary = pd.concat([pd.DataFrame([fw_row]), summary], ignore_index=True)
    all12_summary.to_csv(output / "all12_multiseed_summary.csv", index=False)
    report = {
        "protocol": "Original plus two new complete-pipeline random seeds per frozen outer fold; three out-of-fold predictions averaged per test fruit.",
        "traits": traits,
        "additional_fits": len(metadata_rows),
        "pipeline_instances_including_original": len(traits) * 15,
        "interpretation_boundary": "Training-seed replication estimates optimization sensitivity; it does not add biological replicates or justify replacing cultivar-cluster inference with seed-based error bars.",
        "summary": summary.to_dict(orient="records"),
    }
    (output / "alltrait_multiseed_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    all12_report = {
        "protocol": "FW: original plus four new seeds; other 11 traits: original plus two new seeds. Every repeat reran the complete train-internal selection and final-fitting pipeline within each frozen outer fold.",
        "traits": all12_summary["trait"].tolist(),
        "additional_fits": len(metadata_rows) + 20,
        "pipeline_instances_including_original": int(all12_summary["total_pipeline_instances_including_original"].sum()),
        "interpretation_boundary": "Optimization-seed replication does not add biological units; cultivar-cluster inference and the frozen primary predictions remain authoritative.",
        "summary": all12_summary.to_dict(orient="records"),
    }
    (output / "all12_multiseed_summary.json").write_text(json.dumps(all12_report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
