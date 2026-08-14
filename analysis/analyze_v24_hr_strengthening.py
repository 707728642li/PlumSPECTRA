from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import multipletests


MODEL_COLUMNS = {
    "Global PLSR": "y_global_pls",
    "Cultivar-aware PLSR": "y_domain_pls",
    "Nested RBF-SVR": "y_domain_svr",
    "Residual CNN": "y_deep",
    "Deep-kernel ensemble": "y_deep_kernel",
}


def stable_seed(*values: object) -> int:
    raw = "|".join(map(str, values)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - pred))))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.sum(np.square(y - np.mean(y))))
    return float(1.0 - np.sum(np.square(y - pred)) / denominator) if denominator > 0 else np.nan


def ccc(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    covariance = float(np.cov(y, pred, ddof=1)[0, 1])
    denominator = float(
        np.var(y, ddof=1) + np.var(pred, ddof=1) + (np.mean(y) - np.mean(pred)) ** 2
    )
    return float(2.0 * covariance / denominator) if denominator > 0 else np.nan


def fit_adapter(
    y: np.ndarray,
    pred: np.ndarray,
    mode: str,
    slope_prior_strength: float,
) -> tuple[float, float]:
    if mode == "intercept":
        return 1.0, float(np.mean(y - pred))
    if mode != "shrunken_affine":
        raise ValueError(mode)
    x_mean = float(np.mean(pred))
    y_mean = float(np.mean(y))
    x_centered = pred - x_mean
    y_centered = y - y_mean
    ss_x = float(np.sum(x_centered**2))
    scale = max(ss_x / max(len(pred) - 1, 1), np.finfo(float).eps)
    penalty = slope_prior_strength * scale
    slope = float((np.sum(x_centered * y_centered) + penalty) / (ss_x + penalty))
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def summarize_fewshot(repeated: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "trait", "shots", "adapter", "aggregation"]
    summary = (
        repeated.groupby(keys, observed=True)
        .agg(
            repeats=("repeat", "nunique"),
            n_evaluation_mean=("n_evaluation", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            rmse_ci025=("rmse", lambda x: x.quantile(0.025)),
            rmse_ci975=("rmse", lambda x: x.quantile(0.975)),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            r2_ci025=("r2", lambda x: x.quantile(0.025)),
            r2_ci975=("r2", lambda x: x.quantile(0.975)),
            ccc_mean=("ccc", "mean"),
            ccc_ci025=("ccc", lambda x: x.quantile(0.025)),
            ccc_ci975=("ccc", lambda x: x.quantile(0.975)),
            rmse_gain_pct_mean=("rmse_gain_pct", "mean"),
            rmse_gain_pct_ci025=("rmse_gain_pct", lambda x: x.quantile(0.025)),
            rmse_gain_pct_ci975=("rmse_gain_pct", lambda x: x.quantile(0.975)),
        )
        .reset_index()
    )
    return summary


def analyze_fewshot(
    predictions_path: Path,
    output: Path,
    shots: list[int],
    repeats: int,
    slope_prior_strength: float,
) -> dict[str, object]:
    frame = pd.read_parquet(predictions_path)
    required = {"sample_id", "trait", "batch_id", "y_true", *MODEL_COLUMNS.values()}
    if missing := required - set(frame.columns):
        raise ValueError(f"Cross-batch predictions lack columns: {sorted(missing)}")
    if frame.duplicated(["sample_id", "trait"]).any():
        raise ValueError("Cross-batch predictions contain duplicate fruit/trait rows")

    rows: list[dict[str, object]] = []
    for trait, trait_frame in frame.groupby("trait", observed=True):
        batch_frames = {
            str(batch): group.sort_values("sample_id").reset_index(drop=True)
            for batch, group in trait_frame.groupby("batch_id", observed=True)
        }
        for shot in shots:
            if shot and any(shot >= len(group) - 2 for group in batch_frames.values()):
                raise ValueError(f"Shot count {shot} leaves too few evaluation fruits for {trait}")
            active_repeats = [0] if shot == 0 else range(1, repeats + 1)
            adapters = ["none"] if shot == 0 else ["intercept", "shrunken_affine"]
            for repeat in active_repeats:
                masks: dict[str, np.ndarray] = {}
                for batch, group in batch_frames.items():
                    calibration = np.zeros(len(group), dtype=bool)
                    if shot:
                        rng = np.random.default_rng(stable_seed("v24", trait, batch, shot, repeat))
                        calibration[rng.choice(len(group), size=shot, replace=False)] = True
                    masks[batch] = calibration
                for model, column in MODEL_COLUMNS.items():
                    for adapter in adapters:
                        pooled_truth: list[np.ndarray] = []
                        pooled_pred: list[np.ndarray] = []
                        pooled_unadapted: list[np.ndarray] = []
                        batch_metrics: list[tuple[int, float, float, float, float]] = []
                        for batch, group in batch_frames.items():
                            calibration = masks[batch]
                            evaluation = ~calibration
                            truth = group.loc[evaluation, "y_true"].to_numpy(float)
                            pred0 = group.loc[evaluation, column].to_numpy(float)
                            if adapter == "none":
                                pred = pred0
                            else:
                                slope, intercept = fit_adapter(
                                    group.loc[calibration, "y_true"].to_numpy(float),
                                    group.loc[calibration, column].to_numpy(float),
                                    adapter,
                                    slope_prior_strength,
                                )
                                pred = slope * pred0 + intercept
                            base_rmse = rmse(truth, pred0)
                            adapted_rmse = rmse(truth, pred)
                            gain = 100.0 * (base_rmse - adapted_rmse) / base_rmse
                            batch_metrics.append((len(truth), adapted_rmse, r2(truth, pred), ccc(truth, pred), gain))
                            pooled_truth.append(truth)
                            pooled_pred.append(pred)
                            pooled_unadapted.append(pred0)

                        truth = np.concatenate(pooled_truth)
                        pred = np.concatenate(pooled_pred)
                        pred0 = np.concatenate(pooled_unadapted)
                        base_rmse = rmse(truth, pred0)
                        pooled_rmse = rmse(truth, pred)
                        rows.append(
                            {
                                "model": model,
                                "trait": trait,
                                "shots": shot,
                                "adapter": adapter,
                                "repeat": repeat,
                                "aggregation": "pooled",
                                "n_evaluation": len(truth),
                                "rmse": pooled_rmse,
                                "r2": r2(truth, pred),
                                "ccc": ccc(truth, pred),
                                "rmse_gain_pct": 100.0 * (base_rmse - pooled_rmse) / base_rmse,
                            }
                        )
                        batch_array = np.asarray(batch_metrics, dtype=float)
                        rows.append(
                            {
                                "model": model,
                                "trait": trait,
                                "shots": shot,
                                "adapter": adapter,
                                "repeat": repeat,
                                "aggregation": "batch_macro",
                                "n_evaluation": int(np.sum(batch_array[:, 0])),
                                "rmse": float(np.mean(batch_array[:, 1])),
                                "r2": float(np.mean(batch_array[:, 2])),
                                "ccc": float(np.mean(batch_array[:, 3])),
                                "rmse_gain_pct": float(np.mean(batch_array[:, 4])),
                            }
                        )

                if shot:
                    pooled_truth = []
                    pooled_pred = []
                    batch_metrics = []
                    for batch, group in batch_frames.items():
                        calibration = masks[batch]
                        evaluation = ~calibration
                        truth = group.loc[evaluation, "y_true"].to_numpy(float)
                        batch_mean = float(group.loc[calibration, "y_true"].mean())
                        pred = np.full(len(truth), batch_mean, dtype=float)
                        batch_metrics.append(
                            (len(truth), rmse(truth, pred), r2(truth, pred), ccc(truth, pred), 0.0)
                        )
                        pooled_truth.append(truth)
                        pooled_pred.append(pred)
                    truth = np.concatenate(pooled_truth)
                    pred = np.concatenate(pooled_pred)
                    rows.append(
                        {
                            "model": "Batch-mean null (no spectra)",
                            "trait": trait,
                            "shots": shot,
                            "adapter": "batch_mean_null",
                            "repeat": repeat,
                            "aggregation": "pooled",
                            "n_evaluation": len(truth),
                            "rmse": rmse(truth, pred),
                            "r2": r2(truth, pred),
                            "ccc": ccc(truth, pred),
                            "rmse_gain_pct": 0.0,
                        }
                    )
                    batch_array = np.asarray(batch_metrics, dtype=float)
                    rows.append(
                        {
                            "model": "Batch-mean null (no spectra)",
                            "trait": trait,
                            "shots": shot,
                            "adapter": "batch_mean_null",
                            "repeat": repeat,
                            "aggregation": "batch_macro",
                            "n_evaluation": int(np.sum(batch_array[:, 0])),
                            "rmse": float(np.mean(batch_array[:, 1])),
                            "r2": float(np.mean(batch_array[:, 2])),
                            "ccc": float(np.mean(batch_array[:, 3])),
                            "rmse_gain_pct": 0.0,
                        }
                    )
    repeated = pd.DataFrame(rows)
    repeated.to_parquet(output / "fewshot_repeat_metrics.parquet", index=False, compression="zstd")
    summary = summarize_fewshot(repeated)
    summary.to_csv(output / "fewshot_summary.csv", index=False)

    success_rows: list[dict[str, object]] = []
    candidates = summary.loc[
        (summary["shots"] > 0) & (summary["aggregation"] == "batch_macro")
    ].sort_values("shots")
    for (model, trait, adapter), group in candidates.groupby(["model", "trait", "adapter"], observed=True):
        strict = group.loc[(group["r2_ci025"] > 0) & (group["rmse_gain_pct_ci025"] > 0)]
        directional = group.loc[(group["r2_mean"] > 0) & (group["rmse_gain_pct_mean"] > 0)]
        success_rows.append(
            {
                "model": model,
                "trait": trait,
                "adapter": adapter,
                "minimum_shots_strict": int(strict.iloc[0]["shots"]) if len(strict) else np.nan,
                "minimum_shots_directional": int(directional.iloc[0]["shots"]) if len(directional) else np.nan,
                "strict_definition": "batch-macro R2 2.5th percentile > 0 and RMSE-gain 2.5th percentile > 0",
            }
        )
    minimum = pd.DataFrame(success_rows)
    minimum.to_csv(output / "fewshot_minimum_shots.csv", index=False)

    return {
        "fruits": int(frame["sample_id"].nunique()),
        "traits": int(frame["trait"].nunique()),
        "batches": int(frame["batch_id"].nunique()),
        "models": [*MODEL_COLUMNS, "Batch-mean null (no spectra)"],
        "shots": shots,
        "repeats": repeats,
        "adapters": ["intercept", "shrunken_affine"],
        "slope_prior_strength": slope_prior_strength,
        "fairness": "identical deterministic calibration fruits across models and adapters; calibration fruits excluded from evaluation",
        "inference_warning": "Five observed batches from two cultivars; intervals quantify calibration-subsampling sensitivity, not population-wide biological replication.",
    }


def analyze_multiplicity(
    predictions_path: Path,
    output: Path,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    frame = pd.read_csv(predictions_path)
    required = {
        "cultivar_ascii", "trait", "y_true", "y_final", "y_global_pls", "y_domain_pls", "y_domain_svr"
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Integrated predictions lack columns: {sorted(missing)}")
    cultivars = sorted(frame["cultivar_ascii"].unique())
    traits = list(dict.fromkeys(frame["trait"].tolist()))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cultivars), size=(iterations, len(cultivars)), dtype=np.int16)
    counts = np.zeros((iterations, len(cultivars)), dtype=np.float32)
    row_indices = np.arange(iterations)[:, None]
    np.add.at(counts, (row_indices, draws), 1.0)

    contrasts: list[dict[str, object]] = []
    boot_vectors: list[np.ndarray] = []
    boot_vector_lookup: dict[tuple[str, str], np.ndarray] = {}
    baselines = {
        "Global PLSR": "y_global_pls",
        "Cultivar-aware PLSR": "y_domain_pls",
        "Nested RBF-SVR": "y_domain_svr",
    }
    for trait in traits:
        group = frame.loc[frame["trait"] == trait]
        by_cultivar: list[dict[str, float]] = []
        for cultivar in cultivars:
            part = group.loc[group["cultivar_ascii"] == cultivar]
            row: dict[str, float] = {"n": float(len(part))}
            row["candidate"] = float(np.square(part["y_true"] - part["y_final"]).sum())
            for baseline, column in baselines.items():
                row[baseline] = float(np.square(part["y_true"] - part[column]).sum())
            by_cultivar.append(row)
        n = np.asarray([row["n"] for row in by_cultivar], dtype=float)
        sse_candidate = np.asarray([row["candidate"] for row in by_cultivar], dtype=float)
        boot_n = counts @ n
        boot_candidate_rmse = np.sqrt((counts @ sse_candidate) / boot_n)
        candidate_rmse = float(np.sqrt(np.sum(sse_candidate) / np.sum(n)))
        for baseline, column in baselines.items():
            sse_baseline = np.asarray([row[baseline] for row in by_cultivar], dtype=float)
            baseline_rmse = float(np.sqrt(np.sum(sse_baseline) / np.sum(n)))
            point = 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse
            boot_baseline_rmse = np.sqrt((counts @ sse_baseline) / boot_n)
            boot_effect = 100.0 * (boot_baseline_rmse - boot_candidate_rmse) / boot_baseline_rmse
            p_raw = (1.0 + float(np.sum(boot_effect <= 0.0))) / (iterations + 1.0)
            boot_vectors.append(boot_effect.astype(np.float32))
            boot_vector_lookup[(str(trait), baseline)] = boot_effect.astype(np.float32)
            contrasts.append(
                {
                    "trait": trait,
                    "candidate": "PlumSPECTRA final",
                    "baseline": baseline,
                    "n_fruits": int(np.sum(n)),
                    "n_cultivars": len(cultivars),
                    "candidate_rmse": candidate_rmse,
                    "baseline_rmse": baseline_rmse,
                    "relative_rmse_improvement_pct": point,
                    "bootstrap_ci95_low_pct": float(np.quantile(boot_effect, 0.025)),
                    "bootstrap_ci95_high_pct": float(np.quantile(boot_effect, 0.975)),
                    "bootstrap_probability_better": float(np.mean(boot_effect > 0.0)),
                    "p_raw_one_sided": p_raw,
                }
            )

    result = pd.DataFrame(contrasts)
    result["p_bh_fdr"] = multipletests(result["p_raw_one_sided"], method="fdr_bh")[1]
    result["p_holm"] = multipletests(result["p_raw_one_sided"], method="holm")[1]
    result["supported_bh_fdr_0_05"] = result["p_bh_fdr"] < 0.05
    result["supported_holm_0_05"] = result["p_holm"] < 0.05

    boot_matrix = np.column_stack(boot_vectors).astype(np.float32)
    points = result["relative_rmse_improvement_pct"].to_numpy(float)
    standard_errors = np.std(boot_matrix, axis=0, ddof=1)
    studentized = np.divide(
        boot_matrix - points[None, :],
        standard_errors[None, :],
        out=np.zeros_like(boot_matrix),
        where=standard_errors[None, :] > 0,
    )
    critical = float(np.quantile(np.max(np.abs(studentized), axis=1), 0.95))
    result["simultaneous_ci95_low_pct"] = points - critical * standard_errors
    result["simultaneous_ci95_high_pct"] = points + critical * standard_errors
    result["supported_simultaneous_0_05"] = result["simultaneous_ci95_low_pct"] > 0
    result["simultaneous_critical_value"] = critical
    result.to_csv(output / "multiplicity_adjusted_contrasts.csv", index=False)
    family_frames: list[pd.DataFrame] = []
    family_summary: dict[str, dict[str, float | int]] = {}
    for baseline in baselines:
        mask = result["baseline"].eq(baseline).to_numpy()
        family = result.loc[mask].copy()
        family_boot = boot_matrix[:, mask]
        family_points = points[mask]
        family_se = standard_errors[mask]
        family_studentized = np.divide(
            family_boot - family_points[None, :],
            family_se[None, :],
            out=np.zeros_like(family_boot),
            where=family_se[None, :] > 0,
        )
        family_critical = float(
            np.quantile(np.max(np.abs(family_studentized), axis=1), 0.95)
        )
        family["family_definition"] = f"12 traits versus {baseline}"
        family["p_bh_within_12_contrast_family"] = multipletests(
            family["p_raw_one_sided"], method="fdr_bh"
        )[1]
        family["p_holm_within_12_contrast_family"] = multipletests(
            family["p_raw_one_sided"], method="holm"
        )[1]
        family["simultaneous_ci95_low_within_12_contrast_family"] = (
            family_points - family_critical * family_se
        )
        family["simultaneous_ci95_high_within_12_contrast_family"] = (
            family_points + family_critical * family_se
        )
        family["simultaneous_critical_value_within_12_contrast_family"] = family_critical
        family_frames.append(family)
        family_summary[baseline] = {
            "contrasts": int(len(family)),
            "supported_raw": int((family["p_raw_one_sided"] < 0.05).sum()),
            "supported_bh_fdr": int((family["p_bh_within_12_contrast_family"] < 0.05).sum()),
            "supported_holm": int((family["p_holm_within_12_contrast_family"] < 0.05).sum()),
            "supported_simultaneous": int(
                (family["simultaneous_ci95_low_within_12_contrast_family"] > 0).sum()
            ),
            "simultaneous_critical_value": family_critical,
        }
    pd.concat(family_frames, ignore_index=True).to_csv(
        output / "multiplicity_baseline_family_sensitivity.csv", index=False
    )

    # Post-hoc conservative sensitivity requested during external review:
    # compare the final model with the lowest-RMSE non-neural comparator for
    # each trait, then control multiplicity across those 12 selected contrasts.
    # B50 is eligible here when it is present, but it is deliberately not added
    # to the frozen 36-contrast primary family above.
    strongest_baselines = dict(baselines)
    if "y_b50" in frame.columns:
        strongest_baselines["No-neural B50"] = "y_b50"
        for trait in traits:
            group = frame.loc[frame["trait"] == trait]
            cultivars_trait = sorted(group["cultivar_ascii"].unique())
            by_cultivar = []
            for cultivar in cultivars_trait:
                part = group.loc[group["cultivar_ascii"] == cultivar]
                by_cultivar.append(
                    {
                        "n": float(len(part)),
                        "candidate": float(np.square(part["y_true"] - part["y_final"]).sum()),
                        "No-neural B50": float(np.square(part["y_true"] - part["y_b50"]).sum()),
                    }
                )
            n = np.asarray([row["n"] for row in by_cultivar], dtype=float)
            sse_candidate = np.asarray([row["candidate"] for row in by_cultivar], dtype=float)
            sse_baseline = np.asarray([row["No-neural B50"] for row in by_cultivar], dtype=float)
            boot_n = counts @ n
            boot_candidate_rmse = np.sqrt((counts @ sse_candidate) / boot_n)
            boot_baseline_rmse = np.sqrt((counts @ sse_baseline) / boot_n)
            boot_vector_lookup[(str(trait), "No-neural B50")] = (
                100.0 * (boot_baseline_rmse - boot_candidate_rmse) / boot_baseline_rmse
            ).astype(np.float32)

    strongest_rows: list[dict[str, object]] = []
    strongest_vectors: list[np.ndarray] = []
    for trait in traits:
        group = frame.loc[frame["trait"] == trait]
        candidate_rmse = float(
            np.sqrt(np.mean(np.square(group["y_true"].to_numpy(float) - group["y_final"].to_numpy(float))))
        )
        baseline_rmse = {
            name: float(
                np.sqrt(
                    np.mean(
                        np.square(
                            group["y_true"].to_numpy(float) - group[column].to_numpy(float)
                        )
                    )
                )
            )
            for name, column in strongest_baselines.items()
        }
        strongest_name = min(baseline_rmse, key=baseline_rmse.get)
        strongest_rmse = baseline_rmse[strongest_name]
        boot_effect = boot_vector_lookup[(str(trait), strongest_name)].astype(float)
        strongest_vectors.append(boot_effect.astype(np.float32))
        strongest_rows.append(
            {
                "trait": trait,
                "candidate": "PlumSPECTRA final",
                "strongest_baseline": strongest_name,
                "selection_rule": "lowest pooled out-of-fold RMSE among global PLSR, cultivar-aware PLSR, nested RBF-SVR, and B50",
                "family_status": "post-hoc conservative 12-contrast sensitivity",
                "n_fruits": int(len(group)),
                "n_cultivars": int(group["cultivar_ascii"].nunique()),
                "candidate_rmse": candidate_rmse,
                "strongest_baseline_rmse": strongest_rmse,
                "relative_rmse_improvement_pct": 100.0 * (strongest_rmse - candidate_rmse) / strongest_rmse,
                "bootstrap_ci95_low_pct": float(np.quantile(boot_effect, 0.025)),
                "bootstrap_ci95_high_pct": float(np.quantile(boot_effect, 0.975)),
                "bootstrap_probability_better": float(np.mean(boot_effect > 0.0)),
                "p_raw_one_sided": (1.0 + float(np.sum(boot_effect <= 0.0))) / (iterations + 1.0),
            }
        )
    strongest = pd.DataFrame(strongest_rows)
    strongest["p_bh_fdr"] = multipletests(strongest["p_raw_one_sided"], method="fdr_bh")[1]
    strongest["p_holm"] = multipletests(strongest["p_raw_one_sided"], method="holm")[1]
    strongest_matrix = np.column_stack(strongest_vectors).astype(np.float32)
    strongest_points = strongest["relative_rmse_improvement_pct"].to_numpy(float)
    strongest_se = np.std(strongest_matrix, axis=0, ddof=1)
    strongest_studentized = np.divide(
        strongest_matrix - strongest_points[None, :],
        strongest_se[None, :],
        out=np.zeros_like(strongest_matrix),
        where=strongest_se[None, :] > 0,
    )
    strongest_critical = float(
        np.quantile(np.max(np.abs(strongest_studentized), axis=1), 0.95)
    )
    strongest["simultaneous_ci95_low_pct"] = strongest_points - strongest_critical * strongest_se
    strongest["simultaneous_ci95_high_pct"] = strongest_points + strongest_critical * strongest_se
    strongest["supported_raw_0_05"] = strongest["p_raw_one_sided"] < 0.05
    strongest["supported_bh_fdr_0_05"] = strongest["p_bh_fdr"] < 0.05
    strongest["supported_holm_0_05"] = strongest["p_holm"] < 0.05
    strongest["supported_simultaneous_0_05"] = strongest["simultaneous_ci95_low_pct"] > 0
    strongest["simultaneous_critical_value"] = strongest_critical
    strongest.to_csv(output / "multiplicity_strongest_baseline_family.csv", index=False)
    return {
        "iterations": iterations,
        "seed": seed,
        "shared_resampling_matrix": True,
        "cultivar_clusters": len(cultivars),
        "contrasts": len(result),
        "supported_raw": int((result["p_raw_one_sided"] < 0.05).sum()),
        "supported_bh_fdr": int(result["supported_bh_fdr_0_05"].sum()),
        "supported_holm": int(result["supported_holm_0_05"].sum()),
        "supported_simultaneous": int(result["supported_simultaneous_0_05"].sum()),
        "simultaneous_method": "max-|studentized bootstrap deviation| family-wise 95% confidence intervals across all 36 contrasts",
        "baseline_specific_12_contrast_family_sensitivity": family_summary,
        "posthoc_strongest_baseline_12_contrast_family": {
            "eligible_baselines": list(strongest_baselines),
            "contrasts": int(len(strongest)),
            "supported_raw": int(strongest["supported_raw_0_05"].sum()),
            "supported_bh_fdr": int(strongest["supported_bh_fdr_0_05"].sum()),
            "supported_holm": int(strongest["supported_holm_0_05"].sum()),
            "supported_simultaneous": int(strongest["supported_simultaneous_0_05"].sum()),
            "simultaneous_critical_value": strongest_critical,
            "warning": "Comparator selection and this 12-contrast family were defined post hoc as a conservative external-review sensitivity analysis.",
        },
        "family_definition_warning": "The 36-contrast and three 12-contrast families are analysis-stage definitions, not preregistered families.",
        "warning": "Cultivar-cluster resampling protects within-cultivar dependence; 15 clusters still limit population-level generalization.",
    }


def analyze_cultivar_texture(phenotype_path: Path, output: Path) -> dict[str, object]:
    phenotype = pd.read_csv(phenotype_path)
    texture = phenotype.loc[phenotype["family"] == "Mechanical texture"].copy()
    medians = texture.pivot_table(
        index=["cultivar_ascii", "cultivar_code"], columns="trait", values="value", aggfunc="median"
    )
    trait_order = [trait for trait in texture["trait"].drop_duplicates() if trait in medians.columns]
    medians = medians[trait_order]
    center = medians.median(axis=0)
    mad = (medians - center).abs().median(axis=0)
    scale = (1.4826 * mad).replace(0, np.nan)
    robust_z = ((medians - center) / scale).fillna(0.0)
    robust_z_clipped = robust_z.clip(-3.0, 3.0)

    linkage_matrix = linkage(robust_z.to_numpy(float), method="ward", metric="euclidean")
    silhouette_rows: list[dict[str, object]] = []
    best_k = 2
    best_score = -np.inf
    for k in range(2, min(6, len(medians) - 1) + 1):
        labels = fcluster(linkage_matrix, k, criterion="maxclust")
        score = float(silhouette_score(robust_z.to_numpy(float), labels))
        silhouette_rows.append({"k": k, "silhouette": score})
        if score > best_score:
            best_score = score
            best_k = k
    labels = fcluster(linkage_matrix, best_k, criterion="maxclust")

    # Leave-one-cultivar-out stability audit. Cluster labels are aligned to the
    # full-data solution by maximum overlap before counting membership changes.
    # This prevents arbitrary integer label permutations from being reported as
    # biological instability.
    full_label_map = dict(zip(medians.index, labels, strict=True))
    leave_one_rows: list[dict[str, object]] = []
    for removed_index in medians.index:
        subset = medians.drop(index=removed_index)
        # Keep the full-cohort robust transformation fixed so this perturbation
        # audits clustering membership rather than conflating it with a second
        # perturbation of every feature's centre and scale.
        subset_z = robust_z.drop(index=removed_index)
        subset_linkage = linkage(subset_z.to_numpy(float), method="ward", metric="euclidean")
        subset_labels_raw = fcluster(subset_linkage, best_k, criterion="maxclust")
        original_labels = np.asarray([full_label_map[index] for index in subset.index], dtype=int)
        new_values = sorted(np.unique(subset_labels_raw).tolist())
        original_values = sorted(np.unique(original_labels).tolist())
        contingency = np.zeros((len(new_values), len(original_values)), dtype=int)
        for i, new_label in enumerate(new_values):
            for j, original_label in enumerate(original_values):
                contingency[i, j] = int(
                    np.sum((subset_labels_raw == new_label) & (original_labels == original_label))
                )
        row_ind, col_ind = linear_sum_assignment(-contingency)
        alignment = {
            new_values[row]: original_values[column]
            for row, column in zip(row_ind, col_ind, strict=True)
        }
        aligned_labels = np.asarray([alignment.get(value, value) for value in subset_labels_raw])
        subset_silhouette = (
            float(silhouette_score(subset_z.to_numpy(float), subset_labels_raw))
            if len(np.unique(subset_labels_raw)) > 1
            else np.nan
        )
        cluster_sizes = sorted(pd.Series(subset_labels_raw).value_counts().astype(int).tolist())
        leave_one_rows.append(
            {
                "removed_cultivar_ascii": removed_index[0],
                "removed_cultivar_code": removed_index[1],
                "fixed_cluster_count": best_k,
                "feature_scaling": "full-cohort robust centre and MAD held fixed",
                "remaining_cultivars": len(subset),
                "membership_changes_after_label_alignment": int(
                    np.sum(aligned_labels != original_labels)
                ),
                "silhouette": subset_silhouette,
                "cluster_sizes": ";".join(map(str, cluster_sizes)),
            }
        )
    leave_one = pd.DataFrame(leave_one_rows)
    leave_one.to_csv(output / "cultivar_texture_cluster_leave_one_out.csv", index=False)

    profile = medians.reset_index().melt(
        id_vars=["cultivar_ascii", "cultivar_code"], var_name="trait", value_name="cultivar_median"
    )
    z_long = robust_z.reset_index().melt(
        id_vars=["cultivar_ascii", "cultivar_code"], var_name="trait", value_name="robust_z"
    )
    z_clip_long = robust_z_clipped.reset_index().melt(
        id_vars=["cultivar_ascii", "cultivar_code"], var_name="trait", value_name="robust_z_clipped"
    )
    profile = profile.merge(z_long, on=["cultivar_ascii", "cultivar_code", "trait"]).merge(
        z_clip_long, on=["cultivar_ascii", "cultivar_code", "trait"]
    )
    cluster_map = pd.DataFrame(
        {
            "cultivar_ascii": medians.index.get_level_values("cultivar_ascii"),
            "cultivar_code": medians.index.get_level_values("cultivar_code"),
            "texture_cluster": labels,
        }
    )
    profile = profile.merge(cluster_map, on=["cultivar_ascii", "cultivar_code"])
    profile.to_csv(output / "cultivar_texture_profiles.csv", index=False)
    cluster_map.to_csv(output / "cultivar_texture_clusters.csv", index=False)
    pd.DataFrame(silhouette_rows).to_csv(output / "cultivar_texture_cluster_selection.csv", index=False)

    diversity_rows: list[dict[str, object]] = []
    for trait, group in texture.groupby("trait", observed=True):
        overall_mean = float(group["value"].mean())
        total_ss = float(np.square(group["value"] - overall_mean).sum())
        group_stats = group.groupby(["cultivar_ascii", "cultivar_code"], observed=True)["value"].agg(["count", "mean", "median"])
        between_ss = float(np.sum(group_stats["count"] * np.square(group_stats["mean"] - overall_mean)))
        highest = group_stats["median"].idxmax()
        lowest = group_stats["median"].idxmin()
        diversity_rows.append(
            {
                "trait": trait,
                "n_fruits": len(group),
                "n_cultivars": group["cultivar_ascii"].nunique(),
                "cultivar_associated_eta_squared": between_ss / total_ss if total_ss > 0 else np.nan,
                "overall_median": float(group["value"].median()),
                "overall_iqr": float(group["value"].quantile(0.75) - group["value"].quantile(0.25)),
                "lowest_median_cultivar_ascii": lowest[0],
                "lowest_median_cultivar_code": lowest[1],
                "lowest_cultivar_median": float(group_stats.loc[lowest, "median"]),
                "highest_median_cultivar_ascii": highest[0],
                "highest_median_cultivar_code": highest[1],
                "highest_cultivar_median": float(group_stats.loc[highest, "median"]),
            }
        )
    diversity = pd.DataFrame(diversity_rows)
    diversity.to_csv(output / "cultivar_texture_diversity.csv", index=False)
    return {
        "fruits": int(texture["sample_id"].nunique()),
        "cultivars": int(texture["cultivar_ascii"].nunique()),
        "traits": int(texture["trait"].nunique()),
        "selected_clusters": best_k,
        "selected_silhouette": best_score,
        "leave_one_max_membership_changes": int(
            leave_one["membership_changes_after_label_alignment"].max()
        ),
        "leave_one_unstable_removals": leave_one.loc[
            leave_one["membership_changes_after_label_alignment"] > 0,
            "removed_cultivar_ascii",
        ].astype(str).tolist(),
        "interpretation": "Descriptive cultivar-associated mechanical phenotype structure; not a genetic or environment-independent cultivar effect because cultivar and acquisition batch are partly confounded.",
    }


def analyze_qc(qc_path: Path, decision_path: Path, output: Path) -> dict[str, object]:
    qc = pd.read_csv(qc_path)
    selected_columns = [
        "cultivar_ascii", "fruit_count", "batch_count", "analysis_excluded_fraction",
        "spectral_severe_fraction", "median_trait_icc", "traits_with_icc_below_0_70",
        "median_replicate_cv", "independent_failure_domains", "exclude_from_primary_modeling", "decision",
    ]
    qc[selected_columns].to_csv(output / "cultivar_qc_decisions_condensed.csv", index=False)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    return {
        "cultivars_audited": int(len(qc)),
        "whole_cultivars_excluded": qc.loc[qc["exclude_from_primary_modeling"], "cultivar_ascii"].tolist(),
        "exclusion_rule": decision.get("whole_cultivar_rule"),
        "residuals_used_for_exclusion": decision.get("model_residuals_used"),
        "fruits_before": decision.get("fruit_count_before"),
        "fruits_after": decision.get("fruit_count_after_whole_cultivar_exclusion"),
        "excluded_fraction": decision.get("whole_cultivar_excluded_fraction"),
        "retained_with_sensitivity_review": decision.get("retained_with_sensitivity_review", []),
        "important_limitation": decision.get("important_limitation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("results/v24_hr_strengthening/analysis"))
    parser.add_argument("--shots", default="0,5,10,20,40")
    parser.add_argument("--fewshot-repeats", type=int, default=500)
    parser.add_argument("--bootstrap-iterations", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument("--slope-prior-strength", type=float, default=5.0)
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "crossbatch_predictions": root / "results/v21_crossbatch/final_analysis/v21_merged_predictions.parquet",
        "integrated_predictions": root / "results/v22_integrated/figure_data/predictions_all12.csv",
        "phenotype_long": root / "results/v22_integrated/figure_data/phenotype_long.csv",
        "cultivar_qc": root / "results/v4/cultivar_quality_audit/cultivar_measurement_quality_audit.csv",
        "cultivar_qc_decision": root / "results/v4/cultivar_quality_audit/cultivar_exclusion_decision.json",
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    report = {
        "analysis_version": "v24_hr_strengthening",
        "fewshot": analyze_fewshot(
            inputs["crossbatch_predictions"], output, [int(x) for x in args.shots.split(",")],
            args.fewshot_repeats, args.slope_prior_strength,
        ),
        "multiplicity": analyze_multiplicity(
            inputs["integrated_predictions"], output, args.bootstrap_iterations, args.bootstrap_seed
        ),
        "cultivar_texture": analyze_cultivar_texture(inputs["phenotype_long"], output),
        "quality_control": analyze_qc(inputs["cultivar_qc"], inputs["cultivar_qc_decision"], output),
        "provenance_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in [Path(__file__).resolve(), *inputs.values()]
        },
    }
    (output / "v24_hr_strengthening_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
