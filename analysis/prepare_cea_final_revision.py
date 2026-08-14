"""Prepare evidence tables introduced by the final CEA text/logic revision.

The primary addition is a post-hoc 12-trait comparison against the best
branch-excluded baseline.  PlumSPECTRA's two individual branch predictors
(residual CNN and nested RBF-SVR) are excluded from this pool; the eligible
rows are global PLSR, cultivar-aware PLSR and the distinct no-neural B50
ensemble.  B50 shares the nested-SVR prediction but is not an individual
PlumSPECTRA branch.  The cultivar-cluster bootstrap is identical in scale and
seed to the existing release analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/v25_external_review_corrections/final_analysis"
OUTPUT = ROOT / "results/cea_final_revision"
FIGURE_DATA = ROOT / "results/v26_claudecode_integration/figure_data"
PREDICTIONS = EVIDENCE / "v25_integrated_predictions.parquet"
ITERATIONS = 1_000_000
SEED = 20260808

TRAITS = ["FW", "SSC", "pH", "SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"]
BASELINES = {
    "Global PLSR": "y_global_pls",
    "Cultivar-aware PLSR": "y_domain_pls",
    "No-neural B50": "y_b50",
}


def cluster_sse(frame: pd.DataFrame, prediction: str, cultivars: list[str]) -> np.ndarray:
    return np.asarray(
        [
            np.square(
                part["y_true"].to_numpy(float) - part[prediction].to_numpy(float)
            ).sum()
            for cultivar in cultivars
            for part in [frame.loc[frame["cultivar_ascii"].eq(cultivar)]]
        ],
        dtype=float,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(PREDICTIONS)
    required = {"cultivar_ascii", "trait", "y_true", "y_final", *BASELINES.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    cultivars = sorted(frame["cultivar_ascii"].unique())
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(cultivars), size=(ITERATIONS, len(cultivars)), dtype=np.int16)
    counts = np.zeros((ITERATIONS, len(cultivars)), dtype=np.float32)
    np.add.at(counts, (np.arange(ITERATIONS)[:, None], draws), 1.0)
    del draws

    rows: list[dict[str, object]] = []
    boot_vectors: list[np.ndarray] = []
    for trait in TRAITS:
        group = frame.loc[frame["trait"].eq(trait)]
        n = np.asarray(
            [float(group["cultivar_ascii"].eq(cultivar).sum()) for cultivar in cultivars],
            dtype=float,
        )
        boot_n = counts @ n
        candidate_sse = cluster_sse(group, "y_final", cultivars)
        candidate_rmse = float(np.sqrt(candidate_sse.sum() / n.sum()))
        boot_candidate_rmse = np.sqrt((counts @ candidate_sse) / boot_n)

        pooled: dict[str, tuple[float, np.ndarray]] = {}
        for label, column in BASELINES.items():
            sse = cluster_sse(group, column, cultivars)
            pooled[label] = (
                float(np.sqrt(sse.sum() / n.sum())),
                np.sqrt((counts @ sse) / boot_n),
            )
        strongest = min(pooled, key=lambda label: pooled[label][0])
        strongest_rmse, boot_strongest_rmse = pooled[strongest]
        effect = 100.0 * (boot_strongest_rmse - boot_candidate_rmse) / boot_strongest_rmse
        point = 100.0 * (strongest_rmse - candidate_rmse) / strongest_rmse
        boot_vectors.append(effect.astype(np.float32))
        rows.append(
            {
                "trait": trait,
                "candidate": "PlumSPECTRA",
                "strongest_baseline": strongest,
                "selection_rule": (
                    "lowest pooled OOF RMSE among global PLSR, cultivar-aware PLSR "
                    "and no-neural B50; both individual PlumSPECTRA branches excluded"
                ),
                "family_status": "post-hoc branch-excluded 12-contrast sensitivity",
                "n_fruits": int(len(group)),
                "n_cultivars": len(cultivars),
                "candidate_rmse": candidate_rmse,
                "strongest_baseline_rmse": strongest_rmse,
                "relative_rmse_improvement_pct": point,
                "bootstrap_ci95_low_pct": float(np.quantile(effect, 0.025)),
                "bootstrap_ci95_high_pct": float(np.quantile(effect, 0.975)),
                "bootstrap_probability_better": float(np.mean(effect > 0)),
                "p_raw_one_sided": (1.0 + float(np.sum(effect <= 0))) / (ITERATIONS + 1.0),
            }
        )

    result = pd.DataFrame(rows)
    result["p_bh_fdr"] = multipletests(result["p_raw_one_sided"], method="fdr_bh")[1]
    result["p_holm"] = multipletests(result["p_raw_one_sided"], method="holm")[1]
    matrix = np.column_stack(boot_vectors).astype(np.float32)
    points = result["relative_rmse_improvement_pct"].to_numpy(float)
    se = np.std(matrix, axis=0, ddof=1)
    studentized = np.divide(
        matrix - points[None, :], se[None, :], out=np.zeros_like(matrix), where=se[None, :] > 0
    )
    critical = float(np.quantile(np.max(np.abs(studentized), axis=1), 0.95))
    result["simultaneous_ci95_low_pct"] = points - critical * se
    result["simultaneous_ci95_high_pct"] = points + critical * se
    result["supported_raw_0_05"] = result["p_raw_one_sided"] < 0.05
    result["supported_bh_fdr_0_05"] = result["p_bh_fdr"] < 0.05
    result["supported_holm_0_05"] = result["p_holm"] < 0.05
    result["supported_simultaneous_0_05"] = result["simultaneous_ci95_low_pct"] > 0
    result["simultaneous_critical_value"] = critical
    result.to_csv(OUTPUT / "multiplicity_independent_strongest_family.csv", index=False)
    result.to_csv(EVIDENCE / "multiplicity_independent_strongest_family.csv", index=False)
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    result.to_csv(FIGURE_DATA / "multiplicity_independent_strongest_family.csv", index=False)

    pooled = pd.read_csv(EVIDENCE / "pooled_metrics.csv")
    pivot = pooled.loc[
        pooled["model"].isin(["global_pls", "cultivar_mean_null"]),
        ["trait", "model", "rmse", "r2"],
    ].pivot(index="trait", columns="model", values=["rmse", "r2"])
    null_comparison = pd.DataFrame(
        {
            "trait": TRAITS,
            "global_pls_rmse": [pivot.loc[t, ("rmse", "global_pls")] for t in TRAITS],
            "cultivar_mean_null_rmse": [pivot.loc[t, ("rmse", "cultivar_mean_null")] for t in TRAITS],
            "global_pls_r2": [pivot.loc[t, ("r2", "global_pls")] for t in TRAITS],
            "cultivar_mean_null_r2": [pivot.loc[t, ("r2", "cultivar_mean_null")] for t in TRAITS],
        }
    )
    null_comparison["global_pls_worse_than_null"] = (
        null_comparison["global_pls_rmse"] > null_comparison["cultivar_mean_null_rmse"]
    )
    null_comparison.to_csv(OUTPUT / "global_pls_vs_cultivar_mean_null.csv", index=False)

    summary = {
        "bootstrap_iterations": ITERATIONS,
        "bootstrap_seed": SEED,
        "cultivar_clusters": len(cultivars),
        "eligible_branch_excluded_baselines": list(BASELINES),
        "gain_range_pct": [
            float(result["relative_rmse_improvement_pct"].min()),
            float(result["relative_rmse_improvement_pct"].max()),
        ],
        "baseline_counts": result["strongest_baseline"].value_counts().to_dict(),
        "supported_raw": int(result["supported_raw_0_05"].sum()),
        "supported_bh_fdr": int(result["supported_bh_fdr_0_05"].sum()),
        "supported_holm": int(result["supported_holm_0_05"].sum()),
        "supported_simultaneous": int(result["supported_simultaneous_0_05"].sum()),
        "global_pls_worse_than_cultivar_mean_null": int(
            null_comparison["global_pls_worse_than_null"].sum()
        ),
    }
    (OUTPUT / "cea_final_revision_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
