from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def exact_sign_flip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.abs(differences) > 1e-12]
    if len(differences) == 0:
        return 1.0
    observed = float(np.mean(differences))
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(differences))))
    permuted = np.mean(signs * differences[None, :], axis=1)
    return float((np.sum(permuted >= observed - 1e-15) + 1) / (len(permuted) + 1))


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    aliases = {"y_pred": "prediction"}
    return frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-by-seed", type=Path)
    parser.add_argument("--candidate-by-seed", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    args = parser.parse_args()

    keys = ["sample_id", "cultivar_ascii", "cultivar_code", "target", "y_true"]
    baseline = load_predictions(args.baseline).sort_values("sample_id").reset_index(drop=True)
    candidate = load_predictions(args.candidate).sort_values("sample_id").reset_index(drop=True)
    if not baseline[keys].equals(candidate[keys]):
        raise ValueError("Confirmed predictions are not aligned")
    comparison = baseline[keys].copy()
    comparison["baseline_prediction"] = baseline["prediction"].to_numpy(float)
    comparison["candidate_prediction"] = candidate["prediction"].to_numpy(float)

    cultivar_rows: list[dict[str, float | int | str | bool]] = []
    for cultivar, group in comparison.groupby("cultivar_ascii", sort=True):
        truth = group["y_true"].to_numpy(float)
        baseline_prediction = group["baseline_prediction"].to_numpy(float)
        candidate_prediction = group["candidate_prediction"].to_numpy(float)
        baseline_rmse = rmse(truth, baseline_prediction)
        candidate_rmse = rmse(truth, candidate_prediction)
        cultivar_rows.append(
            {
                "cultivar_ascii": cultivar,
                "n": len(group),
                "baseline_rmse": baseline_rmse,
                "candidate_rmse": candidate_rmse,
                "rmse_difference": baseline_rmse - candidate_rmse,
                "candidate_improvement_pct": 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse,
                "candidate_win": candidate_rmse < baseline_rmse,
            }
        )
    cultivar_metrics = pd.DataFrame(cultivar_rows)
    differences = cultivar_metrics["rmse_difference"].to_numpy(float)

    rng = np.random.default_rng(args.bootstrap_seed)
    cultivars = cultivar_metrics["cultivar_ascii"].tolist()
    groups = {cultivar: comparison.loc[comparison["cultivar_ascii"] == cultivar] for cultivar in cultivars}
    pooled_bootstrap = np.empty(args.bootstrap_repetitions, dtype=float)
    macro_bootstrap = np.empty(args.bootstrap_repetitions, dtype=float)
    for repetition in range(args.bootstrap_repetitions):
        sampled = rng.choice(cultivars, size=len(cultivars), replace=True)
        sampled_groups = [groups[cultivar] for cultivar in sampled]
        truth = np.concatenate([group["y_true"].to_numpy(float) for group in sampled_groups])
        baseline_prediction = np.concatenate(
            [group["baseline_prediction"].to_numpy(float) for group in sampled_groups]
        )
        candidate_prediction = np.concatenate(
            [group["candidate_prediction"].to_numpy(float) for group in sampled_groups]
        )
        baseline_rmse = rmse(truth, baseline_prediction)
        candidate_rmse = rmse(truth, candidate_prediction)
        pooled_bootstrap[repetition] = 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse
        fold_improvements = []
        for group in sampled_groups:
            fold_truth = group["y_true"].to_numpy(float)
            fold_baseline = rmse(fold_truth, group["baseline_prediction"].to_numpy(float))
            fold_candidate = rmse(fold_truth, group["candidate_prediction"].to_numpy(float))
            fold_improvements.append(100.0 * (fold_baseline - fold_candidate) / fold_baseline)
        macro_bootstrap[repetition] = float(np.mean(fold_improvements))

    truth = comparison["y_true"].to_numpy(float)
    baseline_prediction = comparison["baseline_prediction"].to_numpy(float)
    candidate_prediction = comparison["candidate_prediction"].to_numpy(float)
    baseline_rmse = rmse(truth, baseline_prediction)
    candidate_rmse = rmse(truth, candidate_prediction)
    wilcoxon_result = wilcoxon(differences, alternative="greater", zero_method="wilcox")
    summary: dict[str, object] = {
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "fruits": int(len(comparison)),
        "cultivars": int(len(cultivar_metrics)),
        "baseline_pooled_rmse": baseline_rmse,
        "candidate_pooled_rmse": candidate_rmse,
        "candidate_pooled_improvement_pct": 100.0 * (baseline_rmse - candidate_rmse) / baseline_rmse,
        "baseline_macro_rmse": float(cultivar_metrics["baseline_rmse"].mean()),
        "candidate_macro_rmse": float(cultivar_metrics["candidate_rmse"].mean()),
        "candidate_macro_improvement_pct": 100.0
        * (
            cultivar_metrics["baseline_rmse"].mean()
            - cultivar_metrics["candidate_rmse"].mean()
        )
        / cultivar_metrics["baseline_rmse"].mean(),
        "candidate_cultivar_wins": int(cultivar_metrics["candidate_win"].sum()),
        "cluster_bootstrap_95ci_and_median": {
            "pooled_improvement_pct": np.quantile(pooled_bootstrap, [0.025, 0.5, 0.975]).tolist(),
            "macro_improvement_pct": np.quantile(macro_bootstrap, [0.025, 0.5, 0.975]).tolist(),
        },
        "cultivar_level_wilcoxon_one_sided": {
            "statistic": float(wilcoxon_result.statistic),
            "p_value": float(wilcoxon_result.pvalue),
        },
        "cultivar_level_exact_sign_flip_one_sided_p": exact_sign_flip_p(differences),
    }

    seed_metrics = pd.DataFrame()
    if args.baseline_by_seed and args.candidate_by_seed:
        baseline_seed = load_predictions(args.baseline_by_seed)
        candidate_seed = load_predictions(args.candidate_by_seed)
        seed_keys = ["sample_id", "cultivar_ascii", "target", "seed", "y_true"]
        baseline_seed = baseline_seed.sort_values(seed_keys).reset_index(drop=True)
        candidate_seed = candidate_seed.sort_values(seed_keys).reset_index(drop=True)
        if not baseline_seed[seed_keys].equals(candidate_seed[seed_keys]):
            raise ValueError("Seed-level predictions are not aligned")
        seed_rows = []
        for seed in sorted(baseline_seed["seed"].unique()):
            mask = baseline_seed["seed"] == seed
            seed_truth = baseline_seed.loc[mask, "y_true"].to_numpy(float)
            seed_baseline = rmse(seed_truth, baseline_seed.loc[mask, "prediction"].to_numpy(float))
            seed_candidate = rmse(seed_truth, candidate_seed.loc[mask, "prediction"].to_numpy(float))
            seed_rows.append(
                {
                    "seed": int(seed),
                    "baseline_rmse": seed_baseline,
                    "candidate_rmse": seed_candidate,
                    "candidate_improvement_pct": 100.0 * (seed_baseline - seed_candidate) / seed_baseline,
                }
            )
        seed_metrics = pd.DataFrame(seed_rows)
        summary["seed_comparison"] = seed_metrics.to_dict("records")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(args.output_dir / "paired_predictions.parquet", index=False, compression="zstd")
    cultivar_metrics.to_csv(args.output_dir / "cultivar_comparison.csv", index=False)
    if not seed_metrics.empty:
        seed_metrics.to_csv(args.output_dir / "seed_comparison.csv", index=False)
    (args.output_dir / "model_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(cultivar_metrics.to_string(index=False))
    if not seed_metrics.empty:
        print(seed_metrics.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
