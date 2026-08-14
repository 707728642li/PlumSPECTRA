from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))


def summarize(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    pooled = rmse(frame["y_true"].to_numpy(), frame[prediction].to_numpy())
    cultivar = frame.groupby("cultivar_ascii", observed=True).apply(
        lambda group: rmse(group["y_true"].to_numpy(), group[prediction].to_numpy()),
        include_groups=False,
    )
    return {"pooled_rmse": pooled, "cultivar_macro_rmse": float(cultivar.mean())}


def cluster_bootstrap(
    frame: pd.DataFrame,
    reference: str,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    grouped = {cultivar: group for cultivar, group in frame.groupby("cultivar_ascii", observed=True)}
    cultivars = sorted(grouped)
    rng = np.random.default_rng(seed)
    pooled_improvement = np.empty(iterations, dtype=float)
    macro_improvement = np.empty(iterations, dtype=float)
    per_cultivar = {
        cultivar: (
            rmse(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            rmse(group["y_true"].to_numpy(), group[reference].to_numpy()),
        )
        for cultivar, group in grouped.items()
    }
    for iteration in range(iterations):
        sampled = rng.choice(cultivars, size=len(cultivars), replace=True)
        parts = [grouped[cultivar] for cultivar in sampled]
        truth = np.concatenate([part["y_true"].to_numpy() for part in parts])
        ai = np.concatenate([part["y_pred"].to_numpy() for part in parts])
        baseline = np.concatenate([part[reference].to_numpy() for part in parts])
        pooled_improvement[iteration] = 100.0 * (1.0 - rmse(truth, ai) / rmse(truth, baseline))
        ai_macro = np.mean([per_cultivar[cultivar][0] for cultivar in sampled])
        baseline_macro = np.mean([per_cultivar[cultivar][1] for cultivar in sampled])
        macro_improvement[iteration] = 100.0 * (1.0 - ai_macro / baseline_macro)
    return pooled_improvement, macro_improvement


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = float(np.mean(differences))
    permutations = np.asarray(list(product([-1.0, 1.0], repeat=len(differences))), dtype=float)
    permuted = np.mean(permutations * differences[None, :], axis=1)
    return float((np.sum(permuted >= observed) + 1) / (len(permuted) + 1)) if observed > 0 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.predictions)

    cultivar_rows = []
    for cultivar, group in frame.groupby("cultivar_ascii", observed=True):
        ai = rmse(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        domain = rmse(group["y_true"].to_numpy(), group["y_pls_anchor"].to_numpy())
        global_pls = rmse(group["y_true"].to_numpy(), group["y_global_pls_anchor"].to_numpy())
        cultivar_rows.append(
            {
                "cultivar_ascii": cultivar,
                "prediction_records": int(len(group)),
                "unique_fruits": int(group["sample_id"].nunique()),
                "ai_rmse": ai,
                "domain_pls_rmse": domain,
                "global_pls_rmse": global_pls,
                "ai_vs_domain_pls_pct": 100.0 * (1.0 - ai / domain),
                "ai_vs_global_pls_pct": 100.0 * (1.0 - ai / global_pls),
            }
        )
    cultivar_table = pd.DataFrame(cultivar_rows).sort_values("ai_vs_global_pls_pct")
    cultivar_table.to_csv(output_dir / "cultivar_metrics.csv", index=False)

    comparisons = {}
    for label, column in [("global_pls", "y_global_pls_anchor"), ("domain_pls", "y_pls_anchor")]:
        pooled, macro = cluster_bootstrap(
            frame,
            column,
            args.bootstrap_iterations,
            seed=20260807 + (0 if label == "global_pls" else 1),
        )
        improvement_column = f"ai_vs_{label}_pct"
        cultivar_improvement = cultivar_table[improvement_column].to_numpy()
        differences = (
            cultivar_table[f"{label}_rmse"].to_numpy() - cultivar_table["ai_rmse"].to_numpy()
        )
        comparisons[label] = {
            "ai": summarize(frame, "y_pred"),
            "baseline": summarize(frame, column),
            "pooled_relative_rmse_improvement_pct": float(
                100.0
                * (
                    1.0
                    - summarize(frame, "y_pred")["pooled_rmse"]
                    / summarize(frame, column)["pooled_rmse"]
                )
            ),
            "cultivar_macro_relative_rmse_improvement_pct": float(
                100.0
                * (
                    1.0
                    - summarize(frame, "y_pred")["cultivar_macro_rmse"]
                    / summarize(frame, column)["cultivar_macro_rmse"]
                )
            ),
            "cultivar_cluster_bootstrap_pooled_ci95_pct": [
                float(np.quantile(pooled, 0.025)),
                float(np.quantile(pooled, 0.975)),
            ],
            "cultivar_cluster_bootstrap_macro_ci95_pct": [
                float(np.quantile(macro, 0.025)),
                float(np.quantile(macro, 0.975)),
            ],
            "cultivar_wins": int(np.sum(cultivar_improvement > 0)),
            "cultivars": int(len(cultivar_improvement)),
            "wilcoxon_greater_p": float(wilcoxon(differences, alternative="greater").pvalue),
            "exact_cultivar_sign_flip_p": exact_sign_flip_pvalue(differences),
        }

    repeat_rows = []
    for repeat, group in frame.groupby("repeat", observed=True):
        ai = rmse(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        domain = rmse(group["y_true"].to_numpy(), group["y_pls_anchor"].to_numpy())
        global_pls = rmse(group["y_true"].to_numpy(), group["y_global_pls_anchor"].to_numpy())
        repeat_rows.append(
            {
                "repeat": int(repeat),
                "ai_rmse": ai,
                "domain_pls_rmse": domain,
                "global_pls_rmse": global_pls,
                "ai_vs_domain_pls_pct": 100.0 * (1.0 - ai / domain),
                "ai_vs_global_pls_pct": 100.0 * (1.0 - ai / global_pls),
            }
        )
    pd.DataFrame(repeat_rows).to_csv(output_dir / "repeat_metrics.csv", index=False)
    report = {
        "validation": "five repeated cultivar-stratified fruit holdouts",
        "prediction_records": int(len(frame)),
        "unique_fruits_in_any_test_split": int(frame["sample_id"].nunique()),
        "cultivars": int(frame["cultivar_ascii"].nunique()),
        "comparisons": comparisons,
        "inference_note": (
            "Cultivar-cluster bootstrap and paired cultivar tests treat cultivar as the independent cluster, "
            "thereby retaining all repeated predictions from the same fruit inside its cultivar cluster."
        ),
        "claim_boundary": (
            "Retrospective known-cultivar validation. Test fruit labels were not used for model fitting or selection; "
            "external year/orchard confirmation remains required."
        ),
    }
    (output_dir / "statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
