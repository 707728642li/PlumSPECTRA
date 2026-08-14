from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(truth)),
        "rmse": rmse(truth, prediction),
        "mae": float(mean_absolute_error(truth, prediction)),
        "bias": float(np.mean(prediction - truth)),
        "r2": float(r2_score(truth, prediction)),
        "pearson_r": float(np.corrcoef(truth, prediction)[0, 1]),
    }


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = float(np.mean(differences))
    if observed <= 0:
        return 1.0
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(float(np.mean(differences * np.asarray(signs))))
    return float((np.sum(np.asarray(values) >= observed - 1e-15) + 1) / (len(values) + 1))


def bootstrap_cultivars(
    predictions: pd.DataFrame,
    fold_scores: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    cultivars = sorted(predictions["cultivar_ascii"].unique().tolist())
    cluster = {cultivar: predictions.loc[predictions["cultivar_ascii"].eq(cultivar)] for cultivar in cultivars}
    fold = fold_scores.set_index("cultivar_ascii")
    pooled_improvements = np.empty(repetitions, dtype=float)
    macro_improvements = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.choice(cultivars, size=len(cultivars), replace=True)
        model_sse = 0.0
        anchor_sse = 0.0
        observations = 0
        model_fold = []
        anchor_fold = []
        for cultivar in selected:
            group = cluster[cultivar]
            truth = group["y_true"].to_numpy(float)
            model_sse += float(np.sum(np.square(truth - group["y_pred"].to_numpy(float))))
            anchor_sse += float(np.sum(np.square(truth - group["y_pls_anchor"].to_numpy(float))))
            observations += len(group)
            model_fold.append(float(fold.loc[cultivar, "model_rmse"]))
            anchor_fold.append(float(fold.loc[cultivar, "plsr_rmse"]))
        model_pooled = np.sqrt(model_sse / observations)
        anchor_pooled = np.sqrt(anchor_sse / observations)
        pooled_improvements[index] = 100.0 * (anchor_pooled - model_pooled) / anchor_pooled
        macro_model = float(np.mean(model_fold))
        macro_anchor = float(np.mean(anchor_fold))
        macro_improvements[index] = 100.0 * (macro_anchor - macro_model) / macro_anchor
    return {
        "pooled_rmse_improvement_pct": np.quantile(pooled_improvements, [0.025, 0.50, 0.975]).tolist(),
        "macro_rmse_improvement_pct": np.quantile(macro_improvements, [0.025, 0.50, 0.975]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata-run-dir",
        type=Path,
        action="append",
        help="Optional source training directory containing runs/*/*/seed_*/metadata.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metadata_run_dirs = (
        [path.resolve() for path in args.metadata_run_dir]
        if args.metadata_run_dir
        else [run_dir]
    )
    run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    ensemble = pd.read_parquet(run_dir / "predictions_ensemble.parquet")
    by_seed = pd.read_parquet(run_dir / "predictions_by_seed.parquet")
    fold_rows = []
    for cultivar, group in ensemble.groupby("cultivar_ascii", observed=True):
        truth = group["y_true"].to_numpy(float)
        model_score = rmse(truth, group["y_pred"].to_numpy(float))
        anchor_score = rmse(truth, group["y_pls_anchor"].to_numpy(float))
        fold_rows.append(
            {
                "cultivar_ascii": cultivar,
                "cultivar_code": group["cultivar_code"].iloc[0],
                "n": int(len(group)),
                "plsr_rmse": anchor_score,
                "model_rmse": model_score,
                "rmse_difference": anchor_score - model_score,
                "rmse_improvement_pct": 100.0 * (anchor_score - model_score) / anchor_score,
                "model_win": bool(model_score < anchor_score),
            }
        )
    folds = pd.DataFrame(fold_rows).sort_values("cultivar_code").reset_index(drop=True)

    seed_rows = []
    for seed, group in by_seed.groupby("seed", observed=True):
        truth = group["y_true"].to_numpy(float)
        model_score = rmse(truth, group["y_pred"].to_numpy(float))
        anchor_score = rmse(truth, group["y_pls_anchor"].to_numpy(float))
        seed_rows.append(
            {
                "seed": int(seed),
                "n": int(len(group)),
                "plsr_rmse": anchor_score,
                "model_rmse": model_score,
                "rmse_improvement_pct": 100.0 * (anchor_score - model_score) / anchor_score,
            }
        )
    seeds = pd.DataFrame(seed_rows).sort_values("seed")

    metadata_rows = []
    for metadata_run_dir in metadata_run_dirs:
        for path in metadata_run_dir.glob("runs/*/*/seed_*/metadata.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata_rows.append(
                {
                    "heldout_cultivar": metadata["heldout_cultivar"],
                    "seed": int(metadata["seed"]),
                    "selected_epoch": int(metadata["selected_epoch"]),
                    "selected_gate": float(metadata["selected_gate"]),
                    "selected_objective_profile": metadata["selected_objective_profile"],
                    "pls_preprocessing": metadata["pls_anchor"]["preprocessing"],
                    "pls_components": int(metadata["pls_anchor"]["n_components"]),
                }
            )
    metadata_table = pd.DataFrame(metadata_rows).sort_values(["heldout_cultivar", "seed"])

    truth = ensemble["y_true"].to_numpy(float)
    model_prediction = ensemble["y_pred"].to_numpy(float)
    anchor_prediction = ensemble["y_pls_anchor"].to_numpy(float)
    model_metrics = metrics(truth, model_prediction)
    anchor_metrics = metrics(truth, anchor_prediction)
    macro_model = float(folds["model_rmse"].mean())
    macro_anchor = float(folds["plsr_rmse"].mean())
    differences = folds["rmse_difference"].to_numpy(float)
    wilcoxon_result = wilcoxon(differences, alternative="greater", zero_method="wilcox")
    bootstrap = bootstrap_cultivars(
        ensemble,
        folds,
        repetitions=args.bootstrap_repetitions,
        seed=args.bootstrap_seed,
    )
    summary = {
        "model": str(run_summary.get("model", "PLUMRAC-MS V4")),
        "comparison": "same-fold nested PLSR anchor",
        "cultivars": int(folds["cultivar_ascii"].nunique()),
        "fruits": int(len(ensemble)),
        "seeds": seeds["seed"].astype(int).tolist(),
        "model_metrics": model_metrics,
        "plsr_metrics": anchor_metrics,
        "pooled_rmse_improvement_pct": 100.0
        * (float(anchor_metrics["rmse"]) - float(model_metrics["rmse"]))
        / float(anchor_metrics["rmse"]),
        "macro_model_rmse": macro_model,
        "macro_plsr_rmse": macro_anchor,
        "macro_rmse_improvement_pct": 100.0 * (macro_anchor - macro_model) / macro_anchor,
        "cultivar_wins": int(folds["model_win"].sum()),
        "cultivar_losses": int((~folds["model_win"]).sum()),
        "cluster_bootstrap_95ci_and_median": bootstrap,
        "cultivar_level_wilcoxon_one_sided": {
            "statistic": float(wilcoxon_result.statistic),
            "p_value": float(wilcoxon_result.pvalue),
        },
        "cultivar_level_exact_sign_flip_one_sided_p": exact_sign_flip_pvalue(differences),
        "gate_selection": {
            "deployment_fixed_gate": run_summary.get("fixed_residual_gate"),
            "metadata_note": (
                "Counts below describe fold-internal candidate gates recorded during training; "
                "deployment_fixed_gate is the gate actually used for reported predictions."
            ),
            "nonzero_fraction": float((metadata_table["selected_gate"] > 0).mean()),
            "counts": {
                str(key): int(value)
                for key, value in metadata_table["selected_gate"].value_counts().sort_index().items()
            },
            "median_selected_epoch": float(metadata_table["selected_epoch"].median()),
            "objective_profile_counts": {
                str(key): int(value)
                for key, value in metadata_table["selected_objective_profile"].value_counts().items()
            },
        },
        "claim_boundary": (
            "Cultivar-cluster inference is primary. This is retrospective nested-LOCO evidence, not an untouched "
            "external validation cohort, because cultivars were exposed in earlier development cycles."
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "cultivar_fold_comparison.csv", index=False)
    seeds.to_csv(output_dir / "seed_stability.csv", index=False)
    metadata_table.to_csv(output_dir / "selection_and_gate_audit.csv", index=False)
    (output_dir / "confirmation_statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(folds.to_string(index=False), flush=True)
    print(seeds.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
