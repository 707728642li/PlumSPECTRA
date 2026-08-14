from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from summarize_v3_confirmation import cultivar_cluster_bootstrap, fold_metrics, regression_metrics, rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rule = protocol["seed_expansion_rule"]
    rows: list[dict[str, object]] = []
    fold_tables: list[pd.DataFrame] = []
    for trait in protocol["screening_traits"]:
        candidate_path = args.screen_root / trait / "predictions_ensemble.parquet"
        if not candidate_path.exists():
            raise FileNotFoundError(f"Incomplete cross-trait screen: {candidate_path}")
        candidate = pd.read_parquet(candidate_path)
        folds = fold_metrics(candidate, "y_pred")
        folds.insert(0, "trait", trait)
        fold_tables.append(folds)
        truth = candidate["y_true"].to_numpy(float)
        ai = candidate["y_pred"].to_numpy(float)
        pls = candidate["y_pls_anchor"].to_numpy(float)
        macro_pls = float(folds["plsr_rmse"].mean())
        macro_ai = float(folds["plumrac_x_rmse"].mean())
        macro_improvement = 100.0 * (macro_pls - macro_ai) / macro_pls
        pooled_improvement = 100.0 * (rmse(truth, pls) - rmse(truth, ai)) / rmse(truth, pls)
        worst = float(folds["rmse_improvement_pct"].min())
        wins = int(folds["plumrac_x_win"].sum())
        expand = bool(
            macro_improvement > float(rule["macro_rmse_improvement_pct_gt"])
            and pooled_improvement > float(rule["pooled_rmse_improvement_pct_gt"])
            and wins >= int(rule["minimum_fold_wins"])
            and worst >= -float(rule["maximum_worst_fold_degradation_pct"])
        )
        bootstrap = cultivar_cluster_bootstrap(folds)

        v2 = pd.read_parquet(args.v2_root / trait / "predictions_ensemble.parquet")
        v2_folds = fold_metrics(v2, "y_pred")
        v2_truth = v2["y_true"].to_numpy(float)
        v2_ai = v2["y_pred"].to_numpy(float)
        rows.append(
            {
                "trait": trait,
                "n": int(len(candidate)),
                "cultivars": int(len(folds)),
                "fold_wins": wins,
                "macro_plsr_rmse": macro_pls,
                "macro_plumrac_x_rmse": macro_ai,
                "macro_rmse_improvement_pct": macro_improvement,
                "pooled_plsr_rmse": rmse(truth, pls),
                "pooled_plumrac_x_rmse": rmse(truth, ai),
                "pooled_rmse_improvement_pct": pooled_improvement,
                "worst_fold_improvement_pct": worst,
                "bootstrap_ci95_lower_pct": bootstrap["ci95_lower_pct"],
                "bootstrap_ci95_upper_pct": bootstrap["ci95_upper_pct"],
                "seed_expansion": expand,
                "plumrac_x_r2": regression_metrics(truth, ai)["r2"],
                "v2_fold_wins": int(v2_folds["plumrac_x_win"].sum()),
                "v2_macro_rmse": float(v2_folds["plumrac_x_rmse"].mean()),
                "v2_pooled_rmse": rmse(v2_truth, v2_ai),
            }
        )
    summary = pd.DataFrame(rows).sort_values("trait").reset_index(drop=True)
    selected = summary.loc[summary["seed_expansion"], "trait"].tolist()
    decision = {
        "protocol": str(args.protocol.resolve()),
        "additional_seed_traits": selected,
        "additional_seeds": protocol["additional_seeds"],
        "rule": rule,
        "selection_is_automatic": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "cross_trait_screen_summary.csv", index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(args.output_dir / "cross_trait_fold_metrics.csv", index=False)
    (args.output_dir / "seed_expansion_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
