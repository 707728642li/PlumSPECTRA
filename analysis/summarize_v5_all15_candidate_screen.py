from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve()
    rows = []
    for statistics_path in sorted(candidate_root.glob("fixed/*/statistics/confirmation_statistics.json")):
        abbreviation = statistics_path.parents[1].name
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        run_summary = json.loads(
            (candidate_root / "fixed" / abbreviation / "summary.json").read_text(encoding="utf-8")
        )
        seed_table = pd.read_csv(statistics_path.parent / "seed_stability.csv")
        pooled_ci = statistics["cluster_bootstrap_95ci_and_median"]["pooled_rmse_improvement_pct"]
        rows.append(
            {
                "trait_abbreviation": abbreviation,
                "target": run_summary["target"],
                "fixed_gate": run_summary["fixed_residual_gate"],
                "cultivars": statistics["cultivars"],
                "fruits": statistics["fruits"],
                "seeds": len(statistics["seeds"]),
                "cultivar_wins": statistics["cultivar_wins"],
                "pooled_plsr_rmse": statistics["plsr_metrics"]["rmse"],
                "pooled_model_rmse": statistics["model_metrics"]["rmse"],
                "pooled_rmse_improvement_pct": statistics["pooled_rmse_improvement_pct"],
                "macro_plsr_rmse": statistics["macro_plsr_rmse"],
                "macro_model_rmse": statistics["macro_model_rmse"],
                "macro_rmse_improvement_pct": statistics["macro_rmse_improvement_pct"],
                "plsr_r2": statistics["plsr_metrics"]["r2"],
                "model_r2": statistics["model_metrics"]["r2"],
                "minimum_seed_improvement_pct": float(seed_table["rmse_improvement_pct"].min()),
                "pooled_cluster_bootstrap_ci_lower": float(pooled_ci[0]),
                "pooled_cluster_bootstrap_ci_upper": float(pooled_ci[2]),
                "wilcoxon_one_sided_p": statistics["cultivar_level_wilcoxon_one_sided"]["p_value"],
                "sign_flip_one_sided_p": statistics["cultivar_level_exact_sign_flip_one_sided_p"],
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No completed fixed-gate candidate results found under {candidate_root}")
    is_multiseed = int(table["seeds"].max()) >= 3
    table["eligible_for_multiseed"] = (
        (table["cultivar_wins"] >= 8)
        & (table["pooled_rmse_improvement_pct"] > 0)
        & (table["macro_rmse_improvement_pct"] > 0)
    )
    table["stable_across_seeds"] = table["eligible_for_multiseed"] & (
        table["minimum_seed_improvement_pct"] > 0
    )
    table["cluster_significant"] = table["pooled_cluster_bootstrap_ci_lower"] > 0
    table = table.sort_values(
        ["stable_across_seeds", "eligible_for_multiseed", "macro_rmse_improvement_pct", "pooled_rmse_improvement_pct"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_name = "all15_multiseed_summary.csv" if is_multiseed else "all15_single_seed_screen.csv"
    decision_name = "all15_multiseed_decision.json" if is_multiseed else "all15_single_seed_decision.json"
    table.to_csv(output_dir / table_name, index=False)
    decision = {
        "scope": (
            "all 15 retained cultivars, three-seed nested leave-one-cultivar-out ensemble"
            if is_multiseed
            else "all 15 retained cultivars, nested leave-one-cultivar-out, seed 20260806"
        ),
        "excluded_cultivars": ["6.11"],
        "multiseed_eligibility": (
            "at least 8/15 cultivar wins and positive pooled and macro RMSE improvements"
        ),
        "eligible_traits": table.loc[table["eligible_for_multiseed"], "trait_abbreviation"].tolist(),
        "stable_across_seed_traits": table.loc[table["stable_across_seeds"], "trait_abbreviation"].tolist(),
        "cluster_significant_traits": table.loc[table["cluster_significant"], "trait_abbreviation"].tolist(),
        "claim_boundary": (
            "Retrospective all-cultivar screen. It is not an untouched external confirmation because all "
            "cultivars were exposed during earlier development cycles."
        ),
    }
    (output_dir / decision_name).write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
