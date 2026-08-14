from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from v2_registry import trait_registry


THRESHOLDS = {
    "minimum_pooled_rmse_improvement_pct": 2.0,
    "minimum_fold_wins": 9,
    "minimum_nonzero_gates": 8,
    "minimum_bootstrap_probability_improvement": 0.80,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen V2 one-seed screen for five-seed expansion."
    )
    parser.add_argument("--model-comparison", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    comparison = pd.read_csv(args.model_comparison.resolve() / "plumrac_vs_plsr.csv")
    gates = pd.read_csv(args.gate_audit.resolve() / "gate_selection_summary.csv")
    merged = comparison.merge(gates, on="trait", validate="one_to_one")
    merged["nonzero_gates"] = merged["gate_0.25"] + merged["gate_0.50"]

    merged["pass_rmse"] = (
        merged["pooled_rmse_improvement_pct"]
        >= THRESHOLDS["minimum_pooled_rmse_improvement_pct"]
    )
    merged["pass_fold_wins"] = merged["fold_wins"] >= THRESHOLDS["minimum_fold_wins"]
    merged["pass_nonzero_gates"] = (
        merged["nonzero_gates"] >= THRESHOLDS["minimum_nonzero_gates"]
    )
    merged["pass_bootstrap_probability"] = (
        merged["probability_improvement"]
        > THRESHOLDS["minimum_bootstrap_probability_improvement"]
    )
    pass_columns = [
        "pass_rmse",
        "pass_fold_wins",
        "pass_nonzero_gates",
        "pass_bootstrap_probability",
    ]
    merged["eligible_for_five_seed_expansion"] = merged[pass_columns].all(axis=1)

    order = trait_registry().loc[
        lambda frame: frame["model_family"] == "endpoint", "abbreviation"
    ].tolist()
    merged["trait"] = pd.Categorical(merged["trait"], order, ordered=True)
    merged = merged.sort_values("trait")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "seed_expansion_decision.csv", index=False)
    report = {
        "policy_status": "frozen before confirmation endpoint inspection",
        "screen_seed": 20260806,
        "additional_seeds_if_eligible": [20260807, 20260808, 20260809, 20260810],
        "thresholds": THRESHOLDS,
        "eligible_traits": merged.loc[
            merged["eligible_for_five_seed_expansion"], "trait"
        ].astype(str).tolist(),
        "decisions": merged.assign(trait=merged["trait"].astype(str)).to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"eligible_traits": report["eligible_traits"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
