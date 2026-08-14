from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="ABBREVIATION=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary_rows = []
    gate_rows = []
    fold_rows = []
    for item in args.run:
        abbreviation, raw_path = item.split("=", maxsplit=1)
        run_dir = Path(raw_path).resolve()
        prediction = pd.read_parquet(run_dir / "predictions_by_seed.parquet")
        targets = prediction["target"].unique().tolist()
        if len(targets) != 1:
            raise ValueError(f"Expected one target in {run_dir}, found {targets}")
        target = targets[0]
        evidence = []
        for path in run_dir.glob("runs/*/*/seed_*/metadata.json"):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            for row in metadata["gate_scores"]:
                evidence.append(
                    {
                        "gate": float(row["gate"]),
                        "selection_score": float(row["selection_score"]),
                        "relative_improvement": float(row["relative_improvement_vs_zero"]),
                        "win_fraction": float(row["group_win_fraction"]),
                        "worst_improvement": float(row["worst_group_improvement"]),
                    }
                )
        evidence_table = pd.DataFrame(evidence)
        aggregate = (
            evidence_table.groupby("gate", as_index=False)
            .agg(
                mean_selection_score=("selection_score", "mean"),
                mean_relative_improvement=("relative_improvement", "mean"),
                mean_win_fraction=("win_fraction", "mean"),
                mean_worst_improvement=("worst_improvement", "mean"),
            )
            .sort_values(["mean_selection_score", "gate"])
        )
        selected_gate = float(aggregate.iloc[0]["gate"])
        aggregate["selected"] = aggregate["gate"].eq(selected_gate)
        aggregate.insert(0, "trait_abbreviation", abbreviation)
        aggregate.insert(1, "target", target)
        gate_rows.append(aggregate)

        prediction = prediction.copy()
        prediction["y_fixed_gate"] = prediction["y_pls_anchor"] + selected_gate * prediction["deep_residual"]
        trait_folds = []
        for cultivar, group in prediction.groupby("cultivar_ascii", observed=True):
            truth = group["y_true"].to_numpy(float)
            model_score = rmse(truth, group["y_fixed_gate"].to_numpy(float))
            anchor_score = rmse(truth, group["y_pls_anchor"].to_numpy(float))
            trait_folds.append(
                {
                    "trait_abbreviation": abbreviation,
                    "target": target,
                    "cultivar_ascii": cultivar,
                    "cultivar_code": group["cultivar_code"].iloc[0],
                    "n": int(len(group)),
                    "fixed_gate": selected_gate,
                    "plsr_rmse": anchor_score,
                    "model_rmse": model_score,
                    "rmse_improvement_pct": 100.0 * (anchor_score - model_score) / anchor_score,
                    "model_win": bool(model_score < anchor_score),
                }
            )
        folds = pd.DataFrame(trait_folds)
        fold_rows.append(folds)
        truth = prediction["y_true"].to_numpy(float)
        model_prediction = prediction["y_fixed_gate"].to_numpy(float)
        anchor_prediction = prediction["y_pls_anchor"].to_numpy(float)
        model_pooled = rmse(truth, model_prediction)
        anchor_pooled = rmse(truth, anchor_prediction)
        model_macro = float(folds["model_rmse"].mean())
        anchor_macro = float(folds["plsr_rmse"].mean())
        summary_rows.append(
            {
                "trait_abbreviation": abbreviation,
                "target": target,
                "n": int(len(prediction)),
                "development_cultivars": int(folds["cultivar_ascii"].nunique()),
                "fixed_gate": selected_gate,
                "fold_wins": int(folds["model_win"].sum()),
                "pooled_plsr_rmse": anchor_pooled,
                "pooled_model_rmse": model_pooled,
                "pooled_rmse_improvement_pct": 100.0 * (anchor_pooled - model_pooled) / anchor_pooled,
                "macro_plsr_rmse": anchor_macro,
                "macro_model_rmse": model_macro,
                "macro_rmse_improvement_pct": 100.0 * (anchor_macro - model_macro) / anchor_macro,
                "plsr_r2": float(r2_score(truth, anchor_prediction)),
                "model_r2": float(r2_score(truth, model_prediction)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["eligible_for_full_confirmation"] = (
        (summary["fold_wins"] >= 3)
        & (summary["pooled_rmse_improvement_pct"] > 0)
        & (summary["macro_rmse_improvement_pct"] > 0)
    )
    summary = summary.sort_values(
        ["eligible_for_full_confirmation", "macro_rmse_improvement_pct", "pooled_rmse_improvement_pct"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "trait_screen_summary.csv", index=False)
    pd.concat(gate_rows, ignore_index=True).to_csv(output_dir / "gate_selection_by_trait.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(output_dir / "fold_scores.csv", index=False)
    decision = {
        "selection_scope": "five historical development cultivars per trait",
        "one_final_model_per_trait": True,
        "gate_selection": "minimum mean source-validation score within each trait",
        "full_confirmation_eligibility": "at least 3/5 wins and positive macro and pooled RMSE improvement",
        "eligible_traits": summary.loc[
            summary["eligible_for_full_confirmation"], "trait_abbreviation"
        ].tolist(),
        "claim_boundary": "Retrospective development screen; no result is an untouched external confirmation.",
    }
    (output_dir / "trait_screen_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
