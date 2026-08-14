from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from v2_registry import trait_registry


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-comparison", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--fewshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison_dir = args.model_comparison.resolve()
    gate_dir = args.gate_audit.resolve()
    fewshot_dir = args.fewshot.resolve()
    registry = trait_registry().loc[lambda frame: frame["model_family"] == "endpoint"]
    order = registry["abbreviation"].tolist()

    models = pd.read_csv(comparison_dir / "model_summary.csv")
    deltas = pd.read_csv(comparison_dir / "plumrac_vs_plsr.csv")
    gates = pd.read_csv(gate_dir / "gate_selection_summary.csv")
    fewshot = pd.read_csv(fewshot_dir / "fewshot_summary.csv")

    zero_rows = []
    for trait in order:
        for model in ["PLSR", "Ridge", "PlumRAC-Net"]:
            row = models.loc[(models["trait"] == trait) & (models["model"] == model)]
            if row.empty:
                continue
            item = row.iloc[0]
            zero_rows.append(
                {
                    "Trait": trait,
                    "Model": model,
                    "RMSE": f"{item['rmse']:.4g}",
                    "R2": f"{item['r2']:.3f}",
                    "CCC": f"{item['ccc']:.3f}",
                    "Centred R2": f"{item['centred_r2']:.3f}",
                    "RPD": f"{item['rpd']:.3f}",
                }
            )
    zero_table = pd.DataFrame(zero_rows)

    best_rows = []
    for trait in order:
        candidates = models.loc[models["trait"] == trait].sort_values("rmse")
        if candidates.empty:
            continue
        best = candidates.iloc[0]
        best_rows.append(
            {
                "Trait": trait,
                "Lowest-RMSE model": best["model"],
                "RMSE": f"{best['rmse']:.4g}",
                "R2": f"{best['r2']:.3f}",
                "CCC": f"{best['ccc']:.3f}",
                "Deployment reading": "positive zero-shot skill" if best["r2"] > 0 else "no pooled zero-shot skill",
            }
        )
    best_table = pd.DataFrame(best_rows)

    delta_table = deltas.copy().set_index("trait").reindex(order).reset_index()
    delta_table = pd.DataFrame(
        {
            "Trait": delta_table["trait"],
            "Fold wins": delta_table["fold_wins"].astype(int).astype(str) + "/" + delta_table["folds"].astype(int).astype(str),
            "Pooled RMSE gain (%)": delta_table["pooled_rmse_improvement_pct"].map(lambda value: f"{value:+.2f}"),
            "Cluster-bootstrap mean (%)": delta_table["rmse_improvement_pct"].map(lambda value: f"{value:+.2f}"),
            "95% cultivar CI": delta_table.apply(lambda row: f"[{row['ci025']:+.2f}, {row['ci975']:+.2f}]", axis=1),
            "Interpretation": delta_table["interpretation"],
        }
    )

    gate_table = gates.set_index("trait").reindex(order).reset_index()
    gate_table = pd.DataFrame(
        {
            "Trait": gate_table["trait"],
            "g=0": gate_table["gate_0"].astype(int),
            "g=0.25": gate_table["gate_0.25"].astype(int),
            "g=0.50": gate_table["gate_0.50"].astype(int),
            "Nonzero (%)": gate_table["nonzero_gate_fraction"].map(lambda value: f"{100 * value:.1f}"),
            "Median epoch": gate_table["median_epoch"].map(lambda value: f"{value:.1f}"),
        }
    )

    selected_fewshot = fewshot.loc[fewshot["shots"].isin([0, 5, 20, 40])].copy()
    selected_fewshot = selected_fewshot.loc[
        ((selected_fewshot["shots"] == 0) & (selected_fewshot["adapter"] == "intercept"))
        | ((selected_fewshot["shots"] > 0) & (selected_fewshot["adapter"].isin(["intercept", "affine"])))
    ]
    selected_fewshot["trait"] = pd.Categorical(selected_fewshot["trait"], order, ordered=True)
    selected_fewshot = selected_fewshot.sort_values(["trait", "model", "shots", "adapter"])
    fewshot_table = pd.DataFrame(
        {
            "Trait": selected_fewshot["trait"].astype(str),
            "Model": selected_fewshot["model"],
            "Shots": selected_fewshot["shots"].astype(int),
            "Adapter": selected_fewshot["adapter"],
            "R2 mean": selected_fewshot["r2_mean"].map(lambda value: f"{value:.3f}"),
            "CCC mean": selected_fewshot["ccc_mean"].map(lambda value: f"{value:.3f}"),
            "RMSE mean": selected_fewshot["rmse_mean"].map(lambda value: f"{value:.4g}"),
        }
    )

    content = "\n\n".join(
        [
            "# PlumRAC-Net V2 autogenerated results tables",
            "## Strict zero-shot metrics\n\n" + markdown_table(zero_table),
            "## Lowest-RMSE model by trait\n\n" + markdown_table(best_table),
            "## PlumRAC-Net versus PLSR\n\n" + markdown_table(delta_table),
            "## Consensus-shrunken gate audit\n\n" + markdown_table(gate_table),
            "## Few-shot calibration\n\n" + markdown_table(fewshot_table),
        ]
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content + "\n", encoding="utf-8")
    summary = {
        "traits": order,
        "clear_improvements": delta_table.loc[delta_table["Interpretation"] == "clear improvement", "Trait"].tolist(),
        "promising": delta_table.loc[delta_table["Interpretation"] == "promising but heterogeneous", "Trait"].tolist(),
        "parity": delta_table.loc[delta_table["Interpretation"] == "safeguarded parity", "Trait"].tolist(),
        "no_improvement": delta_table.loc[delta_table["Interpretation"] == "no improvement", "Trait"].tolist(),
        "best_models": best_table.to_dict(orient="records"),
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
