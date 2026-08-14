from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from v2_registry import cultivar_registry, trait_registry


def parse_specs(values: list[str]) -> dict[str, Path]:
    specs = {}
    for value in values:
        abbreviation, path = value.split("=", 1)
        specs[abbreviation.strip().upper()] = Path(path).resolve()
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="ABBR=PlumRAC output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = parse_specs(args.model)
    code_map = cultivar_registry().set_index("cultivar_ascii")["cultivar_code"].to_dict()
    target_order = trait_registry().loc[lambda frame: frame["model_family"] == "endpoint", "abbreviation"].tolist()
    rows = []
    for abbreviation in target_order:
        if abbreviation not in specs:
            continue
        metadata_paths = list((specs[abbreviation] / "runs" / abbreviation).glob("*/seed_*/metadata.json"))
        for path in metadata_paths:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            selected = next(row for row in metadata["gate_scores"] if row.get("selected"))
            cultivar = str(metadata["heldout_cultivar"])
            rows.append(
                {
                    "trait": abbreviation,
                    "cultivar_ascii": cultivar,
                    "cultivar_code": code_map[cultivar],
                    "seed": int(metadata["seed"]),
                    "selected_gate": float(metadata["selected_gate"]),
                    "objective_profile": str(metadata["selected_objective_profile"]),
                    "selected_epoch": int(metadata["selected_epoch"]),
                    "inner_relative_improvement": float(selected["relative_improvement_vs_zero"]),
                    "inner_win_fraction": float(selected["group_win_fraction"]),
                    "inner_worst_improvement": float(selected["worst_group_improvement"]),
                }
            )
    audit = pd.DataFrame(rows)
    if audit.empty:
        raise ValueError("No fold metadata found")
    audit.to_csv(output / "gate_selection_by_fold.csv", index=False)

    summary_rows = []
    for trait, group in audit.groupby("trait", observed=True):
        counts = group["selected_gate"].value_counts()
        profiles = group["objective_profile"].value_counts()
        summary_rows.append(
            {
                "trait": trait,
                "fold_seed_runs": int(len(group)),
                "gate_0": int(counts.get(0.0, 0)),
                "gate_0.25": int(counts.get(0.25, 0)),
                "gate_0.50": int(counts.get(0.50, 0)),
                "nonzero_gate_fraction": float(np.mean(group["selected_gate"] > 0)),
                "mean_gate": float(group["selected_gate"].mean()),
                "profile_absolute": int(profiles.get("absolute", 0)),
                "profile_balanced": int(profiles.get("balanced", 0)),
                "profile_ranking": int(profiles.get("ranking", 0)),
                "median_epoch": float(group["selected_epoch"].median()),
            }
        )
    summary = pd.DataFrame(summary_rows).set_index("trait").reindex([x for x in target_order if x in specs]).reset_index()
    summary.to_csv(output / "gate_selection_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)
    gate_matrix = audit.pivot_table(index="cultivar_code", columns="trait", values="selected_gate", aggfunc="mean")
    gate_matrix = gate_matrix.reindex(columns=[x for x in target_order if x in gate_matrix.columns])
    gate_matrix = gate_matrix.loc[sorted(gate_matrix.index)]
    sns.heatmap(
        gate_matrix,
        vmin=0,
        vmax=0.5,
        cmap=sns.light_palette("#3F7FA8", as_cmap=True),
        annot=True,
        fmt=".2g",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Residual gate"},
        ax=axes[0],
    )
    axes[0].set_title("A  Fold-internal safe gate", loc="left", fontweight="bold")
    axes[0].set_xlabel("Trait")
    axes[0].set_ylabel("Held-out cultivar / selection")

    x = np.arange(len(summary))
    bottom = np.zeros(len(summary))
    colors = {"gate_0": "#C9D0D4", "gate_0.25": "#79A9C5", "gate_0.50": "#3F7FA8"}
    labels = {"gate_0": "g = 0", "gate_0.25": "g = 0.25", "gate_0.50": "g = 0.50"}
    for column in ["gate_0", "gate_0.25", "gate_0.50"]:
        axes[1].bar(x, summary[column], bottom=bottom, color=colors[column], label=labels[column])
        bottom += summary[column].to_numpy(float)
    axes[1].set_xticks(x, summary["trait"])
    axes[1].set_ylabel("Outer fold-seed runs")
    axes[1].set_title("B  PLSR fallback frequency", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, ncol=3)
    axes[1].spines[["top", "right"]].set_visible(False)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig_v2_gate_audit.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "traits": summary["trait"].tolist(),
        "gate_candidates": [0.0, 0.25, 0.5],
        "acceptance": "all five source validation cultivars improve; none degrade; grouped selection score improves by >=1%",
        "summary": summary.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"traits": report["traits"], "fold_seed_runs": int(len(audit)), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
