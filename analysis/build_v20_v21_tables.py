from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TRAITS = ["SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"]
TRAIT_ORDER = {trait: index for index, trait in enumerate(TRAITS)}
MODEL_ORDER = {
    "global_pls": 0,
    "domain_pls": 1,
    "domain_svr": 2,
    "deep": 3,
    "deep_kernel": 4,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v20-analysis-dir", type=Path, required=True)
    parser.add_argument("--v20-baseline-dir", type=Path, required=True)
    parser.add_argument("--v20-ai-dir", type=Path, required=True)
    parser.add_argument("--v21-analysis-dir", type=Path, required=True)
    parser.add_argument("--endpoint-registry", type=Path, required=True)
    parser.add_argument("--attention-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    v20 = args.v20_analysis_dir.resolve()
    v21 = args.v21_analysis_dir.resolve()

    endpoint_registry_path = args.endpoint_registry.resolve()
    registry = pd.read_csv(endpoint_registry_path)
    # The extraction registry deliberately describes raw ARC features and does
    # not carry publication abbreviations.  Join it to the authoritative V2
    # trait registry rather than duplicating names or relying on row order.
    project_root = endpoint_registry_path.parents[3]
    trait_registry = pd.read_csv(project_root / "configs" / "v2_trait_registry.csv")
    trait_registry = trait_registry[trait_registry["abbreviation"].isin(TRAITS)].copy()
    trait_registry = trait_registry.rename(columns={"unit": "publication_unit"})
    trait_registry["endpoint"] = trait_registry["target"].str.removesuffix("_mean")
    registry = registry.merge(
        trait_registry[
            [
                "endpoint",
                "target",
                "abbreviation",
                "publication_label",
                "publication_unit",
                "priority",
            ]
        ],
        on="endpoint",
        how="inner",
        validate="one_to_one",
    )
    if set(registry["abbreviation"]) != set(TRAITS):
        raise ValueError("Endpoint and trait registries do not resolve all nine frozen traits")
    registry["trait_order"] = registry["abbreviation"].map(TRAIT_ORDER)
    registry = registry.sort_values("trait_order")
    registry.to_csv(output_dir / "Table_1_texture_endpoints.csv", index=False)

    pooled = pd.read_csv(v20 / "pooled_metrics.csv")
    pooled["trait_order"] = pooled["trait"].map(TRAIT_ORDER)
    pooled["model_order"] = pooled["model"].map(MODEL_ORDER)
    table2 = pooled.sort_values(["trait_order", "model_order"])[
        ["trait", "model", "n", "rmse", "mae", "bias", "r2", "pearson_r", "ccc", "rpd", "rpiq"]
    ]
    table2.to_csv(output_dir / "Table_2_v20_model_performance.csv", index=False)

    comparisons = pd.read_csv(v20 / "paired_cluster_bootstrap_comparisons.csv")
    comparisons["trait_order"] = comparisons["trait"].map(TRAIT_ORDER)
    comparisons = comparisons.sort_values(["trait_order", "candidate", "baseline"])
    comparisons.to_csv(output_dir / "Table_3_v20_paired_comparisons.csv", index=False)

    v21_metrics = pd.read_csv(v21 / "pooled_and_batch_macro_metrics.csv")
    v21_metrics["trait_order"] = v21_metrics["trait"].map(TRAIT_ORDER)
    v21_metrics["model_order"] = v21_metrics["model"].map(MODEL_ORDER)
    v21_comparisons = pd.read_csv(v21 / "descriptive_batch_bootstrap_comparisons.csv")
    v21_primary = v21_comparisons[
        v21_comparisons["candidate"] == "deep_kernel"
    ][
        [
            "trait",
            "baseline",
            "relative_batch_macro_improvement_pct",
            "descriptive_ci_low",
            "descriptive_ci_high",
            "bootstrap_probability_candidate_better",
        ]
    ]
    v21_wide = v21_primary.pivot(index="trait", columns="baseline")
    v21_wide.columns = [
        f"{statistic}_vs_{baseline}" for statistic, baseline in v21_wide.columns
    ]
    v21_ensemble = v21_metrics[v21_metrics["model"] == "deep_kernel"].set_index("trait")
    table4 = v21_ensemble[
        ["n", "rmse", "batch_macro_rmse", "mae", "bias", "r2", "pearson_r", "ccc"]
    ].join(v21_wide)
    table4 = table4.loc[TRAITS].reset_index()
    table4.to_csv(output_dir / "Table_4_v21_crossbatch_performance.csv", index=False)

    v20_baseline_dir = args.v20_baseline_dir.resolve()
    baseline_rows = []
    for path in sorted(v20_baseline_dir.glob("*/fold_*/metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        baseline_rows.append(
            {
                "trait": metadata["trait"],
                "target": metadata["target"],
                "outer_fold": metadata["outer_fold"],
                "global_pls_preprocessing": metadata["global_pls_choice"]["preprocessing"],
                "global_pls_components": metadata["global_pls_choice"]["n_components"],
                "domain_pls_preprocessing": metadata["domain_pls_choice"]["preprocessing"],
                "domain_pls_components": metadata["domain_pls_choice"]["n_components"],
                "svr_preprocessing": metadata["domain_svr_choice"]["preprocessing"],
                "svr_C": metadata["domain_svr_choice"]["C"],
                "svr_gamma_factor": metadata["domain_svr_choice"]["gamma_factor"],
                "svr_epsilon_z": metadata["domain_svr_choice"]["epsilon_z"],
            }
        )
    pd.DataFrame(baseline_rows).sort_values(["trait", "outer_fold"]).to_csv(
        output_dir / "Table_S1_v20_nested_hyperparameters.csv", index=False
    )

    ai_rows = []
    for path in sorted(args.v20_ai_dir.resolve().glob("*/fold_*/metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        ai_rows.append(
            {
                "trait": metadata["trait_abbreviation"],
                "target": metadata["target"],
                "outer_fold": metadata["outer_fold"],
                "train_samples": metadata["outer_train_samples"],
                "validation_samples": metadata["validation_samples"],
                "test_samples": metadata["test_samples"],
                "selected_epoch": metadata["selected_epoch"],
                "fixed_gate": metadata["selected_gate"],
                "pls_preprocessing": metadata["final_pls"]["preprocessing"],
                "pls_components": metadata["final_pls"]["n_components"],
                "ai_rmse": metadata["ai_metrics"]["rmse"],
                "domain_pls_rmse": metadata["pls_anchor_metrics"]["rmse"],
                "test_labels_used_for_selection": metadata["test_labels_used_for_selection"],
            }
        )
    pd.DataFrame(ai_rows).sort_values(["trait", "outer_fold"]).to_csv(
        output_dir / "Table_S2_v20_deep_fold_settings.csv", index=False
    )

    pd.read_csv(v20 / "within_cultivar_centered_metrics.csv").to_csv(
        output_dir / "Table_S3_within_cultivar_metrics.csv", index=False
    )
    pd.read_csv(v20 / "cultivar_metrics.csv").to_csv(
        output_dir / "Table_S4_cultivar_metrics.csv", index=False
    )
    pd.read_csv(v21 / "per_batch_metrics.csv").to_csv(
        output_dir / "Table_S5_v21_per_batch_metrics.csv", index=False
    )
    pd.read_csv(args.attention_dir.resolve() / "top_attention_windows.csv").to_csv(
        output_dir / "Table_S6_top_attention_windows.csv", index=False
    )

    manifest = {
        "main_tables": 4,
        "supplementary_tables": 6,
        "v20_prediction_rows": 43_551,
        "v21_prediction_rows": 11_124,
        "all_values_derived_from_frozen_outer_predictions": True,
    }
    (output_dir / "table_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
