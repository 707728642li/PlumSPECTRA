from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eight-trait-table", type=Path, required=True)
    parser.add_argument("--f6-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.eight_trait_table)
    report = json.loads(args.f6_statistics.read_text(encoding="utf-8"))
    global_pls = report["comparisons"]["global_pls"]
    domain_pls = report["comparisons"]["domain_pls"]
    f6 = {
        "trait": "F6",
        "prediction_records": report["prediction_records"],
        "unique_test_fruits": report["unique_fruits_in_any_test_split"],
        "cultivars": report["cultivars"],
        "ai_rmse": global_pls["ai"]["pooled_rmse"],
        "global_pls_rmse": global_pls["baseline"]["pooled_rmse"],
        "ai_vs_global_pls_pct": global_pls["pooled_relative_rmse_improvement_pct"],
        "global_ci95_low_pct": global_pls["cultivar_cluster_bootstrap_pooled_ci95_pct"][0],
        "global_ci95_high_pct": global_pls["cultivar_cluster_bootstrap_pooled_ci95_pct"][1],
        "global_cultivar_wins": global_pls["cultivar_wins"],
        "global_wilcoxon_p": global_pls["wilcoxon_greater_p"],
        "domain_pls_rmse": domain_pls["baseline"]["pooled_rmse"],
        "ai_vs_domain_pls_pct": domain_pls["pooled_relative_rmse_improvement_pct"],
        "domain_ci95_low_pct": domain_pls["cultivar_cluster_bootstrap_pooled_ci95_pct"][0],
        "domain_ci95_high_pct": domain_pls["cultivar_cluster_bootstrap_pooled_ci95_pct"][1],
        "domain_cultivar_wins": domain_pls["cultivar_wins"],
        "domain_wilcoxon_p": domain_pls["wilcoxon_greater_p"],
    }
    table = pd.concat([table, pd.DataFrame([f6])], ignore_index=True).sort_values(
        "ai_vs_global_pls_pct", ascending=False
    )
    if table["trait"].nunique() != 9 or len(table) != 9:
        raise RuntimeError("Expected exactly nine unique texture traits")
    table.to_csv(output_dir / "all_nine_trait_statistics.csv", index=False)
    manifest = {
        "scope": "nine independently trained single-output domain-anchored AI texture models",
        "traits": table["trait"].tolist(),
        "all_traits_global_ci_excludes_zero": bool((table["global_ci95_low_pct"] > 0).all()),
        "all_traits_domain_ci_excludes_zero": bool((table["domain_ci95_low_pct"] > 0).all()),
        "global_improvement_range_pct": [
            float(table["ai_vs_global_pls_pct"].min()),
            float(table["ai_vs_global_pls_pct"].max()),
        ],
        "domain_increment_range_pct": [
            float(table["ai_vs_domain_pls_pct"].min()),
            float(table["ai_vs_domain_pls_pct"].max()),
        ],
        "claim_boundary": (
            "Retrospective repeated known-cultivar fruit validation. Each trait has an independent single-output "
            "model. External year/orchard confirmation remains required."
        ),
    }
    (output_dir / "all_nine_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
