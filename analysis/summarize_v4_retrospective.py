from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_v3_development import score_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", required=True, help="NAME=OUTPUT_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    fold_tables = []
    for item in args.variant:
        if "=" not in item:
            raise ValueError(f"Expected NAME=OUTPUT_DIR, received: {item}")
        name, raw_path = item.split("=", maxsplit=1)
        summary, folds = score_variant(name.strip(), Path(raw_path.strip()))
        summaries.append(summary)
        fold_tables.append(folds)

    table = pd.DataFrame(summaries)
    table["eligible"] = (
        (table["fold_wins"] >= 3)
        & (table["macro_rmse_improvement_pct"] > 0)
        & (table["pooled_rmse_improvement_pct"] > 0)
    )
    table = table.sort_values(
        ["eligible", "macro_candidate_rmse", "pooled_candidate_rmse"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    table["retrospective_rank"] = np.arange(1, len(table) + 1)
    selected = table.iloc[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "variant_summary.csv", index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(args.output_dir / "fold_scores.csv", index=False)
    decision = {
        "selection_scope": "five prespecified historical development cultivars",
        "primary_criterion": "minimum cultivar-macro RMSE among eligible candidates",
        "eligibility": "at least 3/5 PLSR fold wins and positive macro and pooled RMSE improvement",
        "selected_variant": str(selected["variant"]),
        "selected_is_eligible": bool(selected["eligible"]),
        "claim_boundary": (
            "This is retrospective nested-LOCO development because all 16 cultivars were opened in V3. "
            "It is suitable for architecture selection but cannot be described as an untouched external confirmation."
        ),
    }
    (args.output_dir / "selection_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
