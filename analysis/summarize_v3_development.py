from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEVELOPMENT_CULTIVARS = {
    "3.13": "L313",
    "Cuihongli": "CHL",
    "Konglongdan": "KLD",
    "Weiwang": "WW",
    "Weixin": "WX",
}


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def score_variant(name: str, path: Path) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    predictions = pd.read_parquet(path / "predictions_ensemble.parquet")
    predictions = predictions.loc[predictions["cultivar_ascii"].isin(DEVELOPMENT_CULTIVARS)].copy()
    observed = set(predictions["cultivar_ascii"].unique())
    missing = sorted(set(DEVELOPMENT_CULTIVARS) - observed)
    if missing:
        raise ValueError(f"{name} is missing development cultivars: {missing}")

    fold_rows: list[dict[str, float | str | bool]] = []
    for cultivar, group in predictions.groupby("cultivar_ascii", observed=True):
        truth = group["y_true"].to_numpy(float)
        candidate = group["y_pred"].to_numpy(float)
        anchor = group["y_pls_anchor"].to_numpy(float)
        candidate_rmse = rmse(truth, candidate)
        anchor_rmse = rmse(truth, anchor)
        fold_rows.append(
            {
                "variant": name,
                "cultivar_ascii": cultivar,
                "cultivar_code": DEVELOPMENT_CULTIVARS[cultivar],
                "n": int(len(group)),
                "plsr_rmse": anchor_rmse,
                "candidate_rmse": candidate_rmse,
                "rmse_improvement_pct": 100.0 * (anchor_rmse - candidate_rmse) / anchor_rmse,
                "candidate_win": bool(candidate_rmse < anchor_rmse),
            }
        )
    folds = pd.DataFrame(fold_rows).sort_values("cultivar_code")
    truth = predictions["y_true"].to_numpy(float)
    candidate = predictions["y_pred"].to_numpy(float)
    anchor = predictions["y_pls_anchor"].to_numpy(float)
    pooled_candidate = rmse(truth, candidate)
    pooled_anchor = rmse(truth, anchor)
    summary: dict[str, float | int | str] = {
        "variant": name,
        "path": str(path.resolve()),
        "n": int(len(predictions)),
        "development_cultivars": int(folds["cultivar_ascii"].nunique()),
        "fold_wins": int(folds["candidate_win"].sum()),
        "macro_plsr_rmse": float(folds["plsr_rmse"].mean()),
        "macro_candidate_rmse": float(folds["candidate_rmse"].mean()),
        "macro_rmse_improvement_pct": float(
            100.0 * (folds["plsr_rmse"].mean() - folds["candidate_rmse"].mean()) / folds["plsr_rmse"].mean()
        ),
        "pooled_plsr_rmse": pooled_anchor,
        "pooled_candidate_rmse": pooled_candidate,
        "pooled_rmse_improvement_pct": 100.0 * (pooled_anchor - pooled_candidate) / pooled_anchor,
    }
    return summary, folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME=OUTPUT_DIR; repeat for each development variant.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    folds = []
    for item in args.variant:
        if "=" not in item:
            raise ValueError(f"Expected NAME=OUTPUT_DIR, received: {item}")
        name, raw_path = item.split("=", maxsplit=1)
        summary, fold = score_variant(name.strip(), Path(raw_path.strip()))
        summaries.append(summary)
        folds.append(fold)

    summary_table = pd.DataFrame(summaries)
    summary_table["eligible"] = (
        (summary_table["fold_wins"] >= 3)
        & (summary_table["macro_rmse_improvement_pct"] > 0)
        & (summary_table["pooled_rmse_improvement_pct"] > 0)
    )
    summary_table = summary_table.sort_values(
        ["eligible", "macro_candidate_rmse", "pooled_candidate_rmse"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    summary_table["development_rank"] = np.arange(1, len(summary_table) + 1)
    selected = summary_table.iloc[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(args.output_dir / "variant_summary.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(args.output_dir / "fold_scores.csv", index=False)
    decision = {
        "selection_scope": "five prespecified V3 development cultivars only",
        "primary_criterion": "minimum cultivar-macro RMSE",
        "eligibility": "at least 3/5 PLSR fold wins and positive macro and pooled RMSE improvement",
        "selected_variant": str(selected["variant"]),
        "selected_is_eligible": bool(selected["eligible"]),
        "confirmation_boundary": "All remaining cultivars are untouched by this V3 selection table.",
    }
    (args.output_dir / "selection_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(summary_table.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
