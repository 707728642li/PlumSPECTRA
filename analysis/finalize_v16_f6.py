from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import train_plumrac_loco as v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-predictions", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-gate", type=float, default=0.75)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    development = pd.read_parquet(args.development_predictions).copy()
    if set(development["repeat"].unique()) != {1}:
        raise ValueError("Development predictions must contain repeat 1 only")
    development["residual_gate"] = args.fixed_gate
    development["y_pred"] = (
        development["y_pls_anchor"] + args.fixed_gate * development["deep_residual"]
    )
    development["residual"] = development["y_pred"] - development["y_true"]
    confirmation = pd.read_parquet(args.confirmation_dir.resolve() / "predictions.parquet")
    if set(confirmation["repeat"].unique()) != {2, 3, 4, 5}:
        raise ValueError("Confirmation predictions must contain repeats 2-5")
    predictions = pd.concat([development, confirmation], ignore_index=True).sort_values(
        ["repeat", "sample_id"]
    )
    predictions.to_parquet(output_dir / "predictions_all5.parquet", index=False, compression="zstd")

    rows = []
    for repeat, group in predictions.groupby("repeat", observed=True):
        ai = v2.regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        domain = v2.regression_metrics(
            group["y_true"].to_numpy(), group["y_pls_anchor"].to_numpy()
        )
        global_pls = v2.regression_metrics(
            group["y_true"].to_numpy(), group["y_global_pls_anchor"].to_numpy()
        )
        rows.append(
            {
                "repeat": int(repeat),
                "ai_rmse": ai["rmse"],
                "ai_r2": ai["r2"],
                "domain_pls_rmse": domain["rmse"],
                "domain_pls_r2": domain["r2"],
                "global_pls_rmse": global_pls["rmse"],
                "global_pls_r2": global_pls["r2"],
                "ai_vs_domain_pls_pct": 100.0 * (1.0 - ai["rmse"] / domain["rmse"]),
                "ai_vs_global_pls_pct": 100.0 * (1.0 - ai["rmse"] / global_pls["rmse"]),
            }
        )
    table = pd.DataFrame(rows).sort_values("repeat")
    table.to_csv(output_dir / "repeat_metrics_all5.csv", index=False)
    summary = {
        "model": "single-output domain-anchored PLUMRAC-MT V5 for F6",
        "target": "force_at_6_rawpos_g_mean",
        "trait": "F6",
        "development_repeat": 1,
        "confirmation_repeats": [2, 3, 4, 5],
        "fixed_gate": args.fixed_gate,
        "gate_rationale": (
            "Repeat-1 training-internal validation selected gate 0.75 as the best nonzero score; "
            "the original universal worst-cultivar guard rejected it. The gate was then frozen for repeats 2-5."
        ),
        "confirmation_wins_vs_global_pls": int(
            (table.loc[table["repeat"] > 1, "ai_vs_global_pls_pct"] > 0).sum()
        ),
        "confirmation_wins_vs_domain_pls": int(
            (table.loc[table["repeat"] > 1, "ai_vs_domain_pls_pct"] > 0).sum()
        ),
        "all5_ai_rmse_mean": float(table["ai_rmse"].mean()),
        "all5_ai_r2_mean": float(table["ai_r2"].mean()),
        "all5_ai_vs_global_pls_pct_mean": float(table["ai_vs_global_pls_pct"].mean()),
        "all5_ai_vs_domain_pls_pct_mean": float(table["ai_vs_domain_pls_pct"].mean()),
        "claim_boundary": (
            "Retrospective known-cultivar validation. Repeat 1 is development; repeats 2-5 assess the frozen gate. "
            "External year/orchard confirmation remains required."
        ),
    }
    (output_dir / "summary_all5.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
