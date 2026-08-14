from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_v3_confirmation import fold_metrics, regression_metrics, rmse
from v2_registry import trait_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(trait: str, model: str, frame: pd.DataFrame, prediction: str) -> dict[str, object]:
    truth = frame["y_true"].to_numpy(float)
    estimate = frame[prediction].to_numpy(float)
    metrics = regression_metrics(truth, estimate)
    return {"trait": trait, "model": model, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final V3 evidence tables and figures.")
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "v3" / "plumrac_x_cross_trait_screen",
    )
    parser.add_argument(
        "--screen-analysis",
        type=Path,
        default=PROJECT_ROOT / "results" / "v3" / "plumrac_x_cross_trait_screen_analysis",
    )
    parser.add_argument(
        "--rd-full",
        type=Path,
        default=PROJECT_ROOT / "results" / "v3" / "plumrac_x_rd_full",
    )
    parser.add_argument(
        "--v2-comparison",
        type=Path,
        default=PROJECT_ROOT / "results" / "v2" / "model_comparison_final",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "v3" / "final_evidence",
    )
    args = parser.parse_args()

    screen_summary_path = args.screen_analysis / "cross_trait_screen_summary.csv"
    required = [
        screen_summary_path,
        args.rd_full / "rd_predictions_ensemble.parquet",
        args.rd_full / "rd_full_summary.json",
        args.v2_comparison / "model_summary.csv",
        args.v2_comparison / "plumrac_vs_plsr.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V3 finalization inputs are incomplete: {missing}")

    registry = trait_registry().loc[lambda frame: frame["model_family"] == "endpoint"].copy()
    order = registry["abbreviation"].tolist()
    target_by_trait = dict(zip(registry["abbreviation"], registry["target"]))
    v2_metrics = pd.read_csv(args.v2_comparison / "model_summary.csv")
    v2_delta = pd.read_csv(args.v2_comparison / "plumrac_vs_plsr.csv")
    screen = pd.read_csv(screen_summary_path).set_index("trait")

    model_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    prediction_sources: dict[str, str] = {}
    for trait in order:
        for old_model in ["PLSR", "Ridge", "PlumRAC-Net"]:
            selected = v2_metrics.loc[
                (v2_metrics["trait"] == trait) & (v2_metrics["model"] == old_model)
            ]
            if selected.empty:
                raise ValueError(f"Missing V2 metric: {trait}/{old_model}")
            record = selected.iloc[0].to_dict()
            record["model"] = "PlumRAC-Net V2.2" if old_model == "PlumRAC-Net" else old_model
            model_rows.append(record)

        if trait == "RD":
            prediction_path = args.rd_full / "rd_predictions_ensemble.parquet"
            summary = json.loads((args.rd_full / "rd_full_summary.json").read_text(encoding="utf-8"))
            macro_improvement = float(summary["macro_rmse_improvement_pct"])
            pooled_improvement = float(summary["pooled_rmse_improvement_pct"])
            wins = int(summary["fold_wins_vs_plsr"])
            ci_lower = float(summary["cultivar_cluster_bootstrap"]["ci95_lower_pct"])
            ci_upper = float(summary["cultivar_cluster_bootstrap"]["ci95_upper_pct"])
            seeds = len(summary["seeds"])
            scope = "16 cultivars; five development and eleven sealed confirmation"
        else:
            prediction_path = args.screen_root / trait / "predictions_ensemble.parquet"
            item = screen.loc[trait]
            macro_improvement = float(item["macro_rmse_improvement_pct"])
            pooled_improvement = float(item["pooled_rmse_improvement_pct"])
            wins = int(item["fold_wins"])
            ci_lower = float(item["bootstrap_ci95_lower_pct"])
            ci_upper = float(item["bootstrap_ci95_upper_pct"])
            seeds = 1
            scope = "16-cultivar frozen transfer screen"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        predictions = pd.read_parquet(prediction_path)
        if predictions["target"].nunique() != 1 or str(predictions["target"].iloc[0]) != target_by_trait[trait]:
            raise ValueError(f"Unexpected target in {prediction_path}")
        if len(predictions) != 5430 or predictions["sample_id"].nunique() != 5430:
            raise ValueError(f"Incomplete predictions for {trait}: {len(predictions)} rows")
        model_rows.append(metric_row(trait, "PLUMRAC-X V3", predictions, "y_pred"))
        prediction_sources[trait] = str(prediction_path.resolve())
        transfer_rows.append(
            {
                "trait": trait,
                "n": int(len(predictions)),
                "cultivars": int(predictions["cultivar_code"].nunique()),
                "seeds": seeds,
                "fold_wins_vs_plsr": wins,
                "macro_rmse_improvement_pct": macro_improvement,
                "pooled_rmse_improvement_pct": pooled_improvement,
                "cultivar_bootstrap_ci95_lower_pct": ci_lower,
                "cultivar_bootstrap_ci95_upper_pct": ci_upper,
                "scope": scope,
            }
        )

    models = pd.DataFrame(model_rows)
    transfer = pd.DataFrame(transfer_rows)
    models["trait"] = pd.Categorical(models["trait"], order, ordered=True)
    models = models.sort_values(["trait", "rmse", "model"]).reset_index(drop=True)
    models["trait"] = models["trait"].astype(str)

    pls_rmse = models.loc[models["model"] == "PLSR", ["trait", "rmse"]].set_index("trait")["rmse"]
    models["rmse_improvement_vs_plsr_pct"] = models.apply(
        lambda row: 100.0 * (pls_rmse.loc[row["trait"]] - row["rmse"]) / pls_rmse.loc[row["trait"]],
        axis=1,
    )
    observed_best = (
        models.sort_values(["trait", "rmse"]).groupby("trait", sort=False, observed=True).first().reset_index()
    )
    observed_best = observed_best[
        ["trait", "model", "rmse", "r2", "ccc", "rmse_improvement_vs_plsr_pct"]
    ].rename(columns={"model": "lowest_observed_rmse_model"})

    # This table is descriptive, not a new independent selection set. V2 remains the
    # frozen production AI; V3 transfer results must not be used to retune it.
    recommendation = {
        "RD": "PlumRAC-Net V2.2",
        "F6": "PLSR (V2.2 is numerically identical)",
        "LS": "PlumRAC-Net V2.2",
        "LW": "Ridge",
        "MFF": "PLSR",
        "PFD": "Ridge",
        "PRW": "Ridge",
        "SRF": "PlumRAC-Net V2.2 (practical parity)",
        "AF": "PLSR",
    }
    observed_best["evidence_based_deployment_reading"] = observed_best["trait"].map(recommendation)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    models.to_csv(output / "all_model_pooled_metrics.csv", index=False)
    transfer.to_csv(output / "plumrac_x_transfer_summary.csv", index=False)
    observed_best.to_csv(output / "model_selection_reading.csv", index=False)

    plot = models.loc[models["model"].isin(["Ridge", "PlumRAC-Net V2.2", "PLUMRAC-X V3"])].copy()
    palette = {"Ridge": "#6B7280", "PlumRAC-Net V2.2": "#0F766E", "PLUMRAC-X V3": "#D97706"}
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 9.0), gridspec_kw={"height_ratios": [1.5, 1.0]})
    x = np.arange(len(order))
    width = 0.24
    for index, model in enumerate(["Ridge", "PlumRAC-Net V2.2", "PLUMRAC-X V3"]):
        values = plot.loc[plot["model"] == model].set_index("trait").reindex(order)
        axes[0].bar(
            x + (index - 1) * width,
            values["rmse_improvement_vs_plsr_pct"],
            width=width,
            label=model,
            color=palette[model],
        )
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_xticks(x, order)
    axes[0].set_ylabel("Pooled RMSE improvement vs PLSR (%)")
    axes[0].set_title("A  Frozen cross-cultivar model comparison", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="lower left")
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)

    transfer_plot = transfer.set_index("trait").reindex(order)
    axes[1].bar(x, transfer_plot["fold_wins_vs_plsr"], color="#D97706", width=0.62)
    axes[1].axhline(8, color="#111827", linestyle="--", linewidth=1, label="Half of 16 cultivars")
    axes[1].set_xticks(x, order)
    axes[1].set_ylim(0, 16.8)
    axes[1].set_ylabel("Cultivar-fold wins (of 16)")
    axes[1].set_title("B  PLUMRAC-X transfer consistency", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    fig.suptitle("Capacity helps selected development folds but does not remove cultivar-domain shift", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "fig_v3_final_model_evidence.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(output / "fig_v3_final_model_evidence.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    v2_rd = v2_delta.loc[v2_delta["trait"] == "RD"].iloc[0]
    rd_summary = json.loads((args.rd_full / "rd_full_summary.json").read_text(encoding="utf-8"))
    audit = {
        "status": "PASS",
        "traits": order,
        "fruits_per_trait": 5430,
        "cultivars_per_trait": 16,
        "v3_trainable_parameters": 336290,
        "v2_trainable_parameters": 72530,
        "primary_conclusion": (
            "The larger frozen PLUMRAC-X model did not establish broad superiority over PLSR. "
            "The compact V2.2 model remains the best zero-shot AI for RD; cultivar-domain shift, "
            "rather than parameter count or activation choice alone, is the limiting factor."
        ),
        "rd_v2_pooled_improvement_pct": float(v2_rd["pooled_rmse_improvement_pct"]),
        "rd_v3_pooled_improvement_pct": float(rd_summary["pooled_rmse_improvement_pct"]),
        "claim_boundary": (
            "Observed-best and deployment-reading tables summarize completed LOCO evidence; "
            "they are not an additional untouched test set and cannot justify post-hoc retuning."
        ),
        "prediction_sources": prediction_sources,
        "sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256(path)
            for path in [
                screen_summary_path,
                args.rd_full / "rd_predictions_ensemble.parquet",
                args.rd_full / "rd_full_summary.json",
                args.v2_comparison / "model_summary.csv",
                args.v2_comparison / "plumrac_vs_plsr.csv",
            ]
        },
    }
    (output / "final_evidence_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
