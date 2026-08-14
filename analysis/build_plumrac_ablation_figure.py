from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score


COLORS = {
    "PLSR": "#C58C2A",
    "Legacy RAC": "#89939A",
    "PlumRAC-Net": "#3F7CA6",
    "absolute": "#89939A",
    "balanced": "#4F8766",
    "ranking": "#7A6599",
}


def centred_r2(frame: pd.DataFrame, prediction: str) -> float:
    true_c = frame["y_true"] - frame.groupby("cultivar_code")["y_true"].transform("mean")
    pred_c = frame[prediction] - frame.groupby("cultivar_code")[prediction].transform("mean")
    return float(r2_score(true_c, pred_c))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    legacy = pd.read_parquet(args.legacy.resolve())
    v2 = pd.read_parquet(args.v2.resolve())
    keys = set(v2["sample_id"])
    legacy = legacy.loc[legacy["sample_id"].isin(keys)].copy()
    legacy = legacy.set_index("sample_id").loc[v2["sample_id"]].reset_index()
    if not np.allclose(legacy["y_true"], v2["y_true"]):
        raise ValueError("Legacy and V2 predictions are not aligned")

    fold_rows = []
    for cultivar_code, group in v2.groupby("cultivar_code", observed=True):
        legacy_group = legacy.loc[legacy["cultivar_code"] == cultivar_code]
        pls_rmse = float(np.sqrt(mean_squared_error(group["y_true"], group["y_pls_anchor"])))
        legacy_rmse = float(np.sqrt(mean_squared_error(legacy_group["y_true"], legacy_group["y_pred"])))
        v2_rmse = float(np.sqrt(mean_squared_error(group["y_true"], group["y_pred"])))
        fold_rows.append(
            {
                "cultivar_code": cultivar_code,
                "n": len(group),
                "pls_rmse": pls_rmse,
                "legacy_rmse": legacy_rmse,
                "v2_rmse": v2_rmse,
                "legacy_gain": 100.0 * (pls_rmse - legacy_rmse) / pls_rmse,
                "v2_gain": 100.0 * (pls_rmse - v2_rmse) / pls_rmse,
            }
        )
    folds = pd.DataFrame(fold_rows).sort_values("v2_gain")
    folds.to_csv(output / "table_rd_development_ablation.csv", index=False)

    metrics = []
    for label, frame, prediction in [
        ("PLSR", v2, "y_pls_anchor"),
        ("Legacy RAC", legacy, "y_pred"),
        ("PlumRAC-Net", v2, "y_pred"),
    ]:
        metrics.append(
            {
                "model": label,
                "R2": float(r2_score(frame["y_true"], frame[prediction])),
                "Pearson r": float(pearsonr(frame["y_true"], frame[prediction]).statistic),
                "Spearman rho": float(spearmanr(frame["y_true"], frame[prediction]).statistic),
                "Centred R2": centred_r2(frame, prediction),
                "RMSE": float(np.sqrt(mean_squared_error(frame["y_true"], frame[prediction]))),
            }
        )
    metric_table = pd.DataFrame(metrics)
    metric_table.to_csv(output / "table_rd_development_metrics.csv", index=False)

    metadata_rows = []
    v2_root = args.v2.resolve().parent
    for path in (v2_root / "runs" / "RD").glob("*/seed_*/metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        prediction = pd.read_parquet(path.parent / "predictions.parquet")
        metadata_rows.append(
            {
                "cultivar_code": prediction["cultivar_code"].iloc[0],
                "profile": metadata["selected_objective_profile"],
                "gate": metadata["selected_gate"],
                "epoch": metadata["selected_epoch"],
            }
        )
    metadata_table = pd.DataFrame(metadata_rows).set_index("cultivar_code").loc[folds["cultivar_code"]].reset_index()
    metadata_table.to_csv(output / "table_rd_development_selection.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), constrained_layout=True)
    x = np.arange(len(folds))
    width = 0.36
    axes[0, 0].bar(x - width / 2, folds["legacy_gain"], width, color=COLORS["Legacy RAC"], label="Legacy RAC")
    axes[0, 0].bar(x + width / 2, folds["v2_gain"], width, color=COLORS["PlumRAC-Net"], label="PlumRAC-Net")
    axes[0, 0].axhline(0, color="#5B6570", linewidth=0.8)
    axes[0, 0].set_xticks(x, folds["cultivar_code"])
    axes[0, 0].set_ylabel("RMSE improvement over PLSR (%)")
    axes[0, 0].set_title("A  RD development-fold gains", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    dimensions = ["R2", "Pearson r", "Spearman rho", "Centred R2"]
    dim_x = np.arange(len(dimensions))
    bar_width = 0.24
    for index, row in metric_table.iterrows():
        axes[0, 1].bar(
            dim_x + (index - 1) * bar_width,
            [row[value] for value in dimensions],
            bar_width,
            color=COLORS[row["model"]],
            label=row["model"],
        )
    axes[0, 1].axhline(0, color="#5B6570", linewidth=0.8)
    axes[0, 1].set_xticks(dim_x, [r"$R^2$", "Pearson r", r"Spearman $\rho$", r"Centred $R^2$"])
    axes[0, 1].set_title("B  Pooled discrimination", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False, ncol=3, fontsize=8)

    y = np.arange(len(folds))
    for index, row in folds.reset_index(drop=True).iterrows():
        axes[1, 0].plot([row["pls_rmse"], row["v2_rmse"]], [index, index], color="#AAB1B6", linewidth=1.5)
    axes[1, 0].scatter(folds["pls_rmse"], y, color=COLORS["PLSR"], label="PLSR", zorder=3)
    axes[1, 0].scatter(folds["v2_rmse"], y, color=COLORS["PlumRAC-Net"], label="PlumRAC-Net", zorder=3)
    axes[1, 0].set_yticks(y, folds["cultivar_code"])
    axes[1, 0].set_xlabel("RD RMSE (r.u.)")
    axes[1, 0].set_title("C  Paired held-out RMSE", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)

    profile_colors = [COLORS[value] for value in metadata_table["profile"]]
    bars = axes[1, 1].bar(metadata_table["cultivar_code"], metadata_table["gate"], color=profile_colors)
    axes[1, 1].bar_label(bars, labels=[f"e{epoch}" for epoch in metadata_table["epoch"]], padding=3, fontsize=8)
    axes[1, 1].set_ylim(0, 1.45)
    axes[1, 1].set_ylabel("Selected residual gate")
    axes[1, 1].set_title("D  Fold-internal RAC selection", loc="left", fontweight="bold")
    axes[1, 1].legend(
        handles=[Patch(facecolor=COLORS[profile], label=profile) for profile in ["absolute", "balanced", "ranking"]],
        frameon=False,
        ncol=3,
        fontsize=8,
    )

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E1E5E8", linewidth=0.7)
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig_rd_plumrac_ablation.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "trait": "RD",
        "development_folds": folds["cultivar_code"].tolist(),
        "fold_wins_v2_vs_plsr": int((folds["v2_gain"] > 0).sum()),
        "folds": int(len(folds)),
        "pooled_rmse_improvement_pct": float(
            100.0
            * (metric_table.loc[metric_table["model"] == "PLSR", "RMSE"].iloc[0] - metric_table.loc[metric_table["model"] == "PlumRAC-Net", "RMSE"].iloc[0])
            / metric_table.loc[metric_table["model"] == "PLSR", "RMSE"].iloc[0]
        ),
        "interpretation": "RD is the architecture-development endpoint, AF is the gate-stress endpoint, and the remaining seven endpoints are confirmation targets.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
