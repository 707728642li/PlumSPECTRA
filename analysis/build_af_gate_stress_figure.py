from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from v2_registry import add_cultivar_code


COLORS = {"PLSR": "#C98E24", "Three-of-five": "#8A949B", "Unanimous-shrunken": "#3F7FA8"}


def fold_table(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for (cultivar, code), fold in frame.groupby(["cultivar_ascii", "cultivar_code"], observed=True):
        anchor = float(np.sqrt(mean_squared_error(fold["y_true"], fold["y_pls_anchor"])))
        candidate = float(np.sqrt(mean_squared_error(fold["y_true"], fold["y_pred"])))
        rows.append(
            {
                "cultivar_ascii": cultivar,
                "cultivar_code": code,
                "rule": label,
                "gate": float(fold["residual_gate"].iloc[0]),
                "plsr_rmse": anchor,
                "rmse": candidate,
                "gain_pct": 100.0 * (anchor - candidate) / anchor,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unsafe", type=Path, required=True)
    parser.add_argument("--safe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    unsafe = add_cultivar_code(pd.read_parquet(args.unsafe.resolve()))
    safe = add_cultivar_code(pd.read_parquet(args.safe.resolve()))
    keys = ["sample_id", "target"]
    if set(zip(unsafe[keys[0]], unsafe[keys[1]], strict=True)) != set(zip(safe[keys[0]], safe[keys[1]], strict=True)):
        raise ValueError("Unsafe and safe AF stress predictions do not contain identical samples")

    fold_metrics = pd.concat(
        [fold_table(unsafe, "Three-of-five"), fold_table(safe, "Unanimous-shrunken")], ignore_index=True
    )
    order = fold_metrics.loc[fold_metrics["rule"] == "Three-of-five"].sort_values("gain_pct")["cultivar_code"].tolist()
    fold_metrics.to_csv(output / "af_gate_stress_by_cultivar.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    x = np.arange(len(order))
    width = 0.36
    for offset, rule in [(-width / 2, "Three-of-five"), (width / 2, "Unanimous-shrunken")]:
        panel = fold_metrics.loc[fold_metrics["rule"] == rule].set_index("cultivar_code").reindex(order)
        axes[0, 0].bar(x + offset, panel["gain_pct"], width, color=COLORS[rule], label=rule)
        axes[0, 1].bar(x + offset, panel["gate"], width, color=COLORS[rule], label=rule)
    axes[0, 0].axhline(0, color="#58636B", linewidth=0.9)
    axes[0, 0].set_xticks(x, order)
    axes[0, 0].set_ylabel("RMSE improvement over PLSR (%)")
    axes[0, 0].set_title("A  AF held-out stress folds", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_xticks(x, order)
    axes[0, 1].set_ylabel("Selected residual gate")
    axes[0, 1].set_ylim(0, 1.35)
    axes[0, 1].set_title("B  Inner-validation gate", loc="left", fontweight="bold")

    pooled_rows = []
    for label, frame, prediction in [
        ("PLSR", safe, "y_pls_anchor"),
        ("Three-of-five", unsafe, "y_pred"),
        ("Unanimous-shrunken", safe, "y_pred"),
    ]:
        pooled_rows.append(
            {
                "model": label,
                "rmse": float(np.sqrt(mean_squared_error(frame["y_true"], frame[prediction]))),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    axes[1, 0].bar(pooled["model"], pooled["rmse"], color=[COLORS[name] for name in pooled["model"]])
    axes[1, 0].set_ylabel("Pooled AF RMSE (g)")
    axes[1, 0].set_title("C  Five-fold stress-set error", loc="left", fontweight="bold")
    for index, row in pooled.iterrows():
        axes[1, 0].text(index, row["rmse"] + 0.35, f"{row['rmse']:.2f}", ha="center", va="bottom", fontsize=9)

    frl_unsafe = unsafe.loc[unsafe["cultivar_code"] == "FRL"].copy()
    frl_safe = safe.loc[safe["cultivar_code"] == "FRL"].copy()
    axes[1, 1].scatter(
        frl_unsafe["y_true"],
        frl_unsafe["y_pred"],
        s=10,
        alpha=0.32,
        color=COLORS["Three-of-five"],
        label="Three-of-five",
    )
    axes[1, 1].scatter(
        frl_safe["y_true"],
        frl_safe["y_pred"],
        s=10,
        alpha=0.32,
        color=COLORS["Unanimous-shrunken"],
        label="Unanimous-shrunken = PLSR",
    )
    limits = [
        float(min(frl_unsafe["y_true"].min(), frl_unsafe["y_pred"].min(), frl_safe["y_pred"].min())),
        float(max(frl_unsafe["y_true"].max(), frl_unsafe["y_pred"].max(), frl_safe["y_pred"].max())),
    ]
    axes[1, 1].plot(limits, limits, color="#58636B", linewidth=0.9, linestyle="--")
    axes[1, 1].set_xlim(limits)
    axes[1, 1].set_ylim(limits)
    axes[1, 1].set_xlabel("Observed AF (g)")
    axes[1, 1].set_ylabel("Predicted AF (g)")
    axes[1, 1].set_title("D  FRL failure and exact fallback", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, markerscale=1.4)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)

    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig_af_gate_stress.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    unsafe_rmse = float(pooled.loc[pooled["model"] == "Three-of-five", "rmse"].iloc[0])
    safe_rmse = float(pooled.loc[pooled["model"] == "Unanimous-shrunken", "rmse"].iloc[0])
    plsr_rmse = float(pooled.loc[pooled["model"] == "PLSR", "rmse"].iloc[0])
    report = {
        "trait": "AF",
        "role": "pre-confirmation gate stress endpoint",
        "unsafe_rule": "gate <= 1.25; at least 3/5 validation cultivar wins; worst validation degradation <= 10%",
        "safe_rule": "gate <= 0.50; 5/5 validation cultivar wins; no validation cultivar degradation",
        "pooled_plsr_rmse": plsr_rmse,
        "pooled_unsafe_rmse": unsafe_rmse,
        "pooled_safe_rmse": safe_rmse,
        "safe_vs_plsr_pct": 100.0 * (plsr_rmse - safe_rmse) / plsr_rmse,
        "folds": fold_metrics.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
