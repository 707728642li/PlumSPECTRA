from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

from v2_registry import abbreviated_trait, add_cultivar_code


def stable_seed(*values: object) -> int:
    value = "|".join(map(str, values)).encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:8], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_spec(value: str) -> tuple[str, Path]:
    model, path = value.split("=", 1)
    return model.strip(), Path(path).resolve()


def fit_adapter(calibration: pd.DataFrame, mode: str, slope_prior_strength: float) -> tuple[float, float]:
    x = calibration["y_pred"].to_numpy(float)
    y = calibration["y_true"].to_numpy(float)
    if mode == "intercept":
        return 1.0, float(np.mean(y - x))
    if mode != "affine":
        raise ValueError(mode)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    ss_x = float(np.sum(x_centered**2))
    scale = max(ss_x / max(len(x) - 1, 1), np.finfo(float).eps)
    penalty = slope_prior_strength * scale
    slope = float((np.sum(x_centered * y_centered) + penalty) / (ss_x + penalty))
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", action="append", required=True, help="MODEL=predictions.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", default="0,5,10,20,40")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--slope-prior-strength", type=float, default=5.0)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shots = [int(value) for value in args.shots.split(",")]
    model_frames: dict[str, pd.DataFrame] = {}
    input_paths: list[Path] = []
    for value in args.predictions:
        model, path = parse_spec(value)
        input_paths.append(path)
        frame = pd.read_parquet(path)
        required = {"sample_id", "cultivar_ascii", "target", "y_true", "y_pred"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing required columns for {model}: {sorted(required - set(frame.columns))}")
        frame = frame[list(required)].copy()
        if frame.duplicated(["sample_id", "target"]).any():
            raise ValueError(f"Duplicate sample/target rows for {model}")
        model_frames[model] = add_cultivar_code(frame)

    common_keys: set[tuple[str, str]] | None = None
    for frame in model_frames.values():
        keys = set(zip(frame["sample_id"], frame["target"], strict=True))
        common_keys = keys if common_keys is None else common_keys & keys
    if not common_keys:
        raise ValueError("Prediction inputs have no common sample/target rows")

    rows: list[dict[str, object]] = []
    for model, frame in model_frames.items():
        keep = pd.MultiIndex.from_frame(frame[["sample_id", "target"]]).isin(pd.MultiIndex.from_tuples(common_keys))
        frame = frame.loc[keep].copy()
        for target, target_frame in frame.groupby("target", observed=True):
            trait = abbreviated_trait(str(target))
            for shot in shots:
                # Duplicate the identical zero-shot point for both adapter
                # trajectories so each calibration curve remains continuous.
                modes = ["intercept", "affine"]
                for repeat in range(1, args.repeats + 1):
                    for mode in modes:
                        parts: list[pd.DataFrame] = []
                        for cultivar, fold in target_frame.groupby("cultivar_ascii", observed=True):
                            fold = fold.reset_index(drop=True)
                            if shot == 0:
                                evaluation = fold.copy()
                                slope, intercept = 1.0, 0.0
                            else:
                                if shot >= len(fold) - 2:
                                    raise ValueError(f"Shot count {shot} leaves too few evaluation fruits for {trait}/{cultivar}")
                                rng = np.random.default_rng(stable_seed(target, cultivar, shot, repeat, 20260806))
                                positions = rng.choice(len(fold), size=shot, replace=False)
                                calibration_mask = np.zeros(len(fold), dtype=bool)
                                calibration_mask[positions] = True
                                calibration = fold.loc[calibration_mask]
                                evaluation = fold.loc[~calibration_mask].copy()
                                slope, intercept = fit_adapter(calibration, mode, args.slope_prior_strength)
                            evaluation["y_adapted"] = slope * evaluation["y_pred"] + intercept
                            parts.append(evaluation)
                        pooled = pd.concat(parts, ignore_index=True)
                        rmse = float(np.sqrt(mean_squared_error(pooled["y_true"], pooled["y_adapted"])))
                        truth = pooled["y_true"].to_numpy(float)
                        adapted = pooled["y_adapted"].to_numpy(float)
                        covariance = float(np.cov(truth, adapted, ddof=1)[0, 1])
                        ccc = float(
                            2.0
                            * covariance
                            / max(
                                np.var(truth, ddof=1)
                                + np.var(adapted, ddof=1)
                                + (np.mean(truth) - np.mean(adapted)) ** 2,
                                np.finfo(float).eps,
                            )
                        )
                        rows.append(
                            {
                                "model": model,
                                "target": target,
                                "trait": trait,
                                "shots": shot,
                                "adapter": mode,
                                "repeat": repeat,
                                "n_evaluation": int(len(pooled)),
                                "rmse": rmse,
                                "r2": float(r2_score(pooled["y_true"], pooled["y_adapted"])),
                                "ccc": ccc,
                            }
                        )
    repeated = pd.DataFrame(rows)
    repeated.to_parquet(output / "fewshot_repeat_metrics.parquet", index=False, compression="zstd")
    summary = (
        repeated.groupby(["model", "target", "trait", "shots", "adapter"], observed=True)
        .agg(
            repeats=("repeat", "nunique"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            rmse_ci025=("rmse", lambda value: value.quantile(0.025)),
            rmse_ci975=("rmse", lambda value: value.quantile(0.975)),
            r2_mean=("r2", "mean"),
            r2_sd=("r2", "std"),
            r2_ci025=("r2", lambda value: value.quantile(0.025)),
            r2_ci975=("r2", lambda value: value.quantile(0.975)),
            ccc_mean=("ccc", "mean"),
            ccc_sd=("ccc", "std"),
            ccc_ci025=("ccc", lambda value: value.quantile(0.025)),
            ccc_ci975=("ccc", lambda value: value.quantile(0.975)),
        )
        .reset_index()
    )
    summary.to_csv(output / "fewshot_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    traits = sorted(summary["trait"].unique())
    cols = min(3, len(traits))
    rows_n = int(np.ceil(len(traits) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.1 * cols, 3.0 * rows_n), squeeze=False, constrained_layout=True)
    palette = {"PLSR": "#C58C2A", "PlumRAC-Net": "#3F7CA6"}
    markers = {"intercept": "o", "affine": "s"}
    for axis, trait in zip(axes.flat, traits):
        panel = summary.loc[summary["trait"] == trait]
        for (model, adapter), group in panel.groupby(["model", "adapter"], observed=True):
            group = group.sort_values("shots")
            label = f"{model} + {adapter}"
            axis.plot(group["shots"], group["rmse_mean"], marker=markers[adapter], label=label, color=palette.get(model, "#75818A"), linestyle="--" if adapter == "affine" else "-")
            axis.fill_between(group["shots"], group["rmse_ci025"], group["rmse_ci975"], color=palette.get(model, "#75818A"), alpha=0.10)
        axis.set_title(trait, fontweight="bold")
        axis.set_xlabel("Labelled target-cultivar fruits")
        axis.set_ylabel("RMSE")
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(traits) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"fig_v2_fewshot.{suffix}", dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    report = {
        "models": sorted(model_frames),
        "shots": shots,
        "repeats": args.repeats,
        "adapters": ["intercept", "shrunken affine"],
        "slope_prior_strength": args.slope_prior_strength,
        "calibration_policy": "identical deterministic calibration fruits across models; calibration fruits excluded from evaluation",
        "summary": summary.to_dict(orient="records"),
    }
    project_root = Path(__file__).resolve().parents[1]
    provenance_paths = [
        Path(__file__).resolve(),
        project_root / "configs" / "v2_nomenclature.csv",
        project_root / "configs" / "v2_trait_registry.csv",
        project_root / "environment-lock.txt",
        *input_paths,
    ]
    report["provenance_sha256"] = {
        str(path.relative_to(project_root) if path.is_relative_to(project_root) else path): sha256_file(path)
        for path in provenance_paths
        if path.exists()
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"models": sorted(model_frames), "traits": traits, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
