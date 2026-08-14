from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CODES = {"L611", "LA191", "FTL", "FWHH", "FRL", "L31", "NL", "QCL", "WD", "WJ", "ZSKLD"}
EXPECTED_SEEDS = {20260806, 20260807, 20260808, 20260809}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - p))))


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    residual = y - p
    denominator = float(np.sum(np.square(y - y.mean())))
    r2 = float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0 else float("nan")
    pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else float("nan")
    mean_y = float(y.mean())
    mean_p = float(p.mean())
    var_y = float(np.var(y))
    var_p = float(np.var(p))
    covariance = float(np.mean((y - mean_y) * (p - mean_p)))
    ccc_denominator = var_y + var_p + (mean_y - mean_p) ** 2
    ccc = float(2.0 * covariance / ccc_denominator) if ccc_denominator > 0 else float("nan")
    return {
        "n": int(len(y)),
        "rmse": rmse(y, p),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(p - y)),
        "r2": r2,
        "pearson_r": pearson,
        "ccc": ccc,
    }


def fold_metrics(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code, group in frame.groupby("cultivar_code", observed=True):
        truth = group["y_true"].to_numpy(float)
        candidate = group[prediction_column].to_numpy(float)
        anchor = group["y_pls_anchor"].to_numpy(float)
        ai_rmse = rmse(truth, candidate)
        pls_rmse = rmse(truth, anchor)
        rows.append(
            {
                "cultivar_code": str(code),
                "cultivar_ascii": str(group["cultivar_ascii"].iloc[0]),
                "n": int(len(group)),
                "plsr_rmse": pls_rmse,
                "plumrac_x_rmse": ai_rmse,
                "paired_rmse_gain": pls_rmse - ai_rmse,
                "rmse_improvement_pct": 100.0 * (pls_rmse - ai_rmse) / pls_rmse,
                "plumrac_x_win": bool(ai_rmse < pls_rmse),
            }
        )
    return pd.DataFrame(rows).sort_values("cultivar_code").reset_index(drop=True)


def cultivar_cluster_bootstrap(folds: pd.DataFrame, repeats: int = 100_000) -> dict[str, float | int]:
    rng = np.random.default_rng(20260806)
    pls = folds["plsr_rmse"].to_numpy(float)
    ai = folds["plumrac_x_rmse"].to_numpy(float)
    draws = rng.integers(0, len(folds), size=(repeats, len(folds)))
    sampled_pls = pls[draws].mean(axis=1)
    sampled_ai = ai[draws].mean(axis=1)
    improvements = 100.0 * (sampled_pls - sampled_ai) / sampled_pls
    return {
        "repeats": repeats,
        "seed": 20260806,
        "mean_improvement_pct": float(improvements.mean()),
        "ci95_lower_pct": float(np.quantile(improvements, 0.025)),
        "ci95_upper_pct": float(np.quantile(improvements, 0.975)),
        "probability_improvement": float(np.mean(improvements > 0)),
    }


def exact_sign_flip_test(folds: pd.DataFrame) -> dict[str, float | int]:
    gains = folds["paired_rmse_gain"].to_numpy(float)
    observed = float(gains.mean())
    null = np.array(
        [np.mean(gains * np.asarray(signs, dtype=float)) for signs in itertools.product((-1.0, 1.0), repeat=len(gains))]
    )
    return {
        "folds": int(len(gains)),
        "permutations": int(len(null)),
        "observed_mean_rmse_gain": observed,
        "p_one_sided": float(np.mean(null >= observed)),
        "p_two_sided": float(np.mean(np.abs(null) >= abs(observed))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.confirmation_dir / "predictions_by_seed.parquet"
    predictions = pd.read_parquet(source)
    observed_codes = set(predictions["cultivar_code"].astype(str).unique())
    observed_seeds = set(predictions["seed"].astype(int).unique())
    if observed_codes != EXPECTED_CODES:
        raise ValueError(f"Confirmation codes differ from frozen set: {sorted(observed_codes)}")
    if observed_seeds != EXPECTED_SEEDS:
        raise ValueError(f"Confirmation seeds differ from frozen set: {sorted(observed_seeds)}")
    counts = predictions.groupby("sample_id", observed=True)["seed"].nunique()
    if not (counts == len(EXPECTED_SEEDS)).all():
        raise ValueError("Every confirmation sample must have one prediction from every frozen seed")
    invariants = predictions.groupby("sample_id", observed=True).agg(
        y_true_range=("y_true", lambda x: float(x.max() - x.min())),
        anchor_range=("y_pls_anchor", lambda x: float(x.max() - x.min())),
    )
    if float(invariants.to_numpy().max()) > 1e-7:
        raise ValueError("Truth or PLSR anchor changed across seeds")

    ensemble = (
        predictions.groupby(["sample_id", "cultivar_ascii", "cultivar_code", "target"], observed=True, as_index=False)
        .agg(y_true=("y_true", "first"), y_pls_anchor=("y_pls_anchor", "first"), y_pred=("y_pred", "mean"))
    )
    folds = fold_metrics(ensemble, "y_pred")
    macro_pls = float(folds["plsr_rmse"].mean())
    macro_ai = float(folds["plumrac_x_rmse"].mean())
    truth = ensemble["y_true"].to_numpy(float)
    ai = ensemble["y_pred"].to_numpy(float)
    pls = ensemble["y_pls_anchor"].to_numpy(float)

    per_seed_rows: list[dict[str, object]] = []
    for seed, group in predictions.groupby("seed", observed=True):
        seed_folds = fold_metrics(group, "y_pred")
        seed_macro_pls = float(seed_folds["plsr_rmse"].mean())
        seed_macro_ai = float(seed_folds["plumrac_x_rmse"].mean())
        per_seed_rows.append(
            {
                "seed": int(seed),
                "fold_wins": int(seed_folds["plumrac_x_win"].sum()),
                "macro_plsr_rmse": seed_macro_pls,
                "macro_plumrac_x_rmse": seed_macro_ai,
                "macro_rmse_improvement_pct": 100.0 * (seed_macro_pls - seed_macro_ai) / seed_macro_pls,
                "pooled_plsr_rmse": rmse(group["y_true"].to_numpy(float), group["y_pls_anchor"].to_numpy(float)),
                "pooled_plumrac_x_rmse": rmse(group["y_true"].to_numpy(float), group["y_pred"].to_numpy(float)),
            }
        )
    per_seed = pd.DataFrame(per_seed_rows).sort_values("seed")
    per_seed["pooled_rmse_improvement_pct"] = 100.0 * (
        per_seed["pooled_plsr_rmse"] - per_seed["pooled_plumrac_x_rmse"]
    ) / per_seed["pooled_plsr_rmse"]

    bootstrap = cultivar_cluster_bootstrap(folds)
    sign_flip = exact_sign_flip_test(folds)
    all_seed_macro_positive = bool((per_seed["macro_rmse_improvement_pct"] > 0).all())
    superiority_supported = bool(
        macro_ai < macro_pls
        and int(folds["plumrac_x_win"].sum()) >= 6
        and float(bootstrap["ci95_lower_pct"]) > 0
        and float(sign_flip["p_two_sided"]) < 0.05
        and all_seed_macro_positive
    )
    summary = {
        "model_family": "PLUMRAC-X",
        "scope": "Eleven sealed confirmation cultivars; no development cultivar included",
        "seeds": sorted(EXPECTED_SEEDS),
        "n_unique_fruits": int(len(ensemble)),
        "confirmation_cultivars": int(len(folds)),
        "fold_wins_vs_plsr": int(folds["plumrac_x_win"].sum()),
        "macro_plsr_rmse": macro_pls,
        "macro_plumrac_x_rmse": macro_ai,
        "macro_rmse_improvement_pct": 100.0 * (macro_pls - macro_ai) / macro_pls,
        "pooled_plsr": regression_metrics(truth, pls),
        "pooled_plumrac_x": regression_metrics(truth, ai),
        "pooled_rmse_improvement_pct": 100.0 * (rmse(truth, pls) - rmse(truth, ai)) / rmse(truth, pls),
        "cultivar_cluster_bootstrap": bootstrap,
        "exact_paired_sign_flip": sign_flip,
        "all_seed_macro_improvements_positive": all_seed_macro_positive,
        "prespecified_superiority_rule": (
            "positive macro gain, >=6/11 fold wins, cultivar-bootstrap lower 95% bound >0, "
            "exact two-sided sign-flip p<0.05, and positive macro gain for every seed"
        ),
        "confirmation_supports_superiority": superiority_supported,
        "provenance_sha256": {
            "predictions_by_seed.parquet": sha256(source),
            "frozen_config": sha256(args.frozen_config),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensemble.to_parquet(args.output_dir / "confirmation_predictions_ensemble.parquet", index=False)
    folds.to_csv(args.output_dir / "confirmation_fold_metrics.csv", index=False)
    per_seed.to_csv(args.output_dir / "confirmation_seed_metrics.csv", index=False)
    (args.output_dir / "confirmation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
