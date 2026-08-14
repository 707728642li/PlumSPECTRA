from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAITS = [
    "skin_break_force_g",
    "skin_break_displacement_raw",
    "skin_break_drop_g",
    "force_at_6_rawpos_g",
    "flesh_force_mean_g",
    "loading_stiffness_g_per_rawpos",
    "loading_work_g_rawpos",
    "post_break_work_g_rawpos",
    "adhesive_force_g",
]

# Predeclared measurement-quality thresholds. None use prediction residuals.
ICC_FAILURE_THRESHOLD = 0.70
MIN_TRAITS_BELOW_ICC_THRESHOLD = 5
MIN_MEDIAN_REPLICATE_CV = 0.10
SPECTRAL_SEVERE_FRACTION_THRESHOLD = 0.05
SESSION_CONDITION_FRACTION_THRESHOLD = 0.05
CURVE_SEVERE_FRACTION_THRESHOLD = 0.02
HARD_FAILURE_FRACTION_THRESHOLD = 0.02
STRONG_BATCH_ETA_SQUARED_THRESHOLD = 0.20
MIN_CULTIVAR_FRUITS = 100
MIN_INDEPENDENT_FAILURE_DOMAINS = 2


def icc_absolute_agreement(rep1: pd.Series, rep2: pd.Series) -> float:
    values = pd.DataFrame(
        {
            "rep1": pd.to_numeric(rep1, errors="coerce"),
            "rep2": pd.to_numeric(rep2, errors="coerce"),
        }
    ).dropna().to_numpy(float)
    n, k = values.shape
    if n < 3:
        return np.nan
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else np.nan


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": groups}).dropna()
    if frame.empty or frame["group"].nunique() < 2:
        return np.nan
    grand = frame["value"].mean()
    total = np.sum((frame["value"] - grand) ** 2)
    between = sum(
        len(group) * (group["value"].mean() - grand) ** 2
        for _, group in frame.groupby("group", observed=True)
    )
    return float(between / total) if total > 0 else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-independent whole-cultivar measurement-quality audit")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ledger = pd.read_parquet(args.ledger)
    rows: list[dict[str, object]] = []
    for cultivar, group in ledger.groupby("cultivar_ascii", observed=True):
        icc_values = {
            trait: icc_absolute_agreement(group[f"rep01_{trait}"], group[f"rep02_{trait}"])
            for trait in TRAITS
        }
        cv_values = {
            trait: float(
                pd.to_numeric(group[f"{trait}_cv"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .median()
            )
            for trait in TRAITS
        }
        batch_effects = [
            eta_squared(group[f"{trait}_mean"], group["batch_id"])
            for trait in TRAITS
        ]
        rows.append(
            {
                "cultivar_ascii": cultivar,
                "fruit_count": int(len(group)),
                "batch_count": int(group["batch_id"].nunique()),
                "analysis_excluded_fraction": float((~group["qc_analysis_include"]).mean()),
                "strict_excluded_fraction": float((~group["qc_primary_include"]).mean()),
                "hard_failure_fraction": float(group["qc_hard_exclude"].mean()),
                "acquisition_severe_fraction": float(group["acquisition_severe"].mean()),
                "replicate_severe_fraction": float(group["replicate_severe"].mean()),
                "spectral_severe_fraction": float(group["spectral_severe"].mean()),
                "session_condition_severe_fraction": float(
                    group.get("session_condition_severe", pd.Series(False, index=group.index)).mean()
                ),
                "curve_severe_fraction": float((group["curve_acquisition_severe_count"] > 0).mean()),
                "median_trait_icc": float(np.nanmedian(list(icc_values.values()))),
                "minimum_trait_icc": float(np.nanmin(list(icc_values.values()))),
                "traits_with_icc_below_0_70": int(sum(value < 0.70 for value in icc_values.values())),
                "traits_with_icc_below_0_50": int(sum(value < 0.50 for value in icc_values.values())),
                "median_replicate_cv": float(np.nanmedian(list(cv_values.values()))),
                "maximum_trait_median_cv": float(np.nanmax(list(cv_values.values()))),
                "median_texture_batch_eta_squared": float(np.nanmedian(batch_effects))
                if np.any(np.isfinite(batch_effects))
                else np.nan,
                **{f"icc_{trait}": value for trait, value in icc_values.items()},
                **{f"median_cv_{trait}": value for trait, value in cv_values.items()},
            }
        )

    audit = pd.DataFrame(rows)
    # A low ICC alone is not sufficient: cultivars with a narrow biological
    # range can have low ICC despite small replicate error.  Require both low
    # reliability across most traits and at least 10% typical replicate CV.
    audit["texture_repeatability_failure"] = (
        (audit["median_trait_icc"] < ICC_FAILURE_THRESHOLD)
        & (audit["traits_with_icc_below_0_70"] >= MIN_TRAITS_BELOW_ICC_THRESHOLD)
        & (audit["median_replicate_cv"] >= MIN_MEDIAN_REPLICATE_CV)
    )
    audit["spectral_quality_failure"] = (
        audit["spectral_severe_fraction"] >= SPECTRAL_SEVERE_FRACTION_THRESHOLD
    )
    audit["session_condition_failure"] = (
        audit["session_condition_severe_fraction"] >= SESSION_CONDITION_FRACTION_THRESHOLD
    )
    audit["curve_acquisition_failure"] = (
        (audit["acquisition_severe_fraction"] >= CURVE_SEVERE_FRACTION_THRESHOLD)
        | (audit["curve_severe_fraction"] >= CURVE_SEVERE_FRACTION_THRESHOLD)
    )
    audit["hard_validity_failure"] = audit["hard_failure_fraction"] >= HARD_FAILURE_FRACTION_THRESHOLD
    audit["strong_batch_shift_review"] = (
        (audit["batch_count"] >= 2)
        & (audit["median_texture_batch_eta_squared"] >= STRONG_BATCH_ETA_SQUARED_THRESHOLD)
    )
    audit["small_cohort_review"] = audit["fruit_count"] < MIN_CULTIVAR_FRUITS
    independent_failure_columns = [
        "texture_repeatability_failure",
        "spectral_quality_failure",
        "session_condition_failure",
        "curve_acquisition_failure",
        "hard_validity_failure",
    ]
    audit["independent_failure_domains"] = audit[independent_failure_columns].sum(axis=1)
    # Whole-cultivar exclusion requires convergent evidence from at least two
    # independent measurement domains.  Model predictions are never consulted.
    audit["exclude_from_primary_modeling"] = (
        audit["independent_failure_domains"] >= MIN_INDEPENDENT_FAILURE_DOMAINS
    )
    audit["decision"] = np.where(
        audit["exclude_from_primary_modeling"],
        "exclude_whole_cultivar",
        np.where(
            audit[["strong_batch_shift_review", "small_cohort_review"]].any(axis=1),
            "retain_with_sensitivity_review",
            "retain",
        ),
    )
    audit = audit.sort_values(
        ["exclude_from_primary_modeling", "independent_failure_domains", "spectral_severe_fraction"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    excluded = audit.loc[audit["exclude_from_primary_modeling"], "cultivar_ascii"].tolist()
    review = audit.loc[audit["decision"].eq("retain_with_sensitivity_review"), "cultivar_ascii"].tolist()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "cultivar_measurement_quality_audit.csv", index=False, encoding="utf-8-sig")
    decision = {
        "model_residuals_used": False,
        "whole_cultivar_rule": (
            "Exclude only when at least two model-independent domains fail. Spectral signal quality and "
            "acquisition-session conditions are separate domains. Low ICC requires corroborating median "
            "replicate CV >= 10% to avoid penalizing narrow biological ranges."
        ),
        "thresholds": {
            "icc_failure": ICC_FAILURE_THRESHOLD,
            "minimum_traits_below_icc": MIN_TRAITS_BELOW_ICC_THRESHOLD,
            "minimum_median_replicate_cv": MIN_MEDIAN_REPLICATE_CV,
            "spectral_severe_fraction": SPECTRAL_SEVERE_FRACTION_THRESHOLD,
            "session_condition_fraction": SESSION_CONDITION_FRACTION_THRESHOLD,
            "curve_severe_fraction": CURVE_SEVERE_FRACTION_THRESHOLD,
            "hard_failure_fraction": HARD_FAILURE_FRACTION_THRESHOLD,
            "minimum_independent_failure_domains": MIN_INDEPENDENT_FAILURE_DOMAINS,
        },
        "excluded_cultivars": excluded,
        "retained_with_sensitivity_review": review,
        "retained_cultivars": audit.loc[~audit["exclude_from_primary_modeling"], "cultivar_ascii"].tolist(),
        "fruit_count_before": int(len(ledger)),
        "fruit_count_after_whole_cultivar_exclusion": int((~ledger["cultivar_ascii"].isin(excluded)).sum()),
        "whole_cultivar_excluded_fraction": float(ledger["cultivar_ascii"].isin(excluded).mean()),
        "important_limitation": (
            "Most cultivars were measured in one batch, so cultivar biology and operator/batch effects cannot be "
            "fully separated. A distinctive but internally reliable cultivar is retained."
        ),
    }
    (output_dir / "cultivar_exclusion_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(audit[["cultivar_ascii", "fruit_count", *independent_failure_columns, "decision"]].to_string(index=False))
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
