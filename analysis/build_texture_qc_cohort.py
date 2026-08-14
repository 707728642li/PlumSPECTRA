from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


QC_VERSION = "0.3.0-review-correction"

# Named, model-independent QC constants. The 0.80 moderate threshold is an
# evidence-ranking tier (top 20%), not a claim that every flagged observation
# is erroneous. Exclusion requires convergent evidence below; severe thresholds
# use robust-z values well beyond ordinary Gaussian tail behaviour.
DEFAULT_MODERATE_EVIDENCE_QUANTILE = 0.80
ROBUST_Z_ELEVATED = 3.0
ROBUST_Z_STRONG = 4.5
ROBUST_Z_EXTREME = 8.0
FRUIT_ACQUISITION_Z_EXTREME = 10.0
MAX_BASELINE_NOISE_FORCE_FRACTION = 0.08
MIN_STRONG_ACQUISITION_CHANNELS = 2
MIN_SEVERE_CURVES_PER_FRUIT = 2
MIN_HARD_FAILURE_CURVES_PER_FRUIT = 1
REPLICATE_REFERENCE_STABILIZER = 0.05
MIN_STRONG_REPLICATE_FAMILIES = 2
MIN_ELEVATED_REPLICATE_FAMILIES = 3
MIN_SECOND_REPLICATE_FAMILY_Z = 2.0
TOP_FAMILY_COUNT_FOR_RMS = 3
MIN_VALID_HEADLINE_ENDPOINTS = 7
MIN_MODERATE_SUPPORT_DOMAINS = 1
MIN_MODERATE_TECHNICAL_DOMAINS = 1
MIN_MULTIDOMAIN_TECHNICAL_FLAGS = 2
MIN_MULTIDOMAIN_SUPPORT_FLAGS = 2

TEXTURE_ENDPOINTS = {
    "skin_break_force_g": ("peak", "headline", "Maximum penetration force", "g"),
    "skin_break_displacement_raw": ("peak", "headline", "Peak-force displacement", "raw_position_unit"),
    "skin_break_drop_g": ("peak", "headline", "Post-peak force drop", "g"),
    "flesh_force_mean_g": ("flesh", "headline", "Post-peak flesh resistance", "g"),
    "force_at_6_rawpos_g": ("flesh", "headline", "Force at 6 raw position units", "g"),
    "loading_stiffness_g_per_rawpos": ("stiffness", "headline", "Loading stiffness", "g/raw_position_unit"),
    "loading_work_g_rawpos": ("energy", "headline", "Total loading work", "g*raw_position_unit"),
    "post_break_work_g_rawpos": ("energy", "headline", "Post-peak work", "g*raw_position_unit"),
    "adhesive_force_g": ("adhesion", "headline", "Adhesive force", "g"),
    "skin_break_work_g_rawpos": ("energy", "secondary", "Work to maximum force", "g*raw_position_unit"),
    "max_loading_force_g": ("peak", "redundant", "Maximum loading force", "g"),
    "force_at_2_rawpos_g": ("flesh", "secondary", "Force at 2 raw position units", "g"),
    "force_at_4_rawpos_g": ("flesh", "exploratory", "Force at 4 raw position units", "g"),
    "fracture_peak_count": ("peak", "exploratory", "Local peak count", "count"),
}

ACQUISITION_FEATURES = [
    "baseline_noise_g",
    "baseline_force_g",
    "loading_speed_rawpos_per_s",
    "duration_s",
    "loading_duration_s",
    "max_displacement_raw",
    "sampling_rate_hz",
    "point_count",
    "contact_time_s",
]

NIR_SPECTRAL_QUALITY_FEATURES = [
    "nir_c_absorbance_min",
    "nir_c_absorbance_max",
    "nir_c_absorbance_mean",
    "nir_c_reference_signal_min",
    "nir_c_sample_signal_min",
    "spectral_pca_score_distance_within_batch",
    "spectral_pca_orthogonal_robust_z",
]

# Environmental and instrument-state values describe an acquisition session.
# They are audited separately and never treated as fruit-level spectral defects.
NIR_SESSION_CONDITION_FEATURES = [
    "nir_c_system_temp_c",
    "nir_c_detector_temp_c",
    "nir_c_humidity_pct",
    "nir_c_lamp_pd",
]

CHEMICAL_FEATURES = ["fruit_weight_g", "soluble_solids_pct", "ph"]


def robust_group_abs_z(values: pd.Series, groups: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": groups})
    median = frame.groupby("group", observed=True)["value"].transform("median")
    abs_dev = (frame["value"] - median).abs()
    mad = abs_dev.groupby(frame["group"], observed=True).transform("median")
    q25 = frame.groupby("group", observed=True)["value"].transform(lambda x: x.quantile(0.25))
    q75 = frame.groupby("group", observed=True)["value"].transform(lambda x: x.quantile(0.75))
    scale = 1.4826 * mad
    iqr_scale = (q75 - q25) / 1.349
    scale = scale.where(scale > 1e-12, iqr_scale)
    global_scale = 1.4826 * np.nanmedian(np.abs(frame["value"] - np.nanmedian(frame["value"])))
    if not np.isfinite(global_scale) or global_scale <= 1e-12:
        global_scale = float(np.nanstd(frame["value"]))
    if not np.isfinite(global_scale) or global_scale <= 1e-12:
        global_scale = 1.0
    scale = scale.fillna(global_scale).where(scale > 1e-12, global_scale)
    return ((frame["value"] - median).abs() / scale).astype(float)


def percentile_rank(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(method="average", pct=True, na_option="bottom").astype(float)


def top_k_rms(frame: pd.DataFrame, k: int = 3) -> pd.Series:
    array = frame.to_numpy(dtype=float)
    array = np.where(np.isfinite(array), array, 0.0)
    k = min(k, array.shape[1])
    selected = np.partition(array, -k, axis=1)[:, -k:]
    return pd.Series(np.sqrt(np.mean(selected**2, axis=1)), index=frame.index)


def semicolon_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["qc_hard_exclude"]:
        reasons.append("hard_validity_failure")
    if row["acquisition_severe"]:
        reasons.append("severe_curve_acquisition_anomaly")
    elif row["acquisition_moderate"]:
        reasons.append("curve_acquisition_anomaly")
    if row["replicate_severe"]:
        reasons.append("severe_multifamily_replicate_discordance")
    elif row["replicate_moderate"]:
        reasons.append("replicate_discordance")
    if row["texture_multivariate_moderate"]:
        reasons.append("texture_multivariate_outlier")
    if row["spectral_moderate"]:
        reasons.append("spectral_qc_outlier")
    if row.get("session_condition_moderate", False):
        reasons.append("acquisition_session_condition_outlier")
    if row["chemical_moderate"]:
        reasons.append("chemical_phenotype_outlier")
    return ";".join(reasons)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a model-independent, auditable texture QC cohort.")
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--curve-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--moderate-quantile",
        type=float,
        default=DEFAULT_MODERATE_EVIDENCE_QUANTILE,
        help=(
            "Within-dataset evidence-rank threshold for the descriptive moderate tier "
            "(default top 20%); this is not a defect threshold and never uses model residuals."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    master = pd.read_parquet(args.master).copy()
    curves = pd.read_parquet(args.curve_features).copy()
    curves = curves.sort_values(["sample_id", "replicate"]).reset_index(drop=True)

    # Curve acquisition evidence is normalized within batch, so a batch-wide operator shift
    # is described separately rather than making every fruit in that batch an outlier.
    acquisition_z_cols: list[str] = []
    for feature in ACQUISITION_FEATURES:
        transformed = pd.to_numeric(curves[feature], errors="coerce")
        if feature in {"baseline_noise_g", "loading_speed_rawpos_per_s", "duration_s", "loading_duration_s", "point_count"}:
            transformed = np.log1p(transformed.clip(lower=0))
        column = f"acq_z_{feature}"
        curves[column] = robust_group_abs_z(transformed, curves["batch_id"])
        acquisition_z_cols.append(column)
    curves["baseline_noise_fraction"] = curves["baseline_noise_g"] / curves["skin_break_force_g"].abs().clip(lower=1.0)
    curves["acquisition_zmax"] = curves[acquisition_z_cols].max(axis=1, skipna=True)
    curves["acquisition_count_z3"] = (curves[acquisition_z_cols] > ROBUST_Z_ELEVATED).sum(axis=1)
    curves["acquisition_count_z4_5"] = (curves[acquisition_z_cols] > ROBUST_Z_STRONG).sum(axis=1)
    curves["curve_hard_failure"] = (
        curves["curve_qc_status"].ne("pass")
        | ~np.isfinite(curves["skin_break_force_g"])
        | ~np.isfinite(curves["max_displacement_raw"])
    )
    curves["curve_acquisition_severe"] = (
        curves["curve_hard_failure"]
        | (curves["acquisition_count_z4_5"] >= MIN_STRONG_ACQUISITION_CHANNELS)
        | (curves["acquisition_zmax"] >= ROBUST_Z_EXTREME)
        | (curves["baseline_noise_fraction"] > MAX_BASELINE_NOISE_FORCE_FRACTION)
    )

    curve_summary = curves.groupby("sample_id", observed=True).agg(
        curve_hard_failure_count=("curve_hard_failure", "sum"),
        curve_acquisition_severe_count=("curve_acquisition_severe", "sum"),
        acquisition_zmax=("acquisition_zmax", "max"),
        acquisition_zmean=("acquisition_zmax", "mean"),
        acquisition_count_z3=("acquisition_count_z3", "sum"),
        acquisition_count_z4_5=("acquisition_count_z4_5", "sum"),
        baseline_noise_fraction_max=("baseline_noise_fraction", "max"),
    )
    ledger = master.merge(curve_summary, left_on="sample_id", right_index=True, how="left", validate="one_to_one")
    ledger["acquisition_stat"] = (
        ledger["acquisition_zmax"].fillna(10.0)
        + 0.35 * ledger["acquisition_count_z3"].fillna(0)
        + 1.00 * ledger["curve_acquisition_severe_count"].fillna(0)
    )
    ledger["acquisition_rank"] = percentile_rank(ledger["acquisition_stat"])
    ledger["acquisition_severe"] = (
        (ledger["curve_acquisition_severe_count"] >= MIN_SEVERE_CURVES_PER_FRUIT)
        | (ledger["curve_hard_failure_count"] >= MIN_HARD_FAILURE_CURVES_PER_FRUIT)
        | (ledger["acquisition_zmax"] >= FRUIT_ACQUISITION_Z_EXTREME)
    )

    # Replicate disagreement is assessed within batch using symmetric relative differences.
    family_z: dict[str, list[str]] = {}
    replicate_detail: dict[str, pd.Series] = {}
    for endpoint, (family, role, _, _) in TEXTURE_ENDPOINTS.items():
        if role not in {"headline", "secondary"}:
            continue
        rep1 = pd.to_numeric(ledger[f"rep01_{endpoint}"], errors="coerce")
        rep2 = pd.to_numeric(ledger[f"rep02_{endpoint}"], errors="coerce")
        batch_reference = (
            pd.concat([rep1.abs(), rep2.abs()], axis=1).mean(axis=1)
            .groupby(ledger["batch_id"], observed=True)
            .transform("median")
        )
        denominator = (
            0.5 * (rep1.abs() + rep2.abs())
            + REPLICATE_REFERENCE_STABILIZER * batch_reference.clip(lower=1e-9)
        )
        symmetric_difference = (rep1 - rep2).abs() / denominator
        difference_z = robust_group_abs_z(np.log1p(symmetric_difference), ledger["batch_id"])
        detail_col = f"replicate_z_{endpoint}"
        replicate_detail[detail_col] = difference_z
        family_z.setdefault(family, []).append(detail_col)
    replicate_frame = pd.DataFrame(replicate_detail, index=ledger.index)
    ledger = pd.concat([ledger, replicate_frame], axis=1)
    replicate_family_cols: list[str] = []
    for family, columns in family_z.items():
        name = f"replicate_family_z_{family}"
        ledger[name] = ledger[columns].max(axis=1, skipna=True)
        replicate_family_cols.append(name)
    ledger["replicate_stat"] = top_k_rms(
        ledger[replicate_family_cols], k=TOP_FAMILY_COUNT_FOR_RMS
    )
    ledger["replicate_rank"] = percentile_rank(ledger["replicate_stat"])
    ledger["replicate_family_count_z3"] = (
        ledger[replicate_family_cols] > ROBUST_Z_ELEVATED
    ).sum(axis=1)
    ledger["replicate_family_count_z4_5"] = (
        ledger[replicate_family_cols] > ROBUST_Z_STRONG
    ).sum(axis=1)
    ledger["replicate_severe"] = (
        (ledger["replicate_family_count_z4_5"] >= MIN_STRONG_REPLICATE_FAMILIES)
        | (ledger["replicate_family_count_z3"] >= MIN_ELEVATED_REPLICATE_FAMILIES)
        | (
            (ledger[replicate_family_cols].max(axis=1) >= ROBUST_Z_EXTREME)
            & (
                ledger[replicate_family_cols]
                .apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
                >= MIN_SECOND_REPLICATE_FAMILY_Z
            )
        )
    )

    # Multivariate texture extremes are support evidence only and cannot independently exclude a fruit.
    texture_family_cols: dict[str, list[str]] = {}
    for endpoint, (family, role, _, _) in TEXTURE_ENDPOINTS.items():
        if role != "headline":
            continue
        source = f"{endpoint}_mean"
        column = f"texture_z_{endpoint}"
        ledger[column] = robust_group_abs_z(ledger[source], ledger["batch_id"])
        texture_family_cols.setdefault(family, []).append(column)
    texture_family_max_cols: list[str] = []
    for family, columns in texture_family_cols.items():
        name = f"texture_family_z_{family}"
        ledger[name] = ledger[columns].max(axis=1, skipna=True)
        texture_family_max_cols.append(name)
    ledger["texture_multivariate_stat"] = top_k_rms(
        ledger[texture_family_max_cols], k=TOP_FAMILY_COUNT_FOR_RMS
    )
    ledger["texture_multivariate_rank"] = percentile_rank(ledger["texture_multivariate_stat"])
    ledger["texture_family_count_z4_5"] = (
        ledger[texture_family_max_cols] > ROBUST_Z_STRONG
    ).sum(axis=1)

    # NIR evidence uses only acquisition/QC values, never spectral prediction residuals.
    nir_z_cols: list[str] = []
    for feature in NIR_SPECTRAL_QUALITY_FEATURES:
        if feature not in ledger.columns:
            continue
        column = f"nir_qc_z_{feature}"
        ledger[column] = robust_group_abs_z(ledger[feature], ledger["batch_id"])
        nir_z_cols.append(column)
    ledger["spectral_stat"] = top_k_rms(
        ledger[nir_z_cols], k=min(TOP_FAMILY_COUNT_FOR_RMS, len(nir_z_cols))
    )
    ledger["spectral_rank"] = percentile_rank(ledger["spectral_stat"])
    ledger["spectral_severe"] = (
        ledger["nir_c_valid"].ne(True)
        | (ledger[nir_z_cols].max(axis=1) >= ROBUST_Z_EXTREME)
        | ledger.get(
            "spectral_pca_orthogonal_robust_z", pd.Series(0, index=ledger.index)
        ).gt(ROBUST_Z_EXTREME)
    )

    session_z_cols: list[str] = []
    for feature in NIR_SESSION_CONDITION_FEATURES:
        if feature not in ledger.columns:
            continue
        column = f"session_qc_z_{feature}"
        ledger[column] = robust_group_abs_z(ledger[feature], ledger["batch_id"])
        session_z_cols.append(column)
    ledger["session_condition_stat"] = top_k_rms(
        ledger[session_z_cols], k=min(TOP_FAMILY_COUNT_FOR_RMS, len(session_z_cols))
    )
    ledger["session_condition_rank"] = percentile_rank(ledger["session_condition_stat"])
    ledger["session_condition_severe"] = ledger[session_z_cols].max(axis=1).ge(
        ROBUST_Z_EXTREME
    )

    # Chemical phenotype extremes are supporting evidence; invalid/missing measurements are
    # target-specific exclusions but not automatic texture exclusions when NIR+texture are valid.
    chemical_z_cols: list[str] = []
    for feature in CHEMICAL_FEATURES:
        column = f"chemical_z_{feature}"
        ledger[column] = robust_group_abs_z(ledger[feature], ledger["batch_id"])
        chemical_z_cols.append(column)
    ledger["chemical_stat"] = ledger[chemical_z_cols].max(axis=1, skipna=True)
    ledger["chemical_rank"] = percentile_rank(ledger["chemical_stat"])

    q = float(args.moderate_quantile)
    ledger["acquisition_moderate"] = ledger["acquisition_rank"].ge(q)
    ledger["replicate_moderate"] = ledger["replicate_rank"].ge(q)
    ledger["texture_multivariate_moderate"] = ledger["texture_multivariate_rank"].ge(q)
    ledger["spectral_moderate"] = ledger["spectral_rank"].ge(q) | ledger.get(
        "spectral_soft_outlier_flag", pd.Series(False, index=ledger.index)
    ).fillna(False).astype(bool)
    ledger["session_condition_moderate"] = ledger["session_condition_rank"].ge(q)
    ledger["chemical_moderate"] = ledger["chemical_rank"].ge(q)

    ledger["qc_hard_exclude"] = (
        ledger["nir_c_valid"].ne(True)
        | ledger["texture_valid_replicates"].lt(2)
        | ledger[[f"{endpoint}_mean" for endpoint, (_, role, _, _) in TEXTURE_ENDPOINTS.items() if role == "headline"]]
        .notna()
        .sum(axis=1)
        .lt(MIN_VALID_HEADLINE_ENDPOINTS)
    )
    ledger["technical_severe"] = ledger["acquisition_severe"] | ledger["replicate_severe"]
    ledger["technical_moderate_count"] = ledger[["acquisition_moderate", "replicate_moderate"]].sum(axis=1)
    ledger["support_moderate_count"] = ledger[
        ["texture_multivariate_moderate", "spectral_moderate", "chemical_moderate"]
    ].sum(axis=1)

    # Consensus logic: technical evidence is mandatory. This prevents biologically unusual but
    # well-measured fruit from being removed merely because their phenotypes are extreme.
    ledger["qc_consensus_exclude"] = (
        ledger["qc_hard_exclude"]
        | (
            ledger["technical_severe"]
            & (ledger["support_moderate_count"] >= MIN_MODERATE_SUPPORT_DOMAINS)
        )
        | (
            (ledger["technical_moderate_count"] >= MIN_MULTIDOMAIN_TECHNICAL_FLAGS)
            & (ledger["support_moderate_count"] >= MIN_MODERATE_SUPPORT_DOMAINS)
        )
        | (
            (ledger["technical_moderate_count"] >= MIN_MODERATE_TECHNICAL_DOMAINS)
            & (ledger["support_moderate_count"] >= MIN_MULTIDOMAIN_SUPPORT_FLAGS)
        )
    )
    ledger["qc_high_confidence_exclude"] = (
        ledger["qc_hard_exclude"]
        | (
            ledger["technical_severe"]
            & (ledger["support_moderate_count"] >= MIN_MODERATE_SUPPORT_DOMAINS)
        )
    )
    ledger["qc_analysis_include"] = ~ledger["qc_high_confidence_exclude"]
    ledger["qc_primary_include"] = ~ledger["qc_consensus_exclude"]
    ledger["qc_sensitivity_include"] = ~ledger["qc_hard_exclude"]
    ledger["qc_evidence_class_count"] = ledger[
        [
            "acquisition_moderate",
            "replicate_moderate",
            "texture_multivariate_moderate",
            "spectral_moderate",
            "chemical_moderate",
        ]
    ].sum(axis=1)
    ledger["qc_reason"] = ledger.apply(semicolon_reasons, axis=1)
    ledger["qc_version"] = QC_VERSION

    endpoint_registry = pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                "mechanical_family": values[0],
                "analysis_role": values[1],
                "display_name": values[2],
                "unit": values[3],
            }
            for endpoint, values in TEXTURE_ENDPOINTS.items()
        ]
    )

    audit_columns = [
        "sample_id",
        "batch_id",
        "cultivar_ascii",
        "cultivar_original",
        "fruit_number",
        "texture_valid_replicates",
        "qc_version",
        "qc_primary_include",
        "qc_analysis_include",
        "qc_sensitivity_include",
        "qc_hard_exclude",
        "qc_high_confidence_exclude",
        "qc_consensus_exclude",
        "qc_reason",
        "qc_evidence_class_count",
        "acquisition_rank",
        "replicate_rank",
        "texture_multivariate_rank",
        "spectral_rank",
        "session_condition_rank",
        "chemical_rank",
        "acquisition_severe",
        "replicate_severe",
        "spectral_severe",
        "session_condition_severe",
        "acquisition_moderate",
        "replicate_moderate",
        "texture_multivariate_moderate",
        "spectral_moderate",
        "session_condition_moderate",
        "chemical_moderate",
        "replicate_family_count_z3",
        "replicate_family_count_z4_5",
        "texture_family_count_z4_5",
        "curve_hard_failure_count",
        "curve_acquisition_severe_count",
        "acquisition_zmax",
        "baseline_noise_fraction_max",
    ]
    audit_columns += replicate_family_cols + texture_family_max_cols + session_z_cols
    audit = ledger[audit_columns].copy()

    primary = ledger[ledger["qc_primary_include"]].copy()
    analysis = ledger[ledger["qc_analysis_include"]].copy()
    sensitivity = ledger[ledger["qc_sensitivity_include"]].copy()
    batch_summary = ledger.groupby(["cultivar_ascii", "batch_id"], observed=True).agg(
        fruit_count=("sample_id", "size"),
        primary_included=("qc_primary_include", "sum"),
        consensus_excluded=("qc_consensus_exclude", "sum"),
        hard_excluded=("qc_hard_exclude", "sum"),
        median_acquisition_stat=("acquisition_stat", "median"),
        median_replicate_stat=("replicate_stat", "median"),
        median_spectral_stat=("spectral_stat", "median"),
        median_session_condition_stat=("session_condition_stat", "median"),
        median_baseline_noise_g=("baseline_noise_g_mean", "median"),
        median_loading_speed=("loading_speed_rawpos_per_s_mean", "median"),
    ).reset_index()
    batch_summary["consensus_excluded_fraction"] = batch_summary["consensus_excluded"] / batch_summary["fruit_count"]

    summary = {
        "qc_version": QC_VERSION,
        "selection_is_model_independent": True,
        "model_residuals_used": False,
        "moderate_evidence_quantile": q,
        "fruit_total": int(len(ledger)),
        "primary_included": int(ledger["qc_primary_include"].sum()),
        "primary_excluded": int(ledger["qc_consensus_exclude"].sum()),
        "primary_excluded_fraction": float(ledger["qc_consensus_exclude"].mean()),
        "sensitivity_included": int(ledger["qc_sensitivity_include"].sum()),
        "analysis_included": int(ledger["qc_analysis_include"].sum()),
        "high_confidence_excluded": int(ledger["qc_high_confidence_exclude"].sum()),
        "high_confidence_excluded_fraction": float(ledger["qc_high_confidence_exclude"].mean()),
        "hard_excluded": int(ledger["qc_hard_exclude"].sum()),
        "hard_excluded_fraction": float(ledger["qc_hard_exclude"].mean()),
        "evidence_counts": {
            column: int(ledger[column].sum())
            for column in [
                "acquisition_severe",
                "replicate_severe",
                "spectral_severe",
                "session_condition_severe",
                "acquisition_moderate",
                "replicate_moderate",
                "texture_multivariate_moderate",
                "spectral_moderate",
                "session_condition_moderate",
                "chemical_moderate",
            ]
        },
        "selection_rule": (
            "The analysis tier excludes hard failures or severe technical acquisition/replicate evidence with independent support. "
            "The strict release tier additionally requires concordant moderate evidence. Biological extremes alone cannot exclude."
        ),
        "headline_endpoints": [
            endpoint for endpoint, (_, role, _, _) in TEXTURE_ENDPOINTS.items() if role == "headline"
        ],
    }

    ledger.to_parquet(output_dir / "texture_qc_ledger.parquet", index=False, compression="zstd")
    audit.to_csv(output_dir / "texture_qc_audit.csv", index=False, encoding="utf-8-sig")
    primary.to_parquet(output_dir / "texture_primary_cohort.parquet", index=False, compression="zstd")
    analysis.to_parquet(output_dir / "texture_analysis_cohort.parquet", index=False, compression="zstd")
    sensitivity.to_parquet(output_dir / "texture_sensitivity_cohort.parquet", index=False, compression="zstd")
    endpoint_registry.to_csv(output_dir / "texture_endpoint_registry.csv", index=False, encoding="utf-8-sig")
    batch_summary.to_csv(output_dir / "texture_batch_qc_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "texture_qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
