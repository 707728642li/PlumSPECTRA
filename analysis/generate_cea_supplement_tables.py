"""Generate the five reviewer-priority supplementary tables from frozen CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/v25_external_review_corrections/final_analysis"
PRACTICAL = ROOT / "results/v28_submission_strengthening/practical_accuracy_context.csv"
OUT = ROOT / "results/cea_final_revision/supplement_key_tables.md"
TRAITS = ["FW", "SSC", "pH", "SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"]


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["trait"] = pd.Categorical(result["trait"], TRAITS, ordered=True)
    return result.sort_values("trait")


def main() -> None:
    pooled = pd.read_csv(EVIDENCE / "pooled_metrics.csv")
    centred = pd.read_csv(EVIDENCE / "within_cultivar_centered_metrics.csv")
    reliability = pd.read_csv(EVIDENCE / "texture_reliability_modeling_cohort.csv")
    strongest = pd.read_csv(EVIDENCE / "multiplicity_independent_strongest_family.csv")
    practical = pd.read_csv(PRACTICAL)

    final = ordered(pooled.loc[pooled.model.eq("plumspectra_corrected")])
    s3a = pd.DataFrame({
        "Trait": final.trait.astype(str), "n": final.n.astype(int).astype(str),
        "RMSE": [fmt(v) for v in final.rmse], "MAE": [fmt(v) for v in final.mae],
        "R²": [fmt(v) for v in final.r2], "RPIQ": [fmt(v) for v in final.rpiq],
    })
    model_map = {
        "global_pls": "Global PLSR", "cultivar_aware_pls": "Cultivar-aware PLSR",
        "no_neural_b50": "No-neural B50", "residual_cnn": "Residual CNN",
    }
    comp = pooled.loc[pooled.model.isin(model_map), ["trait", "model", "rmse"]].copy()
    comp["model"] = comp.model.map(model_map)
    comp = comp.pivot(index="trait", columns="model", values="rmse").reindex(TRAITS)
    s3b = comp.reset_index().rename(columns={"trait": "Trait"})
    for column in s3b.columns[1:]:
        s3b[column] = s3b[column].map(fmt)

    s5 = centred.loc[centred.model.isin({"cultivar_aware_pls", "nested_rbf_svr", "no_neural_b50", "residual_cnn", "plumspectra_corrected"}), ["trait", "model", "r2"]].copy()
    s5["model"] = s5.model.map({**model_map, "nested_rbf_svr": "Nested RBF-SVR", "plumspectra_corrected": "PlumSPECTRA"})
    s5 = s5.pivot(index="trait", columns="model", values="r2").reindex(TRAITS).reset_index().rename(columns={"trait": "Trait"})
    for column in s5.columns[1:]:
        s5[column] = s5[column].map(fmt)

    reliability = ordered(reliability)
    s10 = pd.DataFrame({
        "Trait": reliability.trait.astype(str), "n": reliability.n.astype(int).astype(str),
        "ICC(A,1)": reliability.icc_a1.map(fmt), "Replicate r": reliability.pearson_r.map(fmt),
        "Median CV (%)": (100 * reliability.median_replicate_cv).map(lambda v: fmt(v, 1)),
    })

    strongest = ordered(strongest)
    s18 = pd.DataFrame({
        "Trait": strongest.trait.astype(str), "Selected baseline": strongest.strongest_baseline,
        "Gain (%)": strongest.relative_rmse_improvement_pct.map(lambda v: fmt(v, 2)),
        "Simultaneous 95% CI (%)": [f"{fmt(lo, 2)} to {fmt(hi, 2)}" for lo, hi in zip(strongest.simultaneous_ci95_low_pct, strongest.simultaneous_ci95_high_pct)],
        "Supported": strongest.supported_simultaneous_0_05.map({True: "Yes", False: "No"}),
    })

    practical = ordered(practical)
    s41a = pd.DataFrame({
        "Trait": practical.trait.astype(str), "Unit": practical.unit,
        "Median AE": practical.median_absolute_error.map(fmt), "80th-pct AE": practical.p80_absolute_error.map(fmt),
        "Within 0.5 IQR (%)": practical.within_half_iqr_pct.map(lambda v: fmt(v, 1)),
        "R²": practical.r2.map(fmt), "RPIQ": practical.rpiq.map(fmt),
    })
    rel = practical.loc[practical.duplicate_icc_a1.notna()]
    s41b = pd.DataFrame({
        "Trait": rel.trait.astype(str), "ICC(A,1)": rel.duplicate_icc_a1.map(fmt),
        "Median duplicate CV (%)": rel.median_duplicate_cv_pct.map(lambda v: fmt(v, 1)),
        "R² / ICC (%)": rel.r2_as_pct_of_icc.map(lambda v: fmt(v, 1)),
    })

    blocks = [
        "#### Table S3a. PlumSPECTRA pooled OOF performance\n\n" + markdown(s3a),
        "#### Table S3b. Pooled OOF RMSE of principal comparators\n\n" + markdown(s3b),
        "#### Table S5. Within-cultivar centred R²\n\n" + markdown(s5),
        "#### Table S10. Duplicate texture reliability\n\n" + markdown(s10),
        "#### Table S18. Post hoc strongest branch-excluded baseline family\n\n" + markdown(s18),
        "#### Table S41a. Native-unit prediction context\n\n" + markdown(s41a),
        "#### Table S41b. Prediction relative to duplicate reliability\n\n" + markdown(s41b),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
