from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import rankdata

from v2_registry import cultivar_code_map


TRAITS = [
    ("FW", "fruit_weight_g", "Fruit weight", "g", "Conventional quality"),
    ("SSC", "soluble_solids_pct", "Soluble solids", "%", "Conventional quality"),
    ("pH", "ph", "pH", "", "Conventional quality"),
    ("SRF", "skin_break_force_g_mean", "Maximum penetration force", "gf", "Mechanical texture"),
    ("RD", "skin_break_displacement_raw_mean", "Registered-event position", "archive unit", "Mechanical texture"),
    ("PFD", "skin_break_drop_g_mean", "Registered post-event force drop", "gf", "Mechanical texture"),
    ("MFF", "flesh_force_mean_g_mean", "Mean flesh force", "gf", "Mechanical texture"),
    ("F6", "force_at_6_rawpos_g_mean", "Force at 6 archive position units", "gf", "Mechanical texture"),
    ("LS", "loading_stiffness_g_per_rawpos_mean", "Loading stiffness", "gf archive-unit^-1", "Mechanical texture"),
    ("LW", "loading_work_g_rawpos_mean", "Loading work", "gf archive-unit", "Mechanical texture"),
    ("PRW", "post_break_work_g_rawpos_mean", "Registered post-event work", "gf archive-unit", "Mechanical texture"),
    ("AF", "adhesive_force_g_mean", "Adhesive force", "gf", "Mechanical texture"),
]
TRAIT_META = pd.DataFrame(TRAITS, columns=["trait", "target", "trait_label", "unit", "family"])
TRAIT_ORDER = TRAIT_META["trait"].tolist()
TEXTURE_TARGETS = TRAIT_META.loc[TRAIT_META["family"] == "Mechanical texture", "target"].tolist()
QUALITY_TARGETS = TRAIT_META.loc[TRAIT_META["family"] == "Conventional quality", "target"].tolist()


def add_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(TRAIT_META, on=["trait", "target"], how="left", validate="many_to_one")


def orient_pca(scores: np.ndarray, loadings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for component in range(loadings.shape[0]):
        pivot = int(np.argmax(np.abs(loadings[component])))
        if loadings[component, pivot] < 0:
            loadings[component] *= -1
            scores[:, component] *= -1
    return scores, loadings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--texture-manifest", type=Path)
    parser.add_argument("--qc-ledger", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    TRAIT_META.assign(trait_order=np.arange(1, len(TRAIT_META) + 1)).to_csv(
        output / "trait_registry.csv", index=False
    )
    cultivar_registry = pd.read_csv(project / "configs/v2_nomenclature.csv", dtype=str)
    cultivar_registry.to_csv(output / "cultivar_registry.csv", index=False)
    code_map = cultivar_code_map()

    texture_manifest = (
        args.texture_manifest.resolve()
        if args.texture_manifest is not None
        else project / "results/v20/splits/v20_fivefold_manifest.csv"
    )
    qc_ledger = (
        args.qc_ledger.resolve()
        if args.qc_ledger is not None
        else project / "data/processed/texture_qc/texture_qc_ledger.parquet"
    )
    manifest = pd.read_csv(
        texture_manifest,
        dtype={"sample_id": str, "cultivar_ascii": str},
    )
    ledger = pd.read_parquet(qc_ledger).set_index("sample_id")
    cohort = ledger.loc[manifest["sample_id"]].reset_index()
    cohort = cohort.merge(manifest, on=["sample_id", "cultivar_ascii"], how="inner", validate="one_to_one")
    cohort["cultivar_code"] = cohort["cultivar_ascii"].map(code_map)

    counts = (
        cohort.groupby(["cultivar_ascii", "cultivar_code"], observed=True)
        .size().rename("analysis_samples").reset_index()
        .sort_values("analysis_samples", ascending=False)
    )
    counts["source_linked_samples"] = counts["cultivar_ascii"].map(
        ledger.reset_index().groupby("cultivar_ascii").size()
    )
    counts["qc_excluded_samples"] = counts["source_linked_samples"] - counts["analysis_samples"]
    counts.to_csv(output / "cohort_counts.csv", index=False)

    long_rows = []
    for trait, target, label, unit, family in TRAITS:
        values = pd.to_numeric(cohort[target], errors="coerce")
        validity = f"{target}_valid"
        if validity in cohort.columns:
            values = values.where(cohort[validity].astype(bool))
        long_rows.append(pd.DataFrame({
            "sample_id": cohort["sample_id"],
            "cultivar_ascii": cohort["cultivar_ascii"],
            "cultivar_code": cohort["cultivar_code"],
            "outer_fold": cohort["outer_fold"],
            "trait": trait,
            "target": target,
            "trait_label": label,
            "unit": unit,
            "family": family,
            "value": values,
        }))
    phenotype_long = pd.concat(long_rows, ignore_index=True).dropna(subset=["value"])
    phenotype_long.to_csv(output / "phenotype_long.csv", index=False)

    trait_summary = (
        phenotype_long.groupby(["trait", "target", "trait_label", "unit", "family"], observed=True)
        .agg(
            n=("value", "size"), mean=("value", "mean"), sd=("value", "std"),
            median=("value", "median"), q1=("value", lambda x: x.quantile(0.25)),
            q3=("value", lambda x: x.quantile(0.75)), minimum=("value", "min"), maximum=("value", "max"),
        ).reset_index()
    )
    trait_summary.to_csv(output / "trait_summary.csv", index=False)

    texture_matrix = cohort[TEXTURE_TARGETS].apply(pd.to_numeric, errors="coerce")
    texture_matrix = texture_matrix.fillna(texture_matrix.median())
    standardized = (texture_matrix - texture_matrix.median()) / texture_matrix.apply(
        lambda x: max((x - x.median()).abs().median() * 1.4826, 1e-8)
    )
    pca = PCA(n_components=3, random_state=20260822)
    scores = pca.fit_transform(standardized)
    scores, loadings = orient_pca(scores, pca.components_.copy())
    score_frame = cohort[["sample_id", "cultivar_ascii", "cultivar_code"]].copy()
    for i in range(3):
        score_frame[f"PC{i+1}"] = scores[:, i]
    score_frame.to_csv(output / "texture_pca_scores.csv", index=False)
    loading_frame = pd.DataFrame(loadings.T, columns=["PC1", "PC2", "PC3"])
    loading_frame["target"] = TEXTURE_TARGETS
    loading_frame = loading_frame.merge(TRAIT_META, on="target", how="left")
    loading_frame.to_csv(output / "texture_pca_loadings.csv", index=False)
    pd.DataFrame({
        "component": ["PC1", "PC2", "PC3"],
        "variance_explained": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(output / "texture_pca_variance.csv", index=False)

    wide = phenotype_long.pivot(index="sample_id", columns="trait", values="value").reindex(columns=TRAIT_ORDER)
    corr = wide.corr(method="spearman", min_periods=100)
    corr.to_csv(output / "phenotype_spearman_correlation.csv")

    reliability = pd.read_csv(project / "results/eda/tables/texture_replicate_reliability.csv")
    reliability_map = {
        "SRF": "skin_break_force_g", "RD": "skin_break_displacement_raw",
        "PFD": "skin_break_drop_g", "MFF": "flesh_force_mean_g",
        "F6": "force_at_6_rawpos_g", "LS": "loading_stiffness_g_per_rawpos",
        "LW": "loading_work_g_rawpos", "PRW": "post_break_work_g_rawpos",
        "AF": "adhesive_force_g",
    }
    rel_rows = []
    for trait, feature in reliability_map.items():
        row = reliability.loc[reliability["feature"] == feature].copy()
        if len(row) != 1:
            raise RuntimeError(f"Reliability row not uniquely found for {trait}: {feature}")
        row["trait"] = trait
        rel_rows.append(row)
    add_metadata(pd.concat(rel_rows, ignore_index=True).merge(
        TRAIT_META[["trait", "target"]], on="trait", how="left"
    )).to_csv(output / "texture_reliability.csv", index=False)

    texture_predictions = pd.read_parquet(project / "results/v20/final_analysis/v20_merged_predictions.parquet")
    texture_predictions["y_final"] = texture_predictions["y_deep_kernel"]
    texture_predictions["final_variant"] = "Deep-kernel ensemble"
    quality_predictions = pd.read_parquet(project / "results/v22_integrated/quality_analysis/v22_quality_merged_predictions.parquet")
    quality_predictions["y_final"] = quality_predictions["y_plumspectra"]
    quality_predictions["final_variant"] = quality_predictions["selected_variant"].map({
        "deep": "Residual CNN", "deep_kernel": "Deep-kernel ensemble"
    })
    prediction_columns = [
        "sample_id", "cultivar_ascii", "cultivar_code", "target", "trait", "outer_fold",
        "y_true", "y_global_pls", "y_domain_pls", "y_domain_svr", "y_deep", "y_final", "final_variant",
    ]
    predictions = pd.concat([
        texture_predictions[prediction_columns], quality_predictions[prediction_columns]
    ], ignore_index=True)
    predictions = add_metadata(predictions)
    predictions.to_csv(output / "predictions_all12.csv", index=False)

    def combine_metric_table(filename: str) -> pd.DataFrame:
        texture = pd.read_csv(project / "results/v20/final_analysis" / filename)
        quality = pd.read_csv(project / "results/v22_integrated/quality_analysis" / filename)
        texture["is_final"] = texture["model"].eq("deep_kernel")
        quality["is_final"] = quality["model"].eq("plumspectra")
        combined = pd.concat([texture, quality], ignore_index=True)
        combined["model_display"] = combined["model"].map({
            "global_pls": "Global PLSR", "domain_pls": "Cultivar-aware PLSR",
            "domain_svr": "Nested RBF-SVR", "deep": "Residual CNN",
            "deep_kernel": "Deep-kernel ensemble", "plumspectra": "PlumSPECTRA",
        })
        return add_metadata(combined)

    for filename in ["pooled_metrics.csv", "fold_metrics.csv", "within_cultivar_centered_metrics.csv", "cultivar_metrics.csv"]:
        combine_metric_table(filename).to_csv(output / filename, index=False)

    texture_comparison = pd.read_csv(project / "results/v20/final_analysis/paired_cluster_bootstrap_comparisons.csv")
    texture_comparison = texture_comparison.loc[texture_comparison["candidate"] == "deep_kernel"].copy()
    quality_comparison = pd.read_csv(project / "results/v22_integrated/quality_analysis/paired_cluster_bootstrap_comparisons.csv")
    quality_comparison = quality_comparison.loc[quality_comparison["candidate"] == "plumspectra"].copy()
    comparisons = add_metadata(pd.concat([texture_comparison, quality_comparison], ignore_index=True))
    comparisons["baseline_display"] = comparisons["baseline"].map({
        "global_pls": "Global PLSR", "domain_pls": "Cultivar-aware PLSR", "domain_svr": "Nested RBF-SVR"
    })
    comparisons.to_csv(output / "final_model_cluster_comparisons.csv", index=False)

    pd.read_csv(project / "results/v22_integrated/quality_analysis/train_internal_branch_selection.csv").to_csv(
        output / "quality_branch_selection.csv", index=False
    )
    pd.read_csv(project / "results/v20/final_analysis/deep_svr_complementarity.csv").to_csv(
        output / "texture_complementarity.csv", index=False
    )

    v21_predictions = pd.read_parquet(project / "results/v21_crossbatch/final_analysis/v21_merged_predictions.parquet")
    v21_predictions["y_final"] = v21_predictions["y_deep_kernel"]
    v21_predictions = add_metadata(v21_predictions)
    v21_predictions.to_csv(output / "v21_crossbatch_predictions.csv", index=False)
    for source, destination in [
        ("pooled_and_batch_macro_metrics.csv", "v21_pooled_batch_metrics.csv"),
        ("descriptive_batch_bootstrap_comparisons.csv", "v21_batch_comparisons.csv"),
        ("per_batch_metrics.csv", "v21_per_batch_metrics.csv"),
    ]:
        frame = pd.read_csv(project / "results/v21_crossbatch/final_analysis" / source)
        add_metadata(frame).to_csv(output / destination, index=False)
    batch_counts = (
        v21_predictions.drop_duplicates("sample_id")
        .groupby(["batch_id", "cultivar_ascii"], observed=True).size()
        .rename("samples").reset_index()
    )
    batch_counts["cultivar_code"] = batch_counts["cultivar_ascii"].map(code_map)
    batch_counts.to_csv(output / "v21_batch_counts.csv", index=False)

    attention = pd.read_csv(project / "results/v20/spectral_attention/attention_wavelength_summary.csv")
    attention.to_csv(output / "texture_attention_wavelength.csv", index=False)
    vip = pd.read_csv(project / "results/spectral_interpretation/pls_vip_wavelength_summary.csv")
    vip = vip.loc[vip["target"].isin(QUALITY_TARGETS)].merge(
        TRAIT_META[["target", "trait", "trait_label"]], on="target", how="left"
    )
    vip.to_csv(output / "quality_pls_vip_wavelength.csv", index=False)

    invalid_quality = ledger.loc[manifest["sample_id"]].copy()
    invalid_mask = ~invalid_quality[[f"{target}_valid" for target in QUALITY_TARGETS]].astype(bool).all(axis=1)
    invalid_quality.loc[invalid_mask, [
        "cultivar_ascii", *QUALITY_TARGETS, *[f"{target}_valid" for target in QUALITY_TARGETS]
    ]].reset_index().to_csv(output / "quality_invalid_exclusions.csv", index=False)

    manifest_summary = {
        "model_name": "PlumSPECTRA",
        "model_expansion": "Plum Spectral Phenotyping Ensemble with Cultivar-aware Trait Residual Adaptation",
        "linked_fruit": 5502,
        "original_cultivars": 16,
        "analysis_fruit": int(len(cohort)),
        "analysis_cultivars": int(cohort["cultivar_ascii"].nunique()),
        "quality_complete_case_fruit": int(quality_predictions["sample_id"].nunique()),
        "texture_endpoints": 9,
        "quality_traits": 3,
        "wavelengths": 228,
        "wavelength_min_nm": 901.032471,
        "wavelength_max_nm": 1701.208496,
        "texture_curves": 11004,
        "trainable_parameters_per_network": 142285,
        "outer_folds": 5,
        "texture_predictions": int(len(texture_predictions)),
        "quality_predictions": int(len(quality_predictions)),
        "total_out_of_fold_predictions": int(len(predictions)),
    }
    (output / "dataset_figure_summary.json").write_text(json.dumps(manifest_summary, indent=2), encoding="utf-8")
    print(json.dumps(manifest_summary, indent=2))


if __name__ == "__main__":
    main()
