from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analyze_v24_hr_strengthening import analyze_cultivar_texture, analyze_qc
from prepare_v22_figure_data import TRAIT_META, add_metadata, orient_pca
from v2_registry import cultivar_code_map


MODEL_DISPLAY = {
    "global_pls": "Global PLSR",
    "cultivar_aware_pls": "Cultivar-aware PLSR",
    "nested_rbf_svr": "Nested RBF-SVR",
    "residual_cnn": "Residual CNN",
    "no_neural_b50": "No-neural B50 ensemble",
    "cultivar_mean_null": "Cultivar-mean null",
    "plumspectra_corrected": "PlumSPECTRA",
    "plumspectra_fixed_gate_sensitivity": "Fixed-gate sensitivity",
}


def decorate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model_display"] = result["model"].map(MODEL_DISPLAY)
    result["is_final"] = result["model"].eq("plumspectra_corrected")
    return add_metadata(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crossbatch-analysis-dir", type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    analysis = args.analysis_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Build the unchanged cohort/phenotype/attention layers first, then replace
    # every model-dependent table with the corrected V25 evidence.
    subprocess.run(
        [
            sys.executable,
            str(project / "src" / "prepare_v22_figure_data.py"),
            "--project",
            str(project),
            "--output-dir",
            str(output),
            "--texture-manifest",
            str(project / "results/v25_external_review_corrections/splits/v20_fivefold_manifest.csv"),
            "--qc-ledger",
            str(project / "results/v25_external_review_corrections/qc_rebuild/texture_qc_ledger.parquet"),
        ],
        cwd=project,
        check=True,
    )

    code_map = cultivar_code_map()
    analyze_cultivar_texture(output / "phenotype_long.csv", output)
    analyze_qc(
        project
        / "results/v25_external_review_corrections/qc_cultivar_audit/cultivar_measurement_quality_audit.csv",
        project
        / "results/v25_external_review_corrections/qc_cultivar_audit/cultivar_exclusion_decision.json",
        output,
    )
    predictions = pd.read_parquet(analysis / "v25_integrated_predictions.parquet").copy()
    predictions["cultivar_code"] = predictions["cultivar_ascii"].map(code_map)
    predictions["final_variant"] = predictions["selected_variant"].map(
        {"deep_kernel": "Deep-kernel ensemble", "deep_only": "Residual CNN"}
    )
    predictions = predictions.rename(columns={"y_final": "y_final_v25"})
    predictions["y_final"] = predictions["y_final_v25"]
    predictions = add_metadata(predictions)
    predictions.to_csv(output / "predictions_all12.csv", index=False)

    # Freeze the phenotype PCA to the same final complete-case endpoint matrix
    # used by the V25 endpoint-structure audit.  The legacy V22 compatibility
    # layer used median/MAD scaling; retaining it here would make Fig. 2 report
    # a different, albeit valid, PCA variance from the final audit tables.
    texture_order = ["SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"]
    texture_truth = predictions.loc[
        predictions["trait"].isin(texture_order),
        ["sample_id", "cultivar_ascii", "cultivar_code", "trait", "y_true"],
    ].copy()
    texture_matrix = (
        texture_truth.pivot(index="sample_id", columns="trait", values="y_true")
        .reindex(columns=texture_order)
        .dropna()
    )
    standardized = (texture_matrix - texture_matrix.mean()) / texture_matrix.std(ddof=1)
    pca = PCA(n_components=3, random_state=20260822)
    scores = pca.fit_transform(standardized)
    scores, loadings = orient_pca(scores, pca.components_.copy())
    sample_meta = (
        texture_truth[["sample_id", "cultivar_ascii", "cultivar_code"]]
        .drop_duplicates("sample_id")
        .set_index("sample_id")
        .loc[texture_matrix.index]
        .reset_index()
    )
    for component in range(3):
        sample_meta[f"PC{component + 1}"] = scores[:, component]
    sample_meta.to_csv(output / "texture_pca_scores.csv", index=False)
    loading_frame = pd.DataFrame(
        loadings.T,
        columns=["PC1", "PC2", "PC3"],
    )
    loading_frame["trait"] = texture_order
    loading_frame = loading_frame.merge(
        TRAIT_META,
        on="trait",
        how="left",
        validate="one_to_one",
    )
    loading_frame.to_csv(output / "texture_pca_loadings.csv", index=False)
    pd.DataFrame(
        {
            "component": ["PC1", "PC2", "PC3"],
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(output / "texture_pca_variance.csv", index=False)

    # prepare_v22_figure_data builds a legacy cross-batch layer for backward
    # compatibility. Replace it whenever the corrected V25 held-batch audit is
    # supplied, so no final figure can silently mix protocol versions.
    crossbatch = (
        args.crossbatch_analysis_dir.resolve()
        if args.crossbatch_analysis_dir is not None
        else project / "results/v25_external_review_corrections/crossbatch_final_analysis"
    )
    corrected_crossbatch = crossbatch / "v21_merged_predictions.parquet"
    if corrected_crossbatch.exists():
        crossbatch_predictions = pd.read_parquet(corrected_crossbatch).copy()
        crossbatch_predictions["y_final"] = crossbatch_predictions["y_deep_kernel"]
        add_metadata(crossbatch_predictions).to_csv(
            output / "v21_crossbatch_predictions.csv", index=False
        )
        for source, destination in (
            ("pooled_and_batch_macro_metrics.csv", "v21_pooled_batch_metrics.csv"),
            ("descriptive_batch_bootstrap_comparisons.csv", "v21_batch_comparisons.csv"),
            ("per_batch_metrics.csv", "v21_per_batch_metrics.csv"),
        ):
            add_metadata(pd.read_csv(crossbatch / source)).to_csv(
                output / destination, index=False
            )
        batch_counts = (
            crossbatch_predictions.drop_duplicates("sample_id")
            .groupby(["batch_id", "cultivar_ascii"], observed=True)
            .size()
            .rename("samples")
            .reset_index()
        )
        batch_counts["cultivar_code"] = batch_counts["cultivar_ascii"].map(code_map)
        batch_counts.to_csv(output / "v21_batch_counts.csv", index=False)
    elif args.crossbatch_analysis_dir is not None:
        raise FileNotFoundError(corrected_crossbatch)

    for filename in (
        "pooled_metrics.csv",
        "fold_metrics.csv",
        "within_cultivar_centered_metrics.csv",
        "cultivar_metrics.csv",
    ):
        decorate_metrics(pd.read_csv(analysis / filename)).to_csv(output / filename, index=False)

    multiplicity = pd.read_csv(analysis / "multiplicity_adjusted_contrasts.csv")
    primary = multiplicity.assign(
        target="",
        candidate="plumspectra_corrected",
        baseline_display=multiplicity["baseline"],
        relative_improvement_ci_low=multiplicity["bootstrap_ci95_low_pct"],
        relative_improvement_ci_high=multiplicity["bootstrap_ci95_high_pct"],
        claim_status=multiplicity["bootstrap_ci95_low_pct"].gt(0).map(
            {True: "supported_outperformance", False: "not_supported"}
        ),
    )
    target_map = TRAIT_META.set_index("trait")["target"]
    primary["target"] = primary["trait"].map(target_map)
    primary.to_csv(output / "final_model_cluster_comparisons.csv", index=False)

    for filename in (
        "pooled_null_centered_r2.csv",
        "extended_cluster_comparisons.csv",
        "multiplicity_adjusted_contrasts.csv",
        "multiplicity_baseline_family_sensitivity.csv",
        "multiplicity_strongest_baseline_family.csv",
        "fold_hyperparameter_choices.csv",
        "residual_gate_audit.csv",
        "texture_endpoint_correlation.csv",
        "texture_endpoint_pca_variance.csv",
        "texture_endpoint_pca_loadings.csv",
        "texture_reliability_modeling_cohort.csv",
        "cultivar_batch_counts.csv",
        "within_cultivar_batch_effects.csv",
        "cultivar_611_spectral_domain_decomposition.csv",
        "cultivar_611_repeatability_sensitivity.csv",
        "cultivar_exclusion_performance_sensitivity.csv",
        "fewshot_summary.csv",
        "fewshot_minimum_shots.csv",
        "heldbatch_claim_audit.csv",
        "equal_information_pls2_comparison.csv",
    ):
        source = analysis / filename
        if source.exists():
            shutil.copy2(source, output / filename)

    reliability = pd.read_csv(analysis / "texture_reliability_modeling_cohort.csv")
    reliability = reliability.merge(
        TRAIT_META[["trait", "target", "trait_label", "unit", "family"]],
        on=["trait"],
        how="left",
        validate="one_to_one",
    )
    reliability.to_csv(output / "texture_reliability.csv", index=False)

    hyperparameters = pd.read_csv(analysis / "fold_hyperparameter_choices.csv")
    hyperparameters.to_csv(output / "quality_branch_selection.csv", index=False)
    texture_fruits = int(predictions.loc[predictions["family"].eq("Mechanical texture"), "sample_id"].nunique())
    quality_fruits = int(predictions.loc[predictions["family"].eq("Conventional quality"), "sample_id"].nunique())
    dataset_summary_path = output / "dataset_figure_summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    dataset_summary.update(
        {
            "analysis_fruit": texture_fruits,
            "quality_complete_case_fruit": quality_fruits,
            "texture_predictions": texture_fruits * 9,
            "quality_predictions": quality_fruits * 3,
            "total_out_of_fold_predictions": int(len(predictions)),
            "qc_version": "0.3.0-review-correction",
        }
    )
    dataset_summary_path.write_text(json.dumps(dataset_summary, indent=2), encoding="utf-8")
    manifest = {
        "version": "V25 external-review correction",
        "analysis_dir": str(analysis),
        "model_dependent_tables_replaced": True,
        "prediction_rows": int(len(predictions)),
        "traits": int(predictions["trait"].nunique()),
        "cultivars": int(predictions["cultivar_ascii"].nunique()),
        "model_labels": MODEL_DISPLAY,
        "corrected_crossbatch_layer": corrected_crossbatch.exists(),
        "crossbatch_analysis_dir": str(crossbatch) if corrected_crossbatch.exists() else None,
    }
    (output / "v25_figure_data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
