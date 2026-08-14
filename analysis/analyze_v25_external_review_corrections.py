from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from analyze_texture_qc_impact import ENDPOINTS, icc_a1
from analyze_v22_quality_results import cluster_bootstrap_contrast
from analyze_v24_hr_strengthening import analyze_fewshot, analyze_multiplicity
from train_texture_pls_loco import regression_metrics


MODEL_COLUMNS = {
    "global_pls": "y_global_pls",
    "cultivar_aware_pls": "y_domain_pls",
    "nested_rbf_svr": "y_domain_svr",
    "residual_cnn": "y_deep",
    "no_neural_b50": "y_b50",
    "cultivar_mean_null": "y_cultivar_mean_null",
    "plumspectra_corrected": "y_final",
    "plumspectra_fixed_gate_sensitivity": "y_final_fixed_gate",
}

TEXTURE_TRAITS = {"SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"}
LEGACY_TEXTURE_GATES = {
    "SRF": 0.75,
    "RD": 0.50,
    "PFD": 0.75,
    "MFF": 0.25,
    "F6": 0.75,
    "LS": 0.75,
    "LW": 0.75,
    "PRW": 0.25,
    "AF": 0.50,
}


def stable_seed(*values: object) -> int:
    raw = "|".join(map(str, values)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def read_baselines(texture_dir: Path, quality_dir: Path) -> pd.DataFrame:
    frames = []
    for directory in (texture_dir, quality_dir):
        path = directory / "predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["sample_id", "target"]).any():
        raise ValueError("Corrected baseline predictions contain duplicate sample/target rows")
    return frame


def internal_gate(metadata: dict[str, Any]) -> float:
    selected = [row for row in metadata.get("gate_scores", []) if float(row.get("selected", 0)) == 1.0]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one internally selected gate for {metadata.get('trait_abbreviation')} "
            f"fold {metadata.get('outer_fold')}; found {len(selected)}"
        )
    return float(selected[0]["gate"])


def read_ai(texture_dir: Path, quality_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    gate_rows: list[dict[str, Any]] = []
    for directory, is_texture in ((texture_dir, True), (quality_dir, False)):
        paths = sorted(directory.glob("*/fold_*/predictions.parquet"))
        expected = 45 if is_texture else 15
        if len(paths) != expected:
            raise RuntimeError(f"Expected {expected} AI prediction files under {directory}, found {len(paths)}")
        for path in paths:
            metadata_path = path.with_name("metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("crossfit_anchor_folds", 0)) != 4:
                raise RuntimeError(
                    f"Formal V25 AI must use four cross-fitted anchor folds: {metadata_path}"
                )
            if not bool(metadata.get("domain_aware_anchor_selection", False)):
                raise RuntimeError(
                    f"Formal V25 AI must select the PLS anchor under cultivar-aware inner CV: {metadata_path}"
                )
            if metadata.get("gate_selection_mode") != "training_internal_validation":
                raise RuntimeError(
                    f"Formal V25 AI must use its own training-internal residual gate: {metadata_path}"
                )
            if metadata.get("fixed_gate_requested") is not None:
                raise RuntimeError(f"Formal V25 AI cannot request a fixed legacy gate: {metadata_path}")
            if metadata.get("validation_residual_target_mode") != "observed":
                raise RuntimeError(
                    f"Formal V25 AI must score observed validation residuals: {metadata_path}"
                )
            pred = pd.read_parquet(path).copy()
            pred = pred.rename(columns={"repeat": "outer_fold", "y_pred": "y_deep_recorded"})
            trait = str(metadata["trait_abbreviation"])
            selected_internal = internal_gate(metadata)
            recorded_gate = float(metadata["selected_gate"])
            if is_texture:
                pred["y_deep"] = (
                    pred["y_pls_anchor"].to_numpy(float)
                    + selected_internal * pred["deep_residual"].to_numpy(float)
                )
                legacy_gate = LEGACY_TEXTURE_GATES[trait]
                pred["y_deep_fixed_gate"] = (
                    pred["y_pls_anchor"].to_numpy(float)
                    + legacy_gate * pred["deep_residual"].to_numpy(float)
                )
            else:
                legacy_gate = recorded_gate
                pred["y_deep"] = pred["y_deep_recorded"].to_numpy(float)
                pred["y_deep_fixed_gate"] = pred["y_deep_recorded"].to_numpy(float)
            pred["trait"] = trait
            frames.append(
                pred[
                    [
                        "sample_id",
                        "cultivar_ascii",
                        "target",
                        "trait",
                        "outer_fold",
                        "y_true",
                        "y_deep",
                        "y_deep_fixed_gate",
                        "y_pls_anchor",
                        "deep_residual",
                    ]
                ]
            )
            gate_rows.append(
                {
                    "target": metadata["target"],
                    "trait": trait,
                    "outer_fold": int(metadata["outer_fold"]),
                    "is_texture": is_texture,
                    "internal_validation_gate": selected_internal,
                    "recorded_production_gate": recorded_gate,
                    "legacy_fixed_gate": legacy_gate,
                    "recorded_gate_overrode_internal": not np.isclose(selected_internal, recorded_gate),
                    "legacy_gate_differs_from_internal": not np.isclose(selected_internal, legacy_gate),
                    "crossfit_anchor_folds": int(metadata.get("crossfit_anchor_folds", 0)),
                    "domain_aware_anchor_selection": bool(
                        metadata.get("domain_aware_anchor_selection", False)
                    ),
                    "gate_selection_mode": metadata.get("gate_selection_mode"),
                    "validation_residual_target_mode": metadata.get(
                        "validation_residual_target_mode"
                    ),
                    "selected_epoch": int(metadata["selected_epoch"]),
                }
            )
    ai = pd.concat(frames, ignore_index=True)
    if ai.duplicated(["sample_id", "target"]).any():
        raise ValueError("AI predictions contain duplicate sample/target rows")
    return ai, pd.DataFrame(gate_rows)


def read_hyperparameters(texture_dir: Path, quality_dir: Path, margin_pct: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for directory in (texture_dir, quality_dir):
        for path in sorted(directory.glob("*/fold_*/metadata.json")):
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("svr_search_profile") != "v25_staged":
                raise RuntimeError(f"Formal V25 baseline did not use staged SVR search: {path}")
            if bool(metadata.get("test_labels_used_for_selection", True)):
                raise RuntimeError(f"Baseline metadata does not certify train-only selection: {path}")
            pls = metadata["domain_pls_choice"]
            global_pls = metadata["global_pls_choice"]
            svr = metadata["domain_svr_choice"]
            pls_grid = [int(value) for value in metadata.get("pls_component_grid", [4, 8, 12, 16, 24])]
            if pls_grid != [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24]:
                raise RuntimeError(f"Formal V25 baseline has the wrong PLS component grid: {path}")
            search = metadata.get(
                "svr_search_space",
                {
                    "C": [3.0, 10.0, 30.0, 100.0],
                    "gamma_factor": [0.25, 0.5, 1.0, 2.0],
                    "epsilon_z": [0.03, 0.08],
                },
            )
            hard_epsilon = search.get("hard_limits", {}).get("epsilon_z", [None, None])
            if hard_epsilon and hard_epsilon[-1] is not None and float(hard_epsilon[-1]) >= 10.0:
                declared_epsilon = {float(value) for value in search.get("epsilon_z", [])}
                required_superset_points = {1.5, 2.4, 4.0, 8.0}
                if not required_superset_points.issubset(declared_epsilon):
                    raise RuntimeError(
                        "Wide-epsilon baseline is not a strict superset of the original V25 "
                        f"search ({sorted(required_superset_points - declared_epsilon)} missing): {path}"
                    )
            inner_svr_path = path.with_name("inner_svr_cv.csv")
            if inner_svr_path.exists():
                inner_svr = pd.read_csv(inner_svr_path)
                inner_svr = inner_svr.loc[
                    inner_svr["stage"].ne("preprocessing_screen")
                    & inner_svr["preprocessing"].eq(svr["preprocessing"])
                ]
                c_grid = sorted(inner_svr["C"].astype(float).unique().tolist())
                gamma_grid = sorted(inner_svr["gamma_factor"].astype(float).unique().tolist())
                epsilon_grid = sorted(inner_svr["epsilon_z"].astype(float).unique().tolist())
            else:
                c_grid = [float(value) for value in search.get("C", [])]
                gamma_grid = [float(value) for value in search.get("gamma_factor", [])]
                epsilon_grid = [float(value) for value in search.get("epsilon_z", [])]
            degradation = 100.0 * (float(svr["domain_cv_rmse"]) / float(pls["domain_cv_rmse"]) - 1.0)
            global_pls_components = int(global_pls["n_components"])
            domain_pls_components = int(pls["n_components"])
            pls_grid_min = min(pls_grid)
            pls_grid_max = max(pls_grid)
            rows.append(
                {
                    "target": metadata["target"],
                    "trait": metadata["trait"],
                    "outer_fold": int(metadata["outer_fold"]),
                    "global_pls_preprocessing": global_pls["preprocessing"],
                    "global_pls_components": global_pls_components,
                    "global_pls_cv_rmse": float(global_pls["global_cv_rmse"]),
                    "global_pls_component_lower_boundary": global_pls_components == pls_grid_min,
                    "global_pls_component_upper_boundary": global_pls_components == pls_grid_max,
                    "global_pls_component_boundary": global_pls_components in {pls_grid_min, pls_grid_max},
                    "domain_pls_preprocessing": pls["preprocessing"],
                    "domain_pls_components": domain_pls_components,
                    "domain_pls_cv_rmse": float(pls["domain_cv_rmse"]),
                    # One component is the irreducible lower bound of PLSR and is
                    # therefore not evidence of a truncated hyperparameter search.
                    # The upper-bound count is the relevant under-tuning audit.
                    "domain_pls_component_lower_boundary": domain_pls_components == pls_grid_min,
                    "domain_pls_component_upper_boundary": domain_pls_components == pls_grid_max,
                    "domain_pls_component_boundary": domain_pls_components in {pls_grid_min, pls_grid_max},
                    "svr_preprocessing": svr["preprocessing"],
                    "svr_C": float(svr["C"]),
                    "svr_gamma_factor": float(svr["gamma_factor"]),
                    "svr_epsilon_z": float(svr["epsilon_z"]),
                    "svr_cv_rmse": float(svr["domain_cv_rmse"]),
                    "svr_C_boundary": bool(c_grid) and float(svr["C"]) in {min(c_grid), max(c_grid)},
                    "svr_gamma_boundary": bool(gamma_grid) and float(svr["gamma_factor"]) in {min(gamma_grid), max(gamma_grid)},
                    "svr_epsilon_boundary": bool(epsilon_grid) and float(svr["epsilon_z"]) in {min(epsilon_grid), max(epsilon_grid)},
                    "domain_svr_degradation_vs_pls_pct": degradation,
                    "kernel_weight": 0.5 if degradation <= margin_pct else 0.0,
                    "selected_variant": "deep_kernel" if degradation <= margin_pct else "deep_only",
                    "selection_uses_outer_test_labels": False,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 60:
        raise RuntimeError(f"Expected 60 hyperparameter rows, found {len(result)}")
    return result


def add_cultivar_mean_null(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("target", observed=True):
        group = group.copy()
        cultivar_sum = group.groupby("cultivar_ascii", observed=True)["y_true"].transform("sum")
        cultivar_n = group.groupby("cultivar_ascii", observed=True)["y_true"].transform("size")
        fold_sum = group.groupby(["cultivar_ascii", "outer_fold"], observed=True)["y_true"].transform("sum")
        fold_n = group.groupby(["cultivar_ascii", "outer_fold"], observed=True)["y_true"].transform("size")
        denominator = cultivar_n - fold_n
        if (denominator <= 0).any():
            raise ValueError("Cultivar-mean null has a cultivar represented in only one outer fold")
        group["y_cultivar_mean_null"] = (cultivar_sum - fold_sum) / denominator
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def centered_metrics(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    truth = frame["y_true"] - frame.groupby("cultivar_ascii", observed=True)["y_true"].transform("mean")
    prediction = frame[column] - frame.groupby("cultivar_ascii", observed=True)[column].transform("mean")
    return regression_metrics(truth.to_numpy(float), prediction.to_numpy(float))


def model_metrics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled: list[dict[str, Any]] = []
    centered: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    cultivar_rows: list[dict[str, Any]] = []
    for (target, trait), group in frame.groupby(["target", "trait"], observed=True):
        for model, column in MODEL_COLUMNS.items():
            pooled.append({"target": target, "trait": trait, "model": model, **regression_metrics(group["y_true"], group[column])})
            centered.append({"target": target, "trait": trait, "model": model, **centered_metrics(group, column)})
            for outer_fold, fold in group.groupby("outer_fold", observed=True):
                fold_rows.append(
                    {
                        "target": target,
                        "trait": trait,
                        "model": model,
                        "outer_fold": int(outer_fold),
                        **regression_metrics(fold["y_true"], fold[column]),
                    }
                )
            for cultivar, cultivar_group in group.groupby("cultivar_ascii", observed=True):
                cultivar_rows.append(
                    {
                        "target": target,
                        "trait": trait,
                        "model": model,
                        "cultivar_ascii": str(cultivar),
                        **regression_metrics(cultivar_group["y_true"], cultivar_group[column]),
                    }
                )
    return (
        pd.DataFrame(pooled),
        pd.DataFrame(centered),
        pd.DataFrame(fold_rows),
        pd.DataFrame(cultivar_rows),
    )


def extended_comparisons(frame: pd.DataFrame, iterations: int) -> pd.DataFrame:
    comparisons = [
        ("plumspectra_corrected", "cultivar_mean_null"),
        ("plumspectra_corrected", "no_neural_b50"),
        ("residual_cnn", "nested_rbf_svr"),
        ("plumspectra_corrected", "residual_cnn"),
        ("plumspectra_fixed_gate_sensitivity", "plumspectra_corrected"),
    ]
    rows = []
    for (target, trait), group in frame.groupby(["target", "trait"], observed=True):
        for candidate, baseline in comparisons:
            rows.append(
                {
                    "target": target,
                    "trait": trait,
                    "candidate": candidate,
                    "baseline": baseline,
                    **cluster_bootstrap_contrast(
                        group,
                        MODEL_COLUMNS[candidate],
                        MODEL_COLUMNS[baseline],
                        iterations,
                        stable_seed("v25_external_review", target, candidate, baseline),
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["claim_status"] = np.where(
        result["relative_improvement_ci_low"] > 0,
        "supported_outperformance",
        np.where(
            result["relative_improvement_ci_high"] < 0,
            "supported_inferiority",
            "statistical_parity",
        ),
    )
    return result


def analyze_equal_information_pls2(
    frame: pd.DataFrame, predictions_path: Path, iterations: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pls2 = pd.read_parquet(predictions_path)
    required = {
        "sample_id", "cultivar_ascii", "target", "trait", "outer_fold", "y_true", "y_domain_pls2"
    }
    if missing := required - set(pls2.columns):
        raise ValueError(f"PLS2 predictions lack columns: {sorted(missing)}")
    common = frame.merge(
        pls2[list(required)],
        on=["sample_id", "cultivar_ascii", "target", "trait", "outer_fold"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_pls2"),
    )
    expected_rows = int(pls2["sample_id"].nunique()) * 12
    if len(common) != expected_rows or common["trait"].nunique() != 12:
        raise RuntimeError(
            f"Expected a complete 12-trait PLS2 comparison ({expected_rows:,} rows), "
            f"found {len(common):,} rows and {common['trait'].nunique()} traits"
        )
    if not np.allclose(common["y_true"], common["y_true_pls2"], rtol=0, atol=1e-8):
        raise ValueError("PLS2 and V25 truth values disagree")
    rows = []
    for (target, trait), group in common.groupby(["target", "trait"], observed=True):
        rows.append(
            {
                "target": target,
                "trait": trait,
                "candidate": "plumspectra_corrected",
                "baseline": "cultivar_aware_pls2_equal_information",
                **cluster_bootstrap_contrast(
                    group,
                    "y_final",
                    "y_domain_pls2",
                    iterations,
                    stable_seed("v25_pls2", target),
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["claim_status"] = np.where(
        result["relative_improvement_ci_low"] > 0,
        "supported_outperformance",
        np.where(result["relative_improvement_ci_high"] < 0, "supported_inferiority", "statistical_parity"),
    )
    return result, {
        "common_fruits": int(common["sample_id"].nunique()),
        "prediction_rows": int(len(common)),
        "supported_outperformance": int(result["claim_status"].eq("supported_outperformance").sum()),
        "supported_inferiority": int(result["claim_status"].eq("supported_inferiority").sum()),
        "statistical_parity": int(result["claim_status"].eq("statistical_parity").sum()),
    }


def endpoint_structure(frame: pd.DataFrame, output: Path) -> dict[str, Any]:
    texture = frame.loc[frame["trait"].isin(TEXTURE_TRAITS), ["sample_id", "trait", "y_true"]]
    matrix = texture.pivot(index="sample_id", columns="trait", values="y_true").dropna()
    order = [trait for trait in ("SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF") if trait in matrix]
    matrix = matrix[order]
    correlation = matrix.corr()
    correlation.to_csv(output / "texture_endpoint_correlation.csv")
    z = (matrix - matrix.mean()) / matrix.std(ddof=1)
    pca = PCA().fit(z)
    pca_rows = pd.DataFrame(
        {
            "component": np.arange(1, len(order) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    pca_rows.to_csv(output / "texture_endpoint_pca_variance.csv", index=False)
    loadings = pd.DataFrame(pca.components_.T, index=order, columns=[f"PC{i}" for i in range(1, len(order) + 1)])
    loadings.to_csv(output / "texture_endpoint_pca_loadings.csv")
    eigenvalues = pca.explained_variance_
    participation_ratio = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    return {
        "complete_fruits": int(len(matrix)),
        "pc1_variance_pct": float(100 * pca.explained_variance_ratio_[0]),
        "first_3_pc_variance_pct": float(100 * pca.explained_variance_ratio_[:3].sum()),
        "participation_ratio": participation_ratio,
        "pairs_abs_r_above_0_90": int(
            np.triu(np.abs(correlation.to_numpy()) > 0.90, k=1).sum()
        ),
    }


def qc_batch_reliability(
    frame: pd.DataFrame,
    ledger_path: Path,
    texture_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    ledger = pd.read_parquet(ledger_path)
    manifest = pd.read_csv(texture_manifest_path, dtype={"sample_id": str})
    cohort = ledger.set_index("sample_id").loc[manifest["sample_id"]].reset_index()
    reliability_rows = []
    endpoint_to_trait = dict(zip(ENDPOINTS, ("SRF", "RD", "PFD", "F6", "MFF", "LS", "LW", "PRW", "AF")))
    for endpoint in ENDPOINTS:
        replicate_pair = (
            cohort[[f"rep01_{endpoint}", f"rep02_{endpoint}"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
        )
        reliability_rows.append(
            {
                "endpoint": endpoint,
                "trait": endpoint_to_trait[endpoint],
                "n": int(len(replicate_pair)),
                "icc_a1": icc_a1(cohort[f"rep01_{endpoint}"], cohort[f"rep02_{endpoint}"]),
                "pearson_r": float(replicate_pair.corr(method="pearson").iloc[0, 1]),
                "median_replicate_cv": float(
                    pd.to_numeric(cohort[f"{endpoint}_cv"], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .median()
                ),
            }
        )
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(output / "texture_reliability_modeling_cohort.csv", index=False)

    integrated = frame.merge(ledger[["sample_id", "batch_id"]], on="sample_id", how="left", validate="many_to_one")
    batch_counts = (
        integrated[["sample_id", "cultivar_ascii", "batch_id"]]
        .drop_duplicates()
        .groupby("cultivar_ascii", observed=True)
        .agg(fruits=("sample_id", "size"), batches=("batch_id", "nunique"))
        .reset_index()
    )
    batch_counts.to_csv(output / "cultivar_batch_counts.csv", index=False)
    batch_rows = []
    for cultivar in ("Konglongdan", "Weiwang"):
        cultivar_frame = integrated.loc[
            integrated["cultivar_ascii"].eq(cultivar) & integrated["trait"].isin(TEXTURE_TRAITS)
        ]
        for trait, group in cultivar_frame.groupby("trait", observed=True):
            grand = float(group["y_true"].mean())
            total = float(np.square(group["y_true"] - grand).sum())
            between = float(
                sum(
                    len(part) * (float(part["y_true"].mean()) - grand) ** 2
                    for _, part in group.groupby("batch_id", observed=True)
                )
            )
            batch_rows.append(
                {
                    "cultivar_ascii": cultivar,
                    "trait": trait,
                    "n": int(len(group)),
                    "batches": int(group["batch_id"].nunique()),
                    "batch_eta_squared": between / total if total > 0 else np.nan,
                    "batch_mean_min": float(group.groupby("batch_id", observed=True)["y_true"].mean().min()),
                    "batch_mean_max": float(group.groupby("batch_id", observed=True)["y_true"].mean().max()),
                }
            )
    pd.DataFrame(batch_rows).to_csv(output / "within_cultivar_batch_effects.csv", index=False)

    spectral_signal_columns = [
        column
        for column in (
            "nir_qc_z_nir_c_absorbance_min",
            "nir_qc_z_nir_c_absorbance_max",
            "nir_qc_z_nir_c_absorbance_mean",
            "nir_qc_z_nir_c_reference_signal_min",
            "nir_qc_z_nir_c_sample_signal_min",
        )
        if column in ledger.columns
    ]
    session_columns = [
        column
        for column in (
            "session_qc_z_nir_c_system_temp_c",
            "session_qc_z_nir_c_detector_temp_c",
            "session_qc_z_nir_c_humidity_pct",
            "session_qc_z_nir_c_lamp_pd",
            # Backward-compatible V24 names. V25 v0.3 deliberately moved
            # these acquisition-session variables out of the spectral domain.
            "nir_qc_z_nir_c_system_temp_c",
            "nir_qc_z_nir_c_detector_temp_c",
            "nir_qc_z_nir_c_humidity_pct",
            "nir_qc_z_nir_c_lamp_pd",
        )
        if column in ledger.columns
    ]
    cultivar_611 = ledger.loc[ledger["cultivar_ascii"].astype(str).eq("6.11")].copy()
    stored = cultivar_611["spectral_severe"].fillna(False).astype(bool)
    humidity_column = next(
        (
            column
            for column in (
                "session_qc_z_nir_c_humidity_pct",
                "nir_qc_z_nir_c_humidity_pct",
            )
            if column in cultivar_611.columns
        ),
        None,
    )
    humidity = (
        pd.to_numeric(cultivar_611[humidity_column], errors="coerce").ge(8)
        if humidity_column is not None
        else pd.Series(False, index=cultivar_611.index)
    )
    signal = cultivar_611[spectral_signal_columns].apply(pd.to_numeric, errors="coerce").max(axis=1).ge(8)
    session = cultivar_611[session_columns].apply(pd.to_numeric, errors="coerce").max(axis=1).ge(8)
    pca_flag = pd.to_numeric(cultivar_611["spectral_pca_orthogonal_robust_z"], errors="coerce").ge(8)
    invalid = ~cultivar_611["nir_c_valid"].fillna(False).astype(bool)
    humidity_audit = pd.DataFrame(
        [
            {
                "cultivar_ascii": "6.11",
                "fruits": int(len(cultivar_611)),
                "stored_spectral_severe": int(stored.sum()),
                "humidity_trigger": int(humidity.sum()),
                "spectral_signal_trigger": int(signal.sum()),
                "session_condition_trigger": int(session.sum()),
                "pca_orthogonal_trigger": int(pca_flag.sum()),
                "invalid_nir_trigger": int(invalid.sum()),
                "humidity_only_trigger": int((humidity & ~signal & ~pca_flag & ~invalid).sum()),
                "spectral_quality_trigger": int((signal | pca_flag | invalid).sum()),
                "spectral_quality_fraction": float((signal | pca_flag | invalid).mean()),
            }
        ]
    )
    humidity_audit.to_csv(output / "cultivar_611_spectral_domain_decomposition.csv", index=False)
    return {
        "icc_range_modeling_cohort": [float(reliability["icc_a1"].min()), float(reliability["icc_a1"].max())],
        "median_cv_range_modeling_cohort": [
            float(reliability["median_replicate_cv"].min()),
            float(reliability["median_replicate_cv"].max()),
        ],
        "single_batch_cultivars": int((batch_counts["batches"] == 1).sum()),
        "multi_batch_cultivars": int((batch_counts["batches"] > 1).sum()),
        "cultivar_611_spectral_quality_fraction": float(humidity_audit.iloc[0]["spectral_quality_fraction"]),
    }


def cohort_sensitivity(
    frame: pd.DataFrame,
    ledger_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Quantify a stricter retained-cultivar sensitivity without post-outcome deletion.

    The reviewer-proposed retention of 6.11 changes the frozen sample universe
    and cannot honestly be reconstructed from 15-cultivar OOF predictions. We
    therefore report its eligible count and repeatability evidence explicitly,
    while limiting model-performance recalculation to the valid subset
    sensitivity that removes Weijin and Cuihongli from the same OOF predictions.
    """
    scenario_frames = {
        "primary_15_cultivars": frame,
        "stricter_13_cultivars_excluding_Weijin_Cuihongli": frame.loc[
            ~frame["cultivar_ascii"].isin(["Weijin", "Cuihongli"])
        ],
    }
    rows: list[dict[str, Any]] = []
    for scenario, scenario_frame in scenario_frames.items():
        for (target, trait), group in scenario_frame.groupby(["target", "trait"], observed=True):
            for model, column in MODEL_COLUMNS.items():
                rows.append(
                    {
                        "scenario": scenario,
                        "target": target,
                        "trait": trait,
                        "model": model,
                        "fruits": int(group["sample_id"].nunique()),
                        "cultivars": int(group["cultivar_ascii"].nunique()),
                        **regression_metrics(group["y_true"], group[column]),
                    }
                )
    pd.DataFrame(rows).to_csv(output / "cultivar_exclusion_performance_sensitivity.csv", index=False)

    ledger = pd.read_parquet(ledger_path)
    target_columns = [
        "skin_break_force_g_mean",
        "skin_break_displacement_raw_mean",
        "skin_break_drop_g_mean",
        "flesh_force_mean_g_mean",
        "force_at_6_rawpos_g_mean",
        "loading_stiffness_g_per_rawpos_mean",
        "loading_work_g_rawpos_mean",
        "post_break_work_g_rawpos_mean",
        "adhesive_force_g_mean",
    ]
    primary_complete = (
        ledger["qc_primary_include"].astype(bool)
        & ledger[target_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    )
    cultivar_611 = ledger.loc[ledger["cultivar_ascii"].astype(str).eq("6.11")].copy()
    endpoint_rows = []
    for endpoint in ENDPOINTS:
        pair = cultivar_611[[f"rep01_{endpoint}", f"rep02_{endpoint}"]].apply(
            pd.to_numeric, errors="coerce"
        )
        endpoint_rows.append(
            {
                "cultivar_ascii": "6.11",
                "endpoint": endpoint,
                "icc_a1": icc_a1(pair.iloc[:, 0], pair.iloc[:, 1]),
                "median_replicate_cv": float(
                    pd.to_numeric(cultivar_611[f"{endpoint}_cv"], errors="coerce").median()
                ),
            }
        )
    pd.DataFrame(endpoint_rows).to_csv(output / "cultivar_611_repeatability_sensitivity.csv", index=False)
    return {
        "primary_complete_texture_fruits_before_whole_cultivar_exclusion": int(primary_complete.sum()),
        "cultivar_611_primary_complete_texture_fruits": int(
            (primary_complete & ledger["cultivar_ascii"].astype(str).eq("6.11")).sum()
        ),
        "retaining_611_requires_new_split_and_full_refit": True,
        "stricter_subset_fruits": int(
            frame.loc[
                frame["trait"].isin(TEXTURE_TRAITS)
                & ~frame["cultivar_ascii"].isin(["Weijin", "Cuihongli"]),
                "sample_id",
            ].nunique()
        ),
        "stricter_subset_cultivars": 13,
    }


def heldbatch_claim_audit(output: Path) -> dict[str, Any]:
    summary = pd.read_csv(output / "fewshot_summary.csv")

    def select(
        model: str, shots: int, adapter: str, aggregation: str, prefix: str
    ) -> pd.DataFrame:
        columns = [
            "trait",
            "rmse_mean",
            "r2_mean",
            "r2_ci025",
            "r2_ci975",
            "rmse_gain_pct_mean",
        ]
        part = summary.loc[
            summary["model"].eq(model)
            & summary["shots"].eq(shots)
            & summary["adapter"].eq(adapter)
            & summary["aggregation"].eq(aggregation),
            columns,
        ].copy()
        return part.rename(columns={column: f"{prefix}_{column}" for column in columns if column != "trait"})

    audit = select("Deep-kernel ensemble", 0, "none", "batch_macro", "zero_ai_macro")
    for part in (
        select("Global PLSR", 0, "none", "batch_macro", "zero_global_macro"),
        select("Deep-kernel ensemble", 0, "none", "pooled", "zero_ai_pooled"),
        select("Deep-kernel ensemble", 40, "intercept", "pooled", "shot40_ai_pooled"),
        select("Deep-kernel ensemble", 40, "intercept", "batch_macro", "shot40_ai_macro"),
        select("Batch-mean null (no spectra)", 40, "batch_mean_null", "pooled", "shot40_null_pooled"),
        select("Batch-mean null (no spectra)", 40, "batch_mean_null", "batch_macro", "shot40_null_macro"),
    ):
        audit = audit.merge(part, on="trait", how="inner", validate="one_to_one")
    audit["zero_ai_rmse_gain_vs_global_macro_pct"] = 100.0 * (
        1.0
        - audit["zero_ai_macro_rmse_mean"] / audit["zero_global_macro_rmse_mean"]
    )
    audit["shot40_ai_rmse_gain_vs_null_pooled_pct"] = 100.0 * (
        1.0
        - audit["shot40_ai_pooled_rmse_mean"] / audit["shot40_null_pooled_rmse_mean"]
    )
    audit["shot40_ai_rmse_gain_vs_null_macro_pct"] = 100.0 * (
        1.0
        - audit["shot40_ai_macro_rmse_mean"] / audit["shot40_null_macro_rmse_mean"]
    )
    audit.to_csv(output / "heldbatch_claim_audit.csv", index=False)
    return {
        "traits": int(len(audit)),
        "zero_shot_ai_worse_than_global_batch_macro": int(
            (audit["zero_ai_rmse_gain_vs_global_macro_pct"] < 0).sum()
        ),
        "shot40_ai_worse_than_batch_mean_null_pooled": int(
            (audit["shot40_ai_rmse_gain_vs_null_pooled_pct"] < 0).sum()
        ),
        "shot40_ai_worse_than_batch_mean_null_batch_macro": int(
            (audit["shot40_ai_rmse_gain_vs_null_macro_pct"] < 0).sum()
        ),
        "shot40_ai_median_pooled_r2": float(audit["shot40_ai_pooled_r2_mean"].median()),
        "shot40_null_median_pooled_r2": float(audit["shot40_null_pooled_r2_mean"].median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texture-baseline-dir", type=Path, required=True)
    parser.add_argument("--quality-baseline-dir", type=Path, required=True)
    parser.add_argument("--texture-ai-dir", type=Path, required=True)
    parser.add_argument("--quality-ai-dir", type=Path, required=True)
    parser.add_argument("--texture-manifest", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--crossbatch-predictions", type=Path)
    parser.add_argument("--multitrait-pls2-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-eligibility-margin-pct", type=float, default=5.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1_000_000)
    parser.add_argument("--extended-bootstrap-iterations", type=int, default=200_000)
    parser.add_argument("--fewshot-repeats", type=int, default=500)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = read_baselines(args.texture_baseline_dir.resolve(), args.quality_baseline_dir.resolve())
    ai, gate_audit = read_ai(args.texture_ai_dir.resolve(), args.quality_ai_dir.resolve())
    hyperparameters = read_hyperparameters(
        args.texture_baseline_dir.resolve(),
        args.quality_baseline_dir.resolve(),
        args.kernel_eligibility_margin_pct,
    )
    merged = baseline.merge(
        ai,
        on=["sample_id", "cultivar_ascii", "target", "trait", "outer_fold"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ai"),
    )
    trait_counts = merged.groupby("trait", observed=True)["sample_id"].nunique()
    quality_traits = sorted(set(trait_counts.index) - TEXTURE_TRAITS)
    if (
        merged["trait"].nunique() != 12
        or trait_counts.loc[list(TEXTURE_TRAITS)].nunique() != 1
        or trait_counts.loc[quality_traits].nunique() != 1
    ):
        raise RuntimeError(
            "Integrated predictions do not contain 12 complete trait-specific outer-fold panels: "
            f"{trait_counts.to_dict()}"
        )
    if not np.allclose(merged["y_true"], merged["y_true_ai"], rtol=0, atol=1e-8):
        raise ValueError("Baseline and AI truth columns differ")
    merged = merged.drop(columns="y_true_ai")
    merged = merged.merge(
        hyperparameters[["target", "outer_fold", "kernel_weight", "selected_variant"]],
        on=["target", "outer_fold"],
        how="left",
        validate="many_to_one",
    )
    merged["y_b50"] = 0.5 * merged["y_domain_pls"] + 0.5 * merged["y_domain_svr"]
    merged["y_final"] = (
        (1.0 - merged["kernel_weight"]) * merged["y_deep"]
        + merged["kernel_weight"] * merged["y_domain_svr"]
    )
    merged["y_final_fixed_gate"] = (
        (1.0 - merged["kernel_weight"]) * merged["y_deep_fixed_gate"]
        + merged["kernel_weight"] * merged["y_domain_svr"]
    )
    merged = add_cultivar_mean_null(merged)
    merged.to_parquet(output / "v25_integrated_predictions.parquet", index=False, compression="zstd")
    merged.to_csv(output / "v25_integrated_predictions.csv", index=False)
    gate_audit.to_csv(output / "residual_gate_audit.csv", index=False)
    hyperparameters.to_csv(output / "fold_hyperparameter_choices.csv", index=False)

    pooled, centered, fold_metrics, cultivar_metrics = model_metrics(merged)
    pooled.to_csv(output / "pooled_metrics.csv", index=False)
    centered.to_csv(output / "within_cultivar_centered_metrics.csv", index=False)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    cultivar_metrics.to_csv(output / "cultivar_metrics.csv", index=False)
    final_centered = centered.loc[centered["model"].eq("plumspectra_corrected")].copy()
    final_cultivar = cultivar_metrics.loc[
        cultivar_metrics["model"].eq("plumspectra_corrected")
    ].copy()
    within_signal = {
        "negative_centered_r2_traits": sorted(
            final_centered.loc[final_centered["r2"] < 0, "trait"].astype(str).tolist()
        ),
        "negative_centered_r2_trait_count": int((final_centered["r2"] < 0).sum()),
        "cultivar_trait_cells": int(len(final_cultivar)),
        "negative_cultivar_pearson_cells": int((final_cultivar["pearson_r"] < 0).sum()),
        "negative_cultivar_r2_cells": int((final_cultivar["r2"] < 0).sum()),
    }
    (output / "within_cultivar_signal_summary.json").write_text(
        json.dumps(within_signal, indent=2), encoding="utf-8"
    )
    core = (
        pooled.pivot(index=["target", "trait"], columns="model", values="r2")
        .join(
            centered.loc[centered["model"].eq("plumspectra_corrected")]
            .set_index(["target", "trait"])[["r2"]]
            .rename(columns={"r2": "plumspectra_centered_r2"})
        )
        .reset_index()
    )
    core.to_csv(output / "pooled_null_centered_r2.csv", index=False)

    extended = extended_comparisons(merged, args.extended_bootstrap_iterations)
    extended.to_csv(output / "extended_cluster_comparisons.csv", index=False)
    multiplicity = analyze_multiplicity(
        output / "v25_integrated_predictions.csv",
        output,
        args.bootstrap_iterations,
        20260810,
    )
    pls2_summary: dict[str, Any] | None = None
    if args.multitrait_pls2_predictions is not None:
        pls2_comparison, pls2_summary = analyze_equal_information_pls2(
            merged,
            args.multitrait_pls2_predictions.resolve(),
            args.extended_bootstrap_iterations,
        )
        pls2_comparison.to_csv(output / "equal_information_pls2_comparison.csv", index=False)

    endpoint_summary = endpoint_structure(merged, output)
    qc_summary = qc_batch_reliability(
        merged,
        args.qc_ledger.resolve(),
        args.texture_manifest.resolve(),
        output,
    )
    cohort_sensitivity_summary = cohort_sensitivity(merged, args.qc_ledger.resolve(), output)
    fewshot_summary: dict[str, Any] | None = None
    heldbatch_claim_summary: dict[str, Any] | None = None
    if args.crossbatch_predictions is not None:
        fewshot_summary = analyze_fewshot(
            args.crossbatch_predictions.resolve(),
            output,
            [0, 5, 10, 20, 40, 80],
            args.fewshot_repeats,
            20.0,
        )
        heldbatch_claim_summary = heldbatch_claim_audit(output)

    summary = {
        "protocol": "V25 external-review correction using corrected nested baselines and fold-internal residual gates",
        "prediction_rows": int(len(merged)),
        "fruits_texture": int(merged.loc[merged["trait"].isin(TEXTURE_TRAITS), "sample_id"].nunique()),
        "fruits_conventional": int(merged.loc[~merged["trait"].isin(TEXTURE_TRAITS), "sample_id"].nunique()),
        "traits": int(merged["trait"].nunique()),
        "cultivars": int(merged["cultivar_ascii"].nunique()),
        "texture_recorded_gate_overrides_internal_count": int(
            gate_audit.loc[gate_audit["is_texture"], "recorded_gate_overrode_internal"].sum()
        ),
        "legacy_fixed_gate_differs_from_internal_count": int(
            gate_audit.loc[gate_audit["is_texture"], "legacy_gate_differs_from_internal"].sum()
        ),
        "texture_crossfit_anchor_folds": sorted(
            gate_audit.loc[gate_audit["is_texture"], "crossfit_anchor_folds"].unique().tolist()
        ),
        "quality_crossfit_anchor_folds": sorted(
            gate_audit.loc[~gate_audit["is_texture"], "crossfit_anchor_folds"].unique().tolist()
        ),
        "domain_pls_component_lower_boundary_count": int(
            hyperparameters["domain_pls_component_lower_boundary"].sum()
        ),
        "domain_pls_component_upper_boundary_count": int(
            hyperparameters["domain_pls_component_upper_boundary"].sum()
        ),
        "domain_pls_component_boundary_count_legacy_combined": int(
            hyperparameters["domain_pls_component_boundary"].sum()
        ),
        "svr_C_boundary_count": int(hyperparameters["svr_C_boundary"].sum()),
        "svr_gamma_boundary_count": int(hyperparameters["svr_gamma_boundary"].sum()),
        "svr_epsilon_boundary_count": int(hyperparameters["svr_epsilon_boundary"].sum()),
        "multiplicity": multiplicity,
        "endpoint_structure": endpoint_summary,
        "qc_batch_reliability": qc_summary,
        "cohort_sensitivity": cohort_sensitivity_summary,
        "fewshot": fewshot_summary,
        "heldbatch_claim_audit": heldbatch_claim_summary,
        "equal_information_pls2": pls2_summary,
        "within_cultivar_signal": within_signal,
        "test_labels_used_for_selection": False,
        "claim_boundary": "same-session interpolation among 15 registered cultivars; external batch and unseen-cultivar transfer are separate negative/sensitivity analyses",
    }
    (output / "v25_correction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
