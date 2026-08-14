from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

import train_plumrac_loco as v2
import train_plumrac_v4_phy as v4
import train_plumrac_v5_auxpretrain as v5
from train_texture_pls_random import COMPONENTS as PLS_COMPONENTS, select_hyperparameters
from v2_registry import abbreviated_trait, add_cultivar_code


QUALITY_TARGETS = ["fruit_weight_g", "soluble_solids_pct", "ph"]
SUPPORTED_TARGETS = [*v2.DEFAULT_TARGETS, *QUALITY_TARGETS]


def prepare_anchor_targets(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: list[np.ndarray],
    preprocessing: str,
    n_components: int,
    cultivar_calibration: bool,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray, float, float, dict[str, float]]:
    target_mean = float(np.mean(y[fit_indices]))
    target_sd = max(float(np.std(y[fit_indices], ddof=1)), 1e-6)
    anchor_model = v2.fit_anchor(arrays, y, fit_indices, preprocessing, n_components)
    fit_prediction = anchor_model.predict(arrays[preprocessing][fit_indices]).ravel()
    cultivar_offsets: dict[str, float] = {}
    if cultivar_calibration:
        fit_residual = y[fit_indices] - fit_prediction
        for cultivar in np.unique(groups[fit_indices]):
            mask = groups[fit_indices] == cultivar
            cultivar_offsets[str(cultivar)] = float(np.mean(fit_residual[mask]))
    anchor_prediction = np.full(len(y), np.nan, dtype=np.float32)
    for indices in predict_indices:
        prediction = anchor_model.predict(arrays[preprocessing][indices]).ravel()
        if cultivar_calibration:
            prediction = prediction + np.asarray(
                [cultivar_offsets[str(cultivar)] for cultivar in groups[indices]], dtype=float
            )
        anchor_prediction[indices] = prediction
    residual_target = np.zeros(len(y), dtype=np.float32)
    residual_target[fit_indices] = (
        (y[fit_indices] - anchor_prediction[fit_indices]) / target_sd
    ).astype(np.float32)
    anchor_standardized = np.zeros(len(y), dtype=np.float32)
    valid = np.isfinite(anchor_prediction)
    anchor_standardized[valid] = (
        (anchor_prediction[valid] - target_mean) / target_sd
    ).astype(np.float32)
    return (
        anchor_model,
        anchor_prediction,
        residual_target,
        anchor_standardized,
        target_mean,
        target_sd,
        cultivar_offsets,
    )


def prepare_crossfit_anchor_targets(
    arrays: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: list[np.ndarray],
    preprocessing: str,
    n_components: int,
    cultivar_calibration: bool,
    n_splits: int,
    seed: int,
) -> tuple[
    object,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    dict[str, float],
    dict[str, object],
]:
    """Build honest OOF neural targets and a full-training deployment anchor.

    Neural training fruits receive PLS predictions from models that did not see
    those fruits. Validation/test fruits receive predictions from the PLS model
    fitted to the complete corresponding training partition. Cultivar offsets
    are likewise estimated only from the training side of each split.
    """
    if n_splits < 2:
        raise ValueError("cross-fitted anchors require at least two folds")
    target_mean = float(np.mean(y[fit_indices]))
    target_sd = max(float(np.std(y[fit_indices], ddof=1)), 1e-6)

    full_model = v2.fit_anchor(arrays, y, fit_indices, preprocessing, n_components)
    full_fit_prediction = full_model.predict(arrays[preprocessing][fit_indices]).ravel()
    full_offsets: dict[str, float] = {}
    if cultivar_calibration:
        full_residual = y[fit_indices] - full_fit_prediction
        for cultivar in np.unique(groups[fit_indices]):
            mask = groups[fit_indices] == cultivar
            full_offsets[str(cultivar)] = float(np.mean(full_residual[mask]))

    anchor_prediction = np.full(len(y), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (relative_train, relative_valid) in enumerate(
        splitter.split(fit_indices, groups[fit_indices]), start=1
    ):
        fold_train = fit_indices[relative_train]
        fold_valid = fit_indices[relative_valid]
        fold_model = v2.fit_anchor(arrays, y, fold_train, preprocessing, n_components)
        fold_train_prediction = fold_model.predict(arrays[preprocessing][fold_train]).ravel()
        fold_offsets: dict[str, float] = {}
        if cultivar_calibration:
            fold_train_residual = y[fold_train] - fold_train_prediction
            for cultivar in np.unique(groups[fold_train]):
                mask = groups[fold_train] == cultivar
                fold_offsets[str(cultivar)] = float(np.mean(fold_train_residual[mask]))
        fold_prediction = fold_model.predict(arrays[preprocessing][fold_valid]).ravel()
        if cultivar_calibration:
            fold_prediction = fold_prediction + np.asarray(
                [fold_offsets[str(cultivar)] for cultivar in groups[fold_valid]], dtype=float
            )
        anchor_prediction[fold_valid] = fold_prediction.astype(np.float32)
        fold_rows.append(
            {
                "fold": fold,
                "train_samples": int(len(fold_train)),
                "validation_samples": int(len(fold_valid)),
                "rmse": float(np.sqrt(np.mean((fold_prediction - y[fold_valid]) ** 2))),
            }
        )

    fit_mask = np.zeros(len(y), dtype=bool)
    fit_mask[fit_indices] = True
    for indices in predict_indices:
        deployment_indices = np.asarray(indices)[~fit_mask[np.asarray(indices)]]
        if len(deployment_indices) == 0:
            continue
        prediction = full_model.predict(arrays[preprocessing][deployment_indices]).ravel()
        if cultivar_calibration:
            prediction = prediction + np.asarray(
                [full_offsets[str(cultivar)] for cultivar in groups[deployment_indices]], dtype=float
            )
        anchor_prediction[deployment_indices] = prediction.astype(np.float32)

    if not np.all(np.isfinite(anchor_prediction[fit_indices])):
        raise RuntimeError("cross-fitting did not produce an anchor for every training fruit")
    residual_target = np.zeros(len(y), dtype=np.float32)
    residual_target[fit_indices] = (
        (y[fit_indices] - anchor_prediction[fit_indices]) / target_sd
    ).astype(np.float32)
    anchor_standardized = np.zeros(len(y), dtype=np.float32)
    valid = np.isfinite(anchor_prediction)
    anchor_standardized[valid] = (
        (anchor_prediction[valid] - target_mean) / target_sd
    ).astype(np.float32)

    calibrated_full_fit = full_fit_prediction.copy()
    if cultivar_calibration:
        calibrated_full_fit += np.asarray(
            [full_offsets[str(cultivar)] for cultivar in groups[fit_indices]], dtype=float
        )
    diagnostics: dict[str, object] = {
        "method": "cultivar-stratified out-of-fold PLS neural targets",
        "n_splits": int(n_splits),
        "seed": int(seed),
        "folds": fold_rows,
        "oof_rmse": float(
            np.sqrt(np.mean((anchor_prediction[fit_indices] - y[fit_indices]) ** 2))
        ),
        "full_fit_rmse": float(np.sqrt(np.mean((calibrated_full_fit - y[fit_indices]) ** 2))),
        "oof_residual_sd": float(np.std(y[fit_indices] - anchor_prediction[fit_indices], ddof=1)),
        "full_fit_residual_sd": float(np.std(y[fit_indices] - calibrated_full_fit, ddof=1)),
    }
    return (
        full_model,
        anchor_prediction,
        residual_target,
        anchor_standardized,
        target_mean,
        target_sd,
        full_offsets,
        diagnostics,
    )


def assign_observed_residual_targets(
    residual_target: np.ndarray,
    y: np.ndarray,
    anchor_prediction: np.ndarray,
    indices: np.ndarray,
    target_sd: float,
) -> None:
    """Populate supervised residuals for an explicitly authorized split.

    Anchor construction deliberately initializes targets only for the fitting
    partition so that held-out responses cannot be consumed accidentally.
    Neural checkpoint selection is a supervised validation operation, however,
    and therefore requires the true validation residual.  Keeping this write
    explicit at the call site preserves the test-label boundary.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if not np.all(np.isfinite(anchor_prediction[indices])):
        raise ValueError("Cannot create residual targets from non-finite anchor predictions")
    residual_target[indices] = (
        (y[indices] - anchor_prediction[indices]) / max(float(target_sd), 1e-6)
    ).astype(np.float32)


def predict_indices(
    model: torch.nn.Module,
    channel_builder: torch.nn.Module,
    raw: np.ndarray,
    residual_target: np.ndarray,
    anchor_standardized: np.ndarray,
    group_index: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        v2.SpectrumDataset(raw, residual_target, anchor_standardized, group_index, indices),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    prediction, _, _ = v2.predict_residual(model, channel_builder, loader, device)
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", required=True, choices=SUPPORTED_TARGETS)
    parser.add_argument("--cohort", choices=["analysis", "primary", "sensitivity"], default="primary")
    parser.add_argument("--exclude-cultivars", default="")
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument(
        "--outer-fold-manifest",
        type=Path,
        help=(
            "Optional frozen CSV with sample_id, cultivar_ascii and outer_fold. "
            "When supplied it replaces the repeated random outer split."
        ),
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        help="Test fold to use from --outer-fold-manifest.",
    )
    parser.add_argument(
        "--allow-manifest-subset",
        action="store_true",
        help=(
            "Allow a frozen manifest to define a strict subset of the otherwise "
            "eligible cohort. This is used for a common complete-case cohort across "
            "multiple targets; extra eligible fruit remain unused."
        ),
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=48)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--center-weight", type=float, default=0.20)
    parser.add_argument("--rank-weight", type=float, default=0.08)
    parser.add_argument("--sampler-power", type=float, default=0.50)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--fixed-gate", type=float)
    parser.add_argument(
        "--validation-residual-target-mode",
        choices=["observed", "legacy_zero"],
        default="observed",
        help=(
            "Use the true train-internal validation residual for checkpoint selection. "
            "legacy_zero reproduces the historical V17 scoring defect and must not be "
            "used for new model development."
        ),
    )
    parser.add_argument(
        "--crossfit-anchor-folds",
        type=int,
        default=0,
        help="Use this many cultivar-stratified OOF folds for honest neural residual targets; 0 preserves V5.",
    )
    parser.add_argument(
        "--cultivar-anchor-calibration",
        action="store_true",
        help="Estimate cultivar-specific PLS residual offsets from training fruit only.",
    )
    parser.add_argument(
        "--domain-aware-anchor-selection",
        action="store_true",
        help="Select PLS preprocessing/components by train-internal cultivar-calibrated CV RMSE.",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if (args.outer_fold_manifest is None) != (args.outer_fold is None):
        raise ValueError("--outer-fold-manifest and --outer-fold must be supplied together")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.parquet"
    metadata_path = output_dir / "metadata.json"
    if prediction_path.exists() and metadata_path.exists():
        print(f"repeat {args.repeat} already complete", flush=True)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("PLUMRAC-MT V5 stratified training requires CUDA")
    seed = 20260806 + args.repeat
    config = v2.RACConfig(
        width=args.width,
        blocks=args.blocks,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        center_weight=args.center_weight,
        rank_weight=args.rank_weight,
        sampler_power=args.sampler_power,
        validation_cultivars=0,
        min_gate_improvement=0.0,
        min_gate_win_fraction=0.50,
        max_gate_worst_degradation=0.05,
        max_residual_gate=1.0,
        structure_on_final=True,
        attention_tail=True,
    )

    multimodal_dir = args.multimodal_dir.resolve()
    raw = np.load(multimodal_dir / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(multimodal_dir / "wavelength_nm.npy").astype(np.float32)
    row_index = pd.read_csv(multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    sample_ids = aligned["sample_id"].astype(str).to_numpy()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    y = pd.to_numeric(aligned[args.target], errors="coerce").to_numpy(float)
    cohort_column = {
        "analysis": "qc_analysis_include",
        "primary": "qc_primary_include",
        "sensitivity": "qc_sensitivity_include",
    }[args.cohort]
    eligible = aligned[cohort_column].to_numpy(bool) & np.isfinite(y)
    validity_column = f"{args.target}_valid"
    if validity_column in aligned.columns:
        eligible &= aligned[validity_column].astype(bool).to_numpy()
    all_group_names = sorted(np.unique(groups).tolist())
    excluded = sorted({value.strip() for value in args.exclude_cultivars.split(",") if value.strip()})
    unknown = sorted(set(excluded) - set(all_group_names))
    if unknown:
        raise ValueError(f"Unknown cultivar exclusions: {unknown}")
    if excluded:
        eligible &= ~np.isin(groups, excluded)
    eligible_indices = np.flatnonzero(eligible)
    group_map = {group: index for index, group in enumerate(all_group_names)}
    group_index = np.asarray([group_map[group] for group in groups], dtype=np.int64)

    outer_manifest_path: Path | None = None
    outer_manifest_sha256: str | None = None
    if args.outer_fold_manifest is not None:
        outer_manifest_path = args.outer_fold_manifest.resolve()
        outer_manifest = pd.read_csv(outer_manifest_path)
        required_manifest_columns = {"sample_id", "cultivar_ascii", "outer_fold"}
        missing_columns = sorted(required_manifest_columns - set(outer_manifest.columns))
        if missing_columns:
            raise ValueError(f"Outer fold manifest is missing columns: {missing_columns}")
        if outer_manifest["sample_id"].astype(str).duplicated().any():
            raise ValueError("Outer fold manifest contains duplicate sample_id values")
        outer_manifest["sample_id"] = outer_manifest["sample_id"].astype(str)
        outer_manifest["cultivar_ascii"] = outer_manifest["cultivar_ascii"].astype(str)
        manifest_ids = set(outer_manifest["sample_id"])
        eligible_ids = set(sample_ids[eligible_indices])
        if args.allow_manifest_subset:
            extra_manifest_ids = manifest_ids - eligible_ids
            if extra_manifest_ids:
                raise ValueError(
                    "Outer fold manifest contains samples outside the eligible cohort: "
                    f"extra={len(extra_manifest_ids)}"
                )
            eligible_indices = np.asarray(
                [index for index in eligible_indices if sample_ids[index] in manifest_ids],
                dtype=np.int64,
            )
            eligible_ids = set(sample_ids[eligible_indices])
        if manifest_ids != eligible_ids:
            raise ValueError(
                "Outer fold manifest does not exactly cover the eligible cohort: "
                f"missing={len(eligible_ids - manifest_ids)}, extra={len(manifest_ids - eligible_ids)}"
            )
        aligned_manifest = outer_manifest.set_index("sample_id").loc[sample_ids[eligible_indices]]
        if not np.array_equal(
            aligned_manifest["cultivar_ascii"].to_numpy(str), groups[eligible_indices]
        ):
            raise ValueError("Cultivar labels in the outer fold manifest do not match the QC ledger")
        fold_values = pd.to_numeric(aligned_manifest["outer_fold"], errors="raise").to_numpy(int)
        available_folds = sorted(np.unique(fold_values).tolist())
        if args.outer_fold not in available_folds:
            raise ValueError(
                f"Requested outer fold {args.outer_fold} is not present; available={available_folds}"
            )
        test_mask = fold_values == args.outer_fold
        test_indices = eligible_indices[test_mask]
        outer_train = eligible_indices[~test_mask]
        outer_manifest_sha256 = hashlib.sha256(outer_manifest_path.read_bytes()).hexdigest()
        split_mode = "frozen_nonoverlapping_outer_fold"
    else:
        outer_train, test_indices = train_test_split(
            eligible_indices,
            test_size=args.test_fraction,
            random_state=seed,
            shuffle=True,
            stratify=groups[eligible_indices],
        )
        split_mode = "repeated_cultivar_stratified_random_holdout"
    inner_train, validation_indices = train_test_split(
        outer_train,
        test_size=args.validation_fraction,
        random_state=seed + 100_003,
        shuffle=True,
        stratify=groups[outer_train],
    )
    arrays = v2.preprocess_all(raw, wavelength)
    clean_channels = v2.build_clean_channels(raw, wavelength)

    v4.CHANNEL_SET = "basic"
    v4.ARCHITECTURE = "multiscale"
    v4.MIXSTYLE_P = 0.0
    v5.GLOBAL_AUXILIARY_TARGETS = list(v5.QUALITY_AUXILIARY_TARGETS)
    v5.GLOBAL_AUXILIARY_Y = np.column_stack(
        [
            pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
            for target in v5.GLOBAL_AUXILIARY_TARGETS
        ]
    )
    v5.PRETRAIN_EPOCHS = args.pretrain_epochs
    v5.PRETRAIN_LEARNING_RATE = 1e-3
    v5.PRETRAIN_WEIGHT_DECAY = 1e-3

    selected_inner_preprocessing, selected_inner_components, inner_cv = select_hyperparameters(
        arrays,
        y,
        groups,
        inner_train,
        seed + 200_003,
        cultivar_calibration=(
            args.domain_aware_anchor_selection and args.cultivar_anchor_calibration
        ),
    )
    if args.crossfit_anchor_folds:
        (
            _,
            anchor_inner,
            residual_inner,
            standardized_inner,
            _,
            inner_sd,
            inner_cultivar_offsets,
            inner_anchor_diagnostics,
        ) = prepare_crossfit_anchor_targets(
            arrays,
            y,
            groups,
            inner_train,
            [validation_indices],
            selected_inner_preprocessing,
            selected_inner_components,
            args.cultivar_anchor_calibration,
            args.crossfit_anchor_folds,
            seed + 250_003,
        )
    else:
        (
            _,
            anchor_inner,
            residual_inner,
            standardized_inner,
            _,
            inner_sd,
            inner_cultivar_offsets,
        ) = prepare_anchor_targets(
            arrays,
            y,
            groups,
            inner_train,
            [inner_train, validation_indices],
            selected_inner_preprocessing,
            selected_inner_components,
            args.cultivar_anchor_calibration,
        )
        inner_anchor_diagnostics = {"method": "in-sample PLS neural targets"}
    if args.validation_residual_target_mode == "observed":
        assign_observed_residual_targets(
            residual_inner,
            y,
            anchor_inner,
            validation_indices,
            inner_sd,
        )
    selection_model, selection_builder, selection_history, best_epoch = (
        v5.train_auxiliary_initialized_residual_model(
            raw,
            wavelength,
            clean_channels,
            residual_inner,
            standardized_inner,
            group_index,
            groups,
            inner_train,
            validation_indices,
            seed + 300_003,
            config,
            device,
        )
    )
    validation_residual_z = predict_indices(
        selection_model,
        selection_builder,
        raw,
        residual_inner,
        standardized_inner,
        group_index,
        validation_indices,
        config.batch_size,
        device,
    )
    selected_gate, gate_rows = v4.select_expanded_residual_gate(
        anchor_inner[validation_indices],
        validation_residual_z * inner_sd,
        y[validation_indices],
        groups[validation_indices],
        inner_sd,
        config.min_gate_improvement,
        config.min_gate_win_fraction,
        config.max_gate_worst_degradation,
        config.max_residual_gate,
    )
    internal_selected_gate = float(selected_gate)
    if args.fixed_gate is not None:
        selected_gate = float(args.fixed_gate)
    pd.DataFrame(inner_cv).to_csv(output_dir / "inner_pls_cv.csv", index=False)
    pd.DataFrame(selection_history).to_csv(output_dir / "selection_history.csv", index=False)

    final_preprocessing, final_components, final_cv = select_hyperparameters(
        arrays,
        y,
        groups,
        outer_train,
        seed,
        cultivar_calibration=(
            args.domain_aware_anchor_selection and args.cultivar_anchor_calibration
        ),
    )
    if args.crossfit_anchor_folds:
        (
            anchor_final_model,
            anchor_final,
            residual_final,
            standardized_final,
            _,
            final_sd,
            final_cultivar_offsets,
            final_anchor_diagnostics,
        ) = prepare_crossfit_anchor_targets(
            arrays,
            y,
            groups,
            outer_train,
            [test_indices],
            final_preprocessing,
            final_components,
            args.cultivar_anchor_calibration,
            args.crossfit_anchor_folds,
            seed + 950_003,
        )
    else:
        (
            anchor_final_model,
            anchor_final,
            residual_final,
            standardized_final,
            _,
            final_sd,
            final_cultivar_offsets,
        ) = prepare_anchor_targets(
            arrays,
            y,
            groups,
            outer_train,
            [outer_train, test_indices],
            final_preprocessing,
            final_components,
            args.cultivar_anchor_calibration,
        )
        final_anchor_diagnostics = {"method": "in-sample PLS neural targets"}
    final_model, final_builder, final_history, _ = v5.train_auxiliary_initialized_residual_model(
        raw,
        wavelength,
        clean_channels,
        residual_final,
        standardized_final,
        group_index,
        groups,
        outer_train,
        None,
        seed + 1_000_003,
        config,
        device,
        fixed_epochs=best_epoch,
    )
    test_residual_z = predict_indices(
        final_model,
        final_builder,
        raw,
        residual_final,
        standardized_final,
        group_index,
        test_indices,
        config.batch_size,
        device,
    )
    test_anchor = anchor_final[test_indices]
    test_global_anchor = anchor_final_model.predict(arrays[final_preprocessing][test_indices]).ravel()
    prediction = test_anchor + selected_gate * test_residual_z * final_sd
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids[test_indices],
            "cultivar_ascii": groups[test_indices],
            "target": args.target,
            "repeat": args.repeat,
            "seed": seed,
            "y_true": y[test_indices],
            "y_pred": prediction,
            "y_global_pls_anchor": test_global_anchor,
            "y_pls_anchor": test_anchor,
            "deep_residual": test_residual_z * final_sd,
            "residual_gate": selected_gate,
        }
    )
    frame["residual"] = frame["y_pred"] - frame["y_true"]
    frame = add_cultivar_code(frame)
    frame.to_parquet(prediction_path, index=False, compression="zstd")
    pd.DataFrame(final_cv).to_csv(output_dir / "final_pls_cv.csv", index=False)
    pd.DataFrame(final_history).to_csv(output_dir / "retrain_history.csv", index=False)
    torch.save(final_model.state_dict(), output_dir / "plumrac_state.pt")
    torch.save(final_builder.state_dict(), output_dir / "channel_builder_state.pt")
    joblib.dump(anchor_final_model, output_dir / "pls_anchor.joblib", compress=3)

    ai_metrics = v2.regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
    pls_metrics = v2.regression_metrics(frame["y_true"].to_numpy(), frame["y_pls_anchor"].to_numpy())
    global_pls_metrics = v2.regression_metrics(
        frame["y_true"].to_numpy(), frame["y_global_pls_anchor"].to_numpy()
    )
    metadata = {
        "model": (
            "PLUMRAC-MT cross-fitted-anchor known-cultivar single-trait model"
            if args.crossfit_anchor_folds
            else "PLUMRAC-MT known-cultivar single-trait model"
        ),
        "validation": (
            "frozen non-overlapping cultivar-stratified outer fold with train-internal validation and final refit"
            if args.outer_fold_manifest is not None
            else "cultivar-stratified fruit holdout with train-internal validation and final refit"
        ),
        "split_mode": split_mode,
        "outer_fold": int(args.outer_fold) if args.outer_fold is not None else None,
        "outer_fold_manifest": str(outer_manifest_path) if outer_manifest_path is not None else None,
        "outer_fold_manifest_sha256": outer_manifest_sha256,
        "target": args.target,
        "trait_abbreviation": abbreviated_trait(args.target),
        "repeat": args.repeat,
        "seed": seed,
        "cohort": args.cohort,
        "model_independent_excluded_cultivars": excluded,
        "eligible_samples": int(len(eligible_indices)),
        "eligible_cultivars": int(len(np.unique(groups[eligible_indices]))),
        "outer_train_samples": int(len(outer_train)),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
        "selected_epoch": int(best_epoch),
        "selected_gate": float(selected_gate),
        "internal_selected_gate": internal_selected_gate,
        "fixed_gate_requested": float(args.fixed_gate) if args.fixed_gate is not None else None,
        "gate_selection_mode": "legacy_fixed_override" if args.fixed_gate is not None else "training_internal_validation",
        "validation_residual_target_mode": args.validation_residual_target_mode,
        "validation_responses_used_for_checkpoint_selection": bool(
            args.validation_residual_target_mode == "observed"
        ),
        "cultivar_anchor_calibration": bool(args.cultivar_anchor_calibration),
        "domain_aware_anchor_selection": bool(args.domain_aware_anchor_selection),
        "crossfit_anchor_folds": int(args.crossfit_anchor_folds),
        "inner_anchor_diagnostics": inner_anchor_diagnostics,
        "final_anchor_diagnostics": final_anchor_diagnostics,
        "inner_cultivar_offsets": inner_cultivar_offsets,
        "final_cultivar_offsets": final_cultivar_offsets,
        "gate_scores": gate_rows,
        "pls_component_grid": list(PLS_COMPONENTS),
        "inner_pls": {
            "preprocessing": selected_inner_preprocessing,
            "n_components": int(selected_inner_components),
        },
        "final_pls": {
            "preprocessing": final_preprocessing,
            "n_components": int(final_components),
        },
        "config": asdict(config),
        "auxiliary_pretraining": {
            "targets": v5.GLOBAL_AUXILIARY_TARGETS,
            "epochs": int(v5.PRETRAIN_EPOCHS),
            "training_indices_only": True,
        },
        "ai_metrics": ai_metrics,
        "global_pls_anchor_metrics": global_pls_metrics,
        "pls_anchor_metrics": pls_metrics,
        "relative_rmse_improvement_pct": float(100.0 * (1.0 - ai_metrics["rmse"] / pls_metrics["rmse"])),
        "relative_rmse_improvement_vs_global_pls_pct": float(
            100.0 * (1.0 - ai_metrics["rmse"] / global_pls_metrics["rmse"])
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "test_labels_used_for_selection": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
