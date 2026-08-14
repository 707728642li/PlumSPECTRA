from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_texture_pls_loco import preprocess_all
from v2_registry import cultivar_registry, trait_registry


ALPHAS = [0.1, 1.0, 10.0, 100.0, 1_000.0]
SHRINKAGES = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_DEVELOPMENT_CODES = ["L313", "CHL", "KLD", "WW", "WX"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fit_pls(x: np.ndarray, y: np.ndarray, indices: np.ndarray, components: int) -> PLSRegression:
    model = PLSRegression(n_components=components, scale=True, max_iter=1000, tol=1e-7)
    model.fit(x[indices], y[indices])
    return model


def context_feature(scores: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            scores.mean(axis=0),
            scores.std(axis=0, ddof=1),
            np.asarray([prediction.mean(), prediction.std(ddof=1)], dtype=float),
        ]
    )


def select_meta_model(features: np.ndarray, offsets: np.ndarray) -> tuple[float, float, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for alpha in ALPHAS:
        crossfit = np.zeros(len(offsets), dtype=float)
        for index in range(len(offsets)):
            train = np.arange(len(offsets)) != index
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[train], offsets[train])
            crossfit[index] = float(model.predict(features[index : index + 1])[0])
        for shrinkage in SHRINKAGES:
            prediction = shrinkage * crossfit
            rows.append(
                {
                    "alpha": alpha,
                    "shrinkage": shrinkage,
                    "domain_rmse": rmse(offsets, prediction),
                    "domain_mae": float(np.mean(np.abs(offsets - prediction))),
                }
            )
    scores = pd.DataFrame(rows).sort_values(["domain_rmse", "alpha", "shrinkage"])
    best = scores.iloc[0]
    return float(best["alpha"]), float(best["shrinkage"]), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--pls-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traits", default="all")
    parser.add_argument("--development-codes", default=",".join(DEFAULT_DEVELOPMENT_CODES))
    parser.add_argument("--context-components", type=int, default=4)
    args = parser.parse_args()

    registry = trait_registry().loc[lambda frame: frame["model_family"] == "endpoint"]
    target_map = registry.set_index("abbreviation")["target"].to_dict()
    traits = list(target_map) if args.traits == "all" else [value.strip().upper() for value in args.traits.split(",")]
    code_table = cultivar_registry().set_index("cultivar_code")
    development_codes = [value.strip().upper() for value in args.development_codes.split(",")]
    development_ascii = [str(code_table.loc[code, "cultivar_ascii"]) for code in development_codes]

    multimodal = args.multimodal_dir.resolve()
    raw = np.load(multimodal / "nir_c_absorbance.npy")
    wavelength = np.load(multimodal / "wavelength_nm.npy")
    arrays = preprocess_all(raw, wavelength)
    snv = arrays["snv"]
    row_index = pd.read_csv(multimodal / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger.resolve()).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    sample_ids = aligned["sample_id"].astype(str).to_numpy()
    include = aligned["qc_analysis_include"].to_numpy(bool)
    selected = pd.read_csv(args.pls_results.resolve() / "selected_hyperparameters.csv")
    outer_predictions = pd.read_parquet(args.pls_results.resolve() / "predictions.parquet")

    prediction_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    meta_score_frames: list[pd.DataFrame] = []
    for trait in traits:
        target = str(target_map[trait])
        y = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
        eligible = include & np.isfinite(y)
        trait_outer = outer_predictions.loc[outer_predictions["target"] == target].set_index("sample_id")
        for heldout_code, heldout in zip(development_codes, development_ascii, strict=True):
            source_indices = np.flatnonzero(eligible & (groups != heldout))
            target_indices = np.flatnonzero(eligible & (groups == heldout))
            selected_row = selected.loc[
                (selected["target"] == target) & (selected["heldout_cultivar"] == heldout)
            ].iloc[0]
            preprocessing = str(selected_row["preprocessing"])
            components = int(selected_row["n_components"])
            x_anchor = arrays[preprocessing]

            pca = PCA(n_components=args.context_components, svd_solver="randomized", random_state=20260806)
            source_context_scores = pca.fit_transform(snv[source_indices])
            target_context_scores = pca.transform(snv[target_indices])
            score_by_index = np.zeros((len(y), args.context_components), dtype=float)
            score_by_index[source_indices] = source_context_scores
            score_by_index[target_indices] = target_context_scores

            source_cultivars = sorted(np.unique(groups[source_indices]))
            context_rows: list[np.ndarray] = []
            offset_rows: list[float] = []
            for pseudo_target in source_cultivars:
                pseudo_indices = np.flatnonzero(eligible & (groups == pseudo_target))
                pseudo_train = np.flatnonzero(eligible & (groups != heldout) & (groups != pseudo_target))
                anchor = fit_pls(x_anchor, y, pseudo_train, components)
                pseudo_prediction = anchor.predict(x_anchor[pseudo_indices]).ravel()
                context_rows.append(context_feature(score_by_index[pseudo_indices], pseudo_prediction))
                offset_rows.append(float(np.mean(y[pseudo_indices] - pseudo_prediction)))
            context_matrix = np.vstack(context_rows)
            offsets = np.asarray(offset_rows)
            alpha, shrinkage, meta_scores = select_meta_model(context_matrix, offsets)
            meta_score_frames.append(meta_scores.assign(trait=trait, heldout_code=heldout_code))
            meta_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            meta_model.fit(context_matrix, offsets)

            sample_order = sample_ids[target_indices]
            saved_anchor = trait_outer.loc[sample_order, "y_pred"].to_numpy(float)
            target_context = context_feature(target_context_scores, saved_anchor)
            raw_offset = float(meta_model.predict(target_context[None, :])[0])
            predicted_offset = shrinkage * raw_offset
            corrected = saved_anchor + predicted_offset
            truth = y[target_indices]
            frame = pd.DataFrame(
                {
                    "sample_id": sample_order,
                    "cultivar_ascii": heldout,
                    "cultivar_code": heldout_code,
                    "trait": trait,
                    "target": target,
                    "y_true": truth,
                    "y_pred_plsr": saved_anchor,
                    "y_pred_context": corrected,
                }
            )
            prediction_frames.append(frame)
            metadata_rows.append(
                {
                    "trait": trait,
                    "cultivar_code": heldout_code,
                    "cultivar_ascii": heldout,
                    "n": int(len(frame)),
                    "preprocessing": preprocessing,
                    "n_components": components,
                    "context_components": args.context_components,
                    "meta_alpha": alpha,
                    "meta_shrinkage": shrinkage,
                    "predicted_offset": predicted_offset,
                    "true_oracle_offset": float(np.mean(truth - saved_anchor)),
                    "plsr_rmse": rmse(truth, saved_anchor),
                    "context_rmse": rmse(truth, corrected),
                    "rmse_improvement_pct": 100.0 * (rmse(truth, saved_anchor) - rmse(truth, corrected)) / rmse(truth, saved_anchor),
                }
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metadata = pd.DataFrame(metadata_rows)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "predictions_development.parquet", index=False, compression="zstd")
    metadata.to_csv(output / "fold_metrics.csv", index=False)
    pd.concat(meta_score_frames, ignore_index=True).to_parquet(output / "meta_inner_scores.parquet", index=False, compression="zstd")

    summary_rows: list[dict[str, object]] = []
    for trait, group in predictions.groupby("trait", observed=True):
        truth = group["y_true"].to_numpy(float)
        baseline = group["y_pred_plsr"].to_numpy(float)
        context = group["y_pred_context"].to_numpy(float)
        baseline_rmse = rmse(truth, baseline)
        context_rmse = rmse(truth, context)
        fold_group = metadata.loc[metadata["trait"] == trait]
        summary_rows.append(
            {
                "trait": trait,
                "development_folds": int(group["cultivar_code"].nunique()),
                "plsr_rmse": baseline_rmse,
                "context_rmse": context_rmse,
                "pooled_rmse_improvement_pct": 100.0 * (baseline_rmse - context_rmse) / baseline_rmse,
                "fold_wins": int((fold_group["context_rmse"] < fold_group["plsr_rmse"]).sum()),
                "plsr_r2": float(r2_score(truth, baseline)),
                "context_r2": float(r2_score(truth, context)),
                "nonzero_context_folds": int((fold_group["meta_shrinkage"] > 0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)
    report = {
        "status": "development-only transductive NIR context diagnostic",
        "development_codes": development_codes,
        "traits": traits,
        "target_labels_used_at_inference": False,
        "target_spectra_used_as_set_context": True,
        "source_offset_targets": "double-held-out PLSR residual means; outer target excluded",
        "claim_boundary": "Results motivate V3 development only and are not confirmation evidence.",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
