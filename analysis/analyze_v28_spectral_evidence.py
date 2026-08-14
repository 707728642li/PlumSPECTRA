from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_plumrac_loco as v2  # noqa: E402
import train_plumrac_v4_phy as v4  # noqa: E402


OUTPUT = ROOT / "results" / "v28_submission_strengthening"
MODEL_ROOT = ROOT / "results" / "v25_external_review_corrections"
MULTIMODAL = ROOT / "data" / "processed" / "multimodal"
FINAL = MODEL_ROOT / "final_analysis"

TRAITS = {
    "FW": {
        "target": "fruit_weight_g",
        "model_dir": MODEL_ROOT / "ai_quality_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v22_quality_fivefold_manifest.csv",
        "unit": "g",
    },
    "SSC": {
        "target": "soluble_solids_pct",
        "model_dir": MODEL_ROOT / "ai_quality_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v22_quality_fivefold_manifest.csv",
        "unit": "percentage points",
    },
    "pH": {
        "target": "ph",
        "model_dir": MODEL_ROOT / "ai_quality_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v22_quality_fivefold_manifest.csv",
        "unit": "pH units",
    },
    "SRF": {
        "target": "skin_break_force_g_mean",
        "model_dir": MODEL_ROOT / "ai_texture_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v20_fivefold_manifest.csv",
        "unit": "N",
    },
    "RD": {
        "target": "skin_break_displacement_raw_mean",
        "model_dir": MODEL_ROOT / "ai_texture_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v20_fivefold_manifest.csv",
        "unit": "APU",
    },
    "PFD": {
        "target": "skin_break_drop_g_mean",
        "model_dir": MODEL_ROOT / "ai_texture_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v20_fivefold_manifest.csv",
        "unit": "N",
    },
    "MFF": {
        "target": "flesh_force_mean_g_mean",
        "model_dir": MODEL_ROOT / "ai_texture_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v20_fivefold_manifest.csv",
        "unit": "N",
    },
    "F6": {
        "target": "force_at_6_rawpos_g_mean",
        "model_dir": MODEL_ROOT / "ai_texture_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v20_fivefold_manifest.csv",
        "unit": "N",
    },
    "LS": {
        "target": "loading_stiffness_g_per_rawpos_mean",
        "model_dir": MODEL_ROOT / "ai_texture_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v20_fivefold_manifest.csv",
        "unit": "N APU^-1",
    },
    "LW": {
        "target": "loading_work_g_rawpos_mean",
        "model_dir": MODEL_ROOT / "ai_texture_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v20_fivefold_manifest.csv",
        "unit": "N APU",
    },
    "PRW": {
        "target": "post_break_work_g_rawpos_mean",
        "model_dir": MODEL_ROOT / "ai_texture_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v20_fivefold_manifest.csv",
        "unit": "N APU",
    },
    "AF": {
        "target": "adhesive_force_g_mean",
        "model_dir": MODEL_ROOT / "ai_texture_domain_anchor_final",
        "manifest": MODEL_ROOT / "splits" / "v20_fivefold_manifest.csv",
        "unit": "N",
    },
}

TRAIT_ORDER = list(TRAITS)
FORCE_TRAITS = {"SRF", "PFD", "MFF", "F6", "AF"}
WORK_TRAITS = {"LW", "PRW"}
STIFFNESS_TRAITS = {"LS"}
GF_TO_N = 0.00980665
WINDOW_EDGES = np.arange(900.0, 1750.1, 50.0)
EDGE_CAUTION_NM = 1685.0
EDGE_CAUTION_LOW_NM = 920.0


def corr_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    numerator = np.sum(x_centered * y_centered[:, None], axis=0)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=0) * np.sum(y_centered**2)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(x.shape[1], np.nan, dtype=float),
        where=denominator > 0,
    )


def summarize_windows(
    frame: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, observed=True, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        fixed = dict(zip(group_cols, keys, strict=True))
        for low, high in zip(WINDOW_EDGES[:-1], WINDOW_EDGES[1:], strict=True):
            selected = group.loc[
                (group["wavelength_nm"] >= low)
                & (group["wavelength_nm"] < high)
            ]
            if selected.empty:
                continue
            row = {
                **fixed,
                "window_low_nm": float(low),
                "window_high_nm": float(min(high, group["wavelength_nm"].max())),
                value_col: float(selected[value_col].mean()),
                "edge_caution": bool(
                    low < EDGE_CAUTION_LOW_NM or high > EDGE_CAUTION_NM
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def extract_current_model_attention(
    raw: np.ndarray,
    wavelength: np.ndarray,
    sample_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v4.CHANNEL_SET = "basic"
    v4.ARCHITECTURE = "multiscale"
    v4.MIXSTYLE_P = 0.0
    rows: list[dict[str, object]] = []
    view_rows: list[dict[str, object]] = []
    sample_to_row = pd.Series(np.arange(len(sample_ids)), index=sample_ids)

    for trait, spec in TRAITS.items():
        manifest = pd.read_csv(spec["manifest"], dtype={"sample_id": str})
        manifest_rows = sample_to_row.reindex(manifest["sample_id"]).to_numpy()
        if np.isnan(manifest_rows).any():
            raise RuntimeError(f"Unaligned sample in {spec['manifest']}")
        manifest = manifest.assign(raw_row=manifest_rows.astype(int))
        for fold in range(1, 6):
            run_dir = Path(spec["model_dir"]) / trait / f"fold_{fold}"
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            config = v2.RACConfig(**metadata["config"])
            builder_state = torch.load(
                run_dir / "channel_builder_state.pt",
                map_location="cpu",
                weights_only=True,
            )
            builder = v2.SpectralChannelBuilder(
                wavelength,
                builder_state["channel_mean"].numpy(),
                builder_state["channel_sd"].numpy(),
                config,
            )
            builder.load_state_dict(builder_state)
            builder.eval()
            model = v4.V4PlumRACNet(
                config.width,
                config.blocks,
                config.dropout,
                config.attention_tail,
            )
            model.load_state_dict(
                torch.load(
                    run_dir / "plumrac_state.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            model.eval()
            indices = manifest.loc[manifest["outer_fold"] == fold, "raw_row"].to_numpy(int)
            attention_parts: list[np.ndarray] = []
            view_parts: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(indices), 512):
                    batch_indices = indices[start : start + 512]
                    channels = builder(torch.from_numpy(raw[batch_indices]), augment=False)
                    summary = torch.cat(
                        [
                            channels.mean(dim=-1),
                            channels.std(dim=-1, correction=0),
                        ],
                        dim=1,
                    )
                    view_weights = 2.0 * model.view_gate.network(summary)
                    features = model.encode(channels)
                    weights = torch.softmax(model.attention_pool(features), dim=-1).squeeze(1)
                    attention_parts.append(weights.numpy())
                    view_parts.append(view_weights.numpy())
            attention = np.concatenate(attention_parts)
            view_weights = np.concatenate(view_parts)
            enrichment = attention.mean(axis=0) * attention.shape[1]
            for wl, value in zip(wavelength, enrichment, strict=True):
                rows.append(
                    {
                        "trait": trait,
                        "outer_fold": fold,
                        "wavelength_nm": float(wl),
                        "attention_enrichment": float(value),
                    }
                )
            view_rows.append(
                {
                    "trait": trait,
                    "outer_fold": fold,
                    "raw_view_weight": float(view_weights[:, 0].mean()),
                    "snv_view_weight": float(view_weights[:, 1].mean()),
                    "sg1_view_weight": float(view_weights[:, 2].mean()),
                    "n_test": int(len(indices)),
                }
            )
            print(f"attention {trait} fold {fold}", flush=True)

    by_fold = pd.DataFrame(rows)
    summary = (
        by_fold.groupby(["trait", "wavelength_nm"], observed=True, as_index=False)
        .agg(
            attention_median=("attention_enrichment", "median"),
            attention_q25=("attention_enrichment", lambda x: x.quantile(0.25)),
            attention_q75=("attention_enrichment", lambda x: x.quantile(0.75)),
            attention_fold_cv=(
                "attention_enrichment",
                lambda x: float(x.std(ddof=1) / max(x.mean(), 1e-12)),
            ),
        )
    )
    summary["trait"] = pd.Categorical(summary["trait"], TRAIT_ORDER, ordered=True)
    summary = summary.sort_values(["trait", "wavelength_nm"])
    return by_fold, summary, pd.DataFrame(view_rows)


def within_cultivar_snv_associations(
    raw: np.ndarray,
    wavelength: np.ndarray,
    aligned_master: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_mean = raw.mean(axis=1, keepdims=True)
    row_sd = raw.std(axis=1, keepdims=True)
    snv = (raw - row_mean) / np.maximum(row_sd, 1e-8)
    rows: list[dict[str, object]] = []

    for trait, spec in TRAITS.items():
        manifest = pd.read_csv(spec["manifest"], dtype={"sample_id": str})
        eligible = set(manifest["sample_id"])
        mask = aligned_master["sample_id"].isin(eligible).to_numpy(copy=True)
        target = pd.to_numeric(aligned_master[spec["target"]], errors="coerce").to_numpy(float)
        mask &= np.isfinite(target)
        cultivars = aligned_master["cultivar_ascii"].astype(str).to_numpy()
        for cultivar in sorted(np.unique(cultivars[mask])):
            cultivar_mask = mask & (cultivars == cultivar)
            if cultivar_mask.sum() < 30:
                continue
            correlations = corr_columns(snv[cultivar_mask], target[cultivar_mask])
            for wl, value in zip(wavelength, correlations, strict=True):
                rows.append(
                    {
                        "trait": trait,
                        "cultivar": cultivar,
                        "n": int(cultivar_mask.sum()),
                        "wavelength_nm": float(wl),
                        "pearson_r": float(value),
                    }
                )
        print(f"association {trait}", flush=True)

    by_cultivar = pd.DataFrame(rows)

    def same_sign_fraction(values: pd.Series) -> float:
        median = float(values.median())
        if median == 0:
            return 0.5
        return float((np.sign(values) == np.sign(median)).mean())

    summary = (
        by_cultivar.groupby(["trait", "wavelength_nm"], observed=True, as_index=False)
        .agg(
            median_r=("pearson_r", "median"),
            q25_r=("pearson_r", lambda x: x.quantile(0.25)),
            q75_r=("pearson_r", lambda x: x.quantile(0.75)),
            median_abs_r=("pearson_r", lambda x: np.median(np.abs(x))),
            direction_consistency=("pearson_r", same_sign_fraction),
            n_cultivars=("cultivar", "nunique"),
        )
    )
    summary["trait"] = pd.Categorical(summary["trait"], TRAIT_ORDER, ordered=True)
    summary = summary.sort_values(["trait", "wavelength_nm"])
    return by_cultivar, summary


def build_window_consensus(
    attention: pd.DataFrame,
    associations: pd.DataFrame,
) -> pd.DataFrame:
    attention_windows = summarize_windows(
        attention,
        "attention_median",
        ["trait"],
    )
    association_windows = summarize_windows(
        associations.assign(abs_median_r=associations["median_r"].abs()),
        "abs_median_r",
        ["trait"],
    )
    direction_windows = summarize_windows(
        associations,
        "direction_consistency",
        ["trait"],
    )
    windows = attention_windows.merge(
        association_windows.drop(columns="edge_caution"),
        on=["trait", "window_low_nm", "window_high_nm"],
        validate="one_to_one",
    ).merge(
        direction_windows.drop(columns="edge_caution"),
        on=["trait", "window_low_nm", "window_high_nm"],
        validate="one_to_one",
    )
    windows["attention_rank"] = windows.groupby("trait", observed=True)[
        "attention_median"
    ].rank(ascending=False, method="min")
    windows["association_rank"] = windows.groupby("trait", observed=True)[
        "abs_median_r"
    ].rank(ascending=False, method="min")
    windows["consensus_rank_sum"] = (
        windows["attention_rank"] + windows["association_rank"]
    )
    windows["consensus_rank_nonedge"] = np.nan
    for trait, index in windows.loc[~windows["edge_caution"]].groupby(
        "trait", observed=True
    ).groups.items():
        windows.loc[index, "consensus_rank_nonedge"] = windows.loc[
            index, "consensus_rank_sum"
        ].rank(ascending=True, method="min")
    windows["trait"] = pd.Categorical(windows["trait"], TRAIT_ORDER, ordered=True)
    return windows.sort_values(["trait", "consensus_rank_sum"])


def publication_scale(trait: str, values: np.ndarray) -> np.ndarray:
    if trait in FORCE_TRAITS | WORK_TRAITS | STIFFNESS_TRAITS:
        return values * GF_TO_N
    return values


def practical_accuracy_table() -> pd.DataFrame:
    predictions = pd.read_parquet(FINAL / "v25_integrated_predictions.parquet")
    metrics = pd.read_csv(FINAL / "pooled_metrics.csv")
    metrics = metrics.loc[metrics["model"] == "plumspectra_corrected"].set_index("trait")
    reliability = pd.read_csv(FINAL / "texture_reliability_modeling_cohort.csv").set_index(
        "trait"
    )
    rows: list[dict[str, object]] = []
    for trait in TRAIT_ORDER:
        frame = predictions.loc[predictions["trait"] == trait]
        truth = publication_scale(trait, frame["y_true"].to_numpy(float))
        pred = publication_scale(trait, frame["y_final"].to_numpy(float))
        absolute_error = np.abs(pred - truth)
        trait_metrics = metrics.loc[trait]
        observed_iqr = float(np.quantile(truth, 0.75) - np.quantile(truth, 0.25))
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        mae = float(np.mean(absolute_error))
        row: dict[str, object] = {
            "trait": trait,
            "unit": TRAITS[trait]["unit"],
            "n": int(len(frame)),
            "observed_iqr": observed_iqr,
            "rmse": rmse,
            "mae": mae,
            "median_absolute_error": float(np.median(absolute_error)),
            "p80_absolute_error": float(np.quantile(absolute_error, 0.80)),
            "rmse_pct_iqr": float(100.0 * rmse / observed_iqr),
            "mae_pct_iqr": float(100.0 * mae / observed_iqr),
            "within_half_iqr_pct": float(100.0 * np.mean(absolute_error <= 0.5 * observed_iqr)),
            "r2": float(trait_metrics["r2"]),
            "rpiq": float(trait_metrics["rpiq"]),
        }
        if trait in reliability.index:
            rel = reliability.loc[trait]
            row["duplicate_icc_a1"] = float(rel["icc_a1"])
            row["median_duplicate_cv_pct"] = float(100.0 * rel["median_replicate_cv"])
            row["r2_as_pct_of_icc"] = float(100.0 * trait_metrics["r2"] / rel["icc_a1"])
        else:
            row["duplicate_icc_a1"] = np.nan
            row["median_duplicate_cv_pct"] = np.nan
            row["r2_as_pct_of_icc"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = np.load(MULTIMODAL / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(MULTIMODAL / "wavelength_nm.npy").astype(np.float32)
    row_index = pd.read_csv(MULTIMODAL / "nir_c_row_index.csv", dtype={"sample_id": str})
    master = pd.read_parquet(MULTIMODAL / "master_samples.parquet")
    aligned_master = row_index[["sample_id"]].merge(
        master,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if aligned_master["cultivar_ascii"].isna().any():
        raise RuntimeError("Failed to align master samples to NIR row index")

    attention_fold, attention_summary, view_weights = extract_current_model_attention(
        raw,
        wavelength,
        row_index["sample_id"].to_numpy(str),
    )
    association_cultivar, association_summary = within_cultivar_snv_associations(
        raw,
        wavelength,
        aligned_master,
    )
    consensus = build_window_consensus(attention_summary, association_summary)
    practical = practical_accuracy_table()

    attention_fold.to_parquet(
        OUTPUT / "current_model_attention_by_fold.parquet",
        index=False,
        compression="zstd",
    )
    attention_summary.to_csv(OUTPUT / "current_model_attention_wavelength.csv", index=False)
    view_weights.to_csv(OUTPUT / "current_model_view_weights.csv", index=False)
    association_cultivar.to_parquet(
        OUTPUT / "within_cultivar_snv_association_by_cultivar.parquet",
        index=False,
        compression="zstd",
    )
    association_summary.to_csv(
        OUTPUT / "within_cultivar_snv_association_wavelength.csv",
        index=False,
    )
    consensus.to_csv(OUTPUT / "wavelength_window_consensus.csv", index=False)
    consensus.loc[
        (~consensus["edge_caution"])
        & (consensus["consensus_rank_nonedge"] <= 3)
    ].to_csv(OUTPUT / "top_nonedge_consensus_windows.csv", index=False)
    practical.to_csv(OUTPUT / "practical_accuracy_context.csv", index=False)

    audit = {
        "release": "V28 submission strengthening",
        "primary_models_interpreted": int(attention_fold[["trait", "outer_fold"]].drop_duplicates().shape[0]),
        "traits": TRAIT_ORDER,
        "wavelengths": int(len(wavelength)),
        "attention_definition": "Mean attention-pooling allocation on each frozen outer-test fold, scaled relative to uniform allocation.",
        "association_definition": "Within-cultivar Pearson correlation between SNV absorbance and observed phenotype, summarized with equal weight across cultivars.",
        "instrument_edge_caution_nm": [EDGE_CAUTION_LOW_NM, EDGE_CAUTION_NM],
        "interpretation_limit": "Attention and correlation are model-based or observational associations under spectral collinearity; neither is a causal or compound-specific assignment.",
        "main_figures_modified": False,
        "outputs": {
            "attention_rows": int(len(attention_summary)),
            "association_rows": int(len(association_summary)),
            "consensus_windows": int(len(consensus)),
            "practical_accuracy_traits": int(len(practical)),
        },
    }
    (OUTPUT / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
