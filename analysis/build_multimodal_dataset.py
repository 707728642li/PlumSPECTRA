from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import PCA


BUILDER_VERSION = "0.1.0"
TARGETS = ["fruit_weight_g", "soluble_solids_pct", "ph"]


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median), axis=0)
    scale = np.where(mad > 1e-12, 1.4826 * mad, 1.0)
    return (values - median) / scale


def grouped_robust_abs_z(frame: pd.DataFrame, value: str, group: str) -> pd.Series:
    median = frame.groupby(group)[value].transform("median")
    absolute_deviation = (frame[value] - median).abs()
    mad = absolute_deviation.groupby(frame[group]).transform("median")
    scale = (1.4826 * mad).where(mad > 1e-12)
    return ((frame[value] - median) / scale).abs()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--nir-dir", type=Path, required=True)
    parser.add_argument("--texture-dir", type=Path, required=True)
    parser.add_argument("--phenotype-corrections", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    nir_dir = args.nir_dir.resolve()
    texture_dir = args.texture_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(dataset_root / "metadata" / "samples.csv", dtype=str, keep_default_na=False)
    samples["sample_id"] = samples["sample_id"].str.upper()
    if samples["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in samples.csv")
    for target in TARGETS:
        samples[f"{target}_source_value"] = samples[target]
    if args.phenotype_corrections is not None:
        corrections = pd.read_csv(args.phenotype_corrections.resolve(), dtype=str, keep_default_na=False)
        for correction in corrections.itertuples(index=False):
            mask = samples["sample_id"].eq(correction.sample_id.upper())
            if mask.sum() != 1:
                raise ValueError(f"Correction sample not unique: {correction.sample_id}")
            if correction.field not in TARGETS:
                raise ValueError(f"Unsupported correction field: {correction.field}")
            actual = samples.loc[mask, correction.field].iat[0]
            if actual != correction.source_value:
                raise ValueError(
                    f"Correction source mismatch for {correction.sample_id}/{correction.field}: "
                    f"expected {correction.source_value!r}, found {actual!r}"
                )
            samples.loc[mask, correction.field] = correction.corrected_value
    for target in TARGETS:
        samples[target] = pd.to_numeric(samples[target], errors="coerce")

    nir_manifest = pd.read_parquet(nir_dir / "nir_scan_manifest.parquet")
    nir_c = nir_manifest.loc[nir_manifest["scan_type"].eq("c")].copy()
    nir_t = nir_manifest.loc[nir_manifest["scan_type"].eq("t")].copy()
    if nir_c["sample_id"].duplicated().any() or nir_t["sample_id"].duplicated().any():
        raise ValueError("Duplicate scan type for a fruit")

    valid_grids = nir_c.loc[nir_c["point_count"].ge(100), "wavelength_grid_hash"]
    canonical_grid_hash = valid_grids.mode().iat[0]
    canonical_point_count = int(
        nir_c.loc[nir_c["wavelength_grid_hash"].eq(canonical_grid_hash), "point_count"].mode().iat[0]
    )
    nir_c["nir_c_valid"] = (
        nir_c["point_count"].eq(canonical_point_count)
        & nir_c["wavelength_grid_hash"].eq(canonical_grid_hash)
        & nir_c["parse_status"].eq("ok")
    )

    c_fields = [
        "sample_id", "nir_c_valid", "parse_status", "warning", "point_count",
        "wavelength_grid_hash", "absorbance_min", "absorbance_max", "absorbance_mean",
        "reference_signal_min", "sample_signal_min", "system_temp_c", "detector_temp_c",
        "humidity_pct", "lamp_pd", "host_date_time", "source_relative_path", "sha256",
    ]
    c_renames = {column: f"nir_c_{column}" for column in c_fields if column not in {"sample_id", "nir_c_valid"}}
    c_meta = nir_c[c_fields].rename(columns=c_renames)
    t_meta = nir_t[["sample_id", "parse_status", "warning", "point_count", "source_relative_path", "sha256"]].rename(
        columns={column: f"nir_t_{column}" for column in ["parse_status", "warning", "point_count", "source_relative_path", "sha256"]}
    )
    t_meta["nir_t_available"] = True

    texture = pd.read_parquet(texture_dir / "texture_sample_features.parquet")
    texture["sample_id"] = texture["sample_id"].str.upper()
    master = samples.merge(c_meta, on="sample_id", how="left", validate="one_to_one")
    master = master.merge(t_meta, on="sample_id", how="left", validate="one_to_one")
    master = master.merge(texture, on=["sample_id", "batch_id"], how="left", validate="one_to_one")
    master["nir_c_valid"] = master["nir_c_valid"].fillna(False).astype(bool)
    master["nir_t_available"] = master["nir_t_available"].fillna(False).astype(bool)
    master["fruit_weight_g_robust_z_within_cultivar"] = grouped_robust_abs_z(
        master, "fruit_weight_g", "cultivar_ascii"
    )
    master["fruit_weight_g_valid"] = (
        master["fruit_weight_g"].notna()
        & master["fruit_weight_g"].between(5, 500, inclusive="both")
        & master["fruit_weight_g_robust_z_within_cultivar"].le(10)
    )
    master["soluble_solids_pct_valid"] = master["soluble_solids_pct"].notna() & master["soluble_solids_pct"].between(0, 40, inclusive="both")
    master["ph_valid"] = master["ph"].notna() & master["ph"].between(2, 7, inclusive="both")
    master["phenotype_valid"] = master[[f"{target}_valid" for target in TARGETS]].all(axis=1)
    master["texture_dual_valid"] = master["texture_valid_replicates"].eq(2)
    for target in TARGETS:
        master[f"include_nir_{target}"] = master["nir_c_valid"] & master[f"{target}_valid"]
    master["include_primary_nir"] = master["phenotype_valid"] & master["nir_c_valid"]
    master["include_primary_multitask"] = master["include_primary_nir"] & master["texture_dual_valid"]

    spectra_table = pq.read_table(nir_dir / "nir_spectra.parquet")
    spectra = spectra_table.to_pandas()
    spectra_c = spectra.loc[
        spectra["scan_type"].eq("c")
        & spectra["point_count"].eq(canonical_point_count)
        & spectra["wavelength_grid_hash"].eq(canonical_grid_hash)
        & spectra["parse_status"].eq("ok")
    ].copy()
    spectra_c = spectra_c.set_index("sample_id").loc[
        master.loc[master["nir_c_valid"], "sample_id"]
    ].reset_index()
    wavelengths = np.asarray(spectra_c.iloc[0]["wavelength_nm"], dtype=np.float32)
    if not all(np.array_equal(np.asarray(row), wavelengths) for row in spectra_c["wavelength_nm"]):
        raise ValueError("Canonical wavelength hash contains unequal grids")
    absorbance = np.stack(spectra_c["absorbance_au"].map(lambda x: np.asarray(x, dtype=np.float32)))
    sample_signal = np.stack(spectra_c["sample_signal"].map(lambda x: np.asarray(x, dtype=np.float32)))
    reference_signal = np.stack(spectra_c["reference_signal"].map(lambda x: np.asarray(x, dtype=np.float32)))

    # Soft spectral anomaly flags are diagnostic only. Biological extremes remain in the
    # primary cohort; only objectively invalid/corrupt measurements are hard-excluded.
    snv_scale = absorbance.std(axis=1, ddof=1, keepdims=True)
    snv = (absorbance - absorbance.mean(axis=1, keepdims=True)) / np.where(snv_scale > 1e-8, snv_scale, 1.0)
    n_components = min(20, snv.shape[1], snv.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=20260806)
    scores = pca.fit_transform(snv)
    reconstruction = pca.inverse_transform(scores)
    orthogonal_distance = np.sqrt(np.mean((snv - reconstruction) ** 2, axis=1))
    soft_flag = np.zeros(len(spectra_c), dtype=bool)
    robust_score_distance = np.zeros(len(spectra_c), dtype=np.float64)
    batch_values = master.set_index("sample_id").loc[spectra_c["sample_id"], "batch_id"].to_numpy()
    for batch_id in np.unique(batch_values):
        idx = np.flatnonzero(batch_values == batch_id)
        z = robust_z(scores[idx, : min(10, n_components)])
        distance = np.sqrt(np.mean(z**2, axis=1))
        robust_score_distance[idx] = distance
        if len(idx) >= 20:
            threshold = np.quantile(distance, 0.995)
            soft_flag[idx] = distance > max(threshold, 5.0)
    od_z = robust_z(orthogonal_distance[:, None]).ravel()
    soft_flag |= od_z > 6.0

    spectral_qc = pd.DataFrame(
        {
            "sample_id": spectra_c["sample_id"],
            "spectral_pca_score_distance_within_batch": robust_score_distance,
            "spectral_pca_orthogonal_distance": orthogonal_distance,
            "spectral_pca_orthogonal_robust_z": od_z,
            "spectral_soft_outlier_flag": soft_flag,
        }
    )
    master = master.merge(spectral_qc, on="sample_id", how="left", validate="one_to_one")
    master["spectral_soft_outlier_flag"] = master["spectral_soft_outlier_flag"].fillna(False).astype(bool)

    master.to_parquet(output_dir / "master_samples.parquet", index=False, compression="zstd")
    master.to_csv(output_dir / "master_samples.csv", index=False, encoding="utf-8-sig")
    np.save(output_dir / "wavelength_nm.npy", wavelengths, allow_pickle=False)
    np.save(output_dir / "nir_c_absorbance.npy", absorbance, allow_pickle=False)
    np.save(output_dir / "nir_c_sample_signal.npy", sample_signal, allow_pickle=False)
    np.save(output_dir / "nir_c_reference_signal.npy", reference_signal, allow_pickle=False)
    pd.DataFrame({"row_index": np.arange(len(spectra_c)), "sample_id": spectra_c["sample_id"]}).to_csv(
        output_dir / "nir_c_row_index.csv", index=False, encoding="utf-8"
    )

    exclusion_rows: list[dict[str, str]] = []
    for row in master.itertuples(index=False):
        reasons: list[str] = []
        for target in TARGETS:
            if not getattr(row, f"{target}_valid"):
                reasons.append(f"invalid_or_missing_{target}")
        if not row.nir_c_valid:
            reasons.append("invalid_nir_c")
        if not row.texture_dual_valid:
            reasons.append("fewer_than_two_valid_texture_replicates")
        if reasons:
            exclusion_rows.append(
                {
                    "sample_id": row.sample_id,
                    "cultivar_ascii": row.cultivar_ascii,
                    "batch_id": row.batch_id,
                    "reasons": ";".join(reasons),
                    "primary_nir_included": str(bool(row.include_primary_nir)).lower(),
                    "primary_multitask_included": str(bool(row.include_primary_multitask)).lower(),
                }
            )
    exclusions = pd.DataFrame(exclusion_rows)
    exclusions.to_csv(output_dir / "analysis_exclusions.csv", index=False, encoding="utf-8-sig")

    cultivar_counts = (
        master.groupby(["cultivar_ascii", "cultivar_original"], dropna=False)
        .agg(
            total_samples=("sample_id", "size"),
            primary_nir_samples=("include_primary_nir", "sum"),
            nir_weight_samples=("include_nir_fruit_weight_g", "sum"),
            nir_ssc_samples=("include_nir_soluble_solids_pct", "sum"),
            nir_ph_samples=("include_nir_ph", "sum"),
            primary_multitask_samples=("include_primary_multitask", "sum"),
            t_scan_samples=("nir_t_available", "sum"),
            spectral_soft_outliers=("spectral_soft_outlier_flag", "sum"),
        )
        .reset_index()
        .sort_values(["primary_nir_samples", "cultivar_ascii"], ascending=[False, True])
    )
    cultivar_counts.to_csv(output_dir / "cultivar_cohort_counts.csv", index=False, encoding="utf-8-sig")

    summary = {
        "builder_version": BUILDER_VERSION,
        "canonical_grid_hash": canonical_grid_hash,
        "canonical_point_count": canonical_point_count,
        "wavelength_start_nm": float(wavelengths[0]),
        "wavelength_end_nm": float(wavelengths[-1]),
        "total_curated_samples": int(len(master)),
        "phenotype_valid_samples": int(master["phenotype_valid"].sum()),
        "target_valid_samples": {target: int(master[f"{target}_valid"].sum()) for target in TARGETS},
        "nir_c_valid_samples": int(master["nir_c_valid"].sum()),
        "nir_target_samples": {target: int(master[f"include_nir_{target}"].sum()) for target in TARGETS},
        "primary_nir_samples": int(master["include_primary_nir"].sum()),
        "texture_dual_valid_samples": int(master["texture_dual_valid"].sum()),
        "primary_multitask_samples": int(master["include_primary_multitask"].sum()),
        "t_scan_samples": int(master["nir_t_available"].sum()),
        "spectral_soft_outlier_flags_retained": int(master["spectral_soft_outlier_flag"].sum()),
        "pca_components_for_soft_qc": n_components,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "hard_exclusion_policy": "Only corrupt/invalid measurement content or implausible phenotype (weight 5-500 g and within-cultivar robust |z| <=10; SSC 0-40%; pH 2-7); no performance-driven removal.",
        "soft_outlier_policy": "Retained in primary analysis and reported for sensitivity analysis.",
    }
    (output_dir / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
