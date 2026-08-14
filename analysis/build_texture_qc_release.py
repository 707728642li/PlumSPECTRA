from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "1.2.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_one(task: tuple[Path, Path, int]) -> None:
    source, target, expected_size = task
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != expected_size:
        shutil.copy2(source, target)
    if target.stat().st_size != expected_size:
        raise IOError(f"Size mismatch after copy: {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-release", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    parent = args.parent_release.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in ["data", "metadata", "quality_control"]:
        (output / directory).mkdir(exist_ok=True)

    ledger = pd.read_parquet(args.qc_ledger)
    complete_primary = (
        ledger["qc_primary_include"].astype(bool)
        & ledger["include_primary_multitask"].astype(bool)
        & ledger["texture_dual_valid"].astype(bool)
    )
    selected_source_ids = set(ledger.loc[complete_primary, "sample_id"])
    if len(selected_source_ids) != 4941:
        raise ValueError(f"Unexpected v1.2 complete-primary size: {len(selected_source_ids)}")

    parent_samples = pd.read_parquet(parent / "metadata" / "samples.parquet")
    samples = parent_samples.loc[parent_samples["source_sample_id"].isin(selected_source_ids)].copy()
    if len(samples) != len(selected_source_ids):
        raise ValueError("Parent release does not contain every selected source sample")
    qc_columns = [
        "sample_id",
        "qc_version",
        "qc_primary_include",
        "qc_analysis_include",
        "qc_sensitivity_include",
        "qc_evidence_class_count",
        "qc_reason",
        "acquisition_rank",
        "replicate_rank",
        "texture_multivariate_rank",
        "spectral_rank",
        "chemical_rank",
    ]
    qc_selected = ledger[qc_columns].rename(columns={"sample_id": "source_sample_id"})
    samples = samples.merge(qc_selected, on="source_sample_id", how="left", validate="one_to_one")
    samples = samples.sort_values(["cultivar_ascii", "canonical_batch", "fruit_number"])
    samples.to_csv(output / "metadata" / "samples.csv", index=False, encoding="utf-8-sig")
    samples.to_parquet(output / "metadata" / "samples.parquet", index=False, compression="zstd")
    samples[["sample_id", "source_sample_id", "cultivar_ascii", "batch_id", "fruit_number"]].to_csv(
        output / "metadata" / "id_crosswalk.csv", index=False, encoding="utf-8-sig"
    )

    parent_files = pd.read_csv(parent / "metadata" / "files.csv")
    selected_files = parent_files.loc[parent_files["sample_id"].isin(set(samples["sample_id"]))].copy()
    tasks = [
        (parent / row.target_relative_path, output / row.target_relative_path, int(row.bytes))
        for row in selected_files.itertuples(index=False)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(copy_one, tasks))
    selected_files.to_csv(output / "metadata" / "files.csv", index=False, encoding="utf-8-sig")

    cultivars = (
        samples.groupby(["cultivar_ascii", "cultivar_original", "cultivar_slug"], as_index=False)
        .agg(n_fruit=("sample_id", "size"), n_batches=("canonical_batch", "nunique"))
        .sort_values("cultivar_ascii")
    )
    cultivars.insert(0, "cultivar_id", [f"cv{index:02d}" for index in range(1, len(cultivars) + 1)])
    cultivars.to_csv(output / "metadata" / "cultivars.csv", index=False, encoding="utf-8-sig")
    batches = (
        samples.groupby(["cultivar_ascii", "cultivar_slug", "canonical_batch", "batch_id"], as_index=False)
        .agg(
            n_fruit=("sample_id", "size"),
            min_fruit_number=("fruit_number", "min"),
            max_fruit_number=("fruit_number", "max"),
        )
    )
    batches.to_csv(output / "metadata" / "batches.csv", index=False, encoding="utf-8-sig")
    for name in ["cultivar_aliases.csv", "data_dictionary.csv"]:
        shutil.copy2(parent / "metadata" / name, output / "metadata" / name)

    processed = output / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    parent_processed = parent / "data" / "processed"
    parent_index = pd.read_csv(parent_processed / "nir_c_sample_index.csv")
    selected_mask = parent_index["sample_id"].isin(set(samples["sample_id"]))
    selected_rows = np.flatnonzero(selected_mask.to_numpy())
    absorbance = np.load(parent_processed / "nir_c_absorbance.npy", mmap_mode="r")
    np.save(processed / "nir_c_absorbance.npy", np.asarray(absorbance[selected_rows], dtype=np.float32))
    shutil.copy2(parent_processed / "wavelength_nm.npy", processed / "wavelength_nm.npy")
    index = parent_index.loc[selected_mask].copy().reset_index(drop=True)
    index["matrix_row"] = np.arange(len(index))
    index.to_csv(processed / "nir_c_sample_index.csv", index=False, encoding="utf-8-sig")

    texture = pd.read_parquet(parent_processed / "texture_features.parquet")
    texture = texture.loc[texture["sample_id"].isin(set(samples["sample_id"]))].copy()
    texture.to_parquet(processed / "texture_features.parquet", index=False, compression="zstd")
    texture_mean = texture[["sample_id"] + [column for column in texture.columns if column.endswith("_mean")]]
    model_table = samples.merge(texture_mean, on="sample_id", how="left", validate="one_to_one")
    model_table.to_parquet(processed / "model_table.parquet", index=False, compression="zstd")
    model_table.to_csv(processed / "model_table.csv", index=False, encoding="utf-8-sig")

    excluded = ledger.loc[~complete_primary].copy()
    excluded[
        [
            "sample_id",
            "batch_id",
            "cultivar_ascii",
            "fruit_number",
            "qc_primary_include",
            "qc_sensitivity_include",
            "include_primary_multitask",
            "texture_dual_valid",
            "qc_reason",
            "qc_evidence_class_count",
        ]
    ].to_csv(output / "quality_control" / "excluded_samples_v1.2.0.csv", index=False, encoding="utf-8-sig")
    ledger[
        [
            "sample_id",
            "batch_id",
            "cultivar_ascii",
            "fruit_number",
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
            "chemical_rank",
        ]
    ].to_csv(output / "quality_control" / "full_qc_ledger.csv", index=False, encoding="utf-8-sig")

    readme = f"""# Research-ready plum NIR and mechanical texture dataset

Version: {VERSION}  
Created: 2026-08-06

This texture-priority release contains {len(samples):,} individual fruits from {samples['cultivar_ascii'].nunique()} cultivar or breeding-selection labels and {samples['batch_id'].nunique()} acquisition batches. Every retained fruit has valid fruit mass, soluble-solids content, pH, a valid `c` NIR spectrum, two valid ARC texture replicates, and passes the model-independent consensus QC policy.

## Scientific role of texture

The acquisition order was intact fruit -> NIR -> fruit mass -> texture analyzer -> soluble solids -> pH. Texture is therefore a primary reference phenotype family, not a nuisance covariate. The processed table includes mechanically interpretable skin rupture, flesh resistance, stiffness, work/energy, and adhesion endpoints.

## QC policy

Raw source data were never deleted. Version 1.2.0 is derived from v1.1.0 using an immutable fruit-level QC ledger. This folder is the deliberately strict curation tier: a fruit is excluded only after a hard validity failure or concordant pre-model evidence involving acquisition stability and/or duplicate-measurement discordance plus independent texture, spectral, or chemical support. Prediction residuals are never used for exclusion. The project uses a less aggressive 5,430-fruit high-confidence cohort for headline inference because forcing the full 10% exclusion worsened most cultivar-held-out predictions; it also retains a 5,500-fruit hard-valid sensitivity cohort. The ledger records all three inclusion decisions.

## Layout

- `data/nir/`: standardized raw NIR files with canonical English filenames.
- `data/texture/arc/`: paired ARC texture files.
- `data/processed/`: aligned NIR matrix, texture endpoints, and modelling table.
- `metadata/`: one-row-per-fruit metadata, cultivar/batch summaries, aliases, and provenance.
- `quality_control/full_qc_ledger.csv`: auditable evidence and inclusion status for all 5,502 matched fruits.

## Important semantics

The chemical outcomes are single-fruit mass in grams, soluble solids in percent, and pH. pH is not titratable acidity. Position-derived texture quantities retain `raw_position_unit` because the ARC files do not independently verify a physical displacement unit.

## Naming and integrity

Canonical sample IDs and filenames are inherited unchanged from v1.1.0. `metadata/id_crosswalk.csv` links canonical IDs to source sample IDs. No source-data license was supplied, so this release does not assert one.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    dataset_yaml = f"""name: NIRs_plums_research_ready_en_v1.2.0
version: {VERSION}
created: 2026-08-06
parent_dataset: NIRs_plums_research_ready_en_v1.1.0
included_samples: {len(samples)}
cultivars_or_selections: {samples['cultivar_ascii'].nunique()}
source_batches: {samples['batch_id'].nunique()}
measurement_files: {len(selected_files)}
sample_id_pattern: '^plum-[a-z0-9-]+-b[0-9]{{2}}-f[0-9]{{4}}$'
required_modalities: [nir_c, texture_arc_rep01, texture_arc_rep02]
phenotypes: [fruit_weight_g, soluble_solids_pct, ph, mechanical_texture_endpoints]
qc_policy: model_independent_multievidence_consensus
license: unspecified
"""
    (output / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    copied_bytes = int(sum((output / path).stat().st_size for path in selected_files["target_relative_path"]))
    validation = {
        "status": "PASS",
        "version": VERSION,
        "included_samples": int(len(samples)),
        "unique_sample_ids": int(samples["sample_id"].nunique()),
        "cultivars_or_selections": int(samples["cultivar_ascii"].nunique()),
        "batches": int(samples["batch_id"].nunique()),
        "measurement_files": int(len(selected_files)),
        "measurement_bytes": copied_bytes,
        "nir_matrix_shape": list(np.load(processed / "nir_c_absorbance.npy", mmap_mode="r").shape),
        "all_primary_qc": bool(samples["qc_primary_include"].all()),
        "all_complete_chemical_targets": bool(
            samples[["fruit_weight_g", "soluble_solids_pct", "ph"]].notna().all(axis=None)
        ),
        "source_fruit_total_in_qc_ledger": int(len(ledger)),
        "strict_release_excluded": int(len(ledger) - len(samples)),
    }
    if validation["unique_sample_ids"] != validation["included_samples"] or not validation["all_primary_qc"]:
        validation["status"] = "FAIL"
    (output / "quality_control" / "validation_report.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    if validation["status"] != "PASS":
        raise ValueError(json.dumps(validation, indent=2))

    checksum_lines = []
    for row in selected_files.sort_values("target_relative_path").itertuples(index=False):
        checksum_lines.append(f"{row.sha256}  {row.target_relative_path}")
    raw_targets = set(selected_files["target_relative_path"])
    for path in sorted(path for path in output.rglob("*") if path.is_file() and path.name != "checksums.sha256"):
        relative = path.relative_to(output).as_posix()
        if relative not in raw_targets:
            checksum_lines.append(f"{sha256(path)}  {relative}")
    (output / "quality_control" / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
