from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


SLUGS = {
    "3.13": "selection-3-13",
    "6.11": "selection-6-11",
    "A181": "a181",
    "Cuihongli": "cuihongli",
    "Fengtangli": "fengtangli",
    "Fengwei Huanghou": "fengwei-huanghou",
    "Furongli": "furongli",
    "Konglongdan": "konglongdan",
    "L31": "l31",
    "Naili": "naili",
    "Qingcuili": "qingcuili",
    "Weidi": "weidi",
    "Weijin": "weijin",
    "Weiwang": "weiwang",
    "Weixin": "weixin",
    "Zaoshu Konglongdan": "zaoshu-konglongdan",
}


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


def exclusion_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if not bool(row["nir_c_valid"]):
        reasons.append("invalid_nir_c")
    for target in ["fruit_weight_g", "soluble_solids_pct", "ph"]:
        if not bool(row[f"{target}_valid"]):
            reasons.append(f"invalid_{target}")
    if not bool(row["texture_dual_valid"]):
        reasons.append("incomplete_valid_texture_pair")
    return ";".join(reasons) or "not_in_complete_core"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dataset", type=Path, required=True)
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--texture-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    parent = args.parent_dataset.resolve()
    multimodal = args.multimodal_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in ["data", "metadata", "quality_control"]:
        (output / directory).mkdir(exist_ok=True)

    master = pd.read_parquet(multimodal / "master_samples.parquet")
    core_mask = master["include_primary_multitask"].astype(bool) & master["texture_dual_valid"].astype(bool)
    core = master.loc[core_mask].copy()
    if len(core) != 5487:
        raise ValueError(f"Unexpected complete-core size: {len(core)}")

    batch_order: dict[tuple[str, str], int] = {}
    for cultivar, frame in core.groupby("cultivar_ascii", sort=True):
        for order, batch in enumerate(sorted(frame["batch_id"].unique()), start=1):
            batch_order[(cultivar, batch)] = order
    core["cultivar_slug"] = core["cultivar_ascii"].map(SLUGS)
    if core["cultivar_slug"].isna().any():
        raise ValueError("Missing cultivar slug")
    core["canonical_batch"] = [
        f"b{batch_order[(cultivar, batch)]:02d}"
        for cultivar, batch in zip(core["cultivar_ascii"], core["batch_id"])
    ]
    core["canonical_sample_id"] = [
        f"plum-{slug}-{batch}-f{int(number):04d}"
        for slug, batch, number in zip(core["cultivar_slug"], core["canonical_batch"], core["fruit_number"])
    ]
    if core["canonical_sample_id"].duplicated().any():
        raise ValueError("Duplicate canonical sample IDs")
    id_map = core.set_index("sample_id")["canonical_sample_id"].to_dict()
    slug_map = core.set_index("sample_id")["cultivar_slug"].to_dict()

    parent_files = pd.read_csv(parent / "metadata" / "files.csv")
    selected_files = parent_files.loc[parent_files["sample_id"].isin(id_map)].copy()
    new_paths: list[str] = []
    for row in selected_files.itertuples(index=False):
        canonical = id_map[row.sample_id]
        slug = slug_map[row.sample_id]
        old_extension = Path(row.target_relative_path).suffix.lower()
        if row.data_type == "nir":
            scan = str(row.measurement).lower()
            relative = Path("data") / "nir" / scan / old_extension.lstrip(".") / slug / f"{canonical}__nir-{scan}{old_extension}"
        elif row.data_type == "texture":
            replicate = int(float(row.replicate))
            relative = Path("data") / "texture" / "arc" / slug / f"{canonical}__texture-rep{replicate:02d}{old_extension}"
        else:
            raise ValueError(f"Unsupported data type: {row.data_type}")
        new_paths.append(relative.as_posix())
    selected_files["source_sample_id"] = selected_files["sample_id"]
    selected_files["sample_id"] = selected_files["source_sample_id"].map(id_map)
    selected_files["parent_relative_path"] = selected_files["target_relative_path"]
    selected_files["target_relative_path"] = new_paths

    tasks = [
        (parent / parent_relative, output / target_relative, int(size))
        for parent_relative, target_relative, size in zip(
            selected_files["parent_relative_path"],
            selected_files["target_relative_path"],
            selected_files["bytes"],
        )
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(copy_one, tasks))

    samples = core[
        [
            "canonical_sample_id",
            "sample_id",
            "cultivar_ascii",
            "cultivar_original",
            "cultivar_slug",
            "canonical_batch",
            "batch_id",
            "material_code",
            "fruit_number",
            "fruit_weight_g",
            "soluble_solids_pct",
            "ph",
            "fruit_weight_g_source_value",
            "soluble_solids_pct_source_value",
            "ph_source_value",
            "nir_scan_types",
            "texture_replicates",
            "spectral_soft_outlier_flag",
        ]
    ].rename(columns={"canonical_sample_id": "sample_id", "sample_id": "source_sample_id"})
    samples.to_csv(output / "metadata" / "samples.csv", index=False)
    samples.to_parquet(output / "metadata" / "samples.parquet", index=False)

    crosswalk = samples[["sample_id", "source_sample_id", "cultivar_ascii", "batch_id", "fruit_number"]]
    crosswalk.to_csv(output / "metadata" / "id_crosswalk.csv", index=False)
    selected_files.to_csv(output / "metadata" / "files.csv", index=False)

    cultivar_summary = (
        samples.groupby(["cultivar_ascii", "cultivar_original", "cultivar_slug"], as_index=False)
        .agg(n_fruit=("sample_id", "size"), n_batches=("canonical_batch", "nunique"))
        .sort_values("cultivar_ascii")
    )
    cultivar_summary.insert(0, "cultivar_id", [f"cv{index:02d}" for index in range(1, len(cultivar_summary) + 1)])
    cultivar_summary.to_csv(output / "metadata" / "cultivars.csv", index=False)
    batch_summary = (
        samples.groupby(["cultivar_ascii", "cultivar_slug", "canonical_batch", "batch_id"], as_index=False)
        .agg(n_fruit=("sample_id", "size"), min_fruit_number=("fruit_number", "min"), max_fruit_number=("fruit_number", "max"))
    )
    batch_summary.to_csv(output / "metadata" / "batches.csv", index=False)

    excluded = master.loc[~core_mask].copy()
    excluded["exclusion_reason_v1_1"] = excluded.apply(exclusion_reason, axis=1)
    excluded[
        [
            "sample_id",
            "batch_id",
            "cultivar_ascii",
            "fruit_number",
            "fruit_weight_g_source_value",
            "soluble_solids_pct_source_value",
            "ph_source_value",
            "exclusion_reason_v1_1",
        ]
    ].to_csv(output / "quality_control" / "additional_exclusions_from_v1.0.csv", index=False)
    shutil.copy2(parent / "quality_control" / "excluded_samples.csv", output / "quality_control" / "source_exclusions_before_v1.0.csv")
    shutil.copy2(parent / "metadata" / "cultivar_aliases.csv", output / "metadata" / "cultivar_aliases.csv")

    row_index = pd.read_csv(multimodal / "nir_c_row_index.csv")
    absorbance = np.load(multimodal / "nir_c_absorbance.npy")
    selected_rows = row_index.index[row_index["sample_id"].isin(id_map)].to_numpy()
    selected_index = row_index.loc[selected_rows].copy()
    selected_index["source_sample_id"] = selected_index["sample_id"]
    selected_index["sample_id"] = selected_index["source_sample_id"].map(id_map)
    selected_index.insert(0, "matrix_row", np.arange(len(selected_index)))
    processed_dir = output / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / "nir_c_absorbance.npy", absorbance[selected_rows].astype(np.float32))
    np.save(processed_dir / "wavelength_nm.npy", np.load(multimodal / "wavelength_nm.npy"))
    selected_index.to_csv(processed_dir / "nir_c_sample_index.csv", index=False)

    texture = pd.read_parquet(args.texture_features)
    texture_core = texture.loc[texture["sample_id"].isin(id_map)].copy()
    texture_core["source_sample_id"] = texture_core["sample_id"]
    texture_core["sample_id"] = texture_core["source_sample_id"].map(id_map)
    texture_core.to_parquet(processed_dir / "texture_features.parquet", index=False)

    texture_mean_columns = [column for column in core.columns if column.endswith("_mean") and column.startswith(tuple([
        "skin_", "max_loading", "force_at_", "flesh_", "loading_", "post_break", "adhesive_", "max_displacement", "fracture_", "baseline_"
    ]))]
    model_table = samples.merge(
        texture_core[["sample_id"] + [column for column in texture_core.columns if column.endswith("_mean")]],
        on="sample_id",
        how="left",
    )
    model_table.to_parquet(processed_dir / "model_table.parquet", index=False)
    model_table.to_csv(processed_dir / "model_table.csv", index=False)

    dictionary_rows = [
        ("sample_id", "Stable canonical fruit identifier", "text"),
        ("source_sample_id", "Identifier in parent clean dataset v1.0.0", "text"),
        ("cultivar_ascii", "Canonical cultivar or breeding-selection label", "text"),
        ("cultivar_original", "Original cultivar label", "text"),
        ("batch_id", "Traceable source batch identifier", "text"),
        ("fruit_number", "Fruit number within source batch", "integer"),
        ("fruit_weight_g", "Single-fruit mass", "g"),
        ("soluble_solids_pct", "Soluble solids content", "%"),
        ("ph", "Measured pH; not titratable acidity", "pH unit"),
        ("spectral_soft_outlier_flag", "Exploratory PCA flag retained in primary analyses", "boolean"),
    ]
    pd.DataFrame(dictionary_rows, columns=["field", "description", "unit_or_type"]).to_csv(
        output / "metadata" / "data_dictionary.csv", index=False
    )

    readme = f"""# Research-ready plum NIR and texture dataset

Version: 1.1.0  
Created: 2026-08-06

This release contains the strict complete core used for cross-cultivar modelling: {len(samples):,} individual fruits from {samples['cultivar_ascii'].nunique()} cultivars or breeding selections and {samples['batch_id'].nunique()} acquisition batches. Every retained fruit has valid fruit weight, soluble-solids content, pH, a valid `c` NIR spectrum, and two valid primary ARC texture replicates. The optional `t` NIR scan is preserved when available.

## Naming

Canonical IDs follow `plum-<cultivar>-bNN-fNNNN`, for example `plum-weiwang-b01-f0034`. All paths and filenames are ASCII lowercase English. `metadata/id_crosswalk.csv` preserves exact linkage to v1.0 source IDs, source batches and fruit numbers.

## Layout

- `data/nir/<scan>/<format>/<cultivar>/`: standardized NIR CSV and DAT files.
- `data/texture/arc/<cultivar>/`: ARC replicates 01 and 02.
- `data/processed/`: aligned NIR matrix, wavelength vector, texture features and model table.
- `metadata/samples.csv`: one row per fruit.
- `metadata/files.csv`: file provenance, sizes and SHA-256 hashes.
- `quality_control/`: exclusions and validation record.

## Phenotype semantics

The three outcomes are single-fruit weight in grams, soluble solids in percent, and pH. The pH column is **not titratable acidity** and must not be reported as acid content.

## Curation policy

This v1.1.0 core adds content-level validation to parent v1.0.0. Objective invalid values, the one zero-filled NIR record, and an incomplete texture replicate pair are excluded. Thirty-five soft PCA spectral flags are retained to avoid performance-driven sample deletion; sensitivity analysis showed that their removal did not materially change conclusions. Three unambiguous repeated-decimal transcription artifacts were corrected, while original strings remain in source-value columns.

## Reuse and integrity

Use `data/processed/nir_c_sample_index.csv` to align rows of `nir_c_absorbance.npy` with `wavelength_nm.npy`. Raw standardized files are independent copies of parent v1.0.0. No license was supplied with the source data, so this release does not assert one.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    dataset_yaml = f"""name: NIRs_plums_research_ready_en_v1.1.0
version: 1.1.0
created: 2026-08-06
parent_dataset: NIRs_plums_clean_en_v1.0.0
included_samples: {len(samples)}
cultivars_or_selections: {samples['cultivar_ascii'].nunique()}
source_batches: {samples['batch_id'].nunique()}
measurement_files: {len(selected_files)}
sample_id_pattern: '^plum-[a-z0-9-]+-b[0-9]{{2}}-f[0-9]{{4}}$'
required_modalities: [nir_c, texture_arc_rep01, texture_arc_rep02]
phenotypes: [fruit_weight_g, soluble_solids_pct, ph]
license: unspecified
"""
    (output / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    required_per_sample = selected_files.copy()
    required_per_sample["requirement"] = [
        f"texture_{int(float(replicate))}"
        if data_type == "texture"
        else f"nir_{measurement}_{Path(target_path).suffix.lower().lstrip('.')}"
        for data_type, replicate, measurement, target_path in zip(
            required_per_sample["data_type"],
            required_per_sample["replicate"],
            required_per_sample["measurement"],
            required_per_sample["target_relative_path"],
        )
    ]
    counts = required_per_sample.groupby("sample_id")["requirement"].agg(set)
    required = {"nir_c_csv", "nir_c_dat", "texture_1", "texture_2"}
    missing_required = {sample_id: sorted(required - present) for sample_id, present in counts.items() if not required <= present}
    copied_bytes = sum((output / path).stat().st_size for path in selected_files["target_relative_path"])
    validation = {
        "status": "PASS" if not missing_required else "FAIL",
        "included_samples": len(samples),
        "unique_sample_ids": int(samples["sample_id"].nunique()),
        "cultivars_or_selections": int(samples["cultivar_ascii"].nunique()),
        "batches": int(samples["batch_id"].nunique()),
        "measurement_files": len(selected_files),
        "measurement_bytes": copied_bytes,
        "additional_exclusions_from_v1_0": len(excluded),
        "soft_spectral_flags_retained": int(samples["spectral_soft_outlier_flag"].sum()),
        "missing_required_files": missing_required,
        "nir_matrix_shape": list(np.load(processed_dir / "nir_c_absorbance.npy", mmap_mode="r").shape),
        "wavelength_range_nm": [float(np.load(processed_dir / "wavelength_nm.npy").min()), float(np.load(processed_dir / "wavelength_nm.npy").max())],
    }
    (output / "quality_control" / "validation_report.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    if validation["status"] != "PASS":
        raise ValueError(json.dumps(validation, indent=2))

    checksum_lines = [
        f"{row.sha256}  {row.target_relative_path}"
        for row in selected_files.sort_values("target_relative_path").itertuples(index=False)
    ]
    generated_files = [
        path for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256" and not str(path.relative_to(output)).startswith("data\\nir")
        and not str(path.relative_to(output)).startswith("data\\texture")
    ]
    existing_targets = set(selected_files["target_relative_path"])
    for path in sorted(generated_files):
        relative = path.relative_to(output).as_posix()
        if relative not in existing_targets:
            checksum_lines.append(f"{sha256(path)}  {relative}")
    (output / "quality_control" / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
