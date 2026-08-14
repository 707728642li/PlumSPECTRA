from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


PARSER_VERSION = "0.1.0"
NAME_PATTERN = re.compile(
    r"^(?P<sample_id>[A-Z0-9_]+-\d{4})_nir_(?P<scan_type>[ct])\.csv$",
    re.IGNORECASE,
)


def parse_float(value: str) -> float:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return math.nan


def clean_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().rstrip(":").lower()).strip("_")


def parse_nir_csv(file: Path, dataset_root: Path) -> dict[str, Any]:
    match = NAME_PATTERN.match(file.name)
    if match is None:
        raise ValueError(f"Non-standard NIR CSV filename: {file.name}")
    sample_id = match.group("sample_id").upper()
    batch_id = sample_id.rsplit("-", 1)[0]
    scan_type = match.group("scan_type").lower()
    raw_bytes = file.read_bytes()
    text = raw_bytes.decode("utf-8-sig")
    metadata: dict[str, str] = {}
    wavelengths: list[float] = []
    absorbance: list[float] = []
    reference: list[float] = []
    sample: list[float] = []
    data_mode = False
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        first = row[0].strip()
        if first == "Wavelength (nm)":
            data_mode = True
            continue
        if not data_mode:
            if first.endswith(":"):
                metadata[clean_key(first)] = "|".join(cell.strip() for cell in row[1:] if cell.strip())
            continue
        if len(row) < 4 or not first:
            continue
        wavelengths.append(float(row[0]))
        absorbance.append(float(row[1]))
        reference.append(float(row[2]))
        sample.append(float(row[3]))

    wave = np.asarray(wavelengths, dtype=np.float64)
    absorb = np.asarray(absorbance, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    sam = np.asarray(sample, dtype=np.float64)
    warnings: list[str] = []
    if len(wave) < 100:
        warnings.append("too_few_wavelengths")
    if not (len(wave) == len(absorb) == len(ref) == len(sam)):
        warnings.append("column_length_mismatch")
    if len(wave) and not np.all(np.diff(wave) > 0):
        warnings.append("wavelength_not_strictly_increasing")
    if not all(np.all(np.isfinite(values)) for values in (wave, absorb, ref, sam)):
        warnings.append("nonfinite_values")
    if len(ref) and (np.min(ref) <= 0 or np.min(sam) <= 0):
        warnings.append("nonpositive_detector_signal")
    if len(absorb) and (np.min(absorb) < -1.0 or np.max(absorb) > 5.0):
        warnings.append("absorbance_outside_plausibility_range")
    grid_hash = hashlib.sha256(np.round(wave, 6).tobytes()).hexdigest() if len(wave) else ""
    return {
        "parser_version": PARSER_VERSION,
        "sample_id": sample_id,
        "batch_id": batch_id,
        "scan_type": scan_type,
        "source_relative_path": file.relative_to(dataset_root).as_posix(),
        "source_bytes": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "parse_status": "ok" if not warnings else "warning",
        "warning": ";".join(warnings),
        "point_count": int(len(wave)),
        "wavelength_start_nm": float(wave[0]) if len(wave) else math.nan,
        "wavelength_end_nm": float(wave[-1]) if len(wave) else math.nan,
        "wavelength_step_median_nm": float(np.median(np.diff(wave))) if len(wave) > 1 else math.nan,
        "wavelength_grid_hash": grid_hash,
        "absorbance_min": float(np.min(absorb)) if len(absorb) else math.nan,
        "absorbance_max": float(np.max(absorb)) if len(absorb) else math.nan,
        "absorbance_mean": float(np.mean(absorb)) if len(absorb) else math.nan,
        "reference_signal_min": float(np.min(ref)) if len(ref) else math.nan,
        "reference_signal_max": float(np.max(ref)) if len(ref) else math.nan,
        "sample_signal_min": float(np.min(sam)) if len(sam) else math.nan,
        "sample_signal_max": float(np.max(sam)) if len(sam) else math.nan,
        "system_temp_c": parse_float(metadata.get("system_temp_c", "")),
        "detector_temp_c": parse_float(metadata.get("detector_temp_c", "")),
        "humidity_pct": parse_float(metadata.get("humidity", "")),
        "lamp_pd": parse_float(metadata.get("lamp_pd", "")),
        "host_date_time": metadata.get("host_date_time", ""),
        "method": metadata.get("method", ""),
        "scan_config_name": metadata.get("scan_config_name", ""),
        "declared_start_wavelength_nm": parse_float(metadata.get("start_wavelength_nm", "")),
        "declared_end_wavelength_nm": parse_float(metadata.get("end_wavelength_nm", "")),
        "pattern_pixel_width_nm": parse_float(metadata.get("pattern_pixel_width_nm", "")),
        "shift_vector_coefficients": metadata.get("shift_vector_coefficients", ""),
        "pixel_to_wavelength_coefficients": metadata.get("pixel_to_wavelength_coefficients", ""),
        "wavelength_nm": wave.astype(np.float32),
        "absorbance_au": absorb.astype(np.float32),
        "reference_signal": ref.astype(np.float32),
        "sample_signal": sam.astype(np.float32),
    }


def write_spectra(rows: list[dict[str, Any]], output_file: Path) -> None:
    scalar_fields = [
        ("parser_version", pa.string()), ("sample_id", pa.string()), ("batch_id", pa.string()),
        ("scan_type", pa.string()), ("source_relative_path", pa.string()), ("source_bytes", pa.int64()),
        ("sha256", pa.string()), ("parse_status", pa.string()), ("warning", pa.string()),
        ("point_count", pa.int32()), ("wavelength_start_nm", pa.float64()), ("wavelength_end_nm", pa.float64()),
        ("wavelength_step_median_nm", pa.float64()), ("wavelength_grid_hash", pa.string()),
        ("absorbance_min", pa.float64()), ("absorbance_max", pa.float64()), ("absorbance_mean", pa.float64()),
        ("reference_signal_min", pa.float64()), ("reference_signal_max", pa.float64()),
        ("sample_signal_min", pa.float64()), ("sample_signal_max", pa.float64()),
        ("system_temp_c", pa.float64()), ("detector_temp_c", pa.float64()), ("humidity_pct", pa.float64()),
        ("lamp_pd", pa.float64()), ("host_date_time", pa.string()), ("method", pa.string()),
        ("scan_config_name", pa.string()), ("declared_start_wavelength_nm", pa.float64()),
        ("declared_end_wavelength_nm", pa.float64()), ("pattern_pixel_width_nm", pa.float64()),
        ("shift_vector_coefficients", pa.string()), ("pixel_to_wavelength_coefficients", pa.string()),
    ]
    list_fields = [
        ("wavelength_nm", pa.list_(pa.float32())),
        ("absorbance_au", pa.list_(pa.float32())),
        ("reference_signal", pa.list_(pa.float32())),
        ("sample_signal", pa.list_(pa.float32())),
    ]
    schema = pa.schema(scalar_fields + list_fields)
    arrays = [
        pa.array([row[field.name] for row in rows], type=field.type)
        for field in schema
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_file, compression="zstd", compression_level=6)


def quantiles(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {key: float(value) for key, value in zip(["min", "q01", "median", "q99", "max"], [values.min(), values.quantile(0.01), values.median(), values.quantile(0.99), values.max()])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    files = sorted((dataset_root / "data" / "nir").rglob("*.csv"), key=lambda p: p.as_posix().lower())
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(tqdm(executor.map(lambda p: parse_nir_csv(p, dataset_root), files), total=len(files), desc="Scanning NIR", unit="file"))
    rows.sort(key=lambda row: (row["batch_id"], row["sample_id"], row["scan_type"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_path = output_dir / "nir_spectra.parquet"
    write_spectra(rows, spectra_path)
    scalar = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, np.ndarray)} for row in rows])
    scalar.to_parquet(output_dir / "nir_scan_manifest.parquet", index=False, compression="zstd")
    grid_counts = scalar.groupby(["scan_type", "wavelength_grid_hash", "point_count", "wavelength_start_nm", "wavelength_end_nm"]).size().reset_index(name="files")
    summary = {
        "parser_version": PARSER_VERSION,
        "files": int(len(scalar)),
        "samples": int(scalar["sample_id"].nunique()),
        "scan_types": {str(k): int(v) for k, v in scalar["scan_type"].value_counts().items()},
        "batches": int(scalar["batch_id"].nunique()),
        "parse_status": {str(k): int(v) for k, v in scalar["parse_status"].value_counts().items()},
        "warnings": scalar.loc[scalar["warning"].ne(""), ["sample_id", "scan_type", "warning"]].to_dict(orient="records"),
        "wavelength_grids": grid_counts.to_dict(orient="records"),
        "numeric_summary": {column: quantiles(scalar[column]) for column in ["point_count", "wavelength_start_nm", "wavelength_end_nm", "absorbance_min", "absorbance_max", "reference_signal_min", "sample_signal_min", "system_temp_c", "detector_temp_c", "humidity_pct", "lamp_pd"]},
    }
    (output_dir / "nir_scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
