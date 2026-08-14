from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm

from arc_parser import (
    PARSER_VERSION,
    decode_values,
    find_five_channel_chains,
    unpack_arc_outer,
)


FEATURE_VERSION = "0.1.0"
# Dataset-specific Exponent archive-to-display conversion. Original embedded
# previews from all 19 acquisition batches are labelled "Force (g)" and their
# plotted peaks match force_raw * 1000. The archives also record a 30,000 g
# load-cell capacity. Here the displayed g is interpreted scientifically as gf;
# this factor must not be assumed for unrelated ARC archive families.
FORCE_SCALE_TO_G = 1000.0
CORE_FEATURES = [
    "skin_break_force_g",
    "skin_break_displacement_raw",
    "skin_break_work_g_rawpos",
    "skin_break_drop_g",
    "max_loading_force_g",
    "force_at_2_rawpos_g",
    "force_at_4_rawpos_g",
    "force_at_6_rawpos_g",
    "flesh_force_mean_g",
    "flesh_force_median_g",
    "flesh_force_sd_g",
    "flesh_force_slope_g_per_rawpos",
    "loading_stiffness_g_per_rawpos",
    "loading_work_g_rawpos",
    "post_break_work_g_rawpos",
    "adhesive_force_g",
    "adhesive_work_g_rawpos",
    "max_displacement_raw",
    "loading_duration_s",
    "loading_speed_rawpos_per_s",
    "fracture_peak_count",
    "baseline_force_g",
    "baseline_noise_g",
]


def smooth_force(force: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    window = max(9, int(round(0.10 * sampling_rate_hz)) | 1)
    if window >= len(force):
        window = max(5, (len(force) - 1) | 1)
    if window < 5:
        return force.copy()
    return savgol_filter(force, window_length=window, polyorder=min(3, window - 2), mode="interp")


def first_sustained(mask: np.ndarray, samples: int) -> int | None:
    if mask.size < samples:
        return None
    hits = np.convolve(mask.astype(np.int16), np.ones(samples, dtype=np.int16), mode="valid")
    indices = np.flatnonzero(hits == samples)
    return int(indices[0]) if indices.size else None


def safe_interp(x: np.ndarray, y: np.ndarray, target: float) -> float:
    if len(x) < 2 or target < np.nanmin(x) or target > np.nanmax(x):
        return math.nan
    order = np.argsort(x)
    sorted_x = x[order]
    sorted_y = y[order]
    unique_x, indices = np.unique(sorted_x, return_index=True)
    if len(unique_x) < 2:
        return math.nan
    return float(np.interp(target, unique_x, sorted_y[indices]))


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10 or float(np.ptp(x[valid])) <= 1e-9:
        return math.nan
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def extract_features(
    relative_time: np.ndarray,
    force_raw: np.ndarray,
    position_raw: np.ndarray,
) -> dict[str, Any]:
    force_g = force_raw * FORCE_SCALE_TO_G
    dt_values = np.diff(relative_time)
    positive_dt = dt_values[dt_values > 0]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else math.nan
    sampling_rate = 1.0 / median_dt if median_dt > 0 else 400.0
    baseline_samples = min(len(force_g), max(20, int(round(0.20 * sampling_rate))))
    baseline = float(np.median(force_g[:baseline_samples]))
    baseline_noise = float(1.4826 * np.median(np.abs(force_g[:baseline_samples] - baseline)))
    force_corrected = force_g - baseline
    smoothed = smooth_force(force_corrected, sampling_rate)

    reversal = int(np.argmin(position_raw))
    sustain = max(8, int(round(0.04 * sampling_rate)))
    threshold = max(10.0, 6.0 * baseline_noise)
    contact = first_sustained(smoothed[: reversal + 1] > threshold, sustain)
    if contact is None:
        positive = np.flatnonzero(smoothed[: reversal + 1] > max(5.0, 3.0 * baseline_noise))
        contact = int(positive[0]) if positive.size else 0

    loading_slice = slice(contact, reversal + 1)
    load_t = relative_time[loading_slice]
    load_force = force_corrected[loading_slice]
    load_smooth = smoothed[loading_slice]
    load_position = position_raw[loading_slice]
    displacement = position_raw[contact] - load_position
    direction_sign = 1.0 if float(np.nanmedian(np.diff(displacement))) >= 0 else -1.0
    displacement *= direction_sign
    displacement -= displacement[0]
    displacement = np.maximum.accumulate(displacement)

    peak_local = int(np.argmax(load_smooth))
    peak_global = contact + peak_local
    peak_force = float(load_smooth[peak_local])
    peak_disp = float(displacement[peak_local])
    max_disp = float(displacement[-1])

    post_window = (displacement >= peak_disp) & (displacement <= peak_disp + 1.0)
    post_min = float(np.min(load_smooth[post_window])) if post_window.any() else peak_force
    peak_drop = max(0.0, peak_force - post_min)

    flesh_mask = (displacement >= peak_disp + 0.75) & (displacement <= max_disp - 0.75)
    if flesh_mask.sum() < 20:
        flesh_mask = (displacement >= max(peak_disp, max_disp * 0.45)) & (displacement <= max_disp * 0.90)
    flesh_force = load_smooth[flesh_mask]
    flesh_disp = displacement[flesh_mask]

    prepeak_mask = (displacement <= peak_disp) & (load_smooth >= 0.2 * peak_force) & (load_smooth <= 0.8 * peak_force)
    prominence = max(10.0, 0.05 * max(peak_force, 1.0))
    peaks, _ = find_peaks(
        load_smooth,
        prominence=prominence,
        distance=max(8, int(round(0.10 * sampling_rate))),
    )

    loading_force_positive = np.maximum(load_smooth, 0.0)
    total_work = float(np.trapezoid(loading_force_positive, displacement))
    skin_work = float(np.trapezoid(loading_force_positive[: peak_local + 1], displacement[: peak_local + 1]))
    post_work = max(0.0, total_work - skin_work)

    unload_t = relative_time[reversal:]
    unload_force = force_corrected[reversal:]
    unload_position = position_raw[reversal:]
    return_displacement = np.abs(unload_position - unload_position[0])
    negative_force = np.maximum(-unload_force, 0.0)
    adhesive_force = float(max(0.0, -np.min(unload_force))) if unload_force.size else math.nan
    adhesive_work = float(np.trapezoid(negative_force, return_displacement)) if unload_force.size > 1 else math.nan

    qc_reasons: list[str] = []
    duration = float(relative_time[-1] - relative_time[0])
    loading_duration = float(load_t[-1] - load_t[0]) if len(load_t) else 0.0
    if duration < 5.0:
        qc_reasons.append("duration_lt_5s")
    if max_disp < 3.0:
        qc_reasons.append("max_displacement_lt_3_rawpos")
    if peak_force < 50.0:
        qc_reasons.append("peak_force_lt_50g")
    if contact >= reversal:
        qc_reasons.append("contact_after_reversal")
    if len(load_t) < 100:
        qc_reasons.append("loading_segment_too_short")
    if not np.all(np.isfinite(force_raw)) or not np.all(np.isfinite(position_raw)):
        qc_reasons.append("nonfinite_values")

    return {
        "curve_qc_status": "exclude" if qc_reasons else "pass",
        "curve_qc_reason": ";".join(qc_reasons),
        "duration_s": duration,
        "sampling_rate_hz": float(sampling_rate),
        "contact_time_s": float(relative_time[contact]),
        "contact_position_raw": float(position_raw[contact]),
        "reversal_time_s": float(relative_time[reversal]),
        "reversal_position_raw": float(position_raw[reversal]),
        "skin_break_time_s": float(relative_time[peak_global]),
        "skin_break_position_raw": float(position_raw[peak_global]),
        "skin_break_force_g": peak_force,
        "skin_break_displacement_raw": peak_disp,
        "skin_break_work_g_rawpos": skin_work,
        "skin_break_drop_g": peak_drop,
        "max_loading_force_g": float(np.max(load_smooth)),
        "force_at_2_rawpos_g": safe_interp(displacement, load_smooth, 2.0),
        "force_at_4_rawpos_g": safe_interp(displacement, load_smooth, 4.0),
        "force_at_6_rawpos_g": safe_interp(displacement, load_smooth, 6.0),
        "flesh_force_mean_g": float(np.mean(flesh_force)) if flesh_force.size else math.nan,
        "flesh_force_median_g": float(np.median(flesh_force)) if flesh_force.size else math.nan,
        "flesh_force_sd_g": float(np.std(flesh_force, ddof=1)) if flesh_force.size > 1 else math.nan,
        "flesh_force_slope_g_per_rawpos": linear_slope(flesh_disp, flesh_force),
        "loading_stiffness_g_per_rawpos": linear_slope(displacement[prepeak_mask], load_smooth[prepeak_mask]),
        "loading_work_g_rawpos": total_work,
        "post_break_work_g_rawpos": post_work,
        "adhesive_force_g": adhesive_force,
        "adhesive_work_g_rawpos": adhesive_work,
        "max_displacement_raw": max_disp,
        "loading_duration_s": loading_duration,
        "loading_speed_rawpos_per_s": max_disp / loading_duration if loading_duration > 0 else math.nan,
        "fracture_peak_count": int(len(peaks)),
        "baseline_force_g": baseline,
        "baseline_noise_g": baseline_noise,
        "force_raw_min": float(np.min(force_raw)),
        "force_raw_max": float(np.max(force_raw)),
        "position_raw_min": float(np.min(position_raw)),
        "position_raw_max": float(np.max(position_raw)),
        "unloading_duration_s": float(unload_t[-1] - unload_t[0]) if len(unload_t) else 0.0,
    }


def process_curve(row: dict[str, Any], dataset_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = dataset_root / Path(row["source_relative_path"])
    raw, is_gzip, gzip_complete = unpack_arc_outer(source.read_bytes())
    chains = find_five_channel_chains(raw)
    if not chains:
        raise ValueError(f"No five-channel chain: {source}")
    channels = chains[0]
    values = {channel.type_id: decode_values(raw, channel).copy() for channel in channels}
    features = extract_features(values[12], values[0], values[1])
    base = {
        "feature_version": FEATURE_VERSION,
        "parser_version": PARSER_VERSION,
        "sample_id": row["sample_id"],
        "batch_id": row["batch_id"],
        "replicate": int(row["replicate"]),
        "source_relative_path": row["source_relative_path"],
        "sha256": row["sha256"],
        "gzip_complete": bool(gzip_complete),
        "source_scan_warning": row.get("warning", "") or "",
        "force_scale_to_g": FORCE_SCALE_TO_G,
        "force_unit": "g",
        "position_unit": "raw_position_unit",
        "point_count": int(len(values[12])),
    }
    feature_row = {**base, **features}
    curve_row = {
        **base,
        "absolute_time_s": values[2].astype(np.float64),
        "relative_time_s": values[12].astype(np.float32),
        "state": values[11].astype(np.int16),
        "force_raw": values[0].astype(np.float32),
        "force_g": (values[0] * FORCE_SCALE_TO_G).astype(np.float32),
        "position_raw": values[1].astype(np.float32),
    }
    return feature_row, curve_row


def curve_schema() -> pa.Schema:
    return pa.schema(
        [
            ("feature_version", pa.string()),
            ("parser_version", pa.string()),
            ("sample_id", pa.string()),
            ("batch_id", pa.string()),
            ("replicate", pa.int16()),
            ("source_relative_path", pa.string()),
            ("sha256", pa.string()),
            ("gzip_complete", pa.bool_()),
            ("source_scan_warning", pa.string()),
            ("force_scale_to_g", pa.float64()),
            ("force_unit", pa.string()),
            ("position_unit", pa.string()),
            ("point_count", pa.int32()),
            ("absolute_time_s", pa.list_(pa.float64())),
            ("relative_time_s", pa.list_(pa.float32())),
            ("state", pa.list_(pa.int16())),
            ("force_raw", pa.list_(pa.float32())),
            ("force_g", pa.list_(pa.float32())),
            ("position_raw", pa.list_(pa.float32())),
        ]
    )


def write_curve_batch(rows: list[dict[str, Any]], output_file: Path) -> None:
    schema = curve_schema()
    arrays = []
    for field in schema:
        values = [row[field.name] for row in rows]
        arrays.append(pa.array(values, type=field.type))
    table = pa.Table.from_arrays(arrays, schema=schema)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_file, compression="zstd", compression_level=6)


def aggregate_samples(curves: pd.DataFrame) -> pd.DataFrame:
    identity = curves[["sample_id", "batch_id"]].drop_duplicates("sample_id").set_index("sample_id")
    result = identity.copy()
    for replicate in (1, 2):
        part = curves[curves["replicate"] == replicate].set_index("sample_id")
        result[f"rep{replicate:02d}_qc_status"] = part["curve_qc_status"]
        result[f"rep{replicate:02d}_qc_reason"] = part["curve_qc_reason"]
        result[f"rep{replicate:02d}_source_warning"] = part["source_scan_warning"]
        for feature in CORE_FEATURES:
            result[f"rep{replicate:02d}_{feature}"] = part[feature]

    valid = curves[curves["curve_qc_status"] == "pass"]
    grouped = valid.groupby("sample_id", sort=False)
    result["texture_valid_replicates"] = grouped.size().reindex(result.index, fill_value=0).astype(int)
    result["texture_sample_qc"] = result["texture_valid_replicates"].map({0: "no_valid", 1: "single_valid", 2: "complete_dual"})
    for feature in CORE_FEATURES:
        stats = grouped[feature].agg(["mean", "std", "min", "max"])
        result[f"{feature}_mean"] = stats["mean"]
        result[f"{feature}_sd"] = stats["std"].fillna(0.0)
        result[f"{feature}_absdiff"] = (stats["max"] - stats["min"]).abs()
        result[f"{feature}_cv"] = result[f"{feature}_sd"] / result[f"{feature}_mean"].abs().replace(0, np.nan)
    return result.reset_index().sort_values(["batch_id", "sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scan-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(args.scan_manifest)
    if not (manifest["parse_status"] == "ok").all():
        raise SystemExit("Scan manifest contains unparsed ARC files")

    all_features: list[dict[str, Any]] = []
    for batch_id, batch in tqdm(list(manifest.groupby("batch_id", sort=True)), desc="Texture batches", unit="batch"):
        records = batch.sort_values(["sample_id", "replicate"]).to_dict(orient="records")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            processed = list(executor.map(lambda row: process_curve(row, dataset_root), records))
        feature_rows = [item[0] for item in processed]
        curve_rows = [item[1] for item in processed]
        all_features.extend(feature_rows)
        write_curve_batch(curve_rows, output_dir / "curves" / f"{batch_id}.parquet")

    curve_features = pd.DataFrame.from_records(all_features).sort_values(["batch_id", "sample_id", "replicate"])
    curve_features.to_parquet(output_dir / "texture_curve_features.parquet", index=False, compression="zstd")
    sample_features = aggregate_samples(curve_features)
    sample_features.to_parquet(output_dir / "texture_sample_features.parquet", index=False, compression="zstd")

    qc_counts = curve_features["curve_qc_status"].value_counts().to_dict()
    sample_qc_counts = sample_features["texture_sample_qc"].value_counts().to_dict()
    unit_report = {
        "force_unit": "g",
        "scientific_force_unit": "gf",
        "force_scale_to_g": FORCE_SCALE_TO_G,
        "raw_force_unit_interpretation": "kgf-equivalent, inferred from original Exponent preview-to-channel ratio",
        "evidence": [
            "All 11004 ARC files contain an original embedded BMP preview; one representative from each of 19 batches was inspected.",
            "All 19 batch representative embedded previews display the y-axis label 'Force (g)'.",
            "Across representative batches, plotted peak magnitudes agree with force_raw multiplied by 1000.",
            "Representative archive metadata records 'Load Cell Capacity (g)' = 30000 and includes gram-based calibration fields.",
            "The displayed g is interpreted as gram-force; 1 kgf = 1000 gf.",
        ],
        "scope_note": "Dataset-specific factor validated against original embedded Exponent previews; not a universal rule for ARC files.",
        "position_unit": "raw_position_unit",
        "position_note": "Position is retained without unit conversion. Its magnitude and motion rate are consistent with millimetres, but no archive text or preview axis independently labels the position channel.",
        "representative_preview_montage": "../interim/arc_scan/representative_previews/all_batches_montage.png",
        "representative_peak_table": "force_scale_representative_evidence.csv",
        "review_evidence_note": "../../../review_package/09_AUTHOR_CONFIRMATION_AND_FORCE_UNIT_EVIDENCE_ZH.md",
    }
    (output_dir / "texture_unit_validation.json").write_text(json.dumps(unit_report, indent=2), encoding="utf-8")
    summary = {
        "feature_version": FEATURE_VERSION,
        "curve_files": int(len(curve_features)),
        "samples": int(sample_features["sample_id"].nunique()),
        "curve_qc": {str(k): int(v) for k, v in qc_counts.items()},
        "sample_qc": {str(k): int(v) for k, v in sample_qc_counts.items()},
        "source_warnings": curve_features.loc[curve_features["source_scan_warning"].ne(""), ["sample_id", "replicate", "source_scan_warning"]].to_dict(orient="records"),
        "excluded_curves": curve_features.loc[curve_features["curve_qc_status"].ne("pass"), ["sample_id", "replicate", "curve_qc_reason"]].to_dict(orient="records"),
        "core_features": CORE_FEATURES,
    }
    (output_dir / "texture_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
