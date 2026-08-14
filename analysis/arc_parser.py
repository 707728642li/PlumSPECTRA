from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


PARSER_VERSION = "0.1.0"
HEADER_SIZE = 78
EXPECTED_TYPES = (2, 12, 11, 0, 1)
TYPE_NAMES = {
    2: "absolute_time_s",
    12: "relative_time_s",
    11: "state",
    0: "force_raw",
    1: "position_raw",
}
NAME_PATTERN = re.compile(
    r"^(?P<sample_id>[A-Z0-9_]+-\d{4})_texture_rep(?P<replicate>\d{2})\.arc$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Channel:
    type_id: int
    header_offset: int
    data_offset: int
    count: int
    stored_max: float
    stored_min: float


def unpack_arc_outer(compressed: bytes) -> tuple[bytes, bool, bool]:
    is_gzip = compressed[:3] == b"\x1f\x8b\x08"
    if not is_gzip:
        return compressed, False, True
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = decoder.decompress(compressed)
    try:
        raw += decoder.flush()
    except zlib.error:
        pass
    return raw, True, bool(decoder.eof)


def bmp_payload(raw: bytes) -> bytes | None:
    if raw[:2] != b"BM" or len(raw) < 6:
        return None
    bmp_size = struct.unpack_from("<I", raw, 2)[0]
    if bmp_size < 14 or bmp_size > len(raw):
        return None
    return raw[:bmp_size]


def read_header(raw: bytes, offset: int) -> Channel | None:
    if offset < 0 or offset + HEADER_SIZE > len(raw):
        return None
    type_id = struct.unpack_from("<H", raw, offset + 22)[0]
    count = struct.unpack_from("<I", raw, offset + 24)[0]
    stored_max = struct.unpack_from("<d", raw, offset + 28)[0]
    stored_min = struct.unpack_from("<d", raw, offset + 36)[0]
    data_offset = offset + HEADER_SIZE
    data_end = data_offset + 8 * count
    if not 2 <= count <= 20_000_000:
        return None
    if data_end > len(raw):
        return None
    if not (math.isfinite(stored_min) and math.isfinite(stored_max)):
        return None
    if stored_min > stored_max:
        return None
    return Channel(type_id, offset, data_offset, count, stored_max, stored_min)


def try_five_channel_chain(raw: bytes, first_offset: int) -> list[Channel] | None:
    channels: list[Channel] = []
    offset = first_offset
    common_count: int | None = None
    for expected_type in EXPECTED_TYPES:
        channel = read_header(raw, offset)
        if channel is None or channel.type_id != expected_type:
            return None
        if common_count is None:
            common_count = channel.count
        elif channel.count != common_count:
            return None
        channels.append(channel)
        offset = channel.data_offset + 8 * channel.count
    return channels


def find_five_channel_chains(raw: bytes) -> list[list[Channel]]:
    needle = struct.pack("<H", EXPECTED_TYPES[0])
    search_from = 0
    chains: list[list[Channel]] = []
    seen_offsets: set[int] = set()
    while True:
        hit = raw.find(needle, search_from)
        if hit < 0:
            break
        first_offset = hit - 22
        if first_offset not in seen_offsets:
            chain = try_five_channel_chain(raw, first_offset)
            if chain is not None:
                chains.append(chain)
                seen_offsets.add(first_offset)
        search_from = hit + 1
    return chains


def decode_values(raw: bytes, channel: Channel) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f8", count=channel.count, offset=channel.data_offset)


def extrema_match(values: np.ndarray, channel: Channel) -> bool:
    return bool(
        np.isclose(np.min(values), channel.stored_min, rtol=1e-7, atol=1e-10)
        and np.isclose(np.max(values), channel.stored_max, rtol=1e-7, atol=1e-10)
    )


def extract_relevant_text(raw: bytes, end_offset: int) -> list[str]:
    region = raw[:end_offset]
    strings: list[str] = []
    for match in re.finditer(rb"[\x20-\x7e]{5,}", region):
        strings.append(match.group().decode("ascii", errors="ignore").strip())
    try:
        utf16 = region.decode("utf-16le", errors="ignore")
        strings.extend(s.strip() for s in re.findall(r"[\x20-\x7e]{5,}", utf16))
    except UnicodeError:
        pass
    cleaned_strings = [" ".join(value.split())[:500] for value in strings if value.strip()]
    priority_patterns = [
        r"TEE32 Archive",
        r"Created with Exponent Version",
        r"Force\s*\([^)]*\)|gram.?force|\bgf\b|Newton|\bkgf\b",
        r"commence the .*test|penetration test|compression test",
        r"^.?P/\d[^;]{0,12};.{0,120}",
        r"Motor Steps / mm",
        r"System Gain \(Filtered Force\)",
        r"Firmware Ver\. \(Comms\)",
    ]
    selected: list[str] = []
    seen: set[str] = set()
    for pattern in priority_patterns:
        matcher = re.compile(pattern, re.IGNORECASE)
        for cleaned in cleaned_strings:
            if matcher.search(cleaned) and cleaned not in seen:
                selected.append(cleaned[:240])
                seen.add(cleaned)
                break
    return selected


def infer_force_unit(text: list[str]) -> tuple[str, str]:
    joined = " | ".join(text)
    patterns = [
        (r"Force\s*\(\s*g\s*\)|gram.?force|\bgf\b", "gf"),
        (r"Force\s*\(\s*N\s*\)|Newton", "N"),
        (r"Force\s*\(\s*kg\s*\)|\bkgf\b", "kgf"),
    ]
    hits = [unit for pattern, unit in patterns if re.search(pattern, joined, re.IGNORECASE)]
    if len(set(hits)) == 1:
        return hits[0], "archive_text"
    if len(set(hits)) > 1:
        return "ambiguous", "conflicting_archive_text"
    return "unconfirmed", "preview_or_external_confirmation_required"


def parse_identity(file: Path) -> tuple[str, str, int]:
    match = NAME_PATTERN.match(file.name)
    if match is None:
        raise ValueError(f"Non-standard ARC filename: {file.name}")
    sample_id = match.group("sample_id").upper()
    batch_id = sample_id.rsplit("-", 1)[0]
    return sample_id, batch_id, int(match.group("replicate"))


def scan_arc(file: Path, dataset_root: Path) -> dict[str, Any]:
    sample_id, batch_id, replicate = parse_identity(file)
    base: dict[str, Any] = {
        "parser_version": PARSER_VERSION,
        "sample_id": sample_id,
        "batch_id": batch_id,
        "replicate": replicate,
        "source_relative_path": file.relative_to(dataset_root).as_posix(),
        "source_bytes": file.stat().st_size,
        "parse_status": "error",
        "warning": "",
    }
    warnings: list[str] = []
    try:
        compressed = file.read_bytes()
        base["sha256"] = hashlib.sha256(compressed).hexdigest()
        raw, is_gzip, gzip_complete = unpack_arc_outer(compressed)
        base["gzip_outer"] = is_gzip
        base["gzip_complete"] = gzip_complete
        if is_gzip and not gzip_complete:
            warnings.append("gzip_truncated_salvaged")
        base["decompressed_bytes"] = len(raw)
        preview = bmp_payload(raw)
        base["bmp_preview"] = preview is not None
        base["bmp_bytes"] = len(preview) if preview is not None else 0
        chains = find_five_channel_chains(raw)
        base["candidate_chain_count"] = len(chains)
        if not chains:
            raise ValueError("five-channel chain 2->12->11->0->1 not found")
        if len(chains) > 1:
            warnings.append(f"multiple_valid_chains={len(chains)}")
        channels = chains[0]
        first_offset = channels[0].header_offset
        values = {ch.type_id: decode_values(raw, ch) for ch in channels}
        extrema = {ch.type_id: extrema_match(values[ch.type_id], ch) for ch in channels}
        rel_time = values[12]
        state = values[11]
        force = values[0]
        position = values[1]
        dt = np.diff(rel_time)
        positive_dt = dt[dt > 0]
        median_dt = float(np.median(positive_dt)) if positive_dt.size else math.nan
        dt_mad = float(np.median(np.abs(positive_dt - median_dt))) if positive_dt.size else math.nan
        sample_rate = 1.0 / median_dt if median_dt > 0 else math.nan
        force_argmax = int(np.argmax(force))
        force_argmin = int(np.argmin(force))
        unique_state = np.unique(state)
        metadata_text = extract_relevant_text(raw, first_offset)
        force_unit, unit_evidence = infer_force_unit(metadata_text)
        base.update(
            {
                "parse_status": "ok",
                "first_channel_offset": first_offset,
                "channel_types": "2;12;11;0;1",
                "point_count": channels[0].count,
                "duration_s": float(rel_time[-1] - rel_time[0]),
                "relative_time_start_s": float(rel_time[0]),
                "relative_time_end_s": float(rel_time[-1]),
                "relative_time_monotonic": bool(np.all(dt >= -1e-12)),
                "median_dt_s": median_dt,
                "dt_mad_s": dt_mad,
                "sampling_rate_hz": sample_rate,
                "sampling_interval_stable": bool(median_dt > 0 and dt_mad / median_dt < 0.01),
                "state_unique_count": int(unique_state.size),
                "state_values": ";".join(f"{x:.12g}" for x in unique_state[:50]),
                "force_raw_min": float(force[force_argmin]),
                "force_raw_max": float(force[force_argmax]),
                "force_raw_absmax": float(np.max(np.abs(force))),
                "force_raw_max_time_s": float(rel_time[force_argmax]),
                "force_raw_min_time_s": float(rel_time[force_argmin]),
                "position_raw_min": float(np.min(position)),
                "position_raw_max": float(np.max(position)),
                "position_raw_start": float(position[0]),
                "position_raw_end": float(position[-1]),
                "force_unit": force_unit,
                "force_unit_evidence": unit_evidence,
                "metadata_markers": " | ".join(metadata_text),
                "all_channel_extrema_match": bool(all(extrema.values())),
                "critical_channel_extrema_match": bool(all(extrema[t] for t in (12, 0, 1))),
                "stored_extrema_match": ";".join(f"{TYPE_NAMES[t]}={str(extrema[t]).lower()}" for t in EXPECTED_TYPES),
                "structure_signature": f"gzip={int(is_gzip)}|bmp={int(preview is not None)}|types=2-12-11-0-1|header=78|offset={first_offset}",
            }
        )
        if not base["relative_time_monotonic"]:
            warnings.append("relative_time_not_monotonic")
        if not base["sampling_interval_stable"]:
            warnings.append("sampling_interval_unstable")
        if not base["critical_channel_extrema_match"]:
            warnings.append("critical_stored_extrema_mismatch")
        if abs(base["relative_time_start_s"]) > max(0.1, 2 * median_dt if math.isfinite(median_dt) else 0.1):
            warnings.append("relative_time_not_near_zero")
        if unique_state.size > 100:
            warnings.append("state_not_discrete")
    except Exception as exc:  # one record per file is required even on failure
        warnings.append(f"{type(exc).__name__}: {exc}")
    base["warning"] = " | ".join(warnings)
    return base


def extract_representative_previews(
    manifest: pd.DataFrame, dataset_root: Path, preview_dir: Path
) -> dict[str, dict[str, Any]]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    ok = manifest[(manifest["parse_status"] == "ok") & manifest["bmp_preview"]].copy()
    ok = ok.sort_values(["batch_id", "sample_id", "replicate"])
    for batch_id, group in ok.groupby("batch_id", sort=True):
        row = group.iloc[0]
        source = dataset_root / Path(row["source_relative_path"])
        raw, _, _ = unpack_arc_outer(source.read_bytes())
        preview = bmp_payload(raw)
        if preview is None:
            continue
        target = preview_dir / f"{batch_id}__{row['sample_id']}__rep{int(row['replicate']):02d}.bmp"
        target.write_bytes(preview)
        records[batch_id] = {
            "sample_id": row["sample_id"],
            "replicate": int(row["replicate"]),
            "source_relative_path": row["source_relative_path"],
            "preview_relative_path": target.relative_to(preview_dir.parent.parent).as_posix(),
        }
    return records


def value_counts_dict(series: pd.Series, limit: int | None = None) -> dict[str, int]:
    counts = series.astype(str).value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def build_summary(manifest: pd.DataFrame, previews: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = manifest[manifest["parse_status"] == "ok"]
    numeric_summary: dict[str, Any] = {}
    for column in ["point_count", "duration_s", "sampling_rate_hz", "force_raw_min", "force_raw_max", "position_raw_min", "position_raw_max"]:
        values = pd.to_numeric(ok[column], errors="coerce").dropna()
        numeric_summary[column] = {
            "min": float(values.min()),
            "q01": float(values.quantile(0.01)),
            "median": float(values.median()),
            "q99": float(values.quantile(0.99)),
            "max": float(values.max()),
        }
    per_batch = []
    for batch_id, group in manifest.groupby("batch_id", sort=True):
        group_ok = group[group["parse_status"] == "ok"]
        per_batch.append(
            {
                "batch_id": batch_id,
                "files": int(len(group)),
                "ok": int(len(group_ok)),
                "errors": int((group["parse_status"] != "ok").sum()),
                "samples": int(group["sample_id"].nunique()),
                "replicates": sorted(int(x) for x in group["replicate"].dropna().unique()),
                "point_count_unique": int(group_ok["point_count"].nunique()),
                "point_count_median": float(pd.to_numeric(group_ok["point_count"], errors="coerce").median()),
                "point_counts_top10": value_counts_dict(group_ok["point_count"], limit=10),
                "sampling_rate_hz_median": float(pd.to_numeric(group_ok["sampling_rate_hz"], errors="coerce").median()),
                "force_unit": value_counts_dict(group_ok["force_unit"]),
                "warning_files": int((group["warning"].fillna("") != "").sum()),
            }
        )
    return {
        "parser_version": PARSER_VERSION,
        "files_total": int(len(manifest)),
        "files_ok": int(len(ok)),
        "files_error": int((manifest["parse_status"] != "ok").sum()),
        "samples": int(manifest["sample_id"].nunique()),
        "batches": int(manifest["batch_id"].nunique()),
        "gzip_outer": value_counts_dict(manifest["gzip_outer"]),
        "bmp_preview": value_counts_dict(manifest["bmp_preview"]),
        "candidate_chain_count": value_counts_dict(manifest["candidate_chain_count"]),
        "structure_signatures": value_counts_dict(manifest["structure_signature"], limit=100),
        "force_units": value_counts_dict(ok["force_unit"]),
        "warning_records": int((manifest["warning"].fillna("") != "").sum()),
        "numeric_summary": numeric_summary,
        "per_batch": per_batch,
        "representative_previews": previews,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Exponent ARC texture archives without exporting curves.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    texture_root = dataset_root / "data" / "texture"
    files = sorted(texture_root.rglob("*.arc"), key=lambda p: p.as_posix().lower())
    if not files:
        raise SystemExit(f"No ARC files found under {texture_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(
            tqdm(
                executor.map(lambda p: scan_arc(p, dataset_root), files),
                total=len(files),
                desc="Scanning ARC",
                unit="file",
            )
        )
    manifest = pd.DataFrame.from_records(records).sort_values(["batch_id", "sample_id", "replicate"])
    manifest_path = output_dir / "arc_scan_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False, compression="zstd")
    failures = manifest[manifest["parse_status"] != "ok"].to_dict(orient="records")
    with (output_dir / "arc_scan_failures.jsonl").open("w", encoding="utf-8") as handle:
        for record in failures:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    previews = extract_representative_previews(
        manifest, dataset_root, output_dir / "representative_previews" / "bmp"
    )
    summary = build_summary(manifest, previews)
    (output_dir / "arc_scan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "files_total": summary["files_total"],
        "files_ok": summary["files_ok"],
        "files_error": summary["files_error"],
        "samples": summary["samples"],
        "batches": summary["batches"],
        "warning_records": summary["warning_records"],
        "force_units": summary["force_units"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
