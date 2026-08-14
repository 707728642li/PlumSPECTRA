from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


def hash_file(item: tuple[Path, str]) -> tuple[str, str, bool, str]:
    path, expected = item
    if not path.is_file():
        return path.as_posix(), expected, False, "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    observed = digest.hexdigest()
    return path.as_posix(), expected, observed == expected, observed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.dataset.resolve()

    checksum_path = root / "quality_control" / "checksums.sha256"
    entries: list[tuple[Path, str]] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        entries.append((root / Path(relative), expected))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        checks = list(executor.map(hash_file, entries))
    failures = [
        {"path": path, "expected": expected, "observed_or_status": observed}
        for path, expected, passed, observed in checks
        if not passed
    ]

    samples = pd.read_csv(root / "metadata" / "samples.csv")
    files = pd.read_csv(root / "metadata" / "files.csv")
    nir = np.load(root / "data" / "processed" / "nir_c_absorbance.npy", mmap_mode="r")
    wavelength = np.load(root / "data" / "processed" / "wavelength_nm.npy")
    index = pd.read_csv(root / "data" / "processed" / "nir_c_sample_index.csv")
    id_pattern = re.compile(r"^plum-[a-z0-9-]+-b\d{2}-f\d{4}$")
    invalid_ids = samples.loc[~samples["sample_id"].astype(str).map(lambda value: bool(id_pattern.fullmatch(value))), "sample_id"].tolist()
    required_counts = files.assign(
        key=[
            f"texture_{int(float(replicate))}" if data_type == "texture" else f"nir_{measurement}_{Path(path).suffix[1:]}"
            for data_type, replicate, measurement, path in zip(files["data_type"], files["replicate"], files["measurement"], files["target_relative_path"])
        ]
    ).groupby("sample_id")["key"].agg(set)
    required = {"nir_c_csv", "nir_c_dat", "texture_1", "texture_2"}
    incomplete = {sample_id: sorted(required - present) for sample_id, present in required_counts.items() if not required <= present}

    report = {
        "status": "PASS" if not failures and not invalid_ids and not incomplete else "FAIL",
        "checksum_entries_verified": len(entries),
        "checksum_failures": failures,
        "samples": len(samples),
        "unique_samples": int(samples["sample_id"].nunique()),
        "cultivars": int(samples["cultivar_ascii"].nunique()),
        "batches": int(samples["batch_id"].nunique()),
        "measurement_files": len(files),
        "invalid_canonical_ids": invalid_ids,
        "incomplete_samples": incomplete,
        "nir_matrix_shape": list(nir.shape),
        "nir_index_rows": len(index),
        "wavelength_points": len(wavelength),
        "finite_nir_fraction": float(np.isfinite(nir).mean()),
        "all_three_targets_finite": bool(samples[["fruit_weight_g", "soluble_solids_pct", "ph"]].notna().all().all()),
    }
    (root / "quality_control" / "independent_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
