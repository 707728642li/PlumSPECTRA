from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


GENERATED = {"MANIFEST.json", "SHA256SUMS.tsv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_files(package: Path) -> list[Path]:
    return sorted(
        (path for path in package.rglob("*") if path.is_file() and path.name not in GENERATED),
        key=lambda path: path.relative_to(package).as_posix(),
    )


def refresh(package: Path) -> dict[str, object]:
    audit = json.loads((package / "v26_integration_audit.json").read_text(encoding="utf-8"))
    records = [
        {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "path": path.relative_to(package).as_posix(),
        }
        for path in payload_files(package)
    ]
    manifest = {
        "release": "V26 integrated visual revision",
        "created": "2026-08-12",
        "target_journal": "Horticulture Research",
        "frozen_baseline": "HR_EXTERNAL_AUDIT_PACKAGE_V25_FINAL_20260810",
        "file_count": len(records),
        "total_bytes": sum(int(item["bytes"]) for item in records),
        "automated_checks": f"{audit['checks_passed']}/{audit['checks_total']} PASS",
        "abstract_words": 246,
        "files": records,
    }
    (package / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (package / "SHA256SUMS.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_ALL)
        writer.writerow(["sha256", "bytes", "path"])
        writer.writerows(
            (item["sha256"], item["bytes"], item["path"]) for item in records
        )
    return manifest


def verify(package: Path, archive: Path | None) -> None:
    manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
    records = manifest["files"]
    missing = [item["path"] for item in records if not (package / item["path"]).is_file()]
    bad_hashes = [
        item["path"]
        for item in records
        if (package / item["path"]).is_file()
        and sha256(package / item["path"]) != item["sha256"]
    ]
    actual = payload_files(package)
    recorded = {item["path"] for item in records}
    unrecorded = [
        path.relative_to(package).as_posix()
        for path in actual
        if path.relative_to(package).as_posix() not in recorded
    ]
    audit = json.loads((package / "v26_integration_audit.json").read_text(encoding="utf-8"))
    assertions = {
        "manifest count": int(manifest["file_count"]) == len(records) == len(actual),
        "no missing files": not missing,
        "no unrecorded files": not unrecorded,
        "all payload hashes match": not bad_hashes,
        "158/158 visual and scientific checks":
            audit["all_passed"] and audit["checks_passed"] == audit["checks_total"] == 158,
        "six main PNG figures": len(list((package / "main_figures").glob("*.png"))) == 6,
        "24 supplementary PNG figures":
            len(list((package / "supplementary_figures").glob("figS*.png"))) == 24,
        "supplementary rendering script included":
            (package / "scripts" / "render_v22_integrated_figures.R").is_file(),
    }
    if archive is not None:
        with zipfile.ZipFile(archive) as zipped:
            assertions["ZIP CRC test"] = zipped.testzip() is None
            assertions["ZIP contains package manifest"] = any(
                name.endswith("/MANIFEST.json") for name in zipped.namelist()
            )
    failures = [name for name, passed in assertions.items() if not passed]
    if failures:
        raise AssertionError(
            f"package verification failed: {failures}; missing={missing}; "
            f"unrecorded={unrecorded}; bad_hashes={bad_hashes}"
        )
    print(
        f"PASS package files={len(actual)} bytes={manifest['total_bytes']} "
        f"checks={audit['checks_passed']}/{audit['checks_total']} "
        f"zip={'verified' if archive else 'not requested'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    if args.refresh:
        refresh(package)
    verify(package, args.archive.resolve() if args.archive else None)


if __name__ == "__main__":
    main()
