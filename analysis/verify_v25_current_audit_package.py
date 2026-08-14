#!/usr/bin/env python3
"""Verify the frozen files in a PlumSPECTRA V25 current-snapshot audit package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "review_package" / "HR_EXTERNAL_AUDIT_PACKAGE_V25_CURRENT_20260810"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", nargs="?", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    package = args.package.resolve()
    manifest_path = package / "MANIFEST_SHA256.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))

    failures: list[dict[str, object]] = []
    tracked = {manifest_path.resolve()}
    for row in manifest:
        path = (package / row["relative_path"]).resolve()
        tracked.add(path)
        observed = {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256(path) if path.is_file() else None,
        }
        expected = {"exists": True, "bytes": int(row["bytes"]), "sha256": row["sha256"]}
        if observed != expected:
            failures.append({"path": row["relative_path"], "observed": observed, "expected": expected})

    self_check_path = package / "AUDIT_PACKAGE_SELF_CHECK.json"
    if not self_check_path.is_file():
        failures.append({"path": "AUDIT_PACKAGE_SELF_CHECK.json", "observed": "missing", "expected": "PASS"})
    else:
        self_check = json.loads(self_check_path.read_text(encoding="utf-8"))
        if self_check.get("status") != "PASS":
            failures.append({"path": "AUDIT_PACKAGE_SELF_CHECK.json", "observed": self_check.get("status"), "expected": "PASS"})

    instruction_path = package / "06_EXTERNAL_REVIEW_INSTRUCTION_ZH.md"
    if not instruction_path.is_file():
        failures.append({"path": instruction_path.name, "observed": "missing", "expected": "present"})
    else:
        instruction = instruction_path.read_text(encoding="utf-8")
        for suffix in ["_codex.md", "_claudecode.md"]:
            if suffix not in instruction:
                failures.append({"path": instruction_path.name, "observed": f"missing {suffix}", "expected": "present"})

    untracked = []
    for path in package.rglob("*"):
        if not path.is_file() or path.resolve() in tracked:
            continue
        if path.name.endswith("_codex.md") or path.name.endswith("_claudecode.md"):
            continue
        untracked.append(path.relative_to(package).as_posix())

    result = {
        "status": "PASS" if not failures and not untracked else "FAIL",
        "package": str(package),
        "manifest_sha256": sha256(manifest_path),
        "tracked_files": len(manifest),
        "failures": failures,
        "untracked_nonreview_files": sorted(untracked),
        "note": "Reviewer reports ending in _codex.md or _claudecode.md are intentionally allowed outside the frozen manifest.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

