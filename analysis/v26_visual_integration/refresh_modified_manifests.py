"""Refresh checksums for files intentionally updated in the v26 release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> tuple[int, str]:
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def update_json_manifest(manifest_path: Path, updates: dict[str, Path]) -> None:
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = doc if isinstance(doc, list) else doc.get("files")
    if entries is None:
        raise KeyError(f"No file entries found in {manifest_path}")
    indexed = {
        entry.get("path", entry.get("file")): entry
        for entry in entries
        if isinstance(entry, dict)
    }
    for name, source in updates.items():
        if name not in indexed:
            raise KeyError(f"{name} not present in {manifest_path}")
        size, sha = digest(source)
        indexed[name]["bytes"] = size
        indexed[name]["sha256"] = sha
    manifest_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    audit = ROOT / "review_package/HR_EXTERNAL_AUDIT_PACKAGE_V26_INTEGRATED_20260812"
    update_json_manifest(
        audit / "MANIFEST.json",
        {
            "main_figures/Figure_6_v26.pdf": audit / "main_figures/Figure_6_v26.pdf",
            "main_figures/Figure_6_v26.png": audit / "main_figures/Figure_6_v26.png",
            "scripts/v26_visual_integration/plot_figure6_v26.R": (
                audit / "scripts/v26_visual_integration/plot_figure6_v26.R"
            ),
            "scripts/v26_visual_integration/prepare_figure_data.py": (
                audit / "scripts/v26_visual_integration/prepare_figure_data.py"
            ),
        },
    )

    submission = ROOT / "manuscript/submission_package"
    update_json_manifest(
        submission / "submission_manifest.json",
        {"Figure_6.pdf": submission / "Figure_6.pdf"},
    )
    checksum_path = submission / "SHA256SUMS.txt"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    _, fig6_sha = digest(submission / "Figure_6.pdf")
    replaced = False
    for index, line in enumerate(lines):
        if line.endswith("  Figure_6.pdf"):
            lines[index] = f"{fig6_sha}  Figure_6.pdf"
            replaced = True
    if not replaced:
        raise KeyError("Figure_6.pdf not present in SHA256SUMS.txt")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("refreshed v26 audit and submission manifests for Figure 6")


if __name__ == "__main__":
    main()
