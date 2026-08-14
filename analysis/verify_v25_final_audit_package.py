#!/usr/bin/env python3
"""Verify provenance and scientific invariants of a frozen V25 audit package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - handled with an actionable error below
    PdfReader = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_first_page_width_inches(path: Path) -> float:
    """Read the first-page PDF MediaBox and return its width in inches."""

    if PdfReader is not None:
        page = PdfReader(str(path)).pages[0]
        return float(page.mediabox.width) / 72.0
    # Cairo-generated production figures expose MediaBox in an uncompressed
    # page dictionary.  This dependency-free fallback keeps package audit
    # usable in the modelling environment, where pyarrow is present but
    # pypdf may not be.
    candidates = [
        str(
            Path.home()
            / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdfinfo.exe"
        ),
        shutil.which("pdfinfo"),
    ]
    for executable in candidates:
        if executable and Path(executable).is_file():
            try:
                result = subprocess.run(
                    [executable, str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, subprocess.CalledProcessError):
                continue
            match = re.search(r"^Page size:\s+([0-9.]+)\s+x\s+", result.stdout, flags=re.MULTILINE)
            if match is not None:
                return float(match.group(1)) / 72.0
    raise RuntimeError(f"Could not read PDF page size without pypdf or pdfinfo: {path}")


def pdf_font_names(path: Path) -> list[str]:
    """Return unique BaseFont names registered anywhere in a PDF."""

    if PdfReader is None:
        raw_fonts = re.findall(rb"/BaseFont\s*/([^\s/<>{}\[\]()]+)", path.read_bytes())
        return sorted({"/" + value.decode("latin-1") for value in raw_fonts})
    fonts: set[str] = set()
    for page in PdfReader(str(path)).pages:
        resources = page.get("/Resources")
        if resources is None or "/Font" not in resources:
            continue
        for font in resources["/Font"].get_object().values():
            fonts.add(str(font.get_object().get("/BaseFont", "")))
    return sorted(fonts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "package",
        type=Path,
        nargs="?",
        default=Path("review_package/HR_EXTERNAL_AUDIT_PACKAGE_V25_FINAL_20260810"),
    )
    args = parser.parse_args()
    package = args.package.resolve()
    manifest_path = package / "MANIFEST_SHA256.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, expected: Any, detail: str = "") -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if observed == expected else "FAIL",
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    missing: list[str] = []
    mismatched: list[str] = []
    for row in manifest:
        path = package / row["relative_path"]
        if not path.is_file():
            missing.append(row["relative_path"])
        elif sha256(path) != row["sha256"]:
            mismatched.append(row["relative_path"])
    check("manifest missing files", len(missing), 0, "; ".join(missing))
    check("manifest hash mismatches", len(mismatched), 0, "; ".join(mismatched))

    self_check = json.loads((package / "AUDIT_PACKAGE_SELF_CHECK.json").read_text(encoding="utf-8"))
    release = json.loads(
        (package / "evidence/release_audit/v25_final_release_audit.json").read_text(encoding="utf-8")
    )
    check("package self-check", self_check["status"], "PASS")
    check("scientific release audit", release["status"], "PASS")
    check("failed scientific release checks", int(release["failed_checks"]), 0)

    integrated = pd.read_parquet(package / "evidence/final_analysis/v25_integrated_predictions.parquet")
    check("integrated rows", len(integrated), 58206)
    check("integrated fruit-trait duplicates", int(integrated.duplicated(["sample_id", "trait"]).sum()), 0)
    check("integrated traits", integrated["trait"].nunique(), 12)
    check("integrated cultivars", integrated["cultivar_ascii"].nunique(), 15)
    check(
        "texture fruit count",
        integrated.loc[integrated["trait"].isin(["SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"]), "sample_id"].nunique(),
        4853,
    )
    check(
        "conventional fruit count",
        integrated.loc[integrated["trait"].isin(["FW", "SSC", "pH"]), "sample_id"].nunique(),
        4843,
    )

    multiseed = pd.read_csv(package / "evidence/multiseed/multiseed_summary.csv")
    check("multiseed rows", len(multiseed), 24)
    check("multiseed traits", multiseed["trait"].nunique(), 12)
    check("multiseed pipeline instances", sorted(multiseed["pipeline_instances"].unique().tolist()), [15])
    multiseed_manifest = json.loads(
        (package / "evidence/multiseed/run_manifest.json").read_text(encoding="utf-8")
    )
    multiseed_failures = multiseed_manifest.get(
        "failed_jobs", multiseed_manifest.get("failures", [])
    )
    check("multiseed launcher failures", len(multiseed_failures), 0)

    cross = pd.read_parquet(package / "evidence/crossbatch/v21_merged_predictions.parquet")
    check("crossbatch rows", len(cross), 11304)
    check("crossbatch batches", cross["batch_id"].nunique(), 5)
    check("crossbatch cultivars", cross["cultivar_ascii"].nunique(), 2)
    check("crossbatch traits", cross["trait"].nunique(), 9)
    cross_ai_manifest = json.loads(
        (package / "evidence/crossbatch/ai_run_manifest.json").read_text(encoding="utf-8")
    )
    cross_ai_jobs = cross_ai_manifest.get("jobs", [])
    check("crossbatch AI launcher jobs", len(cross_ai_jobs), 45)
    check(
        "crossbatch AI incomplete launcher jobs",
        sum(job.get("status") != "completed" for job in cross_ai_jobs),
        0,
    )

    loco = pd.read_parquet(package / "evidence/loco/loco_predictions.parquet")
    check("LOCO rows", len(loco), 43677)
    check("LOCO cultivars", loco["cultivar_ascii"].nunique(), 15)
    check("LOCO targets", loco["target"].nunique(), 9)

    hyper = pd.read_csv(package / "evidence/release_audit/formal_fold_hyperparameters.csv")
    check("formal hyperparameter rows", len(hyper), 60)
    check("PLSR upper-bound hits", int(hyper["domain_pls_component_upper_boundary"].sum()), 0)
    check("SVR C-bound hits", int(hyper["svr_C_boundary"].sum()), 0)
    check("SVR gamma-bound hits", int(hyper["svr_gamma_boundary"].sum()), 0)
    check("SVR epsilon-bound hits", int(hyper["svr_epsilon_boundary"].sum()), 0)

    fold_protocol = package / "evidence/fold_protocol"
    for label, relative, expected in (
        ("texture baseline", "baselines_texture", 45),
        ("quality baseline", "baselines_quality", 15),
        ("texture AI", "ai_texture", 45),
        ("quality AI", "ai_quality", 15),
        ("crossbatch baseline", "crossbatch_baselines", 45),
        ("crossbatch AI", "crossbatch_ai", 45),
    ):
        check(
            f"{label} packaged metadata files",
            len(list((fold_protocol / relative).glob("*/fold_*/metadata.json"))),
            expected,
        )
    for label, relative, expected in (
        ("texture AI", "ai_texture", 45),
        ("quality AI", "ai_quality", 15),
        ("crossbatch AI", "crossbatch_ai", 45),
    ):
        check(
            f"{label} packaged selection histories",
            len(list((fold_protocol / relative).glob("*/fold_*/selection_history.csv"))),
            expected,
        )
        check(
            f"{label} packaged retrain histories",
            len(list((fold_protocol / relative).glob("*/fold_*/retrain_history.csv"))),
            expected,
        )

    instruction = (package / "06_EXTERNAL_REVIEW_INSTRUCTION_ZH.md").read_text(encoding="utf-8")
    check("Codex report suffix instruction", "_codex.md" in instruction, True)
    check("Claude Code report suffix instruction", "_claudecode.md" in instruction, True)
    check("final package path instruction", "V25_FINAL_20260810" in instruction, True)

    manuscript = (package / "documents/Manuscript_V25_source.md").read_text(encoding="utf-8")
    check("unresolved manuscript placeholders", "{{" in manuscript or "}}" in manuscript, False)
    word_pattern = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9'–-]*\b")
    manuscript_words = len(word_pattern.findall(manuscript))
    check("manuscript word tokens within 6000", manuscript_words <= 6000, True, str(manuscript_words))
    abstract_match = re.search(r"(?ms)^## Abstract\s+(.*?)(?=^## )", manuscript)
    abstract_words = len(word_pattern.findall(abstract_match.group(1))) if abstract_match else -1
    check("abstract found", abstract_match is not None, True)
    check("abstract word tokens within 250", 0 <= abstract_words <= 250, True, str(abstract_words))
    main_figure_legends = len(re.findall(r"(?m)^### Figure \d+\.", manuscript))
    check("main figure legend count", main_figure_legends, 6)
    reference_numbers = {
        int(value) for value in re.findall(r"(?m)^(\d+)\.\s+", manuscript)
    }
    cited_numbers: set[int] = set()
    for citation in re.findall(r"\[((?:\d+\s*(?:[,–-]\s*)?)+)\]", manuscript):
        cited_numbers.update(int(value) for value in re.findall(r"\d+", citation))
    check("reference count within 50", len(reference_numbers) <= 50, True, str(len(reference_numbers)))
    check(
        "all bibliography entries cited",
        sorted(reference_numbers - cited_numbers),
        [],
        f"cited={sorted(cited_numbers)}",
    )
    check(
        "all citation numbers resolve",
        sorted(cited_numbers - reference_numbers),
        [],
        f"references={sorted(reference_numbers)}",
    )

    figure_dir = package / "main_figures"
    check("main production files", len(list(figure_dir.glob("*"))), 7)
    widths: dict[str, float] = {}
    figure_fonts: dict[str, list[str]] = {}
    for filename, expected_width in (
        ("Figure_1B_unmerged.pdf", 6.5),
        ("Figure_2.pdf", 10.5),
        ("Figure_3.pdf", 10.5),
        ("Figure_4.pdf", 10.5),
        ("Figure_5.pdf", 10.5),
        ("Figure_6.pdf", 10.5),
    ):
        width = pdf_first_page_width_inches(figure_dir / filename)
        widths[filename] = width
        check(f"{filename} width inches", round(width, 3), expected_width)
        fonts = pdf_font_names(figure_dir / filename)
        figure_fonts[filename] = fonts
        check(
            f"{filename} uses Arial font resources",
            any("Arial" in font for font in fonts),
            True,
            "; ".join(fonts),
        )

    tracked = {row["relative_path"] for row in manifest}
    allowed_untracked = {"MANIFEST_SHA256.csv"}
    untracked: list[str] = []
    for path in package.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        if relative in tracked or relative in allowed_untracked:
            continue
        if relative.endswith("_codex.md") or relative.endswith("_claudecode.md"):
            continue
        untracked.append(relative)
    check("unexpected untracked files", len(untracked), 0, "; ".join(untracked))

    failed = [row for row in checks if row["status"] != "PASS"]
    report = {
        "package": str(package),
        "status": "PASS" if not failed else "FAIL",
        "manifest_entries": len(manifest),
        "checks": len(checks),
        "failed_checks": len(failed),
        "figure_widths_inches": widths,
        "figure_font_resources": figure_fonts,
        "details": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise RuntimeError(json.dumps(failed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
