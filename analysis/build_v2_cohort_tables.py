from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2_registry import add_cultivar_code, trait_registry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger = add_cultivar_code(pd.read_parquet(args.qc_ledger.resolve()))
    ledger["release_complete"] = (
        ledger["qc_primary_include"].astype(bool)
        & ledger["include_primary_multitask"].astype(bool)
        & ledger["texture_dual_valid"].astype(bool)
    )

    cohort = (
        ledger.groupby(["cultivar_code", "cultivar_ascii"], as_index=False)
        .agg(
            linked_fruits=("sample_id", "size"),
            batches=("batch_id", "nunique"),
            analysis_fruits=("qc_analysis_include", "sum"),
            strict_texture_fruits=("qc_primary_include", "sum"),
            release_complete_fruits=("release_complete", "sum"),
        )
        .sort_values(["analysis_fruits", "cultivar_code"], ascending=[False, True])
    )
    cohort["analysis_retention_pct"] = 100.0 * cohort["analysis_fruits"] / cohort["linked_fruits"]
    cohort["strict_retention_pct"] = 100.0 * cohort["strict_texture_fruits"] / cohort["linked_fruits"]
    cohort.to_csv(output / "table_cohort_by_cultivar.csv", index=False)

    traits = trait_registry()
    traits = traits.loc[traits["model_family"] == "endpoint"].copy()
    analysis = ledger.loc[ledger["qc_analysis_include"].astype(bool)]
    rows = []
    for trait in traits.itertuples(index=False):
        values = pd.to_numeric(analysis[trait.target], errors="coerce").dropna().to_numpy(float)
        rows.append(
            {
                "trait": trait.abbreviation,
                "unit": trait.unit,
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "cv_pct": float(100.0 * np.std(values, ddof=1) / np.mean(values)) if np.mean(values) != 0 else np.nan,
            }
        )
    trait_summary = pd.DataFrame(rows)
    trait_summary.to_csv(output / "table_texture_trait_summary.csv", index=False)

    exclusion_reasons = (
        ledger.loc[~ledger["qc_analysis_include"].astype(bool), "qc_reason"]
        .fillna("unspecified")
        .str.split(";")
        .explode()
        .str.strip()
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="fruits_flagged")
    )
    exclusion_reasons.to_csv(output / "table_analysis_exclusion_reasons.csv", index=False)

    report = {
        "linked_fruits": int(len(ledger)),
        "arc_curves": int(2 * len(ledger)),
        "cultivars_or_selections": int(ledger["cultivar_code"].nunique()),
        "batches": int(ledger["batch_id"].nunique()),
        "analysis_fruits": int(ledger["qc_analysis_include"].sum()),
        "strict_texture_fruits": int(ledger["qc_primary_include"].sum()),
        "release_complete_fruits": int(ledger["release_complete"].sum()),
        "analysis_excluded": int((~ledger["qc_analysis_include"].astype(bool)).sum()),
        "strict_excluded": int((~ledger["qc_primary_include"].astype(bool)).sum()),
        "trait_abbreviations": traits["abbreviation"].tolist(),
    }
    project_root = Path(__file__).resolve().parents[1]
    provenance_paths = [
        Path(__file__).resolve(),
        args.qc_ledger.resolve(),
        project_root / "configs" / "v2_nomenclature.csv",
        project_root / "configs" / "v2_trait_registry.csv",
        project_root / "environment-lock.txt",
    ]
    report["provenance_sha256"] = {
        str(path.relative_to(project_root) if path.is_relative_to(project_root) else path): sha256_file(path)
        for path in provenance_paths
        if path.exists()
    }
    (output / "cohort_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
