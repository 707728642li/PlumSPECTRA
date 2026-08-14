from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_plumrac_loco as v2
from v2_registry import abbreviated_trait


def metadata_paths(project_root: Path, trait: str) -> list[Path]:
    if trait == "RD":
        return sorted(
            (project_root / "results/v14/domain_anchor_rd_primary_stratified").glob(
                "repeat_*/metadata.json"
            )
        )
    if trait == "F6":
        return [
            project_root / "results/v15/domain_anchor_trait_screen/F6/metadata.json",
            *sorted(
                (project_root / "results/v16/f6_fixed_gate075_confirmation").glob(
                    "repeat_*/metadata.json"
                )
            ),
        ]
    return [
        project_root / f"results/v15/domain_anchor_trait_screen/{trait}/metadata.json",
        *sorted(
            (project_root / f"results/v15/domain_anchor_confirmation/{trait}").glob(
                "repeat_*/metadata.json"
            )
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    statistics = pd.read_csv(args.statistics)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    evidence = {}
    for target in v2.DEFAULT_TARGETS:
        trait = abbreviated_trait(target)
        paths = metadata_paths(project_root, trait)
        if len(paths) != 5 or not all(path.exists() for path in paths):
            raise RuntimeError(f"Expected five metadata files for {trait}, got {paths}")
        records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        repeats = sorted(int(record["repeat"]) for record in records)
        if repeats != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"Expected repeats 1-5 for {trait}, got {repeats}")
        epochs = [int(record["selected_epoch"]) for record in records]
        gates = [float(record["selected_gate"]) for record in records]
        production_gate = 0.75 if trait == "F6" else float(np.median(gates))
        validation = statistics.loc[statistics["trait"] == trait]
        if len(validation) != 1:
            raise RuntimeError(f"Expected one validation statistics row for {trait}")
        validation_row = validation.iloc[0]
        rows.append(
            {
                "target": target,
                "trait": trait,
                "fixed_epochs": int(np.median(epochs)),
                "fixed_gate": production_gate,
                "validation_ai_vs_global_pls_pct": float(validation_row["ai_vs_global_pls_pct"]),
                "validation_ai_vs_domain_pls_pct": float(validation_row["ai_vs_domain_pls_pct"]),
                "validation_global_ci95_low_pct": float(validation_row["global_ci95_low_pct"]),
                "validation_domain_ci95_low_pct": float(validation_row["domain_ci95_low_pct"]),
                "production_seed": 20260820 + len(rows),
            }
        )
        evidence[trait] = {
            "metadata_files": [str(path.relative_to(project_root)) for path in paths],
            "selected_epochs": epochs,
            "selected_gates": gates,
            "production_epochs_median": int(np.median(epochs)),
            "production_gate": production_gate,
            "f6_gate_exception": trait == "F6",
        }
    plan = pd.DataFrame(rows)
    plan.to_csv(output_dir / "production_plan.csv", index=False)
    manifest = {
        "scope": "nine final independent single-output texture models trained on the full primary cohort",
        "eligible_fruits": 4839,
        "cultivars": 15,
        "model_independent_excluded_cultivars": ["6.11"],
        "epoch_rule": "median selected epoch across five repeated stratified fruit holdouts",
        "gate_rule": (
            "median selected gate across five repeats; F6 uses the frozen 0.75 development gate "
            "confirmed on repeats 2-5"
        ),
        "validation_statistics": str(args.statistics.resolve()),
        "evidence": evidence,
    }
    (output_dir / "production_plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(plan.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
