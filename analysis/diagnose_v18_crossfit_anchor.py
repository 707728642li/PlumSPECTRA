from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import train_plumrac_loco as v2
from train_plumrac_v5_stratified import (
    prepare_anchor_targets,
    prepare_crossfit_anchor_targets,
)
from v2_registry import abbreviated_trait


DEFAULT_TARGETS = [
    "loading_stiffness_g_per_rawpos_mean",
    "skin_break_displacement_raw_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal-dir", type=Path, required=True)
    parser.add_argument("--qc-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--exclude-cultivars", default="6.11")
    args = parser.parse_args()

    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.multimodal_dir / "nir_c_absorbance.npy").astype(np.float32)
    wavelength = np.load(args.multimodal_dir / "wavelength_nm.npy").astype(np.float32)
    row_index = pd.read_csv(args.multimodal_dir / "nir_c_row_index.csv")
    ledger = pd.read_parquet(args.qc_ledger).set_index("sample_id")
    aligned = ledger.loc[row_index["sample_id"]].reset_index()
    groups = aligned["cultivar_ascii"].astype(str).to_numpy()
    arrays = v2.preprocess_all(raw, wavelength)
    excluded = [value.strip() for value in args.exclude_cultivars.split(",") if value.strip()]
    seed = 20260806 + args.repeat

    configuration_locations = {
        "LS": Path("results/v15/domain_anchor_trait_screen/LS/metadata.json"),
        "RD": Path("results/v14/domain_anchor_rd_primary_stratified/repeat_1/metadata.json"),
    }
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    for target in targets:
        trait = abbreviated_trait(target)
        metadata = json.loads(configuration_locations[trait].read_text(encoding="utf-8"))
        y = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
        eligible = aligned["qc_primary_include"].to_numpy(bool) & np.isfinite(y)
        eligible &= ~np.isin(groups, excluded)
        eligible_indices = np.flatnonzero(eligible)
        outer_train, test_indices = train_test_split(
            eligible_indices,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
            stratify=groups[eligible_indices],
        )
        inner_train, validation_indices = train_test_split(
            outer_train,
            test_size=0.15,
            random_state=seed + 100_003,
            shuffle=True,
            stratify=groups[outer_train],
        )
        for partition_name, fit_indices, heldout_indices, config_key, crossfit_seed in [
            (
                "inner_selection",
                inner_train,
                validation_indices,
                "inner_pls",
                seed + 250_003,
            ),
            (
                "outer_refit",
                outer_train,
                test_indices,
                "final_pls",
                seed + 950_003,
            ),
        ]:
            preprocessing = metadata[config_key]["preprocessing"]
            components = int(metadata[config_key]["n_components"])
            (_, old_anchor, _, _, _, _, _) = prepare_anchor_targets(
                arrays,
                y,
                groups,
                fit_indices,
                [fit_indices, heldout_indices],
                preprocessing,
                components,
                True,
            )
            (_, new_anchor, _, _, _, _, _, diagnostic) = prepare_crossfit_anchor_targets(
                arrays,
                y,
                groups,
                fit_indices,
                [heldout_indices],
                preprocessing,
                components,
                True,
                args.folds,
                crossfit_seed,
            )
            old_fit_residual = y[fit_indices] - old_anchor[fit_indices]
            new_fit_residual = y[fit_indices] - new_anchor[fit_indices]
            heldout_residual = y[heldout_indices] - old_anchor[heldout_indices]
            row = {
                "trait": trait,
                "target": target,
                "partition": partition_name,
                "fit_samples": int(len(fit_indices)),
                "heldout_samples": int(len(heldout_indices)),
                "preprocessing": preprocessing,
                "components": components,
                "in_sample_fit_rmse": float(np.sqrt(np.mean(old_fit_residual**2))),
                "crossfit_fit_rmse": float(np.sqrt(np.mean(new_fit_residual**2))),
                "heldout_rmse": float(np.sqrt(np.mean(heldout_residual**2))),
                "crossfit_to_heldout_rmse_ratio": float(
                    np.sqrt(np.mean(new_fit_residual**2))
                    / np.sqrt(np.mean(heldout_residual**2))
                ),
                "in_sample_to_heldout_rmse_ratio": float(
                    np.sqrt(np.mean(old_fit_residual**2))
                    / np.sqrt(np.mean(heldout_residual**2))
                ),
                "crossfit_vs_in_sample_residual_sd_ratio": float(
                    np.std(new_fit_residual, ddof=1) / np.std(old_fit_residual, ddof=1)
                ),
            }
            rows.append(row)
            details[f"{trait}_{partition_name}"] = diagnostic

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "anchor_distribution_diagnostic.csv", index=False)
    manifest = {
        "question": (
            "Do in-sample PLS residual targets understate the deployment-time residual distribution, "
            "and do cross-fitted targets better match held-out errors?"
        ),
        "repeat": args.repeat,
        "folds": args.folds,
        "excluded_cultivars": excluded,
        "results": rows,
        "crossfit_details": details,
    }
    (output_dir / "anchor_distribution_diagnostic.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
