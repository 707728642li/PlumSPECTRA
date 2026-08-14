from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TRAIT_ORDER = ["LS", "SRF", "PFD", "PRW", "LW", "MFF", "RD", "F6", "AF"]
TARGETS = {
    "LS": "loading_stiffness_g_per_rawpos_mean",
    "SRF": "skin_break_force_g_mean",
    "PFD": "skin_break_drop_g_mean",
    "PRW": "post_break_work_g_rawpos_mean",
    "LW": "loading_work_g_rawpos_mean",
    "MFF": "flesh_force_mean_g_mean",
    "RD": "skin_break_displacement_raw_mean",
    "F6": "force_at_6_rawpos_g_mean",
    "AF": "adhesive_force_g_mean",
}


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(prediction, float)) ** 2)))


def r2(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return float(1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def cluster_ci(
    frame: pd.DataFrame,
    candidate: str,
    comparator: str,
    seed: int,
    n_bootstrap: int = 20_000,
) -> tuple[float, float]:
    groups = [group for _, group in frame.groupby("cultivar_ascii", sort=True)]
    candidate_ss = np.asarray(
        [np.sum((group["y_true"] - group[candidate]) ** 2) for group in groups], float
    )
    comparator_ss = np.asarray(
        [np.sum((group["y_true"] - group[comparator]) ** 2) for group in groups], float
    )
    sizes = np.asarray([len(group) for group in groups], float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(n_bootstrap, len(groups)))
    candidate_rmse = np.sqrt(candidate_ss[indices].sum(axis=1) / sizes[indices].sum(axis=1))
    comparator_rmse = np.sqrt(comparator_ss[indices].sum(axis=1) / sizes[indices].sum(axis=1))
    values = 100.0 * (1.0 - candidate_rmse / comparator_rmse)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def load_predictions(project: Path, sources: str | list[str]) -> pd.DataFrame:
    paths = [sources] if isinstance(sources, str) else sources
    frames = [pd.read_parquet(project / path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    required = {
        "sample_id",
        "cultivar_ascii",
        "repeat",
        "y_true",
        "y_pred",
        "y_global_pls_anchor",
        "y_pls_anchor",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction sources are missing columns: {missing}")
    if sorted(frame["repeat"].unique().tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError("Each selected trait must contain repeats 1-5 exactly")
    if frame.duplicated(["repeat", "sample_id"]).any():
        raise ValueError("Duplicate repeat/sample_id prediction records detected")
    return frame.sort_values(["repeat", "sample_id"]).reset_index(drop=True)


def centred_arrays(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    truth_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    for _, group in frame.groupby(["repeat", "cultivar_ascii"], sort=True):
        truth = group["y_true"].to_numpy(float)
        prediction = group[column].to_numpy(float)
        truth_parts.append(truth - truth.mean())
        prediction_parts.append(prediction - prediction.mean())
    return np.concatenate(truth_parts), np.concatenate(prediction_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    if set(manifest["traits"]) != set(TRAIT_ORDER):
        raise ValueError("Prediction manifest must define exactly the nine texture traits")

    ledger = pd.read_parquet(project / "data/processed/texture_qc/texture_qc_ledger.parquet")
    row_index = pd.read_csv(project / "data/processed/multimodal/nir_c_row_index.csv")
    primary = ledger.loc[
        ledger["qc_primary_include"].astype(bool) & (ledger["cultivar_ascii"].astype(str) != "6.11")
    ].copy()
    identity = {
        "ledger_fruits": int(ledger["sample_id"].nunique()),
        "nir_spectra": int(row_index["sample_id"].nunique()),
        "primary_fruits": int(primary["sample_id"].nunique()),
        "primary_cultivars": int(primary["cultivar_ascii"].nunique()),
        "primary_batches": int(primary["batch_id"].nunique()),
        "ledger_without_nir": sorted(set(ledger["sample_id"]) - set(row_index["sample_id"])),
    }
    (output_dir / "revised_data_identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )

    pooled_rows: list[dict[str, object]] = []
    within_rows: list[dict[str, object]] = []
    repeat_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    benchmark_rows: list[dict[str, object]] = []
    selected_frames: dict[str, pd.DataFrame] = {}

    existing_benchmarks = pd.read_csv(
        project / "results/reviewer_recompute/recompute_extra_baselines.csv"
    )
    for trait_index, trait in enumerate(TRAIT_ORDER):
        frame = load_predictions(project, manifest["traits"][trait])
        selected_frames[trait] = frame
        truth = frame["y_true"].to_numpy(float)
        ai = rmse(truth, frame["y_pred"].to_numpy(float))
        domain = rmse(truth, frame["y_pls_anchor"].to_numpy(float))
        global_pls = rmse(truth, frame["y_global_pls_anchor"].to_numpy(float))
        ai_global = 100.0 * (1.0 - ai / global_pls)
        ai_domain = 100.0 * (1.0 - ai / domain)
        domain_global = 100.0 * (1.0 - domain / global_pls)
        global_ci = cluster_ci(frame, "y_pred", "y_global_pls_anchor", 9100 + trait_index)
        domain_ci = cluster_ci(frame, "y_pred", "y_pls_anchor", 9200 + trait_index)
        cultivar_metrics = frame.groupby("cultivar_ascii", sort=True).apply(
            lambda group: pd.Series(
                {
                    "ai": rmse(group["y_true"], group["y_pred"]),
                    "global_pls": rmse(group["y_true"], group["y_global_pls_anchor"]),
                    "domain_pls": rmse(group["y_true"], group["y_pls_anchor"]),
                }
            ),
            include_groups=False,
        )
        pooled_rows.append(
            {
                "trait": trait,
                "records": len(frame),
                "unique_fruits": frame["sample_id"].nunique(),
                "ai_rmse": ai,
                "global_pls_rmse": global_pls,
                "domain_pls_rmse": domain,
                "ai_vs_global_pls_pct": ai_global,
                "ai_vs_global_ci_low": global_ci[0],
                "ai_vs_global_ci_high": global_ci[1],
                "ai_vs_domain_pls_pct": ai_domain,
                "ai_vs_domain_ci_low": domain_ci[0],
                "ai_vs_domain_ci_high": domain_ci[1],
                "domain_pls_vs_global_pls_pct": domain_global,
                "domain_share_of_total_gain_pct": 100.0 * domain_global / ai_global,
                "cultivar_wins_vs_global": int(
                    (cultivar_metrics["ai"] < cultivar_metrics["global_pls"]).sum()
                ),
                "cultivar_wins_vs_domain": int(
                    (cultivar_metrics["ai"] < cultivar_metrics["domain_pls"]).sum()
                ),
                "wilcoxon_vs_global_one_sided_p": float(
                    stats.wilcoxon(
                        cultivar_metrics["global_pls"], cultivar_metrics["ai"], alternative="greater"
                    ).pvalue
                ),
                "wilcoxon_vs_domain_one_sided_p": float(
                    stats.wilcoxon(
                        cultivar_metrics["domain_pls"], cultivar_metrics["ai"], alternative="greater"
                    ).pvalue
                ),
            }
        )

        truth_c, ai_c = centred_arrays(frame, "y_pred")
        _, domain_c = centred_arrays(frame, "y_pls_anchor")
        _, global_c = centred_arrays(frame, "y_global_pls_anchor")
        ai_r = float(stats.pearsonr(truth_c, ai_c).statistic)
        domain_r = float(stats.pearsonr(truth_c, domain_c).statistic)
        within_rows.append(
            {
                "trait": trait,
                "pooled_r2_ai": r2(truth, frame["y_pred"]),
                "within_cultivar_r2_ai": r2(truth_c, ai_c),
                "within_cultivar_r2_domain_pls": r2(truth_c, domain_c),
                "within_cultivar_pearson_ai": ai_r,
                "within_cultivar_pearson_domain_pls": domain_r,
                "within_cultivar_ai_vs_domain_rmse_pct": 100.0
                * (1.0 - rmse(truth_c, ai_c) / rmse(truth_c, domain_c)),
                "within_cultivar_ai_vs_global_rmse_pct": 100.0
                * (1.0 - rmse(truth_c, ai_c) / rmse(truth_c, global_c)),
                "linear_recalibration_ceiling_r2_ai": ai_r**2,
            }
        )

        for subset, repeats in [("development_r1", [1]), ("confirmation_r2_5", [2, 3, 4, 5])]:
            part = frame[frame["repeat"].isin(repeats)]
            repeat_rows.append(
                {
                    "trait": trait,
                    "subset": subset,
                    "records": len(part),
                    "ai_vs_global_pls_pct": 100.0
                    * (1.0 - rmse(part["y_true"], part["y_pred"]) / rmse(part["y_true"], part["y_global_pls_anchor"])),
                    "ai_vs_domain_pls_pct": 100.0
                    * (1.0 - rmse(part["y_true"], part["y_pred"]) / rmse(part["y_true"], part["y_pls_anchor"])),
                }
            )

        if trait in {"MFF", "F6"}:
            for subset, part in [
                ("all", frame),
                ("nonnegative_reference_only", frame[frame["y_true"] >= 0]),
            ]:
                sensitivity_rows.append(
                    {
                        "trait": trait,
                        "subset": subset,
                        "records": len(part),
                        "negative_records_removed": int((frame["y_true"] < 0).sum())
                        if subset != "all"
                        else 0,
                        "ai_vs_global_pls_pct": 100.0
                        * (1.0 - rmse(part["y_true"], part["y_pred"]) / rmse(part["y_true"], part["y_global_pls_anchor"])),
                        "ai_vs_domain_pls_pct": 100.0
                        * (1.0 - rmse(part["y_true"], part["y_pred"]) / rmse(part["y_true"], part["y_pls_anchor"])),
                    }
                )

        old_trait = existing_benchmarks[existing_benchmarks["trait"] == trait].copy()
        old_trait = old_trait[old_trait["model"] != "ai"]
        benchmark_rows.extend(old_trait.to_dict("records"))
        benchmark_rows.append(
            {
                "trait": trait,
                "model": "ai_selected",
                "n": len(frame),
                "rmse": ai,
                "vs_global_pls_pct": ai_global,
                "vs_domain_pls_pct": ai_domain,
                "cultivar_wins_vs_domain": int(
                    (cultivar_metrics["ai"] < cultivar_metrics["domain_pls"]).sum()
                ),
            }
        )

    pooled = pd.DataFrame(pooled_rows)
    pooled["trait"] = pd.Categorical(pooled["trait"], TRAIT_ORDER, ordered=True)
    pooled = pooled.sort_values("trait")
    pooled.to_csv(output_dir / "revised_primary_statistics.csv", index=False)
    pd.DataFrame(within_rows).to_csv(output_dir / "revised_within_cultivar_statistics.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(output_dir / "revised_repeat_confirmation.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "negative_endpoint_sensitivity.csv", index=False)
    pd.DataFrame(benchmark_rows).to_csv(output_dir / "revised_model_benchmark.csv", index=False)

    ai_svr_rows: list[dict[str, object]] = []
    for trait_index, trait in enumerate(TRAIT_ORDER):
        comparator = pd.read_parquet(project / f"results/reviewer_recompute/pred_ai_vs_svr_{trait}.parquet")
        selected = selected_frames[trait][["repeat", "sample_id", "cultivar_ascii", "y_true", "y_pred"]]
        merged = comparator[["repeat", "sample_id", "svr"]].merge(
            selected, on=["repeat", "sample_id"], how="inner", validate="one_to_one"
        )
        if len(merged) != len(selected):
            raise RuntimeError(f"SVR comparison rows do not match selected predictions for {trait}")
        ci = cluster_ci(merged, "y_pred", "svr", 9300 + trait_index)
        per = merged.groupby("cultivar_ascii", sort=True).apply(
            lambda group: pd.Series(
                {
                    "ai": rmse(group["y_true"], group["y_pred"]),
                    "svr": rmse(group["y_true"], group["svr"]),
                }
            ),
            include_groups=False,
        )
        ai_rmse = rmse(merged["y_true"], merged["y_pred"])
        svr_rmse = rmse(merged["y_true"], merged["svr"])
        ai_svr_rows.append(
            {
                "trait": trait,
                "ai_rmse": ai_rmse,
                "untuned_svr_rmse": svr_rmse,
                "ai_vs_untuned_svr_pct": 100.0 * (1.0 - ai_rmse / svr_rmse),
                "cluster_ci_low": ci[0],
                "cluster_ci_high": ci[1],
                "ai_cultivar_wins": int((per["ai"] < per["svr"]).sum()),
                "wilcoxon_two_sided_p": float(
                    stats.wilcoxon(per["svr"], per["ai"], alternative="two-sided").pvalue
                ),
            }
        )
    pd.DataFrame(ai_svr_rows).to_csv(output_dir / "revised_ai_vs_untuned_svr.csv", index=False)

    cross = pd.crosstab(primary["batch_id"].astype(str), primary["cultivar_ascii"].astype(str))
    cross.to_csv(output_dir / "batch_by_cultivar_counts.csv")
    nesting = pd.DataFrame(
        [
            {
                "batch_id": batch,
                "fruits": int(row.sum()),
                "cultivars": int((row > 0).sum()),
                "dominant_cultivar": str(row.idxmax()),
                "dominant_share": float(row.max() / row.sum()),
            }
            for batch, row in cross.iterrows()
        ]
    )
    nesting.to_csv(output_dir / "batch_nesting_audit.csv", index=False)

    phenotype = primary[[TARGETS[trait] for trait in TRAIT_ORDER]].apply(pd.to_numeric, errors="coerce")
    phenotype.columns = TRAIT_ORDER
    phenotype = phenotype.dropna()
    correlation = phenotype.corr()
    correlation.to_csv(output_dir / "texture_endpoint_correlation.csv")
    standardized = (phenotype - phenotype.mean()) / phenotype.std(ddof=1)
    eigenvalues = np.linalg.eigvalsh(np.cov(standardized, rowvar=False))[::-1]
    pca = pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "variance_explained": eigenvalues / eigenvalues.sum(),
            "cumulative_variance": np.cumsum(eigenvalues / eigenvalues.sum()),
        }
    )
    pca.to_csv(output_dir / "texture_endpoint_pca.csv", index=False)
    effective_dimension = float(eigenvalues.sum() ** 2 / np.sum(eigenvalues**2))
    summary = {
        "prediction_manifest": str(args.prediction_manifest.resolve()),
        "identity": identity,
        "batch_perfectly_nested_within_cultivar": bool(
            (nesting["cultivars"] == 1).all() and np.allclose(nesting["dominant_share"], 1.0)
        ),
        "pca_first_three_cumulative_variance": float(pca.loc[2, "cumulative_variance"]),
        "endpoint_effective_dimension": effective_dimension,
        "negative_primary_fruits": {
            "MFF": int((primary[TARGETS["MFF"]] < 0).sum()),
            "F6": int((primary[TARGETS["F6"]] < 0).sum()),
        },
    }
    (output_dir / "revised_evidence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(pooled.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
