from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"
TABLES = MANUSCRIPT / "tables"
RELEASE = ROOT.parent / "NIRs_plums_research_ready_en_v1.1.0"

TARGET_LABEL = {
    "fruit_weight_g": "Weight (g)",
    "soluble_solids_pct": "SSC (%)",
    "ph": "pH",
}
TARGET_ORDER = {target: index for index, target in enumerate(TARGET_LABEL)}
MODEL_LABEL = {
    "pls_direct": "Direct PLSR",
    "hierarchical": "Hierarchical PLSR",
    "cnn": "1D-CNN ensemble",
    "transformer": "Transformer ensemble",
    "cnn_texture_aux": "Texture-auxiliary 1D-CNN",
}


def summary(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def metric(model_summary: dict, target: str) -> dict:
    return model_summary["pooled_ensemble_metrics"][target]


def minus(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace("-", "−")


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(TABLES / name, index=False, encoding="utf-8-sig")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    cnn = summary("results/cnn_multitask_loco/summary.json")
    transformer = summary("results/transformer_multitask_loco/summary.json")
    auxiliary = summary("results/cnn_texture_aux_loco/summary.json")
    expected_seeds = [20260806, 20260807, 20260808]
    for label, record in [("CNN", cnn), ("Transformer", transformer), ("Auxiliary CNN", auxiliary)]:
        if record.get("seeds") != expected_seeds:
            raise RuntimeError(f"{label} is not the expected three-seed ensemble: {record.get('seeds')}")

    replacements: dict[str, str] = {}
    for prefix, record in [("CNN", cnn), ("TR", transformer)]:
        for short, target, digits in [
            ("WEIGHT", "fruit_weight_g", 2),
            ("SSC", "soluble_solids_pct", 2),
            ("PH", "ph", 3),
        ]:
            values = metric(record, target)
            replacements[f"{{{{{prefix}_{short}_RMSE}}}}"] = minus(values["rmse"], digits)
            replacements[f"{{{{{prefix}_{short}_R2}}}}"] = minus(values["r2"], 3)

    for short, target, digits, unit in [
        ("WEIGHT", "fruit_weight_g", 2, " g"),
        ("SSC", "soluble_solids_pct", 3, "%"),
        ("PH", "ph", 3, " pH units"),
    ]:
        delta = metric(auxiliary, target)["rmse"] - metric(cnn, target)["rmse"]
        replacements[f"{{{{AUX_{short}_DELTA}}}}"] = f"{minus(delta, digits)}{unit}"

    text = (MANUSCRIPT / "manuscript_template.md").read_text(encoding="utf-8")
    for token, value in replacements.items():
        text = text.replace(token, value)
    unresolved = [token for token in replacements if token in text]
    if unresolved or "{{" in text:
        raise RuntimeError(f"Unresolved manuscript tokens: {unresolved}")
    (MANUSCRIPT / "manuscript_final.md").write_text(text, encoding="utf-8")

    samples = pd.read_csv(RELEASE / "metadata" / "samples.csv")
    cultivar_rows = []
    for cultivar, group in samples.groupby("cultivar_ascii", sort=False):
        cultivar_rows.append(
            {
                "Cultivar or selection": cultivar,
                "Fruits (n)": len(group),
                "Batches (n)": group["batch_id"].nunique(),
                "Weight, median [IQR] (g)": (
                    f"{group['fruit_weight_g'].median():.1f} "
                    f"[{group['fruit_weight_g'].quantile(.25):.1f}–{group['fruit_weight_g'].quantile(.75):.1f}]"
                ),
                "SSC, median [IQR] (%)": (
                    f"{group['soluble_solids_pct'].median():.1f} "
                    f"[{group['soluble_solids_pct'].quantile(.25):.1f}–{group['soluble_solids_pct'].quantile(.75):.1f}]"
                ),
                "pH, median [IQR]": (
                    f"{group['ph'].median():.2f} "
                    f"[{group['ph'].quantile(.25):.2f}–{group['ph'].quantile(.75):.2f}]"
                ),
            }
        )
    table1 = pd.DataFrame(cultivar_rows).sort_values("Fruits (n)", ascending=False)
    write_csv(table1, "table1_cohort.csv")

    random = pd.read_csv(ROOT / "results/model_comparison/tables/pls_validation_regime_comparison.csv")
    random = random.loc[random["validation"].eq("Random fruit split")].copy()
    random["model"] = "Random-split PLSR"
    random["n"] = pd.NA
    random["mae"] = pd.NA
    random["rpd"] = pd.NA

    zero_rows = []
    paths = {
        "pls_direct": "results/pls_loco/summary.json",
        "hierarchical": "results/hierarchical_pls_loco/summary.json",
        "cnn": "results/cnn_multitask_loco/summary.json",
        "transformer": "results/transformer_multitask_loco/summary.json",
        "cnn_texture_aux": "results/cnn_texture_aux_loco/summary.json",
    }
    for model_name, path in paths.items():
        record = summary(path)
        metrics = record.get("pooled_metrics", record.get("pooled_ensemble_metrics"))
        for target, values in metrics.items():
            zero_rows.append(
                {
                    "model": MODEL_LABEL[model_name],
                    "target": target,
                    "n": values["n"],
                    "rmse": values["rmse"],
                    "mae": values["mae"],
                    "r2": values["r2"],
                    "ccc": values["ccc"],
                    "rpd": values["rpd"],
                }
            )
    table2 = pd.concat(
        [random[["model", "target", "n", "rmse", "mae", "r2", "ccc", "rpd"]], pd.DataFrame(zero_rows)],
        ignore_index=True,
    )
    table2["Target"] = table2["target"].map(TARGET_LABEL)
    table2["Target order"] = table2["target"].map(TARGET_ORDER)
    performance_order = {
        "Random-split PLSR": 0,
        "Direct PLSR": 1,
        "Hierarchical PLSR": 2,
        "1D-CNN ensemble": 3,
        "Texture-auxiliary 1D-CNN": 4,
        "Transformer ensemble": 5,
    }
    table2["Model order"] = table2["model"].map(performance_order)
    table2 = table2.sort_values(["Model order", "Target order"])
    table2["RMSE"] = table2.apply(
        lambda row: f"{row.rmse:.3f}" if row.target == "ph" else f"{row.rmse:.2f}", axis=1
    )
    for output, source in [("MAE", "mae"), ("R²", "r2"), ("CCC", "ccc"), ("RPD", "rpd")]:
        table2[output] = table2[source].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}")
    table2["n"] = table2["n"].map(lambda value: "—" if pd.isna(value) else f"{int(value):,}")
    table2 = table2.rename(columns={"model": "Model / validation", "n": "n"})[
        ["Model / validation", "Target", "n", "RMSE", "MAE", "R²", "CCC", "RPD"]
    ]
    write_csv(table2, "table2_model_performance.csv")

    fewshot = pd.read_csv(ROOT / "results/fewshot_calibration_all/fewshot_summary.csv")
    table3 = fewshot.loc[
        fewshot["model"].eq("hierarchical") & fewshot["shots"].isin([0, 5, 10, 20]),
        ["target", "shots", "rmse_mean", "rmse_ci025", "rmse_ci975", "r2_mean", "ccc_mean"],
    ].copy()
    table3["Target"] = table3["target"].map(TARGET_LABEL)
    table3["Target order"] = table3["target"].map(TARGET_ORDER)
    table3["RMSE mean [95% interval]"] = table3.apply(
        lambda row: (
            f"{row.rmse_mean:.3f} [{row.rmse_ci025:.3f}–{row.rmse_ci975:.3f}]"
            if row.target == "ph"
            else f"{row.rmse_mean:.2f} [{row.rmse_ci025:.2f}–{row.rmse_ci975:.2f}]"
        ),
        axis=1,
    )
    table3["R²"] = table3["r2_mean"].map(lambda value: f"{value:.3f}")
    table3["CCC"] = table3["ccc_mean"].map(lambda value: f"{value:.3f}")
    table3 = table3.sort_values(["shots", "Target order"]).rename(columns={"shots": "Labelled fruits"})[
        ["Labelled fruits", "Target", "RMSE mean [95% interval]", "R²", "CCC"]
    ]
    write_csv(table3, "table3_fewshot_hierarchical.csv")

    conformal = pd.read_csv(ROOT / "results/conformal_intervals/conformal_summary.csv")
    table4 = conformal.loc[conformal["model"].eq("hierarchical")].copy()
    table4["Target"] = table4["target"].map(TARGET_LABEL)
    table4["Target order"] = table4["target"].map(TARGET_ORDER)
    table4["Empirical coverage"] = table4["empirical_coverage_mean"].map(lambda value: f"{value:.3f}")
    table4["95% interval"] = table4.apply(
        lambda row: f"{row.empirical_coverage_ci025:.3f}–{row.empirical_coverage_ci975:.3f}", axis=1
    )
    table4["Mean width"] = table4.apply(
        lambda row: f"{row.mean_interval_width:.3f}" if row.target == "ph" else f"{row.mean_interval_width:.2f}",
        axis=1,
    )
    table4 = table4.sort_values(["calibration_size", "Target order"]).rename(
        columns={
            "calibration_size": "Total labelled fruits",
            "intercept_fit_size": "Intercept fit",
            "conformal_size": "Interval fit",
        }
    )[
        [
            "Total labelled fruits",
            "Intercept fit",
            "Interval fit",
            "Target",
            "Empirical coverage",
            "95% interval",
            "Mean width",
        ]
    ]
    write_csv(table4, "table4_conformal_intervals.csv")

    zero_display = table2.loc[
        table2["Model / validation"].isin(
            ["Direct PLSR", "1D-CNN ensemble", "Transformer ensemble", "Texture-auxiliary 1D-CNN"]
        )
    ]
    fewshot_display = table3.loc[table3["Labelled fruits"].isin([5, 10, 20])]
    result_lines = [
        "# Key cross-cultivar results",
        "",
        "All zero-shot values come from complete leave-one-cultivar-out predictions; deep models are three-seed ensembles. Few-shot values are hierarchical PLSR means across 100 identical calibration draws with labelled fruits removed from evaluation.",
        "",
        "## Zero-shot held-out-cultivar prediction",
        "",
        "| Model | Target | RMSE | R² | CCC | RPD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in zero_display.itertuples(index=False):
        result_lines.append(f"| {row[0]} | {row[1]} | {row[3]} | {row[5]} | {row[6]} | {row[7]} |")
    result_lines.extend(
        [
            "",
            "## Hierarchical few-shot adaptation",
            "",
            "| Labelled fruits | Target | RMSE mean [95% interval] | R² | CCC |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in fewshot_display.itertuples(index=False):
        result_lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    result_lines.extend(
        [
            "",
            "## Deployment interpretation",
            "",
            "Random fruit splitting preserves cultivar-specific predictor and target distributions and therefore overstates performance for a genuinely unseen cultivar. None of the tested deep architectures consistently surpassed nested direct PLSR in zero-shot transfer. Five to ten labelled fruits primarily correct the new cultivar's target mean; about 20 labelled fruits were required for the evaluated nominal 90% conformal intervals to reach approximately 91% empirical coverage.",
            "",
            "Texture endpoints were repeatable and biologically cultivar structured, but destructive texture measurements are not required during NIR inference.",
        ]
    )
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "key_results.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")

    supplementary = MANUSCRIPT / "supplementary_tables"
    supplementary.mkdir(exist_ok=True)
    copies = {
        "tableS1_outlier_sensitivity.csv": ROOT / "results/sensitivity/soft_spectral_outlier_sensitivity.csv",
        "tableS2_paired_model_tests.csv": ROOT / "results/model_comparison/tables/paired_cultivar_model_tests.csv",
        "tableS3_vip_windows.csv": ROOT / "results/spectral_interpretation/pls_vip_window_summary.csv",
    }
    for name, source in copies.items():
        if source.exists():
            pd.read_csv(source).to_csv(supplementary / name, index=False, encoding="utf-8-sig")

    manifest = {
        "manuscript": "manuscript_final.md",
        "deep_seeds": expected_seeds,
        "tables": sorted(path.name for path in TABLES.glob("*.csv")),
        "replacement_values": replacements,
    }
    (MANUSCRIPT / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
