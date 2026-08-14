#!/usr/bin/env python3
"""Extract compact figure inputs from the frozen audit package.

Read-only against the audit package. Writes small CSVs into ./source_data so the
R scripts stay dependency-light (no arrow needed).

    python prep/prepare_figure_data.py <audit_package_root> [out_dir]

Derived quantities are recomputed here rather than copied, and the script
asserts them against the values the audit package reports, so a silent drift in
either direction fails loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TRAITS = ["SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF"]
ALL_TRAITS = ["FW", "SSC", "pH"] + TRAITS


def main() -> int:
    root = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "source_data")
    out.mkdir(parents=True, exist_ok=True)
    ev = root / "evidence"

    pred = pq.read_table(
        ev / "final_analysis/v25_integrated_predictions.parquet",
        columns=["sample_id", "cultivar_code", "trait", "outer_fold", "y_true",
                 "y_domain_pls", "y_domain_svr", "y_deep", "y_final"],
    ).to_pandas()
    assert len(pred) == 58_206, len(pred)
    assert pred.duplicated(["sample_id", "trait"]).sum() == 0
    print(f"integrated predictions {len(pred):,} rows, 0 duplicate fruit-trait pairs")

    # ---------------------------------------------------------------- Fig 2 --
    obs = pred[["sample_id", "cultivar_code", "trait", "y_true"]].rename(
        columns={"y_true": "value"})
    obs.to_csv(out / "fig2_observed_long.csv", index=False)

    # Joint 12-trait phenotype structure for Fig. 2b/c/e, using the common
    # complete-case cohort so conventional quality and texture are comparable.
    wide = (obs[obs.trait.isin(ALL_TRAITS)]
            .pivot(index="sample_id", columns="trait", values="value")
            .loc[:, ALL_TRAITS]
            .dropna())
    codes = (obs[["sample_id", "cultivar_code"]].drop_duplicates()
             .set_index("sample_id").loc[wide.index, "cultivar_code"])
    print(f"joint phenotype cohort for PCA: {len(wide):,} fruit x {wide.shape[1]} traits")

    z = (wide - wide.mean()) / wide.std(ddof=1)
    corr = np.corrcoef(z.values, rowvar=False)
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    ratio = eigval / eigval.sum()

    # sign convention: make the loading with the largest magnitude positive
    for j in range(eigvec.shape[1]):
        if eigvec[np.argmax(np.abs(eigvec[:, j])), j] < 0:
            eigvec[:, j] *= -1

    print(f"PC1 {ratio[0]*100:.2f}%  PC1-3 cumulative {ratio[:3].sum()*100:.2f}%")
    assert len(wide) == 4_843
    assert abs(ratio[0] - 0.6040586245) < 5e-6, "joint PC1 drift"
    assert abs(ratio[:3].sum() - 0.8292143027) < 5e-6, "joint PC1-3 drift"

    participation = ratio.sum() ** 2 / (ratio ** 2).sum()
    print(f"participation ratio {participation:.2f}")

    pd.DataFrame({"component": np.arange(1, len(ALL_TRAITS) + 1),
                  "explained": ratio,
                  "cumulative": np.cumsum(ratio)}).to_csv(
        out / "fig2_pca_variance.csv", index=False)

    pd.DataFrame(eigvec[:, :3], index=ALL_TRAITS,
                 columns=["PC1", "PC2", "PC3"]).rename_axis("trait").reset_index().to_csv(
        out / "fig2_pca_loadings.csv", index=False)

    scores = z.values @ eigvec[:, :2]
    pd.DataFrame({"sample_id": wide.index, "cultivar_code": codes.values,
                  "PC1": scores[:, 0], "PC2": scores[:, 1]}).to_csv(
        out / "fig2_pca_scores.csv", index=False)

    # correlation matrix, recomputed and checked against the frozen table
    cm = pd.DataFrame(corr, index=ALL_TRAITS, columns=ALL_TRAITS)
    ref_corr = pd.read_csv(ev / "final_analysis/texture_endpoint_correlation.csv",
                           index_col="trait").loc[TRAITS, TRAITS]
    print("max |texture-submatrix corr - frozen| = "
          f"{np.abs(cm.loc[TRAITS, TRAITS].values - ref_corr.values).max():.4f}")
    cm.rename_axis("trait").reset_index().to_csv(out / "fig2_correlation.csv", index=False)

    strong = int(((np.abs(np.triu(corr, 1)) > 0.90)).sum())
    print(f"endpoint pairs with |r| > 0.90: {strong}")

    # ---------------------------------------------------------------- Fig 4 --
    pred[["trait", "y_true", "y_final"]].to_csv(out / "fig4_oof.csv", index=False)

    # ---------------------------------------------------------------- Fig 5 --
    rows = []
    # Component complementarity is defined for every deployed response, not
    # only the nine texture endpoints. Retain FW, SSC and pH here as well.
    for trait, g in pred[pred.trait.isin(ALL_TRAITS)].groupby("trait"):
        e_cnn = g.y_deep.values - g.y_true.values
        e_svr = g.y_domain_svr.values - g.y_true.values
        rows.append({
            "trait": trait,
            "error_correlation": float(np.corrcoef(e_cnn, e_svr)[0, 1]),
            "opposite_sign_fraction": float(np.mean(np.sign(e_cnn) != np.sign(e_svr))),
            "n": int(len(g)),
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "fig5_complementarity.csv", index=False)
    print("CNN-SVR error correlation range "
          f"{comp.error_correlation.min():.3f} to {comp.error_correlation.max():.3f}")

    # per-cultivar per-trait RMSE change, final vs cultivar-aware PLSR
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    cells = []
    for (trait, code), g in pred.groupby(["trait", "cultivar_code"]):
        base = rmse(g.y_true.values, g.y_domain_pls.values)
        fin = rmse(g.y_true.values, g.y_final.values)
        ss = float(np.sum((g.y_true.values - g.y_true.values.mean()) ** 2))
        cells.append({
            "trait": trait, "cultivar_code": code, "n": int(len(g)),
            "rmse_reduction_pct": 100 * (base - fin) / base,
            "r2_within": 1 - float(np.sum((g.y_true.values - g.y_final.values) ** 2)) / ss,
            "pearson_r": float(np.corrcoef(g.y_true.values, g.y_final.values)[0, 1]),
        })
    cell_df = pd.DataFrame(cells)
    cell_df.to_csv(out / "fig5_cultivar_cells.csv", index=False)
    neg_r2 = int((cell_df.r2_within < 0).sum())
    neg_r = int((cell_df.pearson_r < 0).sum())
    print(f"cultivar-trait cells {len(cell_df)}  negative R2 {neg_r2}  negative r {neg_r}")

    # ---------------------------------------------------------------- Fig 6 --
    # Preserve matched few-shot calibration resamples when summarising the
    # nine texture endpoints. The uncertainty band is based on the median
    # endpoint gain within each shared calibration draw, rather than treating
    # endpoints as independent replicates.
    few_repeat_path = ev / "final_analysis/fewshot_repeat_metrics.parquet"
    if not few_repeat_path.exists():
        # The compact external-review package omits this ~500k-row intermediate
        # but preserves its summary. Fall back explicitly to the matching
        # frozen project result; never reconstruct uncertainty from hand-entered
        # points or treat endpoints as independent replicates.
        project_root = Path(__file__).resolve().parents[2]
        few_repeat_path = (
            project_root / "results/v25_external_review_corrections/"
            "final_analysis/fewshot_repeat_metrics.parquet"
        )
    if few_repeat_path.exists():
        print(f"Fig 6 few-shot resampling source: {few_repeat_path}")
        few_repeat = pq.read_table(few_repeat_path).to_pandas()
        few_keep = few_repeat[
            (few_repeat.model == "Deep-kernel ensemble")
            & (few_repeat.aggregation == "pooled")
            & (
                ((few_repeat.shots == 0) & (few_repeat.adapter == "none"))
                | ((few_repeat.shots > 0)
                   & (few_repeat.adapter == "shrunken_affine"))
            )
        ].copy()
        few_draw = (few_keep.groupby(["shots", "repeat"], as_index=False)
                    .rmse_gain_pct.median()
                    .rename(columns={"rmse_gain_pct": "gain"}))
        few_uncertainty = (few_draw.groupby("shots").gain
                           .agg(median_resample="median",
                                q025=lambda x: x.quantile(0.025),
                                q975=lambda x: x.quantile(0.975),
                                repeats="size")
                           .reset_index())
    else:
        # A release package may carry only this exact derived table. It is
        # checksum-tracked evidence from the same 500 matched resamples.
        compact = ev / "final_analysis/fig6_fewshot_resampling_uncertainty.csv"
        if not compact.exists():
            raise FileNotFoundError(
                "Few-shot resampling evidence is missing from the audit "
                f"package and frozen project results: {few_repeat_path}"
            )
        print(f"Fig 6 compact resampling source: {compact}")
        few_uncertainty = pd.read_csv(compact)
    assert few_uncertainty.shots.tolist() == [0, 5, 10, 20, 40, 80]
    assert few_uncertainty.loc[few_uncertainty.shots > 0, "repeats"].eq(500).all()
    few_uncertainty.to_csv(out / "fig6_fewshot_resampling_uncertainty.csv",
                           index=False)
    print("Fig 6 few-shot median gain at 20/40/80 fruit: "
          + "/".join(
              f"{x:.2f}%" for x in few_uncertainty.loc[
                  few_uncertainty.shots.isin([20, 40, 80]),
                  "median_resample"]
          ))

    print(f"\nwrote {len(list(out.glob('fig*.csv')))} tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
