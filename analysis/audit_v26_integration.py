from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "review_package" / "HR_EXTERNAL_AUDIT_PACKAGE_V25_FINAL_20260810"
OUT = ROOT / "results" / "v26_claudecode_integration"
FIG = OUT / "figures_integrated"
SUPP = OUT / "supplementary_figures"
ORIGINAL = ROOT / "review_package" / "claudecode_v26_original" / "source_data"
MANUSCRIPT = ROOT / "manuscript" / "manuscript_plumspectra_v26_integrated.md"
SUPPLEMENT = ROOT / "manuscript" / "supplement_plumspectra_v26_integrated.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


checks: list[dict[str, object]] = []


def check(name: str, observed: object, expected: object) -> None:
    passed = observed == expected
    checks.append({"name": name, "pass": passed, "observed": observed, "expected": expected})
    if not passed:
        raise AssertionError(f"{name}: observed={observed!r}, expected={expected!r}")


# The Windows PNG device rasterises the requested 16.06-in Fig. 4 canvas to
# 7,226 pixels at nominal 450 dpi (16.0578 in effective raster height).
expected_heights = [5.8, 10.2055, 7.4, 16.0578, 7.8, 8.0]
expected_widths = [10.5, 13.125, 10.5, 10.5, 10.5, 10.5]
for index, height in enumerate(expected_heights, start=1):
    png = FIG / f"Figure_{index}_v26.png"
    pdf = FIG / f"Figure_{index}_v26.pdf"
    check(f"Figure {index} PNG exists", png.exists(), True)
    check(f"Figure {index} PDF exists", pdf.exists(), True)
    with Image.open(png) as image:
        check(f"Figure {index} source-canvas width",
              image.width, round(expected_widths[index - 1] * 450))
        check(f"Figure {index} height", image.height, round(height * 450))
    embedded_fonts: set[str] = set()
    for page in PdfReader(pdf).pages:
        font_resources = (page.get("/Resources") or {}).get("/Font") or {}
        for font_ref in font_resources.values():
            embedded_fonts.add(str(font_ref.get_object().get("/BaseFont")))
    allowed_fonts = all("Arial" in name or "Symbol" in name for name in embedded_fonts)
    check(f"Figure {index} PDF embeds only Arial/Symbol font families",
          allowed_fonts and bool(embedded_fonts), True)

check("24 supplementary PNG files", len(list(SUPP.glob("figS*.png"))), 24)
check("24 supplementary PDF files", len(list(SUPP.glob("figS*.pdf"))), 24)
check("obsolete V20 attention file excluded", len(list(SUPP.glob("*attention*"))), 0)
for png in sorted(SUPP.glob("figS*.png")):
    with Image.open(png) as image:
        check(f"{png.stem} uses the 10.5-in supplementary width", image.width, 4725)
        check(f"{png.stem} is rendered at 450 dpi",
              round(float(image.info.get("dpi", (0, 0))[0])), 450)
for pdf in sorted(SUPP.glob("figS*.pdf")):
    embedded_fonts: set[str] = set()
    for page in PdfReader(pdf).pages:
        font_resources = (page.get("/Resources") or {}).get("/Font") or {}
        for font_ref in font_resources.values():
            embedded_fonts.add(str(font_ref.get_object().get("/BaseFont")))
    allowed_fonts = all("Arial" in name or "Symbol" in name for name in embedded_fonts)
    check(f"{pdf.stem} PDF embeds only Arial/Symbol font families",
          allowed_fonts and bool(embedded_fonts), True)

derived = OUT / "figure_data"
derived_files = sorted(derived.glob("*.csv"))
check("eight regenerated figure-data tables", len(derived_files), 8)
intentionally_extended_tables = {
    "fig2_correlation.csv", "fig2_pca_loadings.csv",
    "fig2_pca_scores.csv", "fig2_pca_variance.csv",
    "fig5_complementarity.csv",
}
for path in derived_files:
    if path.name in intentionally_extended_tables:
        continue
    reference = ORIGINAL / path.name
    check(f"figure-data hash matches independent extraction: {path.name}",
          sha256(path), sha256(reference))

joint_loadings = pd.read_csv(derived / "fig2_pca_loadings.csv")
joint_scores = pd.read_csv(derived / "fig2_pca_scores.csv")
joint_variance = pd.read_csv(derived / "fig2_pca_variance.csv")
joint_corr = pd.read_csv(derived / "fig2_correlation.csv")
complementarity = pd.read_csv(derived / "fig5_complementarity.csv")
check("Figure 2 joint PCA has twelve trait loadings", len(joint_loadings), 12)
check("Figure 2 joint PCA uses 4,843 complete fruit", len(joint_scores), 4_843)
check("Figure 2 joint correlation matrix has twelve traits", joint_corr.shape, (12, 13))
check("Figure 2 joint PC1 variance rounds to 60.4%",
      round(float(joint_variance.iloc[0]["explained"]) * 100, 1), 60.4)
check("Figure 2 joint PC1-3 variance rounds to 82.9%",
      round(float(joint_variance.iloc[2]["cumulative"]) * 100, 1), 82.9)
check("Figure 5D complementarity includes all twelve traits",
      [len(complementarity), sorted(complementarity["trait"].tolist())],
      [12, sorted(["FW", "SSC", "pH", "SRF", "RD", "PFD", "MFF", "F6",
                   "LS", "LW", "PRW", "AF"])])

ev = AUDIT / "evidence" / "final_analysis"
mult = pd.read_csv(ev / "multiplicity_adjusted_contrasts.csv")
global_rows = mult[mult["baseline"].eq("Global PLSR")]
check("global PLSR contrast has 12 traits", len(global_rows), 12)
check("all individual global-PLSR cluster intervals exclude zero",
      int((global_rows["bootstrap_ci95_low_pct"] > 0).sum()), 12)
check("minimum global-PLSR RMSE reduction rounds to 12.8%",
      round(float(global_rows["relative_rmse_improvement_pct"].min()), 1), 12.8)
check("maximum global-PLSR RMSE reduction rounds to 51.5%",
      round(float(global_rows["relative_rmse_improvement_pct"].max()), 1), 51.5)

sensitivity = pd.read_csv(ev / "multiplicity_baseline_family_sensitivity.csv")
baseline_family_expected = {
    "12 traits versus Global PLSR": 12,
    "12 traits versus Cultivar-aware PLSR": 10,
    "12 traits versus Nested RBF-SVR": 8,
}
for family, expected in baseline_family_expected.items():
    rows = sensitivity[sensitivity["family_definition"].eq(family)]
    supported = int((rows["simultaneous_ci95_low_within_12_contrast_family"] > 0).sum())
    check(f"baseline-specific simultaneous family: {family}", supported, expected)

extended = pd.read_csv(ev / "extended_cluster_comparisons.csv")
final_vs_cnn = extended[
    extended["candidate"].eq("plumspectra_corrected")
    & extended["baseline"].eq("residual_cnn")
]
check("final-vs-CNN component contrasts", len(final_vs_cnn), 12)
check("all final-vs-CNN contrasts are statistical parity",
      int(final_vs_cnn["claim_status"].eq("statistical_parity").sum()), 12)

pooled = pd.read_csv(ev / "pooled_metrics.csv")
wide = pooled[pooled["model"].isin(["residual_cnn", "plumspectra_corrected"])].pivot(
    index="trait", columns="model", values="rmse"
)
check("CNN branch has lower pooled RMSE for four traits",
      int((wide["residual_cnn"] < wide["plumspectra_corrected"]).sum()), 4)

main_text = MANUSCRIPT.read_text(encoding="utf-8")
supp_text = SUPPLEMENT.read_text(encoding="utf-8")
check("manuscript contains central global-PLSR effect range",
      "12.8–51.5%" in main_text, True)
check("manuscript declares ensemble-CNN parity",
      "statistically indistinguishable" in main_text, True)
check("manuscript avoids causal R2-difference language",
      "increment attributable to the spectrum" in main_text, False)
check("supplement removes obsolete attention heading",
      "Texture-model wavelength attention" in supp_text, False)
check("supplement reaches continuous Figure S24 numbering",
      "Supplementary Figure S24. Complete-pipeline optimisation stability" in supp_text, True)

style_text = (ROOT / "src" / "v26_visual_integration" / "plum_figstyle.R").read_text(encoding="utf-8")
figure2_text = (ROOT / "src" / "v26_visual_integration" / "plot_figure2_v26.R").read_text(encoding="utf-8")
figure4_text = (ROOT / "src" / "v26_visual_integration" / "plot_figure4_v26.R").read_text(encoding="utf-8")
figure_scripts = [
    (ROOT / "src" / "v26_visual_integration" / f"plot_figure{i}_v26.R").read_text(encoding="utf-8")
    for i in range(1, 7)
]
check("theme_classic2 is the integrated theme", "theme_classic2 <- function" in style_text, True)
check("10.5 pt source text floor", "pt_data <- minimum_source_text_pt" in style_text, True)
check("Figure 2A uses one cultivar boxplot layer", figure2_text.count("geom_boxplot") == 1, True)
check("Figure 2A exposes every fruit through one jitter layer",
      figure2_text.count("position_jitter") == 1, True)
check("Figure 2A facet labels keep trait and unit on one line",
      'paste0(" (", unit_lab[all_traits], ")")' in figure2_text, True)
check("Figure 2A is one 3-by-4 facet grid",
      'nrow = 3, ncol = 4' in figure2_text, True)
check("Figure 2A draws axes for all twelve facets",
      'axes = "all", axis.labels = "all_y"' in figure2_text, True)
check("Figure 2A keeps cultivar labels only on the bottom row",
      'axis.labels = "all_y"' in figure2_text, True)
check("Figure 2A cultivar labels are 45-degree and 80-percent size",
      'angle = 45, hjust = 1, vjust = 1' in figure2_text and
      'size = pt_data * 0.80' in figure2_text, True)
check("Figure 2A converts force dimensions exactly from gf to N",
      "gf_to_N <- 0.00980665" in figure2_text and
      "value * gf_to_N" in figure2_text, True)
check("Figure 2A preserves uncalibrated position dimensions as APU",
      'RD = "APU"' in figure2_text and 'LS = "N·APU⁻¹"' in figure2_text and
      'LW = "N·APU"' in figure2_text, True)
check("Figure 2B-E uses one four-column row",
      "structure <- panel_b | panel_c | panel_d | panel_e" in figure2_text, True)
check("Figure 2B-C use all twelve traits",
      "cmat[, all_traits]" in figure2_text and
      "levels = rev(all_traits)" in figure2_text, True)
check("Figure 2B-E row is twenty percent taller",
      "plot_layout(heights = c(1, 0.552))" in figure2_text, True)
check("Figure 2C restores the reviewed loading lollipops",
      "geom_segment(aes(x = 0, xend = loading" in figure2_text, True)
check("Figure 2C includes a dashed zero-reference line",
      'geom_vline(xintercept = 0' in figure2_text and 'linetype = "22"' in figure2_text, True)
check("Figure 2D restores the reviewed paired reliability connectors",
      "geom_line(aes(group = trait)" in figure2_text, True)
check("Figure 2D excludes traits without duplicate measurements",
      'mutate(trait = factor(trait, levels = rev(trait_levels)))' in figure2_text and
      'label = "NA"' not in figure2_text, True)
check("Figure 2E restores whole-cohort density contours",
      "stat_density_2d" in figure2_text, True)
check("Figure 2E uses Claude-reviewed light-plum contour styling",
      "linewidth = 0.22, alpha = 0.45, bins = 6" in figure2_text, True)
check("Figure 2E background fruit visibility is increased",
      "alpha = 0.26" in figure2_text, True)
check("Figure 2E presentation window is cropped without refitting PCA",
      "quantile(scr$PC1, c(0.005, 0.975))" in figure2_text and
      'coord_cartesian(xlim = x_lim, ylim = y_lim, clip = "on")' in figure2_text, True)
check("Figure 2E label leaders use visible neutral grey",
      "segment.colour = muted" in figure2_text and "segment.size = 0.32" in figure2_text, True)
check("Figure 4 renders ordinary global PLSR in the main panel", "panel_b_global" in figure4_text, True)
check("Figure 4 no longer says global PLSR is omitted", "Global PLSR (%.0f-%.0f%%) is omitted" in figure4_text, False)
check("Figure 4B and 4C rows receive 25% more height without reducing Figure 4A",
      "plot_layout(heights = c(3, 1.25, 1.25))" in figure4_text, True)
check("Figure 4A facets are square", "aspect.ratio = 1" in figure4_text, True)
check("main-figure scripts contain no prose captions",
      any("caption = paste" in text or "caption = sprintf" in text or
          "labs(caption =" in text for text in figure_scripts), False)

report = {
    "release": "V26 integrated",
    "checks_passed": sum(item["pass"] for item in checks),
    "checks_total": len(checks),
    "all_passed": all(item["pass"] for item in checks),
    "checks": checks,
}
OUT.mkdir(parents=True, exist_ok=True)
target = OUT / "v26_integration_audit.json"
target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"PASS {report['checks_passed']}/{report['checks_total']} -> {target}")
