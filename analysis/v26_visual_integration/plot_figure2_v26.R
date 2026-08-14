## Figure 2 (v27 visual revision) - Phenotype atlas and texture structure.
##
##   a  all twelve traits as cultivar-resolved jitter + boxplots
##   b  twelve-trait phenotype correlation with hierarchical ordering
##   c  PCA loadings for the first three components
##   d  duplicate-measurement reliability
##   e  fruit-level twelve-trait PCA landscape with cultivar centroids
##
## Panel a is one 3 x 4 facet grid, not separate quality/texture blocks. Every
## facet keeps both axis lines; cultivar tick labels appear only on the bottom
## row. Panels b-e restore the Claude-reviewed encodings in one four-column row.
## Plot-internal prose captions are intentionally absent.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr)
  library(tidyr); library(readr); library(forcats); library(stringr)
})

args    <- commandArgs(trailingOnly = TRUE)
root    <- if (length(args) >= 1) args[[1]] else
  "./review_package/HR_EXTERNAL_AUDIT_PACKAGE_V25_FINAL_20260810"
out_dir <- if (length(args) >= 2) args[[2]] else "figures"
style   <- if (length(args) >= 3) args[[3]] else "R/plum_figstyle.R"
src     <- if (length(args) >= 4) args[[4]] else "source_data"
source(style)

ev <- function(...) file.path(root, "evidence", ...)
sd_ <- function(f) file.path(src, f)

all_traits <- c("FW", "SSC", "pH", trait_levels)

## Publication display units. Force dimensions are converted exactly from gf
## to N using standard gravity (1 gf = 0.00980665 N). Position-derived
## quantities retain the explicitly defined archive position unit (APU),
## because the retained ARC files lack the numeric Motor Steps/mm calibration.
## Consequently LS is not labelled N/mm and LW/PRW are not labelled joules.
gf_to_N <- 0.00980665
force_dimension_traits <- c("SRF", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF")
unit_lab <- c(
  FW = "g", SSC = "%", pH = "", SRF = "N", RD = "APU",
  PFD = "N", MFF = "N", F6 = "N", LS = "N·APU⁻¹",
  LW = "N·APU", PRW = "N·APU", AF = "N"
)
facet_lab <- setNames(
  paste0(all_traits,
         ifelse(unit_lab[all_traits] == "", "",
                paste0(" (", unit_lab[all_traits], ")"))),
  all_traits
)

registry <- read_csv(file.path(root, "evidence/config/v2_trait_registry.csv"),
                     show_col_types = FALSE)
stopifnot(setequal(intersect(registry$abbreviation, all_traits), all_traits))

## ------------------------------------------------------------------ data ----
obs   <- read_csv(sd_("fig2_observed_long.csv"), show_col_types = FALSE)
loads <- read_csv(sd_("fig2_pca_loadings.csv"), show_col_types = FALSE)
vari  <- read_csv(sd_("fig2_pca_variance.csv"), show_col_types = FALSE)
scr   <- read_csv(sd_("fig2_pca_scores.csv"), show_col_types = FALSE)
cmat  <- read_csv(sd_("fig2_correlation.csv"), show_col_types = FALSE)
rel   <- read_csv(ev("final_analysis/texture_reliability_modeling_cohort.csv"),
                  show_col_types = FALSE)

stopifnot(n_distinct(obs$trait) == 12L, n_distinct(obs$cultivar_code) == 15L)

## ----------------------------- panel a: one 3 x 4 cultivar distribution grid ---
obs_a <- obs |>
  mutate(value = if_else(trait %in% force_dimension_traits,
                         value * gf_to_N, value),
         trait = factor(trait, levels = all_traits),
         cultivar_code = factor(cultivar_code, levels = names(cultivar_cols)))

panel_a <- ggplot(obs_a, aes(cultivar_code, value, colour = cultivar_code)) +
  geom_boxplot(aes(group = cultivar_code), width = 0.62, outlier.shape = NA,
               linewidth = 0.45, fill = paper, colour = ink) +
  geom_point(position = position_jitter(width = 0.17, height = 0,
                                        seed = 20260822),
             size = 0.26, alpha = 0.20) +
  facet_wrap(
    ~ trait, scales = "free_y", nrow = 3, ncol = 4,
    labeller = as_labeller(facet_lab),
    axes = "all", axis.labels = "all_y"
  ) +
  scale_colour_manual(values = cultivar_cols, drop = FALSE) +
  scale_x_discrete(expand = expansion(add = c(0.55, 0.55))) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.08))) +
  labs(x = NULL, y = "Measured value", tag = "a") +
  theme_classic2() +
  theme(
    ## facet_wrap draws an axis grob for all twelve panels, while all_y keeps
    ## cultivar labels only on the outer (third) row.
    axis.line = element_line(colour = ink, linewidth = 0.55),
    axis.ticks = element_line(colour = ink, linewidth = 0.50),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1,
                               size = pt_data * 0.80),
    strip.text = element_text(size = pt_data, lineheight = 1.0),
    legend.position = "none",
    panel.spacing.x = unit(8, "pt"),
    panel.spacing.y = unit(8, "pt")
  )

## --------------------------------------------------- panel b: correlation ---
m <- as.matrix(cmat[, all_traits]); rownames(m) <- cmat$trait
ord <- hclust(as.dist(1 - abs(m)), method = "average")$order
lev <- rownames(m)[ord]

corr_long <- as.data.frame(m) |>
  mutate(row = rownames(m)) |>
  pivot_longer(-row, names_to = "col", values_to = "r") |>
  mutate(row = factor(row, levels = lev), col = factor(col, levels = lev))

panel_b <- ggplot(corr_long, aes(col, row, fill = r)) +
  geom_tile(colour = paper, linewidth = 0.4) +
  geom_point(data = ~ subset(.x, abs(r) > 0.90 & as.integer(row) < as.integer(col)),
             colour = paper, size = 0.6) +
  scale_fill_gradient2(low = "#3567A5", mid = "#F7F7F7", high = below,
                       midpoint = 0, limits = c(-1, 1), breaks = c(-1, 0, 1),
                       name = "Pearson r") +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  coord_fixed() +
  labs(x = NULL, y = NULL, tag = "b", caption = NULL) +
  theme_plum() +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        legend.position = "right", legend.key.width = unit(8, "pt"),
        legend.key.height = unit(22, "pt"),
        legend.title = element_text(size = pt_data),
        legend.text = element_text(size = pt_data))

## ------------------------------------------------------- panel c: loadings --
load_long <- loads |>
  pivot_longer(-trait, names_to = "pc", values_to = "loading") |>
  mutate(trait = factor(trait, levels = rev(all_traits)),
         pc = factor(pc, levels = c("PC1", "PC2", "PC3")))

pc_lab <- setNames(
  sprintf("PC%d\n%.1f%%", vari$component[1:3], 100 * vari$explained[1:3]),
  c("PC1", "PC2", "PC3")
)

panel_c <- ggplot(load_long, aes(loading, trait, colour = pc)) +
  ## Claude-reviewed zero reference and lollipop encoding.
  geom_vline(xintercept = 0, colour = rule, linewidth = 0.4,
             linetype = "22") +
  geom_segment(aes(x = 0, xend = loading, yend = trait), linewidth = 0.55) +
  geom_point(size = 1.7) +
  facet_wrap(~ pc, nrow = 1, labeller = as_labeller(pc_lab)) +
  scale_colour_manual(values = c(PC1 = plum, PC2 = teal, PC3 = gold),
                      guide = "none") +
  scale_x_continuous(breaks = c(-0.4, 0, 0.4)) +
  labs(x = "PCA loading", y = NULL, tag = "c", caption = NULL) +
  theme_plum() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        panel.spacing.x = unit(4, "pt"))

## ---------------------------------------------------- panel d: reliability --
rel_plot <- rel |>
  mutate(trait = factor(trait, levels = rev(trait_levels))) |>
  select(trait, icc_a1, pearson_r) |>
  pivot_longer(-trait, names_to = "metric", values_to = "value") |>
  mutate(metric = recode(metric,
                         icc_a1 = "Absolute-agreement ICC",
                         pearson_r = "Replicate Pearson r"))

panel_d <- ggplot(rel_plot, aes(value, trait)) +
  ## Restore the reviewed within-endpoint connector between the two metrics.
  geom_line(aes(group = trait), colour = rule, linewidth = 1.4) +
  geom_point(aes(colour = metric), size = 1.8) +
  scale_colour_manual(values = c("Absolute-agreement ICC" = plum,
                                 "Replicate Pearson r" = teal), name = NULL) +
  scale_x_continuous(limits = c(0.78, 1.0), breaks = c(0.8, 0.9, 1.0)) +
  guides(colour = guide_legend(ncol = 1)) +
  labs(x = "Duplicate reliability", y = NULL, tag = "d", caption = NULL) +
  theme_plum() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.line.x = element_line(colour = ink, linewidth = 0.55),
        axis.ticks.x = element_line(colour = ink, linewidth = 0.50),
        legend.position = "bottom", legend.margin = margin(t = -6),
        legend.text = element_text(size = pt_data))

## ------------------------------------------------------- panel e: PCA map ---
cent <- scr |>
  group_by(cultivar_code) |>
  summarise(PC1 = mean(PC1), PC2 = mean(PC2), n = n(), .groups = "drop")
## Presentation-only focus window. PCA fitting, scores, centroids, and density
## estimation all use the complete cohort; only the displayed coordinate window
## is cropped to the central cloud so a few extremes do not compress the signal.
x_lim <- quantile(scr$PC1, c(0.005, 0.975))
y_lim <- quantile(scr$PC2, c(0.005, 0.975))

panel_e <- ggplot(scr, aes(PC1, PC2)) +
  geom_hline(yintercept = 0, colour = rule, linewidth = 0.35) +
  geom_vline(xintercept = 0, colour = rule, linewidth = 0.35) +
  ## All 4,853 fruit enter this layer and all calculations; the coordinate
  ## window below shows 4,567 central scores for legible presentation.
  geom_point(colour = ink, alpha = 0.26, size = 0.44, stroke = 0) +
  ## Claude-reviewed light-plum whole-cohort contours (not confidence ellipses).
  stat_density_2d(colour = plum, linewidth = 0.22, alpha = 0.45, bins = 6) +
  geom_point(data = cent, aes(size = n), colour = plum, alpha = 0.92) +
  scale_size_area(max_size = 4.6, guide = "none") +
  ggrepel::geom_text_repel(
    data = cent, aes(label = cultivar_code), family = font_family,
    colour = ink, size = sz(pt_data), segment.colour = muted,
    segment.size = 0.32, min.segment.length = 0, box.padding = 0.50,
    point.padding = 0.10, force = 10, force_pull = 0.10,
    max.iter = 200000, max.time = 10, max.overlaps = Inf, seed = 26
  ) +
  coord_cartesian(xlim = x_lim, ylim = y_lim, clip = "on") +
  labs(x = sprintf("PC1 (%.1f%%)", 100 * vari$explained[1]),
       y = sprintf("PC2 (%.1f%%)", 100 * vari$explained[2]),
       tag = "e", caption = NULL) +
  theme_plum()

## ----------------------------------------------------------------- layout ---
structure <- panel_b | panel_c | panel_d | panel_e
structure <- structure + plot_layout(widths = c(0.90, 1.25, 0.75, 1.60))

figure2 <- wrap_elements(full = panel_a) / wrap_elements(full = structure) +
  plot_layout(heights = c(1, 0.552)) &
  theme(plot.caption = element_blank())

## Preserve panel a's physical height while increasing the b-e row by 20%.
save_figure(figure2, "Figure_2_v26", out_dir, height = 10.2055,
            width = standard_width * 1.25)

message(sprintf(
  "panel a  %s fruit-trait observations across 12 traits and 15 cultivars",
  format(nrow(obs_a), big.mark = ",")
))
message(sprintf(
  "panel e  %s individual fruit points at alpha %.2f plus 15 cultivar centroids",
  format(nrow(scr), big.mark = ","), 0.26
))
