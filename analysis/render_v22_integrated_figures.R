suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(forcats)
  library(patchwork)
  library(scales)
  library(viridis)
  library(ggrepel)
  library(hexbin)
  library(grid)
  library(magick)
  library(jsonlite)
})

project <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
results_version <- Sys.getenv("PLUM_RESULTS_VERSION", unset = "v22_integrated")
version_root <- file.path(project, "results", results_version)
data_dir <- file.path(version_root, "figure_data")
figure_dir <- file.path(version_root, "figures_r")
supp_figure_dir <- file.path(version_root, "supplement/figures")
supp_table_dir <- file.path(version_root, "supplement/tables")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(supp_figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(supp_table_dir, recursive = TRUE, showWarnings = FALSE)

font_family <- "Arial"
if (.Platform$OS.type == "windows") {
  windowsFonts(Arial = windowsFont("Arial"))
}
# Some grid/patchwork operations open an off-screen PostScript/PDF device to
# measure grobs. Alias that device's Arial name to R's bundled ArialMT AFM
# metrics so the same family resolves consistently for PNG, PDF, and SVG.
if ("ArialMT" %in% names(postscriptFonts())) {
  postscriptFonts(Arial = postscriptFonts("ArialMT")[[1]])
}
if ("ArialMT" %in% names(pdfFonts())) {
  pdfFonts(Arial = pdfFonts("ArialMT")[[1]])
}
standard_width <- 10.5
standard_dpi <- 450
minimum_source_text_pt <- 10.5
zero_lower_expand <- expansion(mult = c(0, 0.04))
data_safe_expand <- expansion(mult = c(0.025, 0.035))
ink <- "#17233B"
muted <- "#65738A"
plum <- "#7B2D5F"
plum_light <- "#C77DAA"
teal <- "#238C86"
cyan <- "#5AB5C2"
gold <- "#D89B35"
orange <- "#E67835"
blue <- "#4679A8"
purple <- "#76528B"
paper <- "#FFFFFF"
grid_col <- "#DDE3EA"
model_cols <- c(
  "Global PLSR" = gold,
  "Cultivar-aware PLSR" = blue,
  "Nested RBF-SVR" = purple,
  "Residual CNN" = teal,
  "PlumSPECTRA" = plum,
  "No-neural B50 ensemble" = "#536878",
  "Cultivar-mean null" = "#202020",
  "Fixed-gate sensitivity" = orange
)
cultivar_cols <- c(
  "L313"="#264653", "LA191"="#2A9D8F", "CHL"="#54A4A8", "FTL"="#8AB17D",
  "FWHH"="#E9C46A", "FRL"="#F4A261", "KLD"="#E76F51", "L31"="#B65B80",
  "NL"="#8C6BB1", "QCL"="#6C7FD8", "WD"="#4C9ED9", "WJ"="#65B5C0",
  "WW"="#3A7D6E", "WX"="#91B949", "ZSKLD"="#C58A32"
)

theme_classic2 <- function(base_size = minimum_source_text_pt) {
  base_size <- max(base_size, minimum_source_text_pt)
  theme_classic(base_size = base_size, base_family = font_family) +
    theme(
      plot.background = element_rect(fill = paper, colour = NA),
      panel.background = element_rect(fill = paper, colour = NA),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(colour = ink, linewidth = 0.55),
      axis.ticks = element_line(colour = ink, linewidth = 0.50),
      axis.ticks.length = unit(2.4, "pt"),
      axis.title = element_text(colour = ink, face = "bold"),
      axis.text = element_text(colour = ink),
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      plot.tag = element_text(colour = ink, face = "bold", size = rel(1.35), hjust = 0, vjust = 1),
      plot.tag.position = "topleft",
      strip.text = element_text(colour = ink, face = "bold"),
      strip.background = element_blank(),
      legend.title = element_text(face = "bold", colour = ink),
      legend.text = element_text(colour = ink),
      legend.key = element_rect(fill = paper, colour = NA),
      legend.key.height = unit(9, "pt"),
      legend.key.width = unit(14, "pt"),
      legend.spacing.x = unit(5, "pt"),
      panel.spacing = unit(7, "pt"),
      plot.margin = margin(9, 9, 8, 9)
    )
}
theme_set(theme_classic2())

# Remove frame lines only from categorical sides; retain quantitative axes and
# statistically meaningful reference lines.
theme_categorical_x <- function() {
  theme(axis.line.x = element_blank(), axis.ticks.x = element_blank())
}
theme_categorical_y <- function() {
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank())
}

publication_dimensions <- function(stem, width, height) {
  if (identical(stem, "fig01b_cohort_depth")) {
    return(c(width = 6.5, height = 7.6, dpi = 500))
  }
  c(width = standard_width, height = height * standard_width / width, dpi = standard_dpi)
}

save_plot <- function(plot, stem, width, height, dpi = standard_dpi, svg = FALSE) {
  dims <- publication_dimensions(stem, width, height)
  width <- unname(dims["width"]); height <- unname(dims["height"]); dpi <- unname(dims["dpi"])
  ggsave(file.path(figure_dir, paste0(stem, ".png")), plot, width = width, height = height,
         units = "in", dpi = dpi, bg = "white", limitsize = FALSE)
  ggsave(file.path(figure_dir, paste0(stem, ".pdf")), plot, width = width, height = height,
         units = "in", device = cairo_pdf, bg = "white", limitsize = FALSE)
  if (svg) {
    ggsave(file.path(figure_dir, paste0(stem, ".svg")), plot, width = width, height = height,
           units = "in", device = svglite::svglite, bg = "white", limitsize = FALSE)
  }
}
save_supp <- function(plot, stem, width, height, dpi = standard_dpi) {
  ## Supplementary panels follow the same journal convention as the main set:
  ## interpretation belongs in the legend, never as prose below a subfigure.
  plot <- plot & theme(plot.caption = element_blank(),
                       plot.title = element_blank(),
                       plot.subtitle = element_blank())
  dims <- publication_dimensions(stem, width, height)
  width <- unname(dims["width"]); height <- unname(dims["height"]); dpi <- unname(dims["dpi"])
  ggsave(file.path(supp_figure_dir, paste0(stem, ".png")), plot, width = width, height = height,
         units = "in", dpi = dpi, bg = "white", limitsize = FALSE)
  ggsave(file.path(supp_figure_dir, paste0(stem, ".pdf")), plot, width = width, height = height,
         units = "in", device = cairo_pdf, bg = "white", limitsize = FALSE)
}

trait_registry <- read_csv(file.path(data_dir, "trait_registry.csv"), show_col_types = FALSE) %>%
  arrange(trait_order)
trait_facet_labels_df <- trait_registry %>%
  mutate(
    unit = replace_na(unit, ""),
    facet_label = if_else(unit == "", trait, paste0(trait, " (", unit, ")"))
  ) %>%
  select(trait, facet_label)
trait_facet_labels <- setNames(trait_facet_labels_df$facet_label, trait_facet_labels_df$trait)
trait_labeller <- as_labeller(trait_facet_labels)
trait_unit_label <- function(trait_code) {
  unit <- trait_registry %>% filter(trait == trait_code) %>% pull(unit)
  if (length(unit) == 0 || is.na(unit) || unit == "") "" else paste0(" (", unit, ")")
}
cultivar_registry <- read_csv(file.path(data_dir, "cultivar_registry.csv"), show_col_types = FALSE) %>%
  select(cultivar_ascii, cultivar_code)
trait_levels <- trait_registry$trait
texture_levels <- trait_registry %>% filter(family == "Mechanical texture") %>% pull(trait)
quality_levels <- trait_registry %>% filter(family == "Conventional quality") %>% pull(trait)
cohort_counts <- read_csv(file.path(data_dir, "cohort_counts.csv"), show_col_types = FALSE)
phenotype <- read_csv(file.path(data_dir, "phenotype_long.csv"), show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = trait_levels), cultivar_code = factor(cultivar_code, levels = names(cultivar_cols)))
predictions <- read_csv(file.path(data_dir, "predictions_all12.csv"), show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = trait_levels), cultivar_code = factor(cultivar_code, levels = names(cultivar_cols)))
pooled <- read_csv(file.path(data_dir, "pooled_metrics.csv"), show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = trait_levels))
fold_metrics <- read_csv(file.path(data_dir, "fold_metrics.csv"), show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = trait_levels))
cultivar_metrics <- read_csv(file.path(data_dir, "cultivar_metrics.csv"), show_col_types = FALSE) %>%
  left_join(cultivar_registry, by = "cultivar_ascii") %>%
  mutate(trait = factor(trait, levels = trait_levels))
comparisons <- read_csv(file.path(data_dir, "final_model_cluster_comparisons.csv"), show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = trait_levels))
summary_info <- fromJSON(file.path(data_dir, "dataset_figure_summary.json"))

finalize_model_names <- function(x) {
  x %>% mutate(model_plot = if_else(is_final, "PlumSPECTRA", model_display))
}

# ---- Figure 1B: R-only cohort panel; ImageGen artwork remains separate ----
workflow_paths <- file.path(
  project, "results/v22_integrated/imagegen/fig01a_candidates",
  c(
    "fig01a_candidate_01_flat_editorial.png",
    "fig01a_candidate_02_soft_isometric.png",
    "fig01a_candidate_03_ink_watercolor.png",
    "fig01a_candidate_04_modular_vector.png",
    "fig01a_candidate_05_papercut_25d.png"
  )
)

count_plot <- cohort_counts %>%
  mutate(cultivar_code = fct_reorder(cultivar_code, analysis_samples)) %>%
  ggplot(aes(analysis_samples, cultivar_code)) +
  geom_col(width = 0.72, fill = plum, alpha = 0.92) +
  geom_point(aes(x = source_linked_samples), shape = 21, size = 2.0, stroke = 0.7,
             colour = ink, fill = "white") +
  scale_x_continuous(limits = c(0, NA), expand = zero_lower_expand) +
  labs(tag = "B", x = "Fruit samples", y = NULL) +
  theme_classic2(9.5) +
  theme_categorical_y() +
  theme(legend.position = "none")
save_plot(count_plot, "fig01b_cohort_depth", 6.5, 7.6, dpi = 500)

# ---- Figure 2: phenotype atlas ----
quality_plot <- phenotype %>% filter(trait %in% quality_levels) %>%
  ggplot(aes(cultivar_code, value, colour = cultivar_code)) +
  geom_boxplot(aes(group = cultivar_code), width = 0.62, outlier.shape = NA, linewidth = 0.45,
               fill = "white", colour = ink) +
  geom_point(position = position_jitter(width = 0.17, height = 0, seed = 20260822),
             size = 0.27, alpha = 0.22) +
  facet_wrap(~trait, scales = "free_y", nrow = 1, labeller = trait_labeller) +
  scale_colour_manual(values = cultivar_cols, drop = FALSE) +
  scale_y_continuous(expand = zero_lower_expand) +
  labs(tag = "A", x = NULL, y = "Measured value") +
  theme_classic2() +
  theme_categorical_x() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 9.0),
        legend.position = "none", panel.grid.major.x = element_blank())

texture_plot <- phenotype %>% filter(trait %in% texture_levels) %>%
  ggplot(aes(cultivar_code, value, colour = cultivar_code)) +
  geom_boxplot(aes(group = cultivar_code), width = 0.62, outlier.shape = NA, linewidth = 0.45,
               fill = "white", colour = ink) +
  geom_point(position = position_jitter(width = 0.17, height = 0, seed = 20260823),
             size = 0.22, alpha = 0.16) +
  facet_wrap(~trait, scales = "free_y", ncol = 3, labeller = trait_labeller) +
  scale_colour_manual(values = cultivar_cols, drop = FALSE) +
  scale_y_continuous(expand = zero_lower_expand) +
  labs(tag = "B", x = NULL, y = "Measured value") +
  theme_classic2() +
  theme_categorical_x() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 8.4),
        legend.position = "none", panel.grid.major.x = element_blank(), strip.text = element_text(size = 10.0))

reliability <- read_csv(file.path(data_dir, "texture_reliability.csv"), show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = rev(texture_levels)))
reliability_plot <- reliability %>%
  select(trait, icc_a1, pearson_r) %>%
  pivot_longer(c(icc_a1, pearson_r), names_to = "metric", values_to = "value") %>%
  mutate(metric = recode(metric, icc_a1 = "Absolute-agreement ICC", pearson_r = "Replicate Pearson r")) %>%
  ggplot(aes(value, trait, colour = metric)) +
  geom_point(size = 2.2) +
  scale_colour_manual(values = c("Absolute-agreement ICC" = plum, "Replicate Pearson r" = teal)) +
  scale_x_continuous(limits = c(0.74, 1), breaks = seq(0.75, 1, 0.05),
                     expand = expansion(mult = c(0, 0))) +
  labs(tag = "C", x = "Reliability", y = NULL, colour = NULL) +
  theme_classic2() + theme_categorical_y() + theme(legend.position = "bottom")

corr <- read_csv(file.path(data_dir, "phenotype_spearman_correlation.csv"), show_col_types = FALSE)
corr_mat <- as.matrix(corr[, -1]); rownames(corr_mat) <- corr[[1]]
corr_mat <- corr_mat[trait_levels, trait_levels]
corr_long <- as.data.frame(corr_mat) %>%
  tibble::rownames_to_column("row_trait") %>%
  pivot_longer(-row_trait, names_to = "column_trait", values_to = "correlation") %>%
  mutate(
    row_trait = factor(row_trait, levels = rev(trait_levels)),
    column_trait = factor(column_trait, levels = trait_levels)
  )
corr_panel <- ggplot(corr_long, aes(column_trait, row_trait, fill = correlation)) +
  geom_tile(colour = "white", linewidth = 0.30) +
  scale_fill_gradient2(low = "#3567A5", mid = "#F7F7F7", high = "#B6424A",
                       midpoint = 0, limits = c(-1, 1), breaks = seq(-1, 1, 0.5)) +
  scale_x_discrete(expand = expansion(add = 0)) +
  scale_y_discrete(expand = expansion(add = 0)) +
  coord_fixed() +
  labs(tag = "D", x = NULL, y = NULL, fill = "Spearman\ncorrelation") +
  theme_classic2() + theme_categorical_x() + theme_categorical_y() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 8.8),
        axis.text.y = element_text(size = 8.8),
        legend.position = "right", legend.key.height = unit(18, "pt"),
        legend.title = element_text(size = 8.8), legend.text = element_text(size = 8.4))

pca_loadings <- read_csv(file.path(data_dir, "texture_pca_loadings.csv"), show_col_types = FALSE) %>%
  select(trait, PC1, PC2, PC3) %>% pivot_longer(starts_with("PC"), names_to = "component", values_to = "loading") %>%
  mutate(trait = factor(trait, levels = texture_levels))
pca_loading_plot <- ggplot(pca_loadings, aes(loading, trait, colour = component)) +
  geom_vline(xintercept = 0, colour = "#AEB7C4", linewidth = 0.4) +
  geom_segment(aes(x = 0, xend = loading, yend = trait), linewidth = 0.65) +
  geom_point(size = 1.9) +
  facet_wrap(~component, nrow = 1) +
  scale_colour_manual(values = c(PC1 = plum, PC2 = teal, PC3 = gold), guide = "none") +
  scale_x_continuous(expand = data_safe_expand) +
  labs(tag = "E", x = "PCA loading", y = NULL) +
  theme_classic2() + theme_categorical_y()

pca_scores <- read_csv(file.path(data_dir, "texture_pca_scores.csv"), show_col_types = FALSE)
pca_var <- read_csv(file.path(data_dir, "texture_pca_variance.csv"), show_col_types = FALSE)
pca_centroids <- pca_scores %>%
  group_by(cultivar_code) %>%
  summarise(PC1 = mean(PC1), PC2 = mean(PC2), .groups = "drop")
pca_score_plot <- ggplot(pca_scores, aes(PC1, PC2, colour = cultivar_code)) +
  geom_point(size = 0.28, alpha = 0.22) +
  geom_point(data = pca_centroids, size = 3.0, shape = 21, stroke = 0.7,
             aes(fill = cultivar_code), colour = "white", show.legend = FALSE) +
  geom_text_repel(data = pca_centroids, aes(label = cultivar_code), size = 3.0,
                  family = font_family, fontface = "bold", colour = ink,
                  box.padding = 0.35, point.padding = 0.25, min.segment.length = 0,
                  max.overlaps = Inf, force = 2, max.time = 3, max.iter = 100000,
                  seed = 20260808, segment.size = 0.35, show.legend = FALSE) +
  scale_colour_manual(values = cultivar_cols, drop = FALSE) +
  scale_fill_manual(values = cultivar_cols, drop = FALSE) +
  scale_x_continuous(expand = data_safe_expand) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(tag = "F", x = sprintf("PC1 (%.1f%%)", 100 * pca_var$variance_explained[1]),
       y = sprintf("PC2 (%.1f%%)", 100 * pca_var$variance_explained[2]), colour = "Cultivar") +
  theme_classic2() + theme(legend.position = "none")

summary_top <- (reliability_plot | corr_panel) + plot_layout(widths = c(0.90, 1.10))
summary_bottom <- (pca_loading_plot | pca_score_plot) + plot_layout(widths = c(0.90, 1.10))
summary_block <- (summary_top / summary_bottom) + plot_layout(heights = c(1.35, 1.0))
fig02 <- quality_plot / texture_plot / summary_block +
  plot_layout(heights = c(0.78, 2.45, 2.45))
save_plot(fig02, "fig02_integrated_phenotype_atlas", 19, 22, dpi = 420)

# ---- Figure 3A: vector architecture ----
box_df <- tribble(
  ~xmin, ~xmax, ~ymin, ~ymax, ~label, ~fill,
  0.3, 1.8, 6.2, 8.7, "228-band NIR\n901–1701 nm\n\nRAW  |  SNV  |  SG1", "#EAF2F8",
  2.3, 4.2, 7.0, 8.7, "Training-selected\nPLSR anchor\n+ cultivar offsets", "#FFF3D9",
  2.3, 4.2, 4.4, 6.2, "12-trait auxiliary\npretraining\n(training fruit only)", "#F4EAF2",
  4.8, 7.4, 4.4, 8.7, "Residual spectral CNN\n\n48-channel stem\n4 multiscale blocks\nkernels 3 / 9 / 21\ndilations 1 / 2 / 4 / 8\nGELU + GroupNorm\nwavelength attention", "#DDF3F0",
  4.8, 7.4, 1.7, 3.6, "Nested RBF-SVR\nnonlinear kernel expert", "#EEE8F5",
  8.0, 10.0, 4.6, 7.7, "Training-internal\nbranch eligibility\n\nresidual CNN\n+ optional 0.5 kernel", "#F7E7EF",
  10.6, 12.5, 4.6, 7.7, "PlumSPECTRA\ntrait-specific head\n\n142,285 parameters\nper residual network", "#E9D5E2",
  13.0, 15.4, 4.6, 7.7, "12 continuous outputs\n\nFW · SSC · pH\n+\n9 texture endpoints", "#EEF1F6"
)
arrow_df <- tribble(
  ~x, ~xend, ~y, ~yend, ~colour,
  1.8, 2.3, 7.45, 7.45, blue,
  1.8, 4.8, 6.65, 6.65, teal,
  4.2, 4.8, 5.3, 5.3, plum,
  4.2, 8.0, 7.85, 6.7, gold,
  7.4, 8.0, 6.4, 6.4, teal,
  7.4, 8.0, 2.65, 5.2, purple,
  10.0, 10.6, 6.15, 6.15, plum,
  12.5, 13.0, 6.15, 6.15, ink
)
architecture_plot <- ggplot() +
  geom_rect(data = box_df, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = fill),
            colour = ink, linewidth = 0.65) +
  geom_segment(data = arrow_df, aes(x = x, xend = xend, y = y, yend = yend, colour = colour),
               linewidth = 1.0, arrow = arrow(length = unit(0.12, "inches"), type = "closed")) +
  geom_text(data = box_df, aes(x = (xmin+xmax)/2, y = (ymin+ymax)/2, label = label),
            family = font_family, colour = ink, lineheight = 1.10, size = 2.75, fontface = "bold") +
  annotate("text", x = 7.85, y = 0.7,
           label = "One independently fitted model per trait  •  five frozen cultivar-stratified outer folds  •  test labels never used for tuning",
           family = font_family, colour = muted, size = 3.15) +
  scale_fill_identity() + scale_colour_identity() +
  coord_cartesian(xlim = c(0, 15.7), ylim = c(0.2, 9.05), clip = "off") +
  labs(tag = "A") +
  theme_void(base_family = font_family) +
  theme(plot.tag = element_text(colour = ink, face = "bold", size = 13, hjust = 0, vjust = 1),
        plot.tag.position = "topleft",
        plot.margin = margin(8, 8, 8, 8))
save_plot(architecture_plot, "fig03a_plumspectra_architecture", 18, 8, dpi = 500, svg = TRUE)

primary_model_levels <- c(
  "Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR",
  "Residual CNN", "PlumSPECTRA"
)
fold_plot <- finalize_model_names(fold_metrics) %>%
  filter(model_plot %in% primary_model_levels)
global_fold <- fold_plot %>% filter(model_plot == "Global PLSR") %>%
  select(trait, outer_fold, global_rmse = rmse)
improvement_fold <- fold_plot %>%
  left_join(global_fold, by = c("trait", "outer_fold")) %>%
  mutate(improvement = 100 * (1 - rmse/global_rmse)) %>%
  filter(model_plot != "Global PLSR") %>%
  group_by(trait, model_plot) %>%
  summarise(mean = mean(improvement), sd = sd(improvement), n = n(), .groups = "drop") %>%
  mutate(se = sd/sqrt(n),
         trait = factor(trait, levels = trait_levels),
         model_plot = factor(model_plot, levels = primary_model_levels[-1]))

improvement_bar <- ggplot(improvement_fold, aes(trait, mean, fill = model_plot)) +
  geom_col(position = position_dodge(width = 0.78), width = 0.68, colour = "white", linewidth = 0.2) +
  geom_errorbar(aes(ymin = mean-se, ymax = mean+se), position = position_dodge(width = 0.78),
                width = 0.18, linewidth = 0.45, colour = ink) +
  scale_fill_manual(values = model_cols, drop = FALSE) +
  scale_y_continuous(
    limits = c(min(0, improvement_fold$mean - improvement_fold$se, na.rm = TRUE), NA),
    expand = zero_lower_expand
  ) +
  labs(tag = "B", x = NULL, y = "RMSE reduction vs global PLSR (%)", fill = NULL) +
  theme_classic2() +
  theme_categorical_x() +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9.2),
        legend.position = "bottom", legend.box = "vertical")

r2_summary <- fold_plot %>%
  group_by(trait, model_plot) %>%
  summarise(mean = mean(r2), sd = sd(r2), n = n(), .groups = "drop") %>%
  mutate(ci = qt(0.975, pmax(n - 1, 1)) * sd/sqrt(n),
         trait = factor(trait, levels = rev(trait_levels)),
         model_plot = factor(model_plot, levels = primary_model_levels))
r2_plot <- ggplot(r2_summary, aes(mean, trait, colour = model_plot)) +
  geom_errorbar(aes(xmin = mean-ci, xmax = mean+ci), orientation = "y",
                position = position_dodge(width = 0.62), width = 0.15, linewidth = 0.42) +
  geom_point(position = position_dodge(width = 0.62), size = 1.8) +
  scale_colour_manual(values = model_cols, drop = FALSE) +
  scale_x_continuous(expand = data_safe_expand) +
  labs(tag = "C", x = expression("Outer-fold " * R^2),
       y = NULL, colour = NULL) +
  theme_classic2() +
  theme_categorical_y() +
  guides(colour = guide_legend(nrow = 2, byrow = TRUE)) +
  theme(legend.position = "bottom", legend.box = "vertical")

final_fold_rmse <- fold_plot %>%
  filter(model_plot == "PlumSPECTRA") %>%
  select(trait, outer_fold, final_rmse = rmse)
comparison_fold_summary <- fold_plot %>%
  filter(model_plot %in% c("Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR")) %>%
  select(trait, outer_fold, baseline_display = model_plot, baseline_rmse = rmse) %>%
  left_join(final_fold_rmse, by = c("trait", "outer_fold")) %>%
  mutate(improvement = 100 * (1 - final_rmse / baseline_rmse)) %>%
  group_by(trait, baseline_display) %>%
  summarise(mean = mean(improvement), se = sd(improvement) / sqrt(n()), n = n(), .groups = "drop") %>%
  mutate(trait = factor(trait, levels = trait_levels),
         baseline_display = factor(baseline_display, levels = c("Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR")))
write_csv(comparison_fold_summary, file.path(data_dir, "fig03d_fivefold_se.csv"))
comparison_levels <- c(
  "Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR",
  "No-neural B50 ensemble", "Cultivar-mean null"
)
comparison_evidence <- comparisons %>%
  transmute(
    trait = factor(trait, levels = trait_levels),
    baseline_display = factor(
      baseline_display,
      levels = comparison_levels
    ),
    improvement = relative_rmse_improvement_pct,
    evidence = if_else(
      relative_improvement_ci_low > 0,
      "95% CI excludes zero",
      "95% CI includes zero"
    )
  )
extended_path <- file.path(data_dir, "extended_cluster_comparisons.csv")
if (file.exists(extended_path)) {
  comparison_evidence <- bind_rows(
    comparison_evidence,
    read_csv(extended_path, show_col_types = FALSE) %>%
      filter(
        candidate == "plumspectra_corrected",
        baseline %in% c("no_neural_b50", "cultivar_mean_null")
      ) %>%
      transmute(
        trait = factor(trait, levels = trait_levels),
        baseline_display = factor(
          recode(baseline,
                 no_neural_b50 = "No-neural B50 ensemble",
                 cultivar_mean_null = "Cultivar-mean null"),
          levels = comparison_levels
        ),
        improvement = relative_rmse_improvement_pct,
        evidence = if_else(
          relative_improvement_ci_low > 0,
          "95% CI excludes zero",
          "95% CI includes zero"
        )
      )
  )
}
comparison_bar <- comparison_evidence %>%
  ggplot(aes(trait, improvement, colour = baseline_display, shape = evidence)) +
  geom_point(position = position_dodge(width = 0.72), size = 2.45, stroke = 0.78) +
  scale_colour_manual(values = c("Global PLSR" = gold, "Cultivar-aware PLSR" = blue,
                                 "Nested RBF-SVR" = purple,
                                 "No-neural B50 ensemble" = "#536878",
                                 "Cultivar-mean null" = "#202020"), drop = FALSE) +
  scale_shape_manual(values = c("95% CI excludes zero" = 16, "95% CI includes zero" = 1)) +
  scale_y_continuous(
    limits = c(min(0, comparison_evidence$improvement, na.rm = TRUE), NA),
    expand = zero_lower_expand
  ) +
  labs(tag = "D", x = NULL, y = "PlumSPECTRA RMSE reduction (%)",
       colour = NULL, shape = NULL) +
  theme_classic2() +
  theme_categorical_x() +
  guides(colour = guide_legend(nrow = 1, order = 1),
         shape = guide_legend(nrow = 1, order = 2)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9.2),
        legend.position = "bottom", legend.box = "vertical")

fig03 <- architecture_plot / (improvement_bar | r2_plot) / comparison_bar +
  plot_layout(heights = c(1.25, 1, 0.92))
save_plot(fig03, "fig03_plumspectra_architecture_performance", 19, 19, dpi = 430)

# ---- Figure 4 and per-cultivar supplements ----
label_metrics <- pooled %>% filter(is_final) %>%
  transmute(trait, metric_label = sprintf("R² = %.2f   r = %.2f", r2, pearson_r))
prediction_labels <- predictions %>% group_by(trait) %>%
  summarise(x = quantile(y_true, 0.03), y = quantile(y_final, 0.97), n = n(), .groups = "drop") %>%
  left_join(label_metrics, by = "trait") %>%
  mutate(label = sprintf("n = %s\n%s", comma(n), metric_label), trait = factor(trait, levels = trait_levels))

prediction_bounds <- predictions %>%
  group_by(trait) %>%
  summarise(bound_min = min(c(y_true, y_final), na.rm = TRUE),
            bound_max = max(c(y_true, y_final), na.rm = TRUE), .groups = "drop") %>%
  rowwise() %>%
  reframe(trait = trait, y_true = c(bound_min, bound_max), y_final = c(bound_min, bound_max)) %>%
  ungroup() %>%
  mutate(trait = factor(trait, levels = trait_levels))

fig04 <- ggplot(predictions, aes(y_true, y_final)) +
  geom_blank(data = prediction_bounds, inherit.aes = FALSE, aes(x = y_true, y = y_final)) +
  geom_hex(bins = 42, colour = NA) +
  geom_abline(slope = 1, intercept = 0, colour = "white", linetype = 2, linewidth = 0.65) +
  geom_label(data = prediction_labels, aes(x = x, y = y, label = label), inherit.aes = FALSE,
             hjust = 0, vjust = 1, size = 2.7, family = font_family, colour = ink,
             fill = alpha("white", 0.86), linewidth = 0) +
  facet_wrap(~trait, scales = "free", ncol = 4, labeller = trait_labeller) +
  scale_x_continuous(expand = data_safe_expand) +
  scale_y_continuous(expand = data_safe_expand) +
  scale_fill_viridis_c(option = "mako", trans = "log10", name = "Fruit density") +
  labs(x = "Observed", y = "Predicted") +
  theme_classic2() +
  theme(legend.position = "bottom", strip.text = element_text(size = 10.0), aspect.ratio = 1)
save_plot(fig04, "fig04_all12_observed_predicted", 18, 14, dpi = 450)

for (i in seq_along(trait_levels)) {
  tr <- trait_levels[i]
  tr_meta <- trait_registry %>% filter(trait == tr)
  sub <- predictions %>% filter(trait == tr)
  sub_bounds <- sub %>%
    group_by(cultivar_code) %>%
    summarise(bound_min = min(c(y_true, y_final), na.rm = TRUE),
              bound_max = max(c(y_true, y_final), na.rm = TRUE), .groups = "drop") %>%
    rowwise() %>%
    reframe(cultivar_code = cultivar_code,
            y_true = c(bound_min, bound_max), y_final = c(bound_min, bound_max)) %>%
    ungroup()
  p <- ggplot(sub, aes(y_true, y_final, colour = cultivar_code)) +
    geom_blank(data = sub_bounds, inherit.aes = FALSE, aes(x = y_true, y = y_final)) +
    geom_abline(slope = 1, intercept = 0, colour = "#9FA9B7", linetype = 2, linewidth = 0.45) +
    geom_point(size = 0.52, alpha = 0.38) +
    facet_wrap(~cultivar_code, scales = "free", ncol = 5) +
    scale_colour_manual(values = cultivar_cols, guide = "none", drop = FALSE) +
    scale_x_continuous(expand = data_safe_expand) +
    scale_y_continuous(expand = data_safe_expand) +
    labs(x = paste0("Observed", trait_unit_label(tr)),
         y = paste0("Predicted", trait_unit_label(tr))) +
    theme_classic2() + theme(strip.text = element_text(size = 10.0), aspect.ratio = 1)
  save_supp(p, sprintf("figS%02d_%s_cultivar_observed_predicted", i + 2, tr), 16, 10.5)
}

# ---- Figure 5: within-cultivar signal and heterogeneity without R2 heatmaps ----
metric_final <- cultivar_metrics %>% finalize_model_names() %>%
  filter(model_plot %in% c("Cultivar-aware PLSR", "PlumSPECTRA")) %>%
  mutate(model_plot = factor(model_plot, levels = c("Cultivar-aware PLSR", "PlumSPECTRA")))
within_r_plot <- ggplot(metric_final, aes(model_plot, pearson_r, colour = model_plot)) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.4) +
  geom_boxplot(width = 0.58, outlier.shape = NA, colour = ink, fill = "white", linewidth = 0.45) +
  geom_point(aes(group = model_plot), position = position_jitter(width = 0.14, seed = 20260824),
             size = 0.45, alpha = 0.45) +
  facet_wrap(~trait, ncol = 4) +
  scale_colour_manual(values = model_cols, guide = "none") +
  scale_y_continuous(expand = data_safe_expand) +
  labs(tag = "A", x = NULL, y = "Within-cultivar Pearson r") +
  theme_classic2() +
  theme_categorical_x() +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 9.2), strip.text = element_text(size = 10.0))

centered_fold <- predictions %>%
  group_by(trait, outer_fold, cultivar_ascii) %>%
  mutate(y_true_c = y_true - mean(y_true), y_pred_c = y_final - mean(y_final)) %>%
  ungroup() %>% group_by(trait, outer_fold) %>%
  summarise(r2_centered = 1 - sum((y_true_c-y_pred_c)^2)/sum(y_true_c^2), .groups = "drop") %>%
  group_by(trait) %>%
  summarise(mean = mean(r2_centered), sd = sd(r2_centered), n = n(), .groups = "drop") %>%
  mutate(ci = qt(0.975, pmax(n-1, 1))*sd/sqrt(n), trait = factor(trait, levels = rev(trait_levels)))
within_r2_plot <- ggplot(centered_fold, aes(mean, trait)) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.45) +
  geom_col(fill = ifelse(centered_fold$mean >= 0, teal, "#C44E52"), width = 0.62) +
  geom_errorbar(aes(xmin = mean-ci, xmax = mean+ci), orientation = "y",
                width = 0.2, colour = ink, linewidth = 0.5) +
  scale_x_continuous(
    limits = c(min(0, centered_fold$mean - centered_fold$ci, na.rm = TRUE), NA),
    expand = zero_lower_expand
  ) +
  labs(tag = "B", x = expression("Within-cultivar centred " * R^2 * " (mean ± 95% CI)"), y = NULL) +
  theme_classic2() + theme_categorical_y()

gain_cultivar <- cultivar_metrics %>% finalize_model_names() %>%
  filter(model_plot %in% c("Cultivar-aware PLSR", "PlumSPECTRA")) %>%
  select(trait, cultivar_ascii, cultivar_code, model_plot, rmse) %>%
  pivot_wider(names_from = model_plot, values_from = rmse) %>%
  mutate(gain = 100*(1 - PlumSPECTRA/`Cultivar-aware PLSR`),
         trait = factor(trait, levels = trait_levels),
         cultivar_code = factor(cultivar_code, levels = names(cultivar_cols)))
heterogeneity_plot <- ggplot(gain_cultivar, aes(gain, cultivar_code, colour = cultivar_code)) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.45) +
  geom_segment(aes(x = 0, xend = gain, yend = cultivar_code), linewidth = 0.45, alpha = 0.65) +
  geom_point(size = 1.25) +
  facet_wrap(~trait, scales = "free_x", ncol = 4) +
  scale_colour_manual(values = cultivar_cols, guide = "none", drop = FALSE) +
  scale_x_continuous(expand = zero_lower_expand) +
  labs(tag = "C", x = "RMSE reduction (%)", y = NULL) +
  theme_classic2() + theme_categorical_y() +
  theme(strip.text = element_text(size = 10.0), axis.text.y = element_text(size = 9.2))

complementarity <- predictions %>%
  filter(trait %in% texture_levels) %>%
  group_by(trait) %>%
  summarise(
    error_correlation = cor(y_deep-y_true, y_domain_svr-y_true),
    opposite_sign = mean((y_deep-y_true)*(y_domain_svr-y_true) < 0), .groups = "drop"
  ) %>% mutate(trait = factor(trait, levels = texture_levels))
complementarity_plot <- ggplot(complementarity, aes(error_correlation, opposite_sign, label = trait, colour = trait)) +
  geom_point(size = 2.4) +
  geom_text_repel(family = font_family, size = 3.0, min.segment.length = 0, box.padding = 0.35) +
  scale_colour_viridis_d(option = "plasma", guide = "none") +
  scale_x_continuous(expand = data_safe_expand) +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = data_safe_expand) +
  labs(tag = "D", x = "CNN–SVR error correlation", y = "Opposite-sign error fraction") +
  theme_classic2()

fig05 <- (within_r_plot | within_r2_plot) / heterogeneity_plot / complementarity_plot +
  plot_layout(heights = c(1.00, 3.10, 0.80))
save_plot(fig05, "fig05_within_cultivar_heterogeneity", 19, 23, dpi = 430)

# ---- Figure 6: cross-batch boundary ----
v21_counts <- read_csv(file.path(data_dir, "v21_batch_counts.csv"), show_col_types = FALSE) %>%
  mutate(batch_id = fct_reorder(batch_id, samples))
batch_count_plot <- ggplot(v21_counts, aes(samples, batch_id, colour = cultivar_code)) +
  geom_segment(aes(x = 0, xend = samples, yend = batch_id), linewidth = 1.2, colour = "#D9DEE6") +
  geom_point(size = 3.2) +
  scale_colour_manual(values = cultivar_cols) +
  scale_x_continuous(limits = c(0, NA), expand = zero_lower_expand) +
  labs(tag = "A", x = "Held-batch fruit", y = NULL, colour = "Cultivar") +
  theme_classic2() + theme_categorical_y() + theme(legend.position = "bottom")

v21_comp <- read_csv(file.path(data_dir, "v21_batch_comparisons.csv"), show_col_types = FALSE) %>%
  filter(candidate == "deep_kernel") %>%
  mutate(trait = factor(trait, levels = texture_levels),
         baseline_display = factor(recode(baseline, global_pls="Global PLSR", domain_pls="Cultivar-aware PLSR", domain_svr="Nested RBF-SVR"),
                                   levels = c("Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR")))
batch_gain_plot <- ggplot(v21_comp, aes(trait, relative_batch_macro_improvement_pct, fill = baseline_display)) +
  geom_col(position = position_dodge(width = 0.76), width = 0.67, colour = "white", linewidth = 0.2) +
  geom_errorbar(aes(ymin = descriptive_ci_low, ymax = descriptive_ci_high),
                position = position_dodge(width = 0.76), width = 0.18, linewidth = 0.45, colour = ink) +
  scale_fill_manual(values = c("Global PLSR"=gold, "Cultivar-aware PLSR"=blue, "Nested RBF-SVR"=purple)) +
  scale_y_continuous(
    limits = c(min(0, v21_comp$descriptive_ci_low, na.rm = TRUE), NA),
    expand = zero_lower_expand
  ) +
  labs(tag = "B", x = NULL, y = "RMSE reduction (%)", fill = NULL) +
  theme_classic2() +
  theme_categorical_x() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9.2), legend.position = "bottom")

internal_r2 <- pooled %>% filter(is_final, trait %in% texture_levels) %>%
  transmute(trait, internal_r2 = r2)
batch_r2 <- read_csv(file.path(data_dir, "v21_pooled_batch_metrics.csv"), show_col_types = FALSE) %>%
  filter(model == "deep_kernel") %>% transmute(trait, batch_r2 = r2)
slope <- inner_join(internal_r2, batch_r2, by = "trait") %>%
  pivot_longer(c(internal_r2, batch_r2), names_to = "setting", values_to = "r2") %>%
  mutate(setting = recode(setting, internal_r2 = "Same-session\nouter folds", batch_r2 = "Held-batch\naudit"),
         setting = factor(setting, levels = c("Same-session\nouter folds", "Held-batch\naudit")),
         trait = factor(trait, levels = texture_levels))
slope_plot <- ggplot(slope, aes(setting, r2, group = trait, colour = trait)) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.45) +
  geom_line(linewidth = 0.65, alpha = 0.8) + geom_point(size = 2.0) +
  scale_colour_viridis_d(option = "turbo", end = 0.9) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(tag = "C", x = NULL, y = expression(R^2), colour = "Trait") +
  theme_classic2() + theme_categorical_x() + theme(legend.position = "bottom")

v21_batch <- read_csv(file.path(data_dir, "v21_per_batch_metrics.csv"), show_col_types = FALSE) %>%
  filter(model %in% c("global_pls", "deep_kernel")) %>%
  select(trait, batch_id, model, rmse) %>% pivot_wider(names_from = model, values_from = rmse) %>%
  mutate(rmse_ratio = deep_kernel/global_pls, trait = factor(trait, levels = texture_levels))
per_batch_plot <- ggplot(v21_batch, aes(rmse_ratio, batch_id, colour = batch_id)) +
  geom_vline(xintercept = 1, colour = ink, linewidth = 0.45, linetype = 2) +
  geom_point(size = 1.7) +
  facet_wrap(~trait, ncol = 3) +
  scale_colour_viridis_d(option = "cividis", guide = "none") +
  scale_x_continuous(expand = data_safe_expand) +
  labs(tag = "D", x = "Deep-kernel / global-PLSR RMSE", y = NULL) +
  theme_classic2() + theme_categorical_y()

v24_analysis_dir <- if (grepl("^v25", results_version)) {
  file.path(version_root, "final_analysis")
} else {
  file.path(project, "results/v24_hr_strengthening/analysis")
}
v24_multiseed_dir <- if (grepl("^v25", results_version)) {
  file.path(version_root, "multiseed_analysis")
} else {
  file.path(project, "results/v24_hr_strengthening/multiseed_analysis")
}
fewshot_path <- if (file.exists(file.path(data_dir, "fewshot_summary.csv"))) {
  file.path(data_dir, "fewshot_summary.csv")
} else {
  file.path(v24_analysis_dir, "fewshot_summary.csv")
}
fewshot <- read_csv(fewshot_path, show_col_types = FALSE)
fewshot_main <- fewshot %>%
  filter(
    aggregation == "pooled",
    (model == "Deep-kernel ensemble" & adapter %in% c("none", "intercept")) |
      (model == "Batch-mean null (no spectra)" & adapter == "batch_mean_null")
  ) %>%
  mutate(
    model_plot = recode(model,
                        `Deep-kernel ensemble` = "Spectral ensemble",
                        `Batch-mean null (no spectra)` = "Batch-mean null"),
    trait = factor(trait, levels = texture_levels)
  )
fewshot_median <- fewshot_main %>%
  group_by(shots, model_plot) %>%
  summarise(r2_mean = median(r2_mean), .groups = "drop")
fewshot_recovery_plot <- ggplot(
  fewshot_main,
  aes(shots, r2_mean, group = interaction(trait, model_plot), colour = model_plot)
) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.45) +
  geom_line(linewidth = 0.45, alpha = 0.36) +
  geom_point(size = 1.15, alpha = 0.48) +
  geom_line(data = fewshot_median,
            aes(shots, r2_mean, colour = model_plot, group = model_plot),
            inherit.aes = FALSE, linewidth = 1.05) +
  geom_point(data = fewshot_median,
             aes(shots, r2_mean, colour = model_plot),
             inherit.aes = FALSE, shape = 23, size = 2.35, stroke = 0.55,
             fill = "white") +
  scale_colour_manual(values = c("Spectral ensemble" = plum, "Batch-mean null" = ink)) +
  scale_x_continuous(breaks = c(0, 5, 10, 20, 40, 80), limits = c(0, 80),
                     expand = expansion(mult = c(0, 0.015))) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(tag = "D", x = "Labelled fruit per held batch",
       y = expression("Pooled held-batch "*R^2), colour = NULL) +
  theme_classic2() + theme(legend.position = "bottom")

fig06 <- (batch_count_plot | batch_gain_plot) / (slope_plot | fewshot_recovery_plot)
save_plot(fig06, "fig06_crossbatch_boundary", 18, 13, dpi = 440)

# ---- Additional supplements ----
fold_lines <- fold_plot %>% filter(model_plot %in% c("Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR", "PlumSPECTRA"))
figS01 <- ggplot(fold_lines, aes(outer_fold, rmse, colour = model_plot, group = model_plot)) +
  geom_line(linewidth = 0.62) + geom_point(size = 1.35) +
  facet_wrap(~trait, scales = "free_y", ncol = 4, labeller = trait_labeller) +
  scale_colour_manual(values = model_cols) +
  scale_x_continuous(breaks = 1:5, limits = c(1, 5), expand = expansion(mult = c(0, 0))) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(x = "Frozen outer fold", y = "RMSE", colour = NULL) +
  theme_classic2() + theme(legend.position = "bottom", strip.text = element_text(size = 10.0))
save_supp(figS01, "figS01_foldwise_rmse_stability", 16, 12)

branch <- read_csv(file.path(data_dir, "quality_branch_selection.csv"), show_col_types = FALSE) %>%
  filter(trait %in% quality_levels) %>%
  mutate(trait = factor(trait, levels = quality_levels))
figS02 <- ggplot(branch, aes(outer_fold, domain_svr_degradation_vs_pls_pct, colour = trait)) +
  geom_hline(yintercept = 5, colour = ink, linetype = 2, linewidth = 0.55) +
  geom_line(linewidth = 0.7) + geom_point(size = 2) +
  facet_wrap(~trait, scales = "free_y", nrow = 1) +
  scale_colour_manual(values = c(FW=plum, SSC=teal, pH=gold), guide = "none") +
  scale_x_continuous(breaks = 1:5, limits = c(1, 5), expand = expansion(mult = c(0, 0))) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(x = "Outer fold", y = "Domain-SVR degradation vs domain-PLSR inner CV (%)") +
  theme_classic2()
save_supp(figS02, "figS02_quality_branch_selection", 13, 5)

attention <- read_csv(file.path(data_dir, "texture_attention_wavelength.csv"), show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = texture_levels))
att_col <- names(attention)[grepl("attention", names(attention)) & grepl("mean|median", names(attention))][1]
if (!is.na(att_col)) {
  figS15 <- ggplot(attention, aes(wavelength_nm, .data[[att_col]], colour = trait)) +
    geom_line(linewidth = 0.58) + facet_wrap(~trait, ncol = 3, scales = "free_y", labeller = trait_labeller) +
    scale_colour_viridis_d(option = "plasma", guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0))) +
    scale_y_continuous(expand = data_safe_expand) +
    labs(x = "Wavelength (nm)", y = "Relative attention") +
    theme_classic2() + theme(strip.text = element_text(size = 10.0))
  save_supp(figS15, "figS15_texture_wavelength_attention", 14, 10)
}
vip <- read_csv(file.path(data_dir, "quality_pls_vip_wavelength.csv"), show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = quality_levels))
figS16 <- ggplot(vip, aes(wavelength_nm, vip_median, colour = trait, fill = trait)) +
  geom_ribbon(aes(ymin = vip_q25, ymax = vip_q75), alpha = 0.16, colour = NA) +
  geom_line(linewidth = 0.65) + geom_hline(yintercept = 1, linetype = 2, colour = ink, linewidth = 0.45) +
  facet_wrap(~trait, nrow = 1, labeller = trait_labeller) +
  scale_colour_manual(values = c(FW=plum, SSC=teal, pH=gold), guide = "none") +
  scale_fill_manual(values = c(FW=plum, SSC=teal, pH=gold), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0))) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(x = "Wavelength (nm)", y = "Median VIP (IQR)") +
  theme_classic2() + theme(strip.text = element_text(size = 10.0))
save_supp(figS16, "figS16_quality_pls_vip", 14, 5)

# ---- Supplementary training dynamics from authentic selection histories ----
history_index <- expand_grid(trait = trait_levels, outer_fold = 1:5) %>%
  mutate(
    history_root = if_else(
      trait %in% quality_levels,
      if (grepl("^v25", results_version)) {
        file.path(version_root, "ai_quality_domain_anchor_final")
      } else {
        file.path(project, "results/v22_integrated/ai")
      },
      if (grepl("^v25", results_version)) {
        file.path(version_root, "ai_texture_domain_anchor_final")
      } else {
        file.path(project, "results/v20/ai")
      }
    ),
    history_path = file.path(history_root, trait, paste0("fold_", outer_fold), "selection_history.csv")
  )
missing_history <- history_index$history_path[!file.exists(history_index$history_path)]
if (length(missing_history) > 0) {
  stop("Missing authentic selection histories: ", paste(missing_history, collapse = "; "))
}
training_history <- bind_rows(lapply(seq_len(nrow(history_index)), function(i) {
  read_csv(history_index$history_path[i], show_col_types = FALSE) %>%
    transmute(
      trait = history_index$trait[i],
      outer_fold = history_index$outer_fold[i],
      epoch = as.integer(epoch),
      train_loss = train_loss,
      validation_residual_score = validation_residual_score,
      learning_rate = learning_rate
    )
})) %>%
  group_by(trait, outer_fold) %>%
  arrange(epoch, .by_group = TRUE) %>%
  mutate(relative_train_loss = train_loss / first(train_loss)) %>%
  ungroup() %>%
  mutate(trait = factor(trait, levels = trait_levels))

training_summary <- training_history %>%
  group_by(trait, epoch) %>%
  summarise(
    loss_median = median(relative_train_loss, na.rm = TRUE),
    loss_q25 = quantile(relative_train_loss, 0.25, na.rm = TRUE),
    loss_q75 = quantile(relative_train_loss, 0.75, na.rm = TRUE),
    validation_median = median(validation_residual_score, na.rm = TRUE),
    validation_q25 = quantile(validation_residual_score, 0.25, na.rm = TRUE),
    validation_q75 = quantile(validation_residual_score, 0.75, na.rm = TRUE),
    folds_present = n_distinct(outer_fold),
    .groups = "drop"
  )

loss_plot <- ggplot(training_history, aes(epoch, relative_train_loss, group = outer_fold)) +
  geom_line(colour = "#B8C0CB", linewidth = 0.40, alpha = 0.52) +
  geom_ribbon(data = training_summary,
              aes(x = epoch, ymin = loss_q25, ymax = loss_q75, group = trait),
              inherit.aes = FALSE, fill = plum_light, alpha = 0.20) +
  geom_line(data = training_summary, aes(x = epoch, y = loss_median, group = trait),
            inherit.aes = FALSE, colour = plum, linewidth = 0.75) +
  facet_wrap(~trait, scales = "free_x", ncol = 4, labeller = trait_labeller) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.01))) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(tag = "A", x = "Epoch", y = "Training loss / epoch-1 loss") +
  theme_classic2() + theme(strip.text = element_text(size = 10.0))

validation_plot <- ggplot(training_history, aes(epoch, validation_residual_score, group = outer_fold)) +
  geom_line(colour = "#B8C0CB", linewidth = 0.40, alpha = 0.52) +
  geom_ribbon(data = training_summary,
              aes(x = epoch, ymin = validation_q25, ymax = validation_q75, group = trait),
              inherit.aes = FALSE, fill = cyan, alpha = 0.18) +
  geom_line(data = training_summary, aes(x = epoch, y = validation_median, group = trait),
            inherit.aes = FALSE, colour = teal, linewidth = 0.75) +
  facet_wrap(~trait, scales = "free", ncol = 4, labeller = trait_labeller) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.01))) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(tag = "B", x = "Epoch", y = "Validation residual score (lower is better)") +
  theme_classic2() + theme(strip.text = element_text(size = 10.0))

figS17 <- loss_plot / validation_plot
save_supp(figS17, "figS17_training_dynamics", 16, 16)
write_csv(training_history, file.path(data_dir, "training_dynamics.csv"))

# Full cultivar-cluster 95% confidence intervals remain available for inference.
# Figure 3D uses point estimates plus filled/open evidence encoding so that a
# noisy five-fold SEM is not mistaken for an inferential confidence interval.
figS18 <- comparisons %>%
  mutate(
    trait = factor(trait, levels = rev(trait_levels)),
    baseline_display = factor(baseline_display,
                              levels = c("Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR"))
  ) %>%
  ggplot(aes(relative_rmse_improvement_pct, trait, colour = baseline_display)) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.45) +
  geom_errorbar(aes(xmin = relative_improvement_ci_low, xmax = relative_improvement_ci_high),
                orientation = "y", position = position_dodge(width = 0.58),
                width = 0.18, linewidth = 0.50) +
  geom_point(position = position_dodge(width = 0.58), size = 2.0) +
  scale_colour_manual(values = c("Global PLSR" = gold, "Cultivar-aware PLSR" = blue,
                                 "Nested RBF-SVR" = purple)) +
  scale_x_continuous(expand = data_safe_expand) +
  labs(x = "PlumSPECTRA RMSE reduction (cultivar-cluster bootstrap 95% CI; %)",
       y = NULL, colour = NULL) +
  theme_classic2() + theme_categorical_y() + theme(legend.position = "bottom")
save_supp(figS18, "figS18_cluster_bootstrap_contrasts", 11, 8)

# FW was the only final/cultivar-aware-PLSR contrast without a positive cluster
# interval. A targeted audit therefore added four complete pipeline seeds to
# the original fit in every frozen outer fold (25 fits total). Raw seed/fold
# points are shown rather than treating correlated seeds as independent data.
v23_analysis_dir <- file.path(project, "results/v23_multiseed/analysis")
if (grepl("^v25", results_version)) {
  fw_seed_fold <- read_csv(file.path(v24_multiseed_dir, "seed_fold_metrics.csv"),
                           show_col_types = FALSE) %>%
    filter(trait == "FW", candidate == "selected_final") %>%
    rename(improvement_vs_domain_pls_pct = improvement_vs_cultivar_aware_plsr_pct) %>%
    mutate(
      seed_label = recode(seed_label, primary = "primary"),
      outer_fold = factor(outer_fold),
      seed_label = factor(seed_label, levels = c("primary", "seed_repeat_101", "seed_repeat_102"))
    )
  fw_multiseed_fold <- read_csv(file.path(v24_multiseed_dir, "multiseed_fold_metrics.csv"),
                                show_col_types = FALSE) %>%
    filter(trait == "FW", candidate == "selected_final") %>%
    rename(improvement_vs_domain_pls_pct = improvement_vs_cultivar_aware_plsr_pct) %>%
    mutate(outer_fold = factor(outer_fold))
  fw_multiseed_contrast <- read_csv(
    file.path(v24_multiseed_dir, "multiseed_cluster_contrasts.csv"),
    show_col_types = FALSE
  ) %>%
    filter(
      trait == "FW", candidate == "selected_final",
      baseline %in% c("Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR")
    ) %>%
    rename(ci95_low = ci95_low_pct, ci95_high = ci95_high_pct) %>%
    mutate(
      baseline_display = factor(
        baseline,
        levels = rev(c("Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR"))
      )
    )
  fw_prediction_mean_label <- "Three-seed prediction mean"
} else {
  fw_seed_fold <- read_csv(file.path(v23_analysis_dir, "fw_seed_fold_metrics.csv"),
                           show_col_types = FALSE) %>%
    mutate(outer_fold = factor(outer_fold),
           seed_label = factor(seed_label, levels = c("original", paste0("seed_repeat_", 101:104))))
  fw_multiseed_fold <- read_csv(file.path(v23_analysis_dir, "fw_multiseed_fold_metrics.csv"),
                                show_col_types = FALSE) %>%
    mutate(outer_fold = factor(outer_fold))
  fw_multiseed_contrast <- read_csv(
    file.path(v23_analysis_dir, "fw_multiseed_cluster_bootstrap.csv"),
    show_col_types = FALSE
  ) %>%
    mutate(
      baseline_display = recode(
        baseline,
        "global_pls" = "Global PLSR",
        "domain_pls" = "Cultivar-aware PLSR",
        "domain_svr" = "Nested RBF-SVR"
      ),
      baseline_display = factor(
        baseline_display,
        levels = rev(c("Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR"))
      )
    )
  fw_prediction_mean_label <- "Five-seed prediction mean"
}

fw_seed_plot <- ggplot(
  fw_seed_fold,
  aes(outer_fold, improvement_vs_domain_pls_pct, colour = seed_label)
) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.42) +
  geom_point(position = position_jitter(width = 0.10, height = 0, seed = 20260823),
             size = 1.65, alpha = 0.78) +
  geom_point(data = fw_multiseed_fold,
             aes(outer_fold, improvement_vs_domain_pls_pct), inherit.aes = FALSE,
             shape = 23, size = 2.7, stroke = 0.65, colour = ink, fill = plum) +
  scale_colour_manual(values = c("original" = ink, "primary" = ink, "seed_repeat_101" = blue,
                                 "seed_repeat_102" = teal, "seed_repeat_103" = purple,
                                 "seed_repeat_104" = gold), guide = "none") +
  scale_y_continuous(expand = data_safe_expand) +
  labs(tag = "A", x = "Frozen outer fold",
       y = "FW RMSE reduction vs domain PLSR (%)") +
  theme_classic2()

fw_strategy_points <- bind_rows(
  fw_seed_fold %>% filter(seed_label %in% c("original", "primary")) %>%
    transmute(strategy = "Original single seed", outer_fold,
              improvement = improvement_vs_domain_pls_pct),
  fw_multiseed_fold %>%
    transmute(strategy = fw_prediction_mean_label, outer_fold,
              improvement = improvement_vs_domain_pls_pct)
) %>%
  mutate(strategy = factor(strategy, levels = c("Original single seed", fw_prediction_mean_label)))
fw_strategy_means <- fw_strategy_points %>%
  group_by(strategy) %>% summarise(improvement = mean(improvement), .groups = "drop")
fw_strategy_plot <- ggplot(fw_strategy_points, aes(strategy, improvement, colour = strategy)) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.42) +
  geom_point(position = position_jitter(width = 0.055, height = 0, seed = 20260823),
             size = 1.75, alpha = 0.72) +
  geom_point(data = fw_strategy_means, shape = 23, size = 3.0, stroke = 0.65,
             colour = ink, fill = plum) +
  scale_colour_manual(values = c("Original single seed" = blue,
                                 "Five-seed prediction mean" = plum,
                                 "Three-seed prediction mean" = plum), guide = "none") +
  scale_y_continuous(expand = data_safe_expand) +
  labs(tag = "B", x = NULL, y = "Foldwise RMSE reduction (%)") +
  theme_classic2() + theme_categorical_x() +
  theme(axis.text.x = element_text(angle = 18, hjust = 1))

fw_cluster_plot <- ggplot(
  fw_multiseed_contrast,
  aes(relative_improvement_pct, baseline_display, colour = baseline_display)
) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.42) +
  geom_errorbar(aes(xmin = ci95_low, xmax = ci95_high), orientation = "y",
                width = 0.16, linewidth = 0.52) +
  geom_point(size = 2.25) +
  scale_colour_manual(values = c("Global PLSR" = gold, "Cultivar-aware PLSR" = blue,
                                 "Nested RBF-SVR" = purple), guide = "none") +
  scale_x_continuous(expand = data_safe_expand) +
  labs(tag = "C", x = "FW RMSE reduction (cluster 95% CI; %)", y = NULL) +
  theme_classic2() + theme_categorical_y()

figS19 <- fw_seed_plot | fw_strategy_plot | fw_cluster_plot
save_supp(figS19, "figS19_FW_multiseed_robustness", 15.5, 6.2)

# ---- V24 Horticulture Research strengthening analyses ----
fewshot_detail <- fewshot %>%
  filter(model == "Deep-kernel ensemble", aggregation == "pooled") %>%
  mutate(
    trait = factor(trait, levels = texture_levels),
    adapter_display = recode(adapter, none = "No calibration", intercept = "Intercept",
                             shrunken_affine = "Shrunken affine"),
    adapter_display = factor(adapter_display,
                             levels = c("No calibration", "Intercept", "Shrunken affine"))
  )
fewshot_r2_plot <- ggplot(fewshot_detail,
                          aes(shots, r2_mean, colour = adapter_display, fill = adapter_display)) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.40) +
  geom_ribbon(aes(ymin = r2_ci025, ymax = r2_ci975), colour = NA, alpha = 0.10) +
  geom_line(linewidth = 0.62) + geom_point(size = 1.35) +
  facet_wrap(~trait, ncol = 3, scales = "free_y", labeller = trait_labeller) +
  scale_colour_manual(values = c("No calibration" = ink, "Intercept" = blue,
                                 "Shrunken affine" = plum)) +
  scale_fill_manual(values = c("No calibration" = ink, "Intercept" = blue,
                               "Shrunken affine" = plum)) +
  scale_x_continuous(breaks = c(0, 5, 10, 20, 40, 80), limits = c(0, 80),
                     expand = expansion(mult = c(0, 0.01))) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(x = "Labelled fruit per held batch", y = expression("Pooled held-batch "*R^2),
       colour = NULL, fill = NULL) +
  theme_classic2() + theme(legend.position = "bottom", strip.text = element_text(size = 10.0))
save_supp(fewshot_r2_plot, "figS20_heldbatch_fewshot_r2", 14, 11)

fewshot_gain_plot <- fewshot %>%
  filter(model == "Deep-kernel ensemble", aggregation == "batch_macro", shots > 0) %>%
  mutate(
    trait = factor(trait, levels = texture_levels),
    adapter_display = recode(adapter, intercept = "Intercept", shrunken_affine = "Shrunken affine")
  ) %>%
  ggplot(aes(shots, rmse_gain_pct_mean, colour = adapter_display, fill = adapter_display)) +
  geom_ribbon(aes(ymin = rmse_gain_pct_ci025, ymax = rmse_gain_pct_ci975),
              colour = NA, alpha = 0.10) +
  geom_line(linewidth = 0.62) + geom_point(size = 1.35) +
  facet_wrap(~trait, ncol = 3, scales = "free_y", labeller = trait_labeller) +
  scale_colour_manual(values = c("Intercept" = blue, "Shrunken affine" = plum)) +
  scale_fill_manual(values = c("Intercept" = blue, "Shrunken affine" = plum)) +
  scale_x_continuous(breaks = c(5, 10, 20, 40, 80), limits = c(5, 80),
                     expand = expansion(mult = c(0, 0.01))) +
  scale_y_continuous(expand = data_safe_expand) +
  labs(x = "Labelled fruit per held batch", y = "Batch-macro RMSE gain from calibration (%)",
       colour = NULL, fill = NULL) +
  theme_classic2() + theme(legend.position = "bottom", strip.text = element_text(size = 10.0))
save_supp(fewshot_gain_plot, "figS21_heldbatch_fewshot_gain", 14, 11)

texture_profiles <- read_csv(file.path(data_dir, "cultivar_texture_profiles.csv"),
                             show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = texture_levels))
cultivar_order_v24 <- texture_profiles %>%
  distinct(cultivar_code, texture_cluster) %>%
  arrange(texture_cluster, cultivar_code) %>% pull(cultivar_code)
texture_profiles <- texture_profiles %>%
  mutate(cultivar_code = factor(cultivar_code, levels = rev(cultivar_order_v24)))
texture_diversity <- read_csv(file.path(data_dir, "cultivar_texture_diversity.csv"),
                              show_col_types = FALSE) %>%
  mutate(trait = factor(trait, levels = rev(texture_levels)))
profile_plot <- ggplot(texture_profiles,
                       aes(trait, cultivar_code, fill = robust_z_clipped,
                           size = abs(robust_z_clipped))) +
  geom_point(shape = 21, colour = ink, stroke = 0.28) +
  scale_fill_gradient2(low = blue, mid = "white", high = plum, midpoint = 0,
                       limits = c(-3, 3), oob = squish, name = "Robust z") +
  scale_size_continuous(range = c(1.2, 5.0), guide = "none") +
  labs(tag = "A", x = NULL, y = NULL) +
  theme_classic2() + theme_categorical_x() + theme_categorical_y() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom")
diversity_plot <- ggplot(texture_diversity,
                         aes(cultivar_associated_eta_squared, trait)) +
  geom_segment(aes(x = 0, xend = cultivar_associated_eta_squared, yend = trait),
               colour = "#D7DDE5", linewidth = 1.0) +
  geom_point(shape = 21, size = 2.5, stroke = 0.55, colour = ink, fill = teal) +
  scale_x_continuous(limits = c(0, 0.75), breaks = seq(0, 0.75, 0.15),
                     expand = zero_lower_expand) +
  labs(tag = "B", x = expression("Cultivar-associated "*eta^2), y = NULL) +
  theme_classic2() + theme_categorical_y()
figS22 <- profile_plot | diversity_plot
save_supp(figS22, "figS22_cultivar_texture_profiles", 15, 8)

multiplicity_path <- if (file.exists(file.path(data_dir, "multiplicity_adjusted_contrasts.csv"))) {
  file.path(data_dir, "multiplicity_adjusted_contrasts.csv")
} else {
  file.path(v24_analysis_dir, "multiplicity_adjusted_contrasts.csv")
}
multiplicity <- read_csv(multiplicity_path,
                         show_col_types = FALSE) %>%
  mutate(
    trait = factor(trait, levels = rev(trait_levels)),
    baseline = factor(baseline, levels = c("Global PLSR", "Cultivar-aware PLSR", "Nested RBF-SVR"))
  )
multiplicity_forest <- ggplot(multiplicity,
                              aes(relative_rmse_improvement_pct, trait, colour = baseline)) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.42) +
  geom_errorbar(aes(xmin = simultaneous_ci95_low_pct, xmax = simultaneous_ci95_high_pct),
                orientation = "y", position = position_dodge(width = 0.58),
                width = 0.17, linewidth = 0.48) +
  geom_point(aes(shape = supported_simultaneous_0_05),
             position = position_dodge(width = 0.58), size = 1.9, stroke = 0.55) +
  scale_colour_manual(values = c("Global PLSR" = gold, "Cultivar-aware PLSR" = blue,
                                 "Nested RBF-SVR" = purple)) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1), guide = "none") +
  scale_x_continuous(expand = data_safe_expand) +
  labs(tag = "A", x = "RMSE reduction (family-wise simultaneous 95% CI; %)",
       y = NULL, colour = NULL) +
  theme_classic2() + theme_categorical_y() + theme(legend.position = "bottom")
support_counts <- tibble(
  method = factor(c("Unadjusted", "Benjamini–Hochberg", "Holm", "Simultaneous CI"),
                  levels = rev(c("Unadjusted", "Benjamini–Hochberg", "Holm", "Simultaneous CI"))),
  supported = c(sum(multiplicity$p_raw_one_sided < 0.05),
                sum(multiplicity$supported_bh_fdr_0_05),
                sum(multiplicity$supported_holm_0_05),
                sum(multiplicity$supported_simultaneous_0_05))
)
support_plot <- ggplot(support_counts, aes(supported, method)) +
  geom_col(width = 0.62, fill = plum) +
  geom_text(aes(label = paste0(supported, "/36")), hjust = -0.15,
            colour = ink, family = font_family, size = 3.1) +
  scale_x_continuous(limits = c(0, 40), breaks = seq(0, 36, 12), expand = zero_lower_expand) +
  labs(tag = "B", x = "Supported model contrasts", y = NULL) +
  theme_classic2() + theme_categorical_y()
figS23 <- (multiplicity_forest | support_plot) + plot_layout(widths = c(2.3, 1.0))
save_supp(figS23, "figS23_multiplicity_adjusted_contrasts", 15, 8)

qc_v24 <- read_csv(file.path(data_dir, "cultivar_qc_decisions_condensed.csv"),
                   show_col_types = FALSE) %>%
  left_join(cultivar_registry, by = "cultivar_ascii") %>%
  mutate(cultivar_code = if_else(cultivar_ascii == "6.11", "6.11", cultivar_code),
         cultivar_code = fct_reorder(cultivar_code, independent_failure_domains))
qc_domain_plot <- ggplot(qc_v24, aes(independent_failure_domains, cultivar_code,
                                     fill = exclude_from_primary_modeling)) +
  geom_point(shape = 21, size = 3.0, colour = ink, stroke = 0.55) +
  scale_fill_manual(values = c(`TRUE` = plum, `FALSE` = teal), guide = "none") +
  scale_x_continuous(limits = c(0, 2.2), breaks = 0:2, expand = zero_lower_expand) +
  coord_cartesian(clip = "off") +
  labs(tag = "A", x = "Independent measurement-quality domains failed", y = NULL) +
  theme_classic2() + theme_categorical_y()
qc_icc_plot <- ggplot(qc_v24, aes(median_trait_icc, cultivar_code,
                                  size = fruit_count, fill = exclude_from_primary_modeling)) +
  geom_point(shape = 21, colour = ink, stroke = 0.5) +
  scale_fill_manual(values = c(`TRUE` = plum, `FALSE` = "white"), guide = "none") +
  scale_size_continuous(range = c(1.8, 5.0), name = "Fruit") +
  scale_x_continuous(expand = data_safe_expand) +
  labs(tag = "B", x = "Median duplicate-measurement ICC", y = NULL) +
  theme_classic2() + theme_categorical_y() + theme(legend.position = "bottom")
figS24 <- qc_domain_plot | qc_icc_plot
save_supp(figS24, "figS24_cultivar_qc_decision", 14, 8)

if (grepl("^v25", results_version)) {
  multiseed_v24 <- read_csv(file.path(v24_multiseed_dir, "multiseed_summary.csv"),
                            show_col_types = FALSE) %>%
    rename(original_fold_se_pct_points = primary_fold_se_pct_points) %>%
    mutate(
      candidate_display = recode(candidate,
                                 deep_only = "Residual CNN",
                                 selected_final = "Selected final model"),
      trait = factor(trait, levels = rev(trait_levels))
    )
} else {
  multiseed_v24 <- read_csv(file.path(v24_multiseed_dir, "all12_multiseed_summary.csv"),
                            show_col_types = FALSE) %>%
    mutate(candidate_display = "Selected final model",
           trait = factor(trait, levels = rev(trait_levels)))
}
multiseed_final <- multiseed_v24 %>% filter(candidate_display == "Selected final model")
multiseed_se_long <- multiseed_final %>%
  select(trait, original_fold_se_pct_points, multiseed_fold_se_pct_points) %>%
  pivot_longer(-trait, names_to = "strategy", values_to = "fold_se") %>%
  mutate(strategy = recode(strategy,
                           original_fold_se_pct_points = "Primary seed",
                           multiseed_fold_se_pct_points = "Prediction mean"))
multiseed_se_plot <- ggplot(multiseed_final, aes(y = trait)) +
  geom_segment(aes(x = original_fold_se_pct_points,
                   xend = multiseed_fold_se_pct_points, yend = trait),
               colour = "#CBD3DE", linewidth = 0.75) +
  geom_point(data = multiseed_se_long, aes(x = fold_se, colour = strategy),
             size = 2.15) +
  scale_colour_manual(values = c("Primary seed" = blue, "Prediction mean" = plum)) +
  scale_x_continuous(expand = data_safe_expand) +
  labs(tag = "A", x = "Foldwise SE of RMSE reduction (percentage points)",
       y = NULL, colour = NULL) +
  theme_classic2() + theme_categorical_y() + theme(legend.position = "bottom")

multiseed_seed_sd_plot <- ggplot(
  multiseed_v24,
  aes(mean_within_fold_seed_sd_pct_points, trait, fill = candidate_display)
) +
  geom_point(shape = 21, size = 2.45, stroke = 0.55, colour = ink,
             position = position_dodge(width = 0.55)) +
  scale_fill_manual(values = c("Residual CNN" = teal, "Selected final model" = plum)) +
  scale_x_continuous(limits = c(0, NA), expand = zero_lower_expand) +
  labs(tag = "B", x = "Within-fold seed SD (percentage points)", y = NULL, fill = NULL) +
  theme_classic2() + theme_categorical_y() + theme(legend.position = "bottom")

multiseed_cluster_plot <- ggplot(multiseed_final,
                                 aes(cluster_relative_improvement_pct, trait)) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.42) +
  geom_errorbar(aes(xmin = cluster_ci95_low_pct, xmax = cluster_ci95_high_pct),
                orientation = "y", width = 0.16, linewidth = 0.50, colour = plum) +
  geom_point(aes(shape = cluster_supported), size = 2.15, stroke = 0.60,
             colour = plum) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1), guide = "none") +
  scale_x_continuous(expand = data_safe_expand) +
  labs(tag = "C", x = "RMSE reduction vs\ncultivar-aware PLSR (cluster 95% CI; %)",
       y = NULL) +
  theme_classic2() + theme_categorical_y()

figS25 <- multiseed_se_plot | multiseed_seed_sd_plot | multiseed_cluster_plot
save_supp(figS25, "figS25_alltrait_multiseed_stability", 16, 8)

# ---- Raw ImageGen graphical-abstract candidates remain separate from R output ----
ga_paths <- file.path(
  project, "results/v22_integrated/imagegen/graphical_abstract_candidates",
  c(
    "graphical_abstract_candidate_01_flat_editorial.png",
    "graphical_abstract_candidate_02_soft_isometric.png",
    "graphical_abstract_candidate_03_vector_rigorous.png",
    "graphical_abstract_candidate_04_detailed_systems.png",
    "graphical_abstract_candidate_05_papercut_25d.png"
  )
)

# ---- Supplementary tables ----
table_files <- c(
  "cohort_counts.csv", "trait_summary.csv", "pooled_metrics.csv", "fold_metrics.csv",
  "within_cultivar_centered_metrics.csv", "cultivar_metrics.csv", "final_model_cluster_comparisons.csv",
  "quality_branch_selection.csv", "quality_invalid_exclusions.csv", "texture_reliability.csv",
  "v21_pooled_batch_metrics.csv", "v21_batch_comparisons.csv", "v21_per_batch_metrics.csv",
  "training_dynamics.csv", "fig03d_fivefold_se.csv"
)
for (f in table_files) {
  file.copy(file.path(data_dir, f), file.path(supp_table_dir, f), overwrite = TRUE)
}
if (!grepl("^v25", results_version)) {
  for (f in c("fw_seed_fold_metrics.csv", "fw_seed_metadata.csv",
              "fw_multiseed_fold_metrics.csv", "fw_multiseed_cluster_bootstrap.csv")) {
    file.copy(file.path(v23_analysis_dir, f), file.path(supp_table_dir, f), overwrite = TRUE)
  }
}
analysis_table_files <- c(
  "fewshot_summary.csv", "fewshot_minimum_shots.csv",
  "multiplicity_adjusted_contrasts.csv", "multiplicity_baseline_family_sensitivity.csv",
  "multiplicity_strongest_baseline_family.csv"
)
for (f in analysis_table_files) {
  source_path <- if (file.exists(file.path(data_dir, f))) file.path(data_dir, f) else file.path(v24_analysis_dir, f)
  file.copy(source_path, file.path(supp_table_dir, f), overwrite = TRUE)
}
for (f in c("cultivar_texture_profiles.csv", "cultivar_texture_clusters.csv",
            "cultivar_texture_diversity.csv", "cultivar_texture_cluster_selection.csv",
            "cultivar_texture_cluster_leave_one_out.csv", "cultivar_qc_decisions_condensed.csv")) {
  if (file.exists(file.path(data_dir, f))) {
    file.copy(file.path(data_dir, f), file.path(supp_table_dir, f), overwrite = TRUE)
  }
}
multiseed_table_files <- if (grepl("^v25", results_version)) {
  c("multiseed_summary.csv", "seed_fold_metrics.csv", "multiseed_fold_metrics.csv",
    "multiseed_cluster_contrasts.csv", "seed_metadata.csv")
} else {
  c("all12_multiseed_summary.csv", "alltrait_seed_fold_metrics.csv",
    "alltrait_multiseed_fold_metrics.csv", "alltrait_multiseed_cluster_bootstrap.csv",
    "alltrait_seed_metadata.csv")
}
for (f in multiseed_table_files) {
  file.copy(file.path(v24_multiseed_dir, f), file.path(supp_table_dir, f), overwrite = TRUE)
}

manifest <- tibble(
  figure = c(
    "Figure 1B (R-only cohort panel)", "Figure 2", "Figure 3", "Figure 3A SVG",
    "Figure 4", "Figure 5", "Figure 6",
    paste0("Figure 1A ImageGen candidate ", 1:5), paste0("Graphical abstract ImageGen candidate ", 1:5)
  ),
  path = c(
    file.path("figures_r", "fig01b_cohort_depth.png"),
    file.path("figures_r", "fig02_integrated_phenotype_atlas.png"),
    file.path("figures_r", "fig03_plumspectra_architecture_performance.png"),
    file.path("figures_r", "fig03a_plumspectra_architecture.png"),
    file.path("figures_r", "fig04_all12_observed_predicted.png"),
    file.path("figures_r", "fig05_within_cultivar_heterogeneity.png"),
    file.path("figures_r", "fig06_crossbatch_boundary.png"),
    file.path("imagegen", "fig01a_candidates", basename(workflow_paths)),
    file.path("imagegen", "graphical_abstract_candidates", basename(ga_paths))
  ),
  source = c(rep("R", 7), rep("ImageGen (unmerged)", 10))
)
write_csv(manifest, file.path(figure_dir, "figure_manifest.csv"))
cat(sprintf("Rendered %d manifest entries, HR-strengthening supplements, and authentic training dynamics.\n", nrow(manifest)))
