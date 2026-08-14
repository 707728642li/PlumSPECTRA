## Supplementary Figure S25 - corrected wavelength evidence.
## Six main figures are unchanged. This supplementary panel interprets the
## 60 frozen primary CNN fits and a model-independent, cultivar-resolved
## association analysis.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(readr)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
project <- if (length(args) >= 1) args[[1]] else
  "."
out_dir <- if (length(args) >= 2) args[[2]] else
  file.path(project, "results", "v28_submission_strengthening", "figures")
style <- if (length(args) >= 3) args[[3]] else
  file.path(project, "src", "v26_visual_integration", "plum_figstyle.R")
source(style)

data_dir <- file.path(project, "results", "v28_submission_strengthening")
trait_order <- c("FW", "SSC", "pH", "SRF", "RD", "PFD",
                 "MFF", "F6", "LS", "LW", "PRW", "AF")

attention <- read_csv(file.path(data_dir, "current_model_attention_wavelength.csv"),
                      show_col_types = FALSE) |>
  mutate(trait = factor(trait, levels = rev(trait_order)))
association <- read_csv(file.path(data_dir, "within_cultivar_snv_association_wavelength.csv"),
                        show_col_types = FALSE) |>
  mutate(trait = factor(trait, levels = rev(trait_order)))

stopifnot(nrow(attention) == 12L * 228L,
          nrow(association) == 12L * 228L,
          length(unique(attention$trait)) == 12L,
          length(unique(association$trait)) == 12L)

edge_lines <- data.frame(x = c(920, 1685))

panel_a <- ggplot(attention, aes(wavelength_nm, trait, fill = attention_median)) +
  geom_raster(interpolate = FALSE) +
  geom_vline(data = edge_lines, aes(xintercept = x),
             inherit.aes = FALSE, colour = muted,
             linewidth = 0.45, linetype = "22") +
  scale_fill_gradient2(
    low = "#3E73A8", mid = paper, high = plum,
    midpoint = 1, limits = c(0.94, 1.14), oob = squish,
    name = "Attention / uniform"
  ) +
  scale_x_continuous(
    breaks = c(900, 1100, 1300, 1500, 1700),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(x = NULL, y = NULL) +
  theme_classic2() +
  theme(
    axis.line = element_line(colour = ink, linewidth = 0.55),
    axis.ticks.y = element_blank(),
    legend.position = "right",
    plot.margin = margin(7, 8, 4, 5)
  )

panel_b <- ggplot(association, aes(wavelength_nm, trait, fill = median_r)) +
  geom_raster(interpolate = FALSE) +
  geom_vline(data = edge_lines, aes(xintercept = x),
             inherit.aes = FALSE, colour = muted,
             linewidth = 0.45, linetype = "22") +
  scale_fill_gradient2(
    low = "#3E73A8", mid = paper, high = plum,
    midpoint = 0, limits = c(-0.30, 0.30), oob = squish,
    name = "Median within-cultivar r"
  ) +
  scale_x_continuous(
    breaks = c(900, 1100, 1300, 1500, 1700),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(x = "Wavelength (nm)", y = NULL) +
  theme_classic2() +
  theme(
    axis.line = element_line(colour = ink, linewidth = 0.55),
    axis.ticks.y = element_blank(),
    legend.position = "right",
    plot.margin = margin(4, 8, 6, 5)
  )

combined <- panel_a / panel_b +
  plot_annotation(tag_levels = "a") +
  plot_layout(heights = c(1, 1), guides = "keep")

save_figure(combined, "Figure_S25_v28_wavelength_evidence", out_dir,
            height = 7.3, width = standard_width)

message(sprintf("attention range %.3f to %.3f",
                min(attention$attention_median), max(attention$attention_median)))
message(sprintf("association range %.3f to %.3f",
                min(association$median_r), max(association$median_r)))
