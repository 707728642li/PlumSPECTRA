## Candidate figure - reference-fruit calibration efficiency.
##
## This is deliberately kept outside the numbered figure build. It asks a
## deployment question rather than merely redrawing the existing few-shot
## curves: where does the return from an additional labelled fruit flatten?

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(readr)
  library(tidyr); library(forcats); library(scales)
})

args     <- commandArgs(trailingOnly = TRUE)
root     <- if (length(args) >= 1) args[[1]] else "."
out_dir  <- if (length(args) >= 2) args[[2]] else
  file.path(root, "results", "v26_claudecode_integration", "figures_candidate")
style    <- if (length(args) >= 3) args[[3]] else
  file.path(root, "src", "v26_visual_integration", "plum_figstyle.R")
source(style)

summary_path <- file.path(
  root, "results", "v25_external_review_corrections", "final_analysis",
  "fewshot_summary.csv")
few <- read_csv(summary_path, show_col_types = FALSE)
uncertainty_path <- file.path(
  out_dir, "Fewshot_calibration_resampling_uncertainty.csv")
curve_uncertainty <- read_csv(uncertainty_path, show_col_types = FALSE)

trait_order <- c("LW", "SRF", "PFD", "AF", "MFF", "F6", "PRW", "RD", "LS")

## Use the final spectral ensemble and the regularised affine adapter. The
## adapter estimates more than a mean shift but shrinks its slope toward one.
## Zero-shot predictions have no adapter and anchor the same frozen model.
cal <- few |>
  filter(model == "Deep-kernel ensemble", aggregation == "pooled",
         (shots == 0 & adapter == "none") |
           (shots > 0 & adapter == "shrunken_affine")) |>
  mutate(trait = factor(trait, levels = trait_order))

stopifnot(n_distinct(cal$trait) == 9L,
          identical(sort(unique(cal$shots)), c(0, 5, 10, 20, 40, 80)))

curve <- cal |>
  group_by(shots) |>
  summarise(
    gain = median(rmse_gain_pct_mean),
    gain_lo = median(rmse_gain_pct_ci025),
    gain_hi = median(rmse_gain_pct_ci975),
    r2 = median(r2_mean),
    .groups = "drop") |>
  arrange(shots) |>
  mutate(
    added_fruit = shots - lag(shots),
    gain_step = gain - lag(gain),
    gain_per_10 = 10 * gain_step / added_fruit,
    captured = gain / max(gain))

## The uncertainty ribbon is computed at the whole-resample level: within each
## matched calibration draw, take the median across all nine texture endpoints,
## then summarize those 500 medians. This preserves the shared calibration
## draw instead of pretending the nine endpoints are independent replicates.
curve <- curve |>
  rename(trait_median_gain = gain) |>
  left_join(curve_uncertainty, by = "shots") |>
  mutate(gain = median_resample) |>
  arrange(shots) |>
  mutate(
    added_fruit = shots - lag(shots),
    gain_step = gain - lag(gain),
    gain_per_10 = 10 * gain_step / added_fruit,
    captured = gain / max(gain))

## A transparent descriptive elbow: each permissible breakpoint is fitted by a
## continuous two-segment linear regression and compared by residual SSE. This
## is not treated as an inferential population parameter because there are only
## six prespecified label budgets and five observed batches.
knot_candidates <- c(5, 10, 20, 40)
knot_fit <- lapply(knot_candidates, function(k) {
  fit <- lm(gain ~ shots + I(pmax(0, shots - k)), data = curve)
  tibble(knot = k, sse = sum(residuals(fit)^2))
}) |>
  bind_rows() |>
  arrange(sse)
elbow <- knot_fit$knot[[1]]
stopifnot(elbow == 10)

## Operational recommendation remains a band, not a magically exact optimum:
## 20 gives stable positive pooled gains across endpoints; 40 adds robustness;
## the subsequent 40 reference fruit add only ~0.5 percentage points median.
recommended_lo <- 20
recommended_hi <- 40

## ------------------------------------------------ panel a: total return ----
panel_a <- ggplot(curve, aes(shots, gain)) +
  annotate("rect", xmin = recommended_lo, xmax = recommended_hi,
           ymin = -Inf, ymax = Inf, fill = teal, alpha = 0.09) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.45) +
  geom_ribbon(aes(ymin = q025, ymax = q975), fill = plum, alpha = 0.13) +
  geom_line(colour = plum, linewidth = 1.25) +
  geom_point(shape = 21, fill = paper, colour = plum,
             stroke = 0.95, size = 3.0) +
  geom_text(data = subset(curve, shots %in% c(5, 20, 40, 80)),
            aes(label = sprintf("%.1f%%", gain),
                hjust = ifelse(shots == 80, 1, 0.5)), nudge_y = 1.05,
            family = font_family, colour = ink, size = sz(pt_data)) +
  annotate("segment", x = 27, xend = 36, y = 7.1, yend = 7.1,
           colour = teal, linewidth = 0.8,
           arrow = arrow(length = unit(0.10, "in"), type = "closed")) +
  annotate("text", x = 31.5, y = 5.6, label = "practical window",
           family = font_family, fontface = "bold", colour = teal,
           size = sz(pt_data)) +
  scale_x_continuous(breaks = curve$shots, expand = expansion(mult = c(0, 0.03))) +
  annotate("text", x = 59, y = 18.55, label = "95% resampling interval",
           hjust = 0.5, family = font_family, colour = plum,
           size = sz(pt_data)) +
  scale_y_continuous(limits = c(0, 20), breaks = seq(0, 20, 5),
                     labels = label_number(suffix = "%"), expand = c(0, 0)) +
  labs(x = "Reference fruit per held batch",
       y = "Median RMSE reduction vs zero-shot", tag = "a") +
  theme_classic2() +
  theme(plot.margin = margin(8, 8, 5, 5))

## ---------------------------------------------- panel b: marginal return ----
marginal <- curve |> filter(shots > 0)
panel_b <- ggplot(marginal, aes(shots, gain_per_10)) +
  annotate("rect", xmin = recommended_lo, xmax = recommended_hi,
           ymin = -Inf, ymax = Inf, fill = teal, alpha = 0.09) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.45) +
  geom_col(width = c(3.7, 3.7, 7.5, 15, 30), fill = plum, alpha = 0.92) +
  geom_text(aes(label = sprintf("%.1f", gain_per_10)), vjust = -0.55,
            family = font_family, colour = ink, size = sz(pt_data)) +
  annotate("segment", x = 18, xend = 10.8, y = 18.5, yend = 15.2,
           colour = ink, linewidth = 0.55,
           arrow = arrow(length = unit(0.085, "in"), type = "closed")) +
  annotate("text", x = 18.8, y = 18.7,
           label = "descriptive elbow: 10 fruit",
           hjust = 0, family = font_family, fontface = "bold",
           colour = ink, size = sz(pt_data)) +
  scale_x_continuous(breaks = marginal$shots, expand = expansion(mult = c(0.02, 0.03))) +
  scale_y_continuous(limits = c(0, 22), breaks = c(0, 5, 10, 15, 20),
                     expand = c(0, 0)) +
  labs(x = "Cumulative reference-fruit budget",
       y = "Marginal RMSE reduction per 10 fruit (%)", tag = "b") +
  theme_classic2() +
  theme(plot.margin = margin(8, 8, 5, 5))

## --------------------------------------- panel c: endpoint consistency ----
heat <- cal |>
  filter(shots > 0) |>
  mutate(
    trait = fct_relevel(trait, trait_order),
    shots = factor(shots, levels = c(5, 10, 20, 40, 80)))

panel_c <- ggplot(heat, aes(shots, fct_rev(trait), fill = rmse_gain_pct_mean)) +
  geom_tile(colour = paper, linewidth = 0.65) +
  geom_text(aes(label = sprintf("%.1f", rmse_gain_pct_mean)),
            family = font_family, size = sz(pt_data), colour = ink) +
  scale_fill_gradientn(
    colours = c("#F4EFF2", "#CF9FB9", plum),
    limits = c(0, 25), oob = squish,
    breaks = c(0, 5, 10, 15, 20, 25),
    name = "RMSE reduction (%)") +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(x = "Reference fruit per held batch", y = "Texture endpoint", tag = "c") +
  theme_classic2() +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        legend.position = "right", legend.key.height = unit(24, "pt"),
        plot.margin = margin(8, 8, 5, 5))

top <- (panel_a | panel_b) + plot_layout(widths = c(1.05, 0.95))
figure <- top / panel_c + plot_layout(heights = c(1, 0.92))

save_figure(figure, "Fewshot_calibration_efficiency_candidate_v1",
            out_dir, height = 7.9)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(curve, file.path(out_dir, "Fewshot_calibration_efficiency_curve.csv"))
write_csv(knot_fit, file.path(out_dir, "Fewshot_calibration_elbow_sensitivity.csv"))

message(sprintf(
  "Descriptive elbow = %d fruit; practical window = %d-%d fruit; 40->80 adds %.2f percentage points median RMSE reduction.",
  elbow, recommended_lo, recommended_hi,
  curve$gain[curve$shots == 80] - curve$gain[curve$shots == 40]))
