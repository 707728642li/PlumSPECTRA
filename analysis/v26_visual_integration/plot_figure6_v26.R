## Figure 6 (v26) - Deployment boundary.
##
##   a  three evaluation regimes on one axis: same-session outer folds,
##      held-batch audit, unseen cultivar (LOCO)
##   b  LOCO per-cultivar R2, 15 held-out cultivars x 9 endpoints  [NEW]
##   c  held-batch macro RMSE change against three registered baselines
##   d  total calibration return with matched-resampling uncertainty
##   e  marginal return and the descriptive calibration elbow
##
## Panel b is the panel the audit package requires in the main text and the
## published Figure 6 does not contain. Panel a additionally separates pooled
## from cultivar-macro LOCO R2, because the pooled statistic retains
## between-cultivar variance and therefore overstates transfer.
##
## Every value is read from the frozen evidence tables. Nothing is hand-entered.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr)
  library(tidyr); library(readr); library(forcats); library(stringr)
})

args    <- commandArgs(trailingOnly = TRUE)
root    <- if (length(args) >= 1) args[[1]] else
  "./review_package/HR_EXTERNAL_AUDIT_PACKAGE_V25_FINAL_20260810"
out_dir <- if (length(args) >= 2) args[[2]] else "figures"
style   <- if (length(args) >= 3) args[[3]] else "R/plum_figstyle.R"
fig_data <- if (length(args) >= 4) args[[4]] else
  file.path(dirname(dirname(normalizePath(style))), "results",
            "v26_claudecode_integration", "figure_data")
source(style)

ev <- function(...) file.path(root, "evidence", ...)

## ------------------------------------------------------------------ data ----
pooled <- read_csv(ev("final_analysis/pooled_metrics.csv"), show_col_types = FALSE)
batch  <- read_csv(ev("crossbatch/pooled_and_batch_macro_metrics.csv"), show_col_types = FALSE)
loco   <- read_csv(ev("loco/loco_fold_metrics.csv"), show_col_types = FALSE)
boot   <- read_csv(ev("crossbatch/descriptive_batch_bootstrap_comparisons.csv"), show_col_types = FALSE)
few    <- read_csv(ev("final_analysis/fewshot_summary.csv"), show_col_types = FALSE)
few_unc <- read_csv(file.path(fig_data, "fig6_fewshot_resampling_uncertainty.csv"),
                    show_col_types = FALSE)

loco <- loco |>
  mutate(trait = unname(loco_trait_map[target])) |>
  filter(!is.na(trait))
stopifnot(length(unique(loco$trait)) == 9L,
          length(unique(loco$heldout_cultivar)) == 15L,
          nrow(loco) == 135L)

same_session <- pooled |>
  filter(model == "plumspectra_corrected", trait %in% trait_levels) |>
  transmute(trait, regime = "Same-session outer folds", r2)

held_batch <- batch |>
  filter(model == "deep_kernel") |>
  transmute(trait, regime = "Held-batch audit", r2)

## LOCO pooled: variance-weighted recomposition over the held-out cultivars,
## i.e. the pooled R2 the manuscript reports (-0.183 to 0.217).
loco_pooled <- loco |>
  group_by(trait) |>
  summarise(
    ss_res = sum((1 - r2) * (n - 1) * y_sd^2),
    grand  = sum(n * y_mean) / sum(n),
    ss_tot = sum((n - 1) * y_sd^2 + n * (y_mean - grand)^2),
    r2     = 1 - ss_res / ss_tot,
    .groups = "drop") |>
  transmute(trait, regime = "Unseen cultivar\n(LOCO PLSR)", r2)

loco_macro <- loco |>
  group_by(trait) |>
  summarise(r2_macro = mean(r2), .groups = "drop")

message(sprintf("LOCO pooled R2 range   %.3f to %.3f", min(loco_pooled$r2), max(loco_pooled$r2)))
message(sprintf("LOCO macro  R2 range   %.3f to %.3f", min(loco_macro$r2_macro), max(loco_macro$r2_macro)))
message(sprintf("held-batch  R2 range   %.3f to %.3f", min(held_batch$r2), max(held_batch$r2)))

regimes <- bind_rows(same_session, held_batch, loco_pooled) |>
  mutate(trait  = factor(trait, levels = trait_levels),
         regime = factor(regime, levels = names(pal_regime)))

## ------------------------------------------------ panel a: three regimes ----
end_lab <- regimes |> filter(regime == "Unseen cultivar\n(LOCO PLSR)")

panel_a <- ggplot(regimes, aes(x = regime, y = r2, group = trait)) +
  geom_hline(yintercept = 0, colour = "#4B5563", linewidth = 0.55,
             linetype = "22") +
  geom_line(data = ~ subset(.x, regime != "Unseen cultivar\n(LOCO PLSR)"),
            colour = muted, linewidth = 0.45, alpha = 0.75) +
  geom_point(data = ~ subset(.x, regime != "Unseen cultivar\n(LOCO PLSR)"),
             aes(colour = regime), size = 2.3, show.legend = FALSE) +
  geom_point(data = ~ subset(.x, regime == "Unseen cultivar\n(LOCO PLSR)"),
             aes(colour = regime), shape = 1, stroke = 0.9, size = 2.5,
             show.legend = FALSE) +
  scale_colour_manual(values = pal_regime) +
  ggrepel::geom_text_repel(
    data = end_lab, aes(label = trait), nudge_x = 0.28, direction = "y",
    hjust = 0, segment.colour = rule, segment.size = 0.3,
    family = font_family, colour = ink, size = sz(pt_data), min.segment.length = 0,
    max.overlaps = Inf, box.padding = 0.10, seed = 26) +
  scale_x_discrete(expand = expansion(add = c(0.35, 0.85))) +
  scale_y_continuous(limits = c(-0.35, 0.80), breaks = seq(-0.25, 0.75, 0.25)) +
  labs(x = NULL, y = expression(paste("Pooled ", italic(R)^2)), tag = "a") +
  theme_plum(9.6)

## --------------------------------------------------- panel b: LOCO matrix ----
cult_order <- loco |>
  group_by(heldout_cultivar) |>
  summarise(m = mean(r2), .groups = "drop") |>
  arrange(m) |> pull(heldout_cultivar)

loco_tile <- loco |>
  mutate(code    = unname(cultivar_code_map[heldout_cultivar]),
         code    = factor(code, levels = unname(cultivar_code_map[cult_order])),
         trait   = factor(trait, levels = trait_levels),
         clipped = pmax(r2, -3),
         off     = r2 < -3,
         off_high = r2 > 0.3)

panel_b <- ggplot(loco_tile, aes(trait, code, fill = clipped)) +
  geom_tile(colour = paper, linewidth = 0.45) +
  geom_point(data = ~ subset(.x, off), colour = paper, size = 0.55) +
  geom_point(data = ~ subset(.x, off_high), colour = ink, shape = 1,
             stroke = 0.8, size = 1.5) +
  scale_fill_gradient2(low = below, mid = "#F3EDEA", high = teal,
                       midpoint = 0, limits = c(-3, 0.3),
                       oob = scales::squish,
                       breaks = c(-3, -1.5, 0, 0.3),
                       labels = c("\u2264 -3", "-1.5", "0", "\u2265 0.3"),
                       name = expression(paste("LOCO ", italic(R)^2))) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(x = NULL, y = "Held-out cultivar", tag = "b") +
  theme_plum(9.0) +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        legend.position = "right", legend.key.height = unit(20, "pt"),
        legend.title = element_text(size = pt_data), legend.text = element_text(size = pt_data))

## ------------------------------------------ panel c: held-batch baselines ----
base_lab <- c(
  global_pls = "Global\nPLSR\n4/9 better",
  domain_pls = "Cultivar-aware\nPLSR\n9/9 better",
  domain_svr = "Nested\nRBF-SVR\n9/9 better")

cmp <- boot |>
  filter(candidate == "deep_kernel") |>
  mutate(trait    = factor(trait, levels = trait_levels),
         baseline = factor(unname(base_lab[baseline]), levels = unname(base_lab)),
         better   = relative_batch_macro_improvement_pct > 0,
         crosses  = descriptive_ci_low < 0 & descriptive_ci_high > 0)

## Counts are integrated into the equal-height three-line facet titles below,
## keeping the final line aligned and removing labels from the data region.
stopifnot(
  all((cmp |> group_by(baseline) |>
         summarise(w = sum(better), .groups = "drop") |> pull(w)) == c(4, 9, 9)))

panel_c <- ggplot(cmp, aes(x = relative_batch_macro_improvement_pct, y = fct_rev(trait))) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.45, linetype = "22") +
  geom_linerange(aes(xmin = descriptive_ci_low, xmax = descriptive_ci_high),
                 colour = rule, linewidth = 1.1) +
  geom_point(aes(colour = crosses, shape = crosses), size = 1.9) +
  scale_colour_manual(values = c(`FALSE` = plum, `TRUE` = muted),
                      labels = c("interval excludes zero", "interval includes zero"),
                      name = NULL) +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1),
                     labels = c("interval excludes zero", "interval includes zero"),
                     name = NULL) +
  facet_wrap(~ baseline, nrow = 1) +
  scale_x_continuous(limits = c(min(cmp$descriptive_ci_low) - 1,
                                max(cmp$descriptive_ci_high) + 1),
                     breaks = scales::breaks_width(10)) +
  labs(x = "Held-batch macro RMSE change (%)", y = NULL, tag = "c") +
  theme_plum(9.0) +
  theme(legend.position = "bottom", axis.line.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.margin = margin(t = -4), strip.text = element_text(size = pt_data))

## -------------------------------------- panels d/e: calibration efficiency --
## Use the final regularised affine adapter. Zero-shot predictions anchor the
## same frozen ensemble. At each label budget, the 95% band is formed from 500
## matched calibration draws after taking the median across the nine texture
## endpoints within each draw.
few_curve <- few |>
  filter(model == "Deep-kernel ensemble", aggregation == "pooled",
         (shots == 0 & adapter == "none") |
           (shots > 0 & adapter == "shrunken_affine")) |>
  group_by(shots) |>
  summarise(trait_median_gain = median(rmse_gain_pct_mean), .groups = "drop") |>
  left_join(few_unc, by = "shots") |>
  arrange(shots) |>
  mutate(
    gain = median_resample,
    added_fruit = shots - lag(shots),
    gain_step = gain - lag(gain),
    gain_per_10 = 10 * gain_step / added_fruit)

stopifnot(identical(few_curve$shots, c(0, 5, 10, 20, 40, 80)),
          all(few_curve$repeats[few_curve$shots > 0] == 500))

knot_candidates <- c(5, 10, 20, 40)
knot_fit <- bind_rows(lapply(knot_candidates, function(k) {
  fit <- lm(gain ~ shots + I(pmax(0, shots - k)), data = few_curve)
  tibble(knot = k, sse = sum(residuals(fit)^2))
})) |> arrange(sse)
elbow <- knot_fit$knot[[1]]
stopifnot(elbow == 10)

panel_d <- ggplot(few_curve, aes(shots, gain)) +
  annotate("rect", xmin = 20, xmax = 40, ymin = -Inf, ymax = Inf,
           fill = teal, alpha = 0.09) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.45) +
  geom_ribbon(aes(ymin = q025, ymax = q975), fill = plum, alpha = 0.13) +
  geom_line(colour = plum, linewidth = 1.15) +
  geom_point(shape = 21, fill = paper, colour = plum,
             stroke = 0.85, size = 2.45) +
  geom_text(data = subset(few_curve, shots %in% c(5, 20, 40, 80)),
            aes(label = sprintf("%.1f%%", gain),
                hjust = case_when(shots == 5 ~ 0, shots == 80 ~ 1,
                                  TRUE ~ 0.5),
                vjust = case_when(shots == 20 ~ 1.8, shots == 40 ~ -0.8,
                                  TRUE ~ -0.7)),
            family = font_family, colour = ink, size = sz(pt_data)) +
  annotate("text", x = 55, y = 19.1, label = "95% resampling interval",
           family = font_family, colour = plum, hjust = 0.5,
           size = sz(pt_data)) +
  annotate("segment", x = 27, xend = 36, y = 7.4, yend = 7.4,
           colour = teal, linewidth = 0.75,
           arrow = arrow(length = unit(0.075, "in"), type = "closed")) +
  annotate("text", x = 31.5, y = 6.0, label = "practical window",
           family = font_family, fontface = "bold", colour = teal,
           size = sz(pt_data)) +
  scale_x_continuous(breaks = c(0, 10, 20, 40, 80),
                     expand = expansion(mult = c(0, 0.02))) +
  scale_y_continuous(limits = c(0, 20), breaks = c(0, 5, 10, 15, 20),
                     labels = label_number(suffix = "%"), expand = c(0, 0)) +
  labs(x = "Reference fruit per held batch",
       y = "Median RMSE reduction", tag = "d") +
  theme_plum(9.0)

marginal <- few_curve |>
  filter(shots > 0) |>
  mutate(budget = factor(shots, levels = c(5, 10, 20, 40, 80)))

panel_e <- ggplot(marginal, aes(budget, gain_per_10)) +
  annotate("rect", xmin = 2.5, xmax = 4.5, ymin = -Inf, ymax = Inf,
           fill = teal, alpha = 0.09) +
  geom_hline(yintercept = 0, colour = ink, linewidth = 0.45) +
  geom_col(width = 0.68, fill = plum, alpha = 0.92) +
  geom_text(aes(label = sprintf("%.1f", gain_per_10)), vjust = -0.55,
            family = font_family, colour = ink, size = sz(pt_data)) +
  annotate("segment", x = 3.45, xend = 2.05, y = 19.4, yend = 15.2,
           colour = ink, linewidth = 0.50,
           arrow = arrow(length = unit(0.075, "in"), type = "closed")) +
  annotate("text", x = 3.55, y = 19.8,
           label = "descriptive elbow: 10 fruit", hjust = 0.5,
           family = font_family, fontface = "bold", colour = ink,
           size = sz(pt_data)) +
  scale_x_discrete(expand = expansion(add = c(0.25, 0.25))) +
  scale_y_continuous(limits = c(0, 22), breaks = c(0, 5, 10, 15, 20),
                     expand = c(0, 0)) +
  labs(x = "Reference-fruit budget",
       y = "Marginal gain per 10 fruit (%)", tag = "e") +
  theme_plum(9.0)

## ----------------------------------------------------------------- layout ---
## Requested six-column mosaic:
##   A A A C C C
##   B B D D E E
## Panel B's colourbar remains part of its two-column patchwork area; B plus
## its legend therefore has exactly the same allocated width as D or E.
design <- "
AAACCC
BBDDEE
"
figure6 <- panel_a + panel_b + panel_c + panel_d + panel_e +
  plot_layout(design = design, heights = c(1, 1)) &
  theme(plot.caption = element_blank())

save_figure(figure6, "Figure_6_v26", out_dir, height = 8.0)
