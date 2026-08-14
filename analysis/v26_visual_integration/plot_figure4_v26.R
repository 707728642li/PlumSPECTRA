## Figure 4 (v26) - Predictive performance and the size of the claim.
##
##   a  out-of-fold agreement for all 58,206 fruit-trait pairs
##   b  ordinary global PLSR contrast plus a five-baseline strength ladder
##   c  multiplicity ladder: four corrections x five family definitions
##   d  the most conservative statement in the paper, per trait
##
## Three defects in the published performance panels are fixed here.
##   1. The old panel drew its statistics in an opaque box that covered the data;
##      annotations here have no fill.
##   2. Outliers destroyed several agreement panels; axes are clipped to the
##      0.5-99.5 percentile and the excluded count is stated, not hidden.
##   3. The ordinary global PLSR contrast is shown prominently rather than
##      being moved to a footnote. A separate aligned block preserves legibility
##      for the 1-6 percent contrasts against stronger baselines and components.

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
all_traits <- c("FW", "SSC", "pH", trait_levels)

## ------------------------------------------------------------------ data ----
oof    <- read_csv(file.path(src, "fig4_oof.csv"), show_col_types = FALSE)
pooled <- read_csv(ev("final_analysis/pooled_metrics.csv"), show_col_types = FALSE)
mult   <- read_csv(ev("final_analysis/multiplicity_adjusted_contrasts.csv"),
                   show_col_types = FALSE)
sens   <- read_csv(ev("final_analysis/multiplicity_baseline_family_sensitivity.csv"),
                   show_col_types = FALSE)
strong <- read_csv(file.path(src, "multiplicity_independent_strongest_family.csv"),
                   show_col_types = FALSE)
ext    <- read_csv(ev("final_analysis/extended_cluster_comparisons.csv"),
                   show_col_types = FALSE)

stopifnot(nrow(oof) == 58206, nrow(mult) == 36, nrow(strong) == 12)

## --------------------------------------------------- panel a: OOF density ---
r2 <- pooled |> filter(model == "plumspectra_corrected") |> transmute(trait, r2)

clip <- oof |> group_by(trait) |>
  summarise(lo = min(quantile(y_true, 0.005), quantile(y_final, 0.005)),
            hi = max(quantile(y_true, 0.995), quantile(y_final, 0.995)),
            .groups = "drop")

oof_a <- oof |> left_join(clip, by = "trait") |>
  mutate(inside = y_true >= lo & y_true <= hi & y_final >= lo & y_final <= hi,
         facet  = factor(trait, levels = all_traits))

marks <- oof_a |> group_by(trait) |>
  summarise(out = sum(!inside), .groups = "drop") |>
  left_join(r2, by = "trait") |> left_join(clip, by = "trait") |>
  mutate(facet = factor(trait, levels = all_traits),
         lab   = sprintf("italic(R)^2 == '%0.2f'", r2))

n_out <- sum(marks$out)

panel_a <- ggplot(filter(oof_a, inside), aes(y_true, y_final)) +
  geom_bin2d(bins = 34) +
  scale_fill_gradient(low = "#DFE5EC", high = ink, transform = "log10",
                      name = "Fruit", breaks = c(1, 10, 100),
                      labels = c("1", "10", "100")) +
  geom_text(data = marks, aes(x = lo, y = hi, label = lab), parse = TRUE,
            inherit.aes = FALSE, hjust = 0, vjust = 1, family = font_family,
            colour = ink, size = sz(pt_data)) +
  ## Draw the 1:1 reference above the density and annotation layers so it
  ## remains visible over the darkest high-density bins.
  geom_abline(slope = 1, intercept = 0, colour = ink, linewidth = 0.8,
              linetype = "22") +
  facet_wrap(~ facet, scales = "free", nrow = 3) +
  scale_x_continuous(n.breaks = 4, labels = label_number(big.mark = ",")) +
  scale_y_continuous(n.breaks = 4, labels = label_number(big.mark = ",")) +
  labs(x = "Observed", y = "Predicted", tag = "a", caption = NULL) +
  theme_plum(8.4) +
  theme(strip.text = element_text(size = pt_data),
        aspect.ratio = 1,
        panel.spacing = unit(4.5, "pt"),
        legend.position = "right", legend.key.width = unit(7, "pt"),
        legend.key.height = unit(15, "pt"),
        legend.title = element_text(size = pt_data),
        legend.text  = element_text(size = pt_data))

## ------------------------------------------------ panel b: effect ladder ----
## Two evidence files, one quantity: percentage RMSE reduction of the final model
## against a named baseline, with a 15-cluster bootstrap interval.
from_mult <- mult |>
  filter(baseline %in% c("Cultivar-aware PLSR", "Nested RBF-SVR")) |>
  transmute(trait, baseline,
            est = relative_rmse_improvement_pct,
            lo  = bootstrap_ci95_low_pct,
            hi  = bootstrap_ci95_high_pct)

from_ext <- ext |>
  filter(candidate == "plumspectra_corrected",
         baseline %in% c("cultivar_mean_null", "no_neural_b50", "residual_cnn")) |>
  transmute(trait,
            baseline = recode(baseline,
                              cultivar_mean_null = "Cultivar-mean null",
                              no_neural_b50      = "No-neural B50",
                              residual_cnn       = "Residual CNN alone"),
            est = relative_rmse_improvement_pct,
            lo  = relative_improvement_ci_low,
            hi  = relative_improvement_ci_high)

## Facet order is measured, not asserted: rank the five comparators by pooled
## out-of-fold RMSE within each endpoint and order by the mean rank, weakest
## (highest RMSE) first. Ordering them by narrative plausibility instead would
## have put the cultivar-mean null before the cultivar-aware PLSR, which the
## data contradict.
base_key <- c(cultivar_mean_null = "Cultivar-mean null",
              cultivar_aware_pls = "Cultivar-aware PLSR",
              no_neural_b50      = "No-neural B50",
              nested_rbf_svr     = "Nested RBF-SVR",
              residual_cnn       = "Residual CNN alone")

strength <- pooled |>
  filter(model %in% names(base_key)) |>
  select(trait, model, rmse) |>
  group_by(trait) |>
  mutate(rk = rank(rmse)) |>          # 1 = lowest RMSE = strongest
  group_by(model) |>
  summarise(mean_rank = mean(rk), .groups = "drop") |>
  arrange(desc(mean_rank))            # weakest first

base_order <- unname(base_key[strength$model])
null_beaten <- pooled |>
  filter(model %in% c("cultivar_mean_null", "cultivar_aware_pls")) |>
  select(trait, model, rmse) |>
  pivot_wider(names_from = model, values_from = rmse) |>
  summarise(k = sum(cultivar_aware_pls > cultivar_mean_null), n = n())

base_note  <- c("Cultivar-mean null"  = "no spectra",
                "Cultivar-aware PLSR" = "chemometric peer",
                "No-neural B50"       = "ensemble, no network",
                "Nested RBF-SVR"      = "kernel branch",
                "Residual CNN alone"  = "network branch")
base_short <- c("Cultivar-mean null" = "Cultivar-mean\nnull",
                "Cultivar-aware PLSR" = "Cultivar-aware\nPLSR",
                "No-neural B50" = "No-neural\nB50",
                "Nested RBF-SVR" = "Nested\nRBF-SVR",
                "Residual CNN alone" = "Residual\nCNN")

xmax_b <- 8
eff <- bind_rows(from_mult, from_ext) |>
  mutate(baseline = factor(baseline, levels = base_order,
                           labels = base_short[base_order]),
         trait = factor(trait, levels = all_traits),
         holds = lo > 0,
         off   = est > xmax_b,
         est_p = pmin(est, xmax_b),
         lo_p  = pmin(lo, xmax_b),
         hi_p  = pmin(hi, xmax_b))

tally <- eff |> group_by(baseline) |>
  summarise(s = sum(holds), n = n(), .groups = "drop")

glob <- mult |> filter(baseline == "Global PLSR") |>
  summarise(lo = min(relative_rmse_improvement_pct),
            hi = max(relative_rmse_improvement_pct))

global_eff <- mult |>
  filter(baseline == "Global PLSR") |>
  transmute(trait = factor(trait, levels = all_traits),
            est = relative_rmse_improvement_pct,
            lo = bootstrap_ci95_low_pct,
            hi = bootstrap_ci95_high_pct,
            holds = bootstrap_ci95_low_pct > 0)

panel_b_strong <- ggplot(eff, aes(est_p, fct_rev(trait))) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.45, linetype = "22") +
  geom_linerange(aes(xmin = lo_p, xmax = hi_p, colour = holds), linewidth = 1.0,
                 alpha = 0.45) +
  geom_point(data = ~ subset(.x, !off),
             aes(colour = holds, shape = holds), size = 1.8, stroke = 0.75) +
  geom_segment(data = ~ subset(.x, off),
               aes(x = xmax_b - 0.85, xend = xmax_b - 0.22,
                   y = fct_rev(trait), yend = fct_rev(trait)),
               inherit.aes = FALSE, colour = plum, linewidth = 0.55,
               arrow = grid::arrow(length = unit(3.2, "pt"), type = "closed")) +
  geom_text(data = ~ subset(.x, off),
            aes(x = xmax_b - 1.05, label = sprintf("%.0f%%", est)),
            hjust = 1.0, vjust = 0.5, family = font_family, colour = muted,
            size = sz(pt_note)) +
  geom_text(data = tally, aes(x = xmax_b, y = 13.7, label = sprintf("%d/%d", s, n)),
            inherit.aes = FALSE, hjust = 1, vjust = 0.4, family = font_family,
            colour = ink, size = sz(pt_data), fontface = "bold") +
  scale_colour_manual(values = c(`TRUE` = plum, `FALSE` = below),
                      labels = c(`TRUE` = "interval excludes zero",
                                 `FALSE` = "interval includes zero"), name = NULL) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1),
                     labels = c(`TRUE` = "interval excludes zero",
                                `FALSE` = "interval includes zero"), name = NULL) +
  facet_wrap(~ baseline, nrow = 1) +
  scale_x_continuous(breaks = seq(0, 8, 2), expand = expansion(mult = 0.02)) +
  coord_cartesian(xlim = c(-2.7, xmax_b), ylim = c(0.4, 14.2), clip = "off") +
  labs(x = "RMSE reduction vs stronger comparator (%)", y = NULL,
       caption = NULL) +
  theme_plum(8.6) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        strip.text = element_text(size = pt_data, lineheight = 1.05),
        panel.spacing.x = unit(7, "pt"),
        legend.position = "bottom", legend.margin = margin(t = -6, b = 0),
        legend.text = element_text(size = pt_data))

panel_b_global <- ggplot(global_eff, aes(est, fct_rev(trait))) +
  geom_linerange(aes(xmin = lo, xmax = hi), colour = plum, linewidth = 1.0,
                 alpha = 0.45) +
  geom_point(colour = plum, shape = 16, size = 1.9) +
  ## Place every estimate beyond the right CI endpoint so labels never cross
  ## either the point estimate or its horizontal uncertainty interval.
  geom_text(aes(x = hi + 1.0, label = sprintf("%.1f", est)), hjust = 0,
            family = font_family, colour = ink, size = sz(pt_data)) +
  annotate("text", x = 73.5, y = 13.7, label = "12/12",
           hjust = 1, family = font_family, colour = plum,
           size = sz(pt_data), fontface = "bold") +
  scale_x_continuous(breaks = c(0, 20, 40, 60),
                     expand = expansion(mult = c(0, 0.02))) +
  coord_cartesian(xlim = c(0, 74), ylim = c(0.4, 14.2), clip = "off") +
  labs(x = "RMSE reduction vs global PLSR (%)", y = NULL, tag = "b") +
  theme_plum() +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = margin(6, 3, 4, 4))

panel_a <- panel_a + theme(plot.caption = element_blank())
panel_b_strong <- panel_b_strong + theme(plot.caption = element_blank())

panel_b <- (panel_b_global | panel_b_strong) +
  plot_layout(widths = c(0.30, 0.70), guides = "collect") &
  theme(legend.position = "bottom")

## ------------------------------------------- panel c: multiplicity ladder ---
fam <- bind_rows(
  tibble(family = "36-comparison family\n(primary)",
         raw  = sum(mult$p_raw_one_sided < 0.05),
         bh   = sum(mult$supported_bh_fdr_0_05),
         holm = sum(mult$supported_holm_0_05),
         sim  = sum(mult$supported_simultaneous_0_05),
         n    = nrow(mult)),
  sens |> group_by(family_definition) |>
    summarise(raw  = sum(p_raw_one_sided < 0.05),
              bh   = sum(p_bh_within_12_contrast_family < 0.05),
              holm = sum(p_holm_within_12_contrast_family < 0.05),
              sim  = sum(simultaneous_ci95_low_within_12_contrast_family > 0),
              n    = n(), .groups = "drop") |>
    transmute(family = str_replace(family_definition, "^12 traits versus ",
                                   "12-contrast family\nvs "),
              raw, bh, holm, sim, n),
  tibble(family = "post hoc strongest branch-excluded\nbaseline per trait",
         raw  = sum(strong$supported_raw_0_05),
         bh   = sum(strong$supported_bh_fdr_0_05),
         holm = sum(strong$supported_holm_0_05),
         sim  = sum(strong$supported_simultaneous_0_05),
         n    = nrow(strong)))

fam_order <- c("36-comparison family\n(primary)",
               "12-contrast family\nvs Global PLSR",
               "12-contrast family\nvs Cultivar-aware PLSR",
               "12-contrast family\nvs Nested RBF-SVR",
               "post hoc strongest branch-excluded\nbaseline per trait")
stopifnot(setequal(fam$family, fam_order))

ladder <- fam |>
  mutate(family = factor(family, levels = fam_order)) |>
  pivot_longer(c(raw, bh, holm, sim), names_to = "corr", values_to = "k") |>
  mutate(corr = factor(corr, levels = c("raw", "bh", "holm", "sim"),
                       labels = c("raw", "BH FDR", "Holm",
                                  "max-studentised\nsimultaneous")),
         frac = k / n,
         strictest = corr == "max-studentised\nsimultaneous")

n_fam <- nlevels(ladder$family)
k_sim <- which(levels(ladder$corr) == "max-studentised\nsimultaneous")

panel_c <- ggplot(ladder, aes(corr, fct_rev(family))) +
  geom_tile(aes(fill = frac), colour = paper, linewidth = 1.8) +
  geom_text(aes(label = sprintf("%d/%d", k, n), colour = frac > 0.80),
            family = font_family, size = sz(pt_data), fontface = "bold") +
  annotate("rect", xmin = k_sim - 0.5, xmax = k_sim + 0.5,
           ymin = 0.5, ymax = n_fam + 0.5, fill = NA, colour = ink,
           linewidth = 0.8) +
  scale_fill_gradient(low = "#F4EBF0", high = plum, limits = c(0.60, 1.0),
                      guide = "none") +
  scale_colour_manual(values = c(`TRUE` = paper, `FALSE` = ink), guide = "none") +
  scale_x_discrete(position = "top", expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL, tag = "c", caption = NULL) +
  theme_plum(8.6) +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        axis.text.x = element_text(size = pt_data, lineheight = 1.0, vjust = 0,
                                   margin = margin(b = 3)),
        axis.text.y = element_text(size = pt_data, lineheight = 1.0))

## ------------------------------ panel d: most conservative per-trait claim ---
sb <- strong |>
  mutate(trait = factor(trait, levels = all_traits),
         holds = supported_simultaneous_0_05,
         strongest_baseline = factor(strongest_baseline,
                                     levels = c("Cultivar-aware PLSR",
                                                "No-neural B50")))

panel_d <- ggplot(sb, aes(relative_rmse_improvement_pct, fct_rev(trait))) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.45, linetype = "22") +
  geom_linerange(aes(xmin = simultaneous_ci95_low_pct,
                     xmax = simultaneous_ci95_high_pct, colour = holds),
                 linewidth = 1.0, alpha = 0.45) +
  geom_point(aes(colour = holds, shape = holds), size = 1.8, stroke = 0.75) +
  geom_text(aes(x = 5.35, label = as.character(strongest_baseline),
                colour = holds), hjust = 0, family = font_family, size = sz(pt_note),
            show.legend = FALSE) +
  scale_colour_manual(values = c(`TRUE` = plum, `FALSE` = below), guide = "none") +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1), guide = "none") +
  scale_x_continuous(breaks = c(-1, 0, 2, 4)) +
  scale_y_discrete(expand = expansion(add = 0)) +
  coord_cartesian(xlim = c(-1.2, 5.2), ylim = c(0.5, 12.5), clip = "off") +
  labs(x = "RMSE reduction vs the strongest branch-excluded baseline (%)",
       y = NULL, tag = "d", caption = NULL) +
  theme_plum(8.6) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = margin(6, 92, 4, 4))

## ----------------------------------------------------------------- layout ---
panel_c <- panel_c +
  labs(title = "c", tag = NULL) +
  theme(plot.caption = element_blank(), plot.title.position = "plot",
        plot.title = element_text(family = font_family, colour = ink,
                                  size = 14, face = "bold", hjust = 0))
panel_d <- panel_d +
  labs(title = "d", tag = NULL) +
  theme(plot.caption = element_blank(), plot.title.position = "plot",
        plot.title = element_text(family = font_family, colour = ink,
                                  size = 14, face = "bold", hjust = 0))
bottom <- panel_c | panel_d
bottom <- bottom + plot_layout(widths = c(0.94, 1.06))

## Keep panel a at its established absolute height while giving the two lower
## rows 25% more vertical space. Panel c shares its row with panel d, so the
## aligned bottom-row canvas increases as a unit.
figure4 <- panel_a / wrap_elements(full = panel_b) / wrap_elements(full = bottom) +
  plot_layout(heights = c(3, 1.25, 1.25)) &
  theme(plot.caption = element_blank())

save_figure(figure4, "Figure_4_v26", out_dir, height = 16.06)

## ------------------------------------------------------------- validation ---
message(sprintf("panel a  R2 range %.3f to %.3f over %d traits; %s points clipped",
                min(r2$r2), max(r2$r2), nrow(r2), format(n_out, big.mark = ",")))
message("panel b  intervals excluding zero: ",
        paste(sprintf("%s %d/%d", str_replace_all(tally$baseline, "\n", " / "),
                      tally$s, tally$n), collapse = "  |  "))
message(sprintf("panel c  strictest criterion: %s",
                paste(sprintf("%s %d/%d", str_replace_all(fam$family, "\n", " "),
                              fam$sim, fam$n), collapse = "  |  ")))
message(sprintf("panel d  %d of %d traits beat their strongest baseline simultaneously",
                sum(sb$holds), nrow(sb)))
