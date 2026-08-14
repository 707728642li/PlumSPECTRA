## Figure 5 (v26) - Identity-conditioned signal and model complementarity.
##
##   a  pooled R2 compared with an out-of-fold cultivar-mean null
##   b  within-cultivar centred R2 for the model ladder
##   c  every cultivar x endpoint cell, 180 of them, with the failures marked
##   d  error complementarity of the two experts the ensemble averages
##
## The published Figure 5 used an unreadable 180-cell lollipop array. This
## revision separates pooled comparison with an OOF cultivar-mean null from
## within-cultivar performance and component-error complementarity.
##
## Every trait ordering in this figure is the same: ascending R2 difference
## from panel a, so a row that is low in one panel is low in all of them.

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

## ------------------------------------------------------------------ data ----
nullc  <- read_csv(ev("final_analysis/pooled_null_centered_r2.csv"),
                   show_col_types = FALSE)
within <- read_csv(ev("final_analysis/within_cultivar_centered_metrics.csv"),
                   show_col_types = FALSE)
cells  <- read_csv(file.path(src, "fig5_cultivar_cells.csv"), show_col_types = FALSE)
comp   <- read_csv(file.path(src, "fig5_complementarity.csv"), show_col_types = FALSE)

stopifnot(nrow(nullc) == 12, nrow(cells) == 180, nrow(comp) == 12)

## One trait order for the whole figure: ascending R2 difference versus the
## out-of-fold cultivar-mean null. This is a predictive contrast, not a causal
## decomposition or a uniquely attributable spectral effect.
dec <- nullc |>
  transmute(trait,
            identity  = cultivar_mean_null,
            pooled    = plumspectra_corrected,
            increment = plumspectra_corrected - cultivar_mean_null,
            centred   = plumspectra_centered_r2) |>
  arrange(increment)
trait_rank <- dec$trait
tf <- function(x) factor(x, levels = trait_rank)

## --------------------------------------------- panel a: null comparison ----
dec_long <- dec |>
  select(trait, identity, increment) |>
  pivot_longer(-trait, names_to = "component", values_to = "r2") |>
  mutate(trait = tf(trait),
         component = factor(component, levels = c("identity", "increment"),
                            labels = c("OOF cultivar-mean null",
                                       "PlumSPECTRA minus null")))

panel_a <- ggplot(dec_long, aes(r2, trait, fill = component)) +
  geom_col(position = position_stack(reverse = TRUE), width = 0.66) +
  geom_text(data = mutate(dec, trait = tf(trait)),
            aes(x = pooled, y = trait, label = sprintf("+%.3f", increment)),
            inherit.aes = FALSE, hjust = -0.16, family = font_family,
            colour = plum, size = sz(pt_data)) +
  scale_fill_manual(values = c("OOF cultivar-mean null" = "#C7CED8",
                               "PlumSPECTRA minus null" = plum), name = NULL) +
  scale_x_continuous(limits = c(0, 1.0), breaks = seq(0, 0.8, 0.2),
                     expand = expansion(mult = c(0, 0))) +
  scale_y_discrete(expand = expansion(add = c(0.6, 1.0))) +
  guides(fill = guide_legend(nrow = 2)) +
  labs(x = expression(paste("Pooled OOF ", italic(R)^2)),
       y = NULL, tag = "a") +
  theme_plum(8.6) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        legend.position = "bottom", legend.margin = margin(t = -6),
        legend.text = element_text(size = pt_data),
        legend.key.height = unit(8, "pt"), legend.key.width = unit(11, "pt"))

## -------------------------------------------- panel b: centred model ladder --
keep_models <- c(cultivar_aware_pls = "Cultivar-aware PLSR",
                 nested_rbf_svr     = "Nested RBF-SVR",
                 no_neural_b50      = "No-neural B50 ensemble",
                 residual_cnn       = "Residual CNN",
                 plumspectra_corrected = "PlumSPECTRA")

lad <- within |>
  filter(model %in% names(keep_models)) |>
  transmute(trait = tf(trait),
            model = factor(keep_models[model], levels = unname(keep_models)),
            r2)

span <- lad |> group_by(trait) |>
  summarise(lo = min(r2), hi = max(r2), .groups = "drop")

panel_b <- ggplot(lad, aes(r2, trait)) +
  geom_vline(xintercept = 0, colour = ink, linewidth = 0.45, linetype = "22") +
  geom_point(aes(colour = model, shape = model), size = 1.9, stroke = 0.8) +
  scale_colour_manual(values = pal_model[unname(keep_models)], name = NULL) +
  scale_shape_manual(values = c(1, 1, 2, 16, 16), name = NULL) +
  scale_x_continuous(limits = c(-0.09, 0.42), breaks = seq(0, 0.4, 0.1)) +
  scale_y_discrete(expand = expansion(add = c(0.6, 1.0))) +
  guides(colour = guide_legend(nrow = 2), shape = guide_legend(nrow = 2)) +
  labs(x = expression(paste("Within-cultivar centred ", italic(R)^2)),
       y = NULL, tag = "b") +
  theme_plum(8.6) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        legend.position = "bottom", legend.margin = margin(t = -6),
        legend.text = element_text(size = pt_data))

## ------------------------------------------------- panel c: 180-cell ledger --
cult_n <- cells |> group_by(cultivar_code) |>
  summarise(n = max(n), .groups = "drop") |> arrange(desc(n))
stopifnot(nrow(cult_n) == 15, sum(cult_n$n) == 4853)
cell_lab <- setNames(sprintf("%s\n%d", cult_n$cultivar_code, cult_n$n),
                     cult_n$cultivar_code)

lim <- c(-0.35, 0.55)
grid <- cells |>
  mutate(trait = tf(trait),
         cultivar_code = factor(cultivar_code, levels = cult_n$cultivar_code),
         squished = r2_within < lim[1] | r2_within > lim[2],
         xi = as.integer(cultivar_code), yi = as.integer(trait))

n_neg_r2 <- sum(grid$r2_within < 0)
n_neg_r  <- sum(grid$pearson_r < 0)
big5   <- cult_n$cultivar_code[1:5]
small5 <- cult_n$cultivar_code[11:15]
fail_big   <- sum(grid$r2_within[grid$cultivar_code %in% big5] < 0)
fail_small <- sum(grid$r2_within[grid$cultivar_code %in% small5] < 0)

panel_c <- ggplot(grid, aes(cultivar_code, trait)) +
  geom_tile(aes(fill = r2_within), colour = paper, linewidth = 0.7) +
  ## redundant encoding 1: a hairline through every cell that failed outright
  geom_segment(data = ~ subset(.x, r2_within < 0),
               aes(x = xi - 0.42, xend = xi + 0.42, y = yi - 0.42, yend = yi + 0.42),
               inherit.aes = FALSE, colour = ink, linewidth = 0.30, alpha = 0.55) +
  ## redundant encoding 2: a heavy border where the correlation itself is negative
  geom_tile(data = ~ subset(.x, pearson_r < 0), fill = NA, colour = ink,
            linewidth = 0.85) +
  ## squished cells are declared, not silently clamped
  geom_point(data = ~ subset(.x, squished), colour = paper, size = 0.7) +
  scale_fill_gradient2(low = below, mid = "#F5F2EF", high = plum, midpoint = 0,
                       limits = lim, oob = scales::squish,
                       breaks = c(-0.3, 0, 0.3), name = "Within-cultivar R\u00b2") +
  scale_x_discrete(labels = cell_lab, expand = c(0, 0), position = "top") +
  scale_y_discrete(expand = c(0, 0)) +
  labs(x = NULL, y = NULL, tag = "c") +
  theme_plum(8.6) +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        axis.text.x = element_text(size = pt_data, lineheight = 1.0,
                                   margin = margin(b = 2)),
        legend.position = "right", legend.key.width = unit(8, "pt"),
        legend.key.height = unit(20, "pt"),
        legend.title = element_text(size = pt_data), legend.text = element_text(size = pt_data))

## ------------------------------------------- panel d: expert complementarity --
cp <- comp |>
  mutate(trait = factor(trait, levels = trait_rank)) |>
  select(trait, `CNN-SVR error correlation` = error_correlation,
         `Opposite-sign errors` = opposite_sign_fraction) |>
  pivot_longer(-trait, names_to = "metric", values_to = "value")

panel_d <- ggplot(cp, aes(value, trait)) +
  geom_vline(xintercept = 0.5, colour = rule, linewidth = 0.45, linetype = "22") +
  geom_point(aes(colour = metric, shape = metric), size = 1.9, stroke = 0.8) +
  scale_colour_manual(values = c("CNN-SVR error correlation" = plum,
                                 "Opposite-sign errors" = teal),
                      name = NULL) +
  scale_shape_manual(values = c("CNN-SVR error correlation" = 16,
                                "Opposite-sign errors" = 1),
                     name = NULL) +
  scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
  scale_y_discrete(expand = expansion(add = c(0.6, 1.0))) +
  guides(colour = guide_legend(nrow = 2), shape = guide_legend(nrow = 2)) +
  labs(x = "Proportion", y = NULL, tag = "d") +
  theme_plum(8.6) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        legend.position = "bottom", legend.margin = margin(t = -6),
        legend.text = element_text(size = pt_data))

## ----------------------------------------------------------------- layout ---
top <- panel_a | panel_b
top <- top + plot_layout(widths = c(1, 1))

bottom <- panel_c | panel_d
bottom <- bottom + plot_layout(widths = c(1.46, 0.54))

figure5 <- top / bottom + plot_layout(heights = c(1.00, 1.12)) &
  theme(plot.caption = element_blank())

save_figure(figure5, "Figure_5_v26", out_dir, height = 7.8)

## ------------------------------------------------------------- validation ---
message(sprintf("panel a  R2 difference versus OOF cultivar-mean null %.3f (%s) to %.3f (%s)",
                dec$increment[1], dec$trait[1],
                dec$increment[12], dec$trait[12]))
message(sprintf("panel b  centred R2 for PlumSPECTRA %.3f to %.3f; %d of 12 below 0.10",
                min(filter(lad, model == "PlumSPECTRA")$r2),
                max(filter(lad, model == "PlumSPECTRA")$r2),
                sum(filter(lad, model == "PlumSPECTRA")$r2 < 0.10)))
message(sprintf("panel c  %d cells, %d negative R2, %d negative r, %d squished",
                nrow(grid), n_neg_r2, n_neg_r, sum(grid$squished)))
message(sprintf("panel d  error correlation %.3f to %.3f",
                min(comp$error_correlation), max(comp$error_correlation)))
