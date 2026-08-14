## Figure 3 (v26) - Model and protocol.
##
##   a  residual architecture as a data flow, with every train-internal
##      selection step marked
##   b  nested cross-validation and the information channels the design forbids
##   c  equal-information comparator matrix
##
## The published Figure 3 spent panel A on a box-and-arrow diagram (with one
## arrow crossing a box) and panels B-D on performance. Performance moves to
## Figure 4; this figure now carries the part a reviewer actually interrogates:
## what was selected where, and why the comparison is fair.
##
## All three panels are schematics of the protocol and contain no measurements.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
})

args    <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 2) args[[2]] else "figures"
style   <- if (length(args) >= 3) args[[3]] else "R/plum_figstyle.R"
source(style)

## helpers -------------------------------------------------------------------
box <- function(x, y, w, h, label, sub = NA, edge = ink, lw = 0.55,
                fill = paper, size = sz(pt_data), colour = ink) {
  list(
    annotate("rect", xmin = x, xmax = x + w, ymin = y, ymax = y + h,
             fill = fill, colour = edge, linewidth = lw),
    annotate("text", x = x + w / 2, y = y + h * (if (is.na(sub)) 0.5 else 0.63),
             label = label, family = font_family, colour = colour,
             size = size, fontface = "bold", lineheight = 0.95),
    if (!is.na(sub))
      annotate("text", x = x + w / 2, y = y + h * 0.26, label = sub,
               family = font_family, colour = muted, size = sz(pt_note),
               lineheight = 0.95)
  )
}
arr <- function(x0, y0, x1, y1, colour = ink, lw = 0.5, lty = 1) {
  annotate("segment", x = x0, xend = x1, y = y0, yend = y1, colour = colour,
           linewidth = lw, linetype = lty,
           arrow = grid::arrow(length = unit(4, "pt"), type = "closed"))
}
txt <- function(x, y, label, size = sz(pt_note), colour = muted, hjust = 0.5,
                face = "plain") {
  annotate("text", x = x, y = y, label = label, family = font_family,
           colour = colour, size = size, hjust = hjust, fontface = face,
           lineheight = 1.05)
}
## small square marking a step that is selected inside the training data only
internal <- function(x, y) list(
  annotate("point", x = x, y = y, shape = 22, size = 1.5,
           fill = teal, colour = teal, stroke = 0))

blank <- function(p, tag) p +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_blank_panel() + labs(tag = tag)

## ---------------------------------------------------- panel a: data flow ----
pa <- ggplot() +
  ## inputs
  box(0.010, 0.60, 0.135, 0.26, "228-band NIR", "901-1701 nm", edge = ink) +
  txt(0.078, 0.545, "RAW  |  SNV  |  SG1", sz(pt_note)) +

  ## anchor
  box(0.185, 0.66, 0.165, 0.22, "cultivar-aware\nPLSR anchor",
      "4-fold cross-fitting", edge = ink) +
  internal(0.196, 0.855) +
  arr(0.147, 0.73, 0.183, 0.75) +
  txt(0.268, 0.615, "residual target", sz(pt_note), teal) +

  ## auxiliary
  box(0.185, 0.24, 0.165, 0.20, "12-trait auxiliary\npretraining",
      "outer-training fruit only", edge = ink) +
  internal(0.196, 0.415) +
  arr(0.147, 0.66, 0.183, 0.36) +

  ## CNN
  box(0.395, 0.36, 0.185, 0.52, "residual\nspectral CNN",
      "48-channel stem\n4 blocks, k 3/9/21\ndilation 1/2/4/8\n142,285 parameters",
      edge = teal, lw = 0.9) +
  arr(0.352, 0.75, 0.393, 0.70, colour = teal) +
  arr(0.352, 0.34, 0.393, 0.48, colour = teal) +

  ## kernel expert
  box(0.395, 0.09, 0.185, 0.18, "nested RBF-SVR",
      "domain-aware kernel expert", edge = ink) +
  internal(0.406, 0.245) +

  ## gate
  box(0.625, 0.34, 0.155, 0.24, "eligibility gate",
      "training-internal\n5% margin", edge = ink) +
  internal(0.636, 0.555) +
  arr(0.582, 0.60, 0.623, 0.52, colour = teal) +
  arr(0.582, 0.18, 0.623, 0.40) +

  ## ensemble
  box(0.825, 0.36, 0.165, 0.22, "0.5 CNN + 0.5 SVR",
      "one model per trait", edge = plum, lw = 0.9) +
  arr(0.782, 0.46, 0.823, 0.46) +
  txt(0.908, 0.275, "12 continuous outputs\nFW  |  SSC  |  pH\n+ 9 texture endpoints",
      sz(pt_data), ink) +

  ## legend for the internal marker
  annotate("point", x = 0.016, y = 0.115, shape = 22, size = 1.5,
           fill = teal, colour = teal, stroke = 0) +
  txt(0.030, 0.115, "selected inside the training data only", sz(pt_note), ink, hjust = 0)

panel_a <- blank(pa, "a")

## ------------------------------------------- panel b: information isolation --
grid_x0 <- 0.360
folds <- expand_grid(outer = 1:5, cell = 1:5) |>
  mutate(role = ifelse(cell == outer, "test", "train"),
         x = grid_x0 + (cell - 1) * 0.072,
         y = 0.905 - (outer - 1) * 0.078)

pb <- ggplot() +
  txt(grid_x0 + 2 * 0.072, 0.975, "five frozen cultivar-stratified outer folds",
      sz(pt_data), ink, face = "bold") +
  geom_tile(data = folds, aes(x, y, fill = role), width = 0.058, height = 0.058,
            colour = paper, linewidth = 0.8) +
  scale_fill_manual(values = c(train = "#D9DEE6", test = plum), guide = "none") +
  annotate("text", x = grid_x0 - 0.048, y = 0.905 - (0:4) * 0.078,
           label = paste("fold", 1:5), family = font_family, colour = muted,
           size = sz(pt_data), hjust = 1) +
  annotate("point", x = grid_x0 + 4 * 0.072 + 0.055, y = 0.905, shape = 22,
           size = 2.4, fill = plum, colour = plum, stroke = 0) +
  txt(grid_x0 + 4 * 0.072 + 0.074, 0.905, "held out", sz(pt_data), muted, hjust = 0) +

  ## what happens inside the training block
  annotate("segment", x = 0.030, xend = 0.985, y = 0.545, yend = 0.545,
           colour = rule, linewidth = 0.5) +
  txt(0.030, 0.500, "inside each training block only", sz(pt_data), ink, hjust = 0,
      face = "bold") +
  {
    lab <- c("4-fold cross-fitting for the PLSR anchor residual",
             "cultivar-stratified validation for epoch and residual gate",
             "3-fold inner CV for SVR C, gamma and epsilon")
    ys <- 0.445 - (seq_along(lab) - 1) * 0.047
    list(
      annotate("point", x = 0.042, y = ys, shape = 22, size = 1.4,
               fill = teal, colour = teal, stroke = 0),
      annotate("text", x = 0.062, y = ys, label = lab, family = font_family,
               colour = muted, size = sz(pt_note), hjust = 0))
  } +
  txt(0.030, 0.295, "60/60 recorded production gates equal the internal gates",
      sz(pt_note), teal, hjust = 0) +

  ## forbidden channels
  annotate("segment", x = 0.030, xend = 0.985, y = 0.245, yend = 0.245,
           colour = rule, linewidth = 0.5) +
  txt(0.030, 0.198, "channels the execution design forbids", sz(pt_data), below,
      hjust = 0, face = "bold") +
  {
    lab <- c("test labels into any selection",
             "test fruit into auxiliary pretraining",
             "test fold into anchor cross-fitting")
    ys <- 0.143 - (seq_along(lab) - 1) * 0.047
    list(
      annotate("segment", x = 0.036, xend = 0.072, y = ys, yend = ys,
               colour = below, linewidth = 0.45, linetype = "22"),
      annotate("point", x = 0.056, y = ys, shape = 4, size = 1.9,
               colour = below, stroke = 0.9),
      annotate("text", x = 0.088, y = ys, label = lab, family = font_family,
               colour = ink, size = sz(pt_note), hjust = 0))
  }

panel_b <- blank(pb, "b")

## ------------------------------------------ panel c: comparator matrix ------
comp <- tibble::tribble(
  ~model,                    ~`12 traits`, ~cultivar, ~nonlinear, ~ensemble, ~role,
  "Global PLSR",                    0L, 0L, 0L, 0L, "lower bound",
  "Cultivar-aware PLSR",            0L, 1L, 0L, 0L, "strengthened chemometrics",
  "Nested RBF-SVR",                 0L, 1L, 1L, 0L, "kernel expert",
  "12-response PLS2",               1L, 1L, 0L, 0L, "equal-information control",
  "Deep-only",                      1L, 1L, 1L, 0L, "neural contribution",
  "No-neural B50",                  0L, 1L, 1L, 1L, "ensemble without a network",
  "Cultivar-mean null",             0L, 1L, 0L, 0L, "no-spectra floor",
  "PlumSPECTRA",                    1L, 1L, 1L, 1L, "final"
)

comp_long <- comp |>
  mutate(model = factor(model, levels = rev(model)),
         highlight = model %in% c("12-response PLS2", "PlumSPECTRA")) |>
  pivot_longer(c(`12 traits`, cultivar, nonlinear, ensemble),
               names_to = "feature", values_to = "has") |>
  mutate(feature = factor(feature,
                          levels = c("12 traits", "cultivar", "nonlinear", "ensemble"),
                          labels = c("sees 12\ntraits", "sees\ncultivar",
                                     "nonlinear", "ensemble")))

roles <- comp |> mutate(model = factor(model, levels = rev(model)))

panel_c <- ggplot(comp_long, aes(feature, model)) +
  geom_tile(aes(fill = highlight), colour = paper, linewidth = 1.0) +
  geom_point(aes(colour = factor(has), shape = factor(has)), size = 2.1,
             stroke = 0.9) +
  scale_fill_manual(values = c(`FALSE` = paper, `TRUE` = "#F3ECF0"), guide = "none") +
  scale_colour_manual(values = c(`0` = rule, `1` = plum), guide = "none") +
  scale_shape_manual(values = c(`0` = 1, `1` = 16), guide = "none") +
  geom_text(data = roles, aes(x = 4.75, y = model, label = role),
            inherit.aes = FALSE, hjust = 0, family = font_family,
            colour = muted, size = sz(pt_note)) +
  scale_x_discrete(expand = expansion(add = c(0.55, 3.6))) +
  scale_y_discrete(expand = expansion(add = 0.6)) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL, tag = "c") +
  theme_plum(8.8) +
  theme(axis.line = element_blank(), axis.ticks = element_blank(),
        plot.margin = margin(6, 6, 4, 4))

## ----------------------------------------------------------------- layout ---
figure3 <- panel_a / (panel_b | panel_c) + plot_layout(heights = c(0.92, 1)) &
  theme(plot.caption = element_blank())

save_figure(figure3, "Figure_3_v26", out_dir, height = 7.4)
