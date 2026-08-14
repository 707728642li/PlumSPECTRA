## Figure 1 (v26) - Paired fruit workflow and cohort ledger.
##
##   a  the measurement sequence on one fruit, with the destructive boundary
##      made explicit: NIR is the only signal acquired before the fruit changes
##   b  cohort ledger from the source ledger down to the two modelling cohorts
##   c  retained fruit and acquisition batches per cultivar
##
## Replaces the rendered-instrument workflow (old Fig 1A) and merges the old
## Fig 1B into one production file, as the journal requires.
##
## Panel a is a schematic and carries no measured values other than the cohort
## size. Panels b and c are read from the frozen audit package.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr)
  library(tidyr); library(readr); library(forcats)
})

args     <- commandArgs(trailingOnly = TRUE)
here     <- dirname(normalizePath(sys.frames()[[1]]$ofile %||% "R/x.R", mustWork = FALSE))
`%||%`   <- function(a, b) if (is.null(a)) b else a
root     <- if (length(args) >= 1) args[[1]] else
  "./review_package/HR_EXTERNAL_AUDIT_PACKAGE_V25_FINAL_20260810"
out_dir  <- if (length(args) >= 2) args[[2]] else "figures"
style    <- if (length(args) >= 3) args[[3]] else "R/plum_figstyle.R"
source(style)

## ------------------------------------------------------------------ data ----
cultivars <- read_csv(file.path(root, "evidence/final_analysis/cultivar_batch_counts.csv"),
                      show_col_types = FALSE) |>
  mutate(code = unname(cultivar_code_map[cultivar_ascii])) |>
  arrange(desc(fruits)) |>
  mutate(code = fct_inorder(code))

stopifnot(!any(is.na(cultivars$code)))
texture_n <- sum(cultivars$fruits)

## Frozen counts, 00_READ_ME_FIRST_ZH.md section 4.
ledger <- tibble::tribble(
  ~stage,                            ~n,     ~kind,
  "Source ledger",                   5502L,  "source",
  "Analysis tier",                   5430L,  "tier",
  "Strict release tier",             4967L,  "tier",
  "Formal texture cohort",           4853L,  "cohort",
  "Conventional complete case",      4843L,  "cohort"
) |>
  mutate(stage = fct_inorder(stage))

stopifnot(ledger$n[ledger$stage == "Formal texture cohort"] == texture_n)

## ------------------------------------------------------- panel a: workflow ---
## Acquisition order per 01_PROJECT_BACKGROUND_TARGET_AND_REQUIREMENTS_ZH.md:
## intact fruit -> NIR -> single-fruit mass -> duplicate penetration -> SSC -> pH.
## Weighing does not alter the fruit, so the destructive boundary falls after
## fruit mass, not after the spectrum.
steps <- tibble::tribble(
  ~i, ~label,            ~detail,                          ~destructive,
  1L, "NIR spectrum",    "228 bands, 901-1701 nm",         FALSE,
  2L, "Fruit mass",      "intact fruit",                   FALSE,
  3L, "Penetration 1",   "position A",                     TRUE,
  4L, "Penetration 2",   "position B",                     TRUE,
  5L, "SSC",             "refractometer",                  TRUE,
  6L, "pH",              "electrode",                      TRUE
) |>
  mutate(y = seq(0.82, 0.18, length.out = n()))

boundary <- mean(c(steps$y[2], steps$y[3]))
x_axis   <- 0.30

panel_a <- ggplot() +
  ## regions
  annotate("rect", xmin = 0.12, xmax = 0.96, ymin = boundary, ymax = 0.88,
           fill = teal, alpha = 0.05) +
  annotate("rect", xmin = 0.12, xmax = 0.96, ymin = 0.10, ymax = boundary,
           fill = below, alpha = 0.05) +
  ## paired-fruit bracket, now vertical to reinforce the top-to-bottom order
  annotate("segment", x = 0.075, xend = 0.075, y = 0.82, yend = 0.18,
           colour = ink, linewidth = 0.4) +
  annotate("segment", x = c(0.075, 0.075), xend = c(0.105, 0.105),
           y = c(0.82, 0.18), yend = c(0.82, 0.18),
           colour = ink, linewidth = 0.4) +
  annotate("text", x = 0.52, y = 0.965,
           label = sprintf("one fruit, one row\n%s fruit measured end to end",
                           format(texture_n, big.mark = ",")),
           family = font_family, colour = ink, size = 3.35,
           lineheight = 1.05, fontface = "bold") +
  ## vertical measurement sequence
  annotate("segment", x = x_axis, xend = x_axis, y = 0.855, yend = 0.145,
           colour = ink, linewidth = 0.55,
           arrow = grid::arrow(length = unit(4, "pt"), type = "closed")) +
  geom_point(data = steps, aes(x = x_axis, y = y, colour = destructive),
             size = 3.0, show.legend = FALSE) +
  geom_text(data = steps, aes(x = 0.40, y = y + 0.020, label = label),
            family = font_family, colour = ink, size = 3.15,
            fontface = "bold", hjust = 0) +
  geom_text(data = steps, aes(x = 0.40, y = y - 0.025, label = detail),
            family = font_family, colour = muted, size = sz(pt_data), hjust = 0) +
  scale_colour_manual(values = c(`FALSE` = teal, `TRUE` = below)) +
  ## the destructive boundary is the point of the panel
  annotate("segment", x = 0.12, xend = 0.96, y = boundary, yend = boundary,
           colour = ink, linewidth = 0.85, linetype = "22") +
  annotate("text", x = 0.34, y = boundary + 0.018,
           label = "destructive boundary", hjust = 0, vjust = 0,
           family = font_family, colour = ink, size = 2.95, fontface = "bold") +
  annotate("text", x = 0.92, y = 0.855,
           label = "intact-fruit\nmeasurements",
           family = font_family, colour = teal, size = 2.85,
           hjust = 1, vjust = 1, lineheight = 1.05) +
  annotate("text", x = 0.92, y = boundary - 0.025,
           label = "destructive reference\nphenotypes",
           family = font_family, colour = below, size = 2.85,
           hjust = 1, vjust = 1, lineheight = 1.05) +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.02, 1.0), expand = c(0, 0)) +
  theme_blank_panel() +
  labs(tag = "a")

## ------------------------------------------------- panel b: cohort ledger ----
lead <- max(ledger$n)
## Panel c reserves a -120-to-0 data-space channel for cultivar and batch
## metadata. Give panel b the same fractional left channel so the x = 0 bar
## origins align exactly when the two panels are stacked.
c_positive_limit <- max(cultivars$fruits) * 1.24
b_left_channel <- lead * 1.24 * 120 / c_positive_limit
ledger_plot <- ledger |>
  mutate(stage = fct_rev(stage),
         fill_col = case_when(kind == "cohort" ~ plum,
                              kind == "tier"   ~ ink,
                              TRUE             ~ muted))

panel_b <- ggplot(ledger_plot, aes(y = stage, x = n)) +
  geom_col(aes(fill = fill_col), width = 0.62, show.legend = FALSE) +
  geom_text(aes(label = format(n, big.mark = ",")),
            hjust = -0.14, family = font_family, colour = ink,
            size = 3.05, fontface = "bold") +
  scale_fill_identity() +
  scale_x_continuous(limits = c(-b_left_channel, lead * 1.24), expand = c(0, 0),
                     breaks = c(0, 2000, 4000),
                     labels = scales::label_comma()) +
  labs(x = "Fruit", y = NULL, tag = "b") +
  theme_plum(9.6) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = margin(6, 4, 2, 4))

## ---------------------------------------------- panel c: cultivar and batch --
## Every cultivar carries its acquisition-batch count, not only the two that
## have more than one. Thirteen single-batch cultivars is the reason the
## held-batch audit could only be run on two cultivars (Figure 6).
single_batch <- sum(cultivars$batches == 1)
cult_plot <- cultivars |>
  mutate(code = fct_rev(code),
         batch_col = ifelse(batches > 1, below, rule))

panel_c <- ggplot(cult_plot, aes(y = code, x = fruits)) +
  geom_col(width = 0.70, fill = plum, alpha = 0.92) +
  geom_text(aes(label = fruits), hjust = -0.20, family = font_family,
            colour = ink, size = sz(pt_data)) +
  geom_point(aes(x = -74, colour = batch_col), size = 3.4 * 1.2) +
  geom_text(aes(x = -74, label = batches), family = font_family,
            colour = ifelse(cult_plot$batches > 1, "white", muted),
            size = sz(pt_data), fontface = "bold") +
  scale_colour_identity() +
  annotate("text", x = -74, y = length(levels(cult_plot$code)) + 1.15,
           label = "batches", family = font_family, colour = muted,
           size = sz(pt_data), fontface = "bold") +
  annotate("text", x = c_positive_limit,
           y = length(levels(cult_plot$code)) + 1.15, hjust = 1,
           label = sprintf("%d of %d cultivars were acquired in a single batch",
                           single_batch, nrow(cultivars)),
           family = font_family, colour = below, size = sz(pt_note)) +
  scale_x_continuous(limits = c(-120, c_positive_limit),
                     expand = c(0, 0), breaks = c(0, 250, 500, 750)) +
  coord_cartesian(clip = "off") +
  labs(x = "Retained fruit", y = NULL, tag = "c") +
  theme_plum(9.0) +
  theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
        plot.margin = margin(12, 6, 2, 4))

## ----------------------------------------------------------------- layout ---
## Requested reading order: panel a occupies the full left column, while the
## right column stacks b over c at approximately 1:2 height.
right <- panel_b / panel_c + plot_layout(heights = c(1, 2))
figure1 <- panel_a | right
figure1 <- figure1 + plot_layout(widths = c(0.46, 0.54)) &
  theme(plot.caption = element_blank())

save_figure(figure1, "Figure_1_v26", out_dir, height = 5.8)
