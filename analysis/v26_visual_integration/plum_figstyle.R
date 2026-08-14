## PlumSPECTRA v26 shared figure style.
## Extends the existing theme_classic2 contract from render_v22_integrated_figures.R:
##   Arial, minimum 10.5 pt source text, classic axes, no panel titles,
##   no panel-internal conclusions.
## New in v26:
##   - a locked semantic palette, including a colour reserved for negative results
##   - a page-height budget that warns instead of silently shrinking the type

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(grid)
})

## ---------------------------------------------------------------- canvas ----
standard_width  <- 10.5     # in, journal production width
max_page_height <- 15.6     # in, full-page multi-panel source canvas at 183-mm print width
standard_dpi    <- 450
font_family     <- "Arial"
minimum_source_text_pt <- 10.5

## Base R's Windows graphics device exposes Arial as the generic `sans`
## family but does not create an `Arial` alias automatically.  Register the
## explicit family used throughout the plotting code so PNG metrics and text
## drawing do not silently fall back to another font.  Cairo PDF resolves the
## same installed TrueType family and embeds it in the production files.
if (.Platform$OS.type == "windows" && !font_family %in% names(windowsFonts())) {
  windowsFonts(Arial = windowsFont("Arial"))
}

## ------------------------------------------------------------ type floor ----
## Figures are drawn at 10.5 in and reproduced at 183 mm (scale 0.686). The
## project-wide contract is a 10.5 pt source floor, equivalent to 7.2 pt at the
## journal page width. Long explanatory prose belongs in the external legend,
## not in the plotting area; this lets the dense matrices retain readable type.
pt_data <- minimum_source_text_pt
pt_note <- minimum_source_text_pt
repro_width_mm <- 183

## ggplot2 expresses geom_text size in mm. Use sz() so every text layer states
## the point size it intends, and an audit is a grep for sz(.
sz <- function(pt) pt / .pt

printed_pt <- function(pt, width_mm = repro_width_mm, drawn_in = standard_width) {
  pt * (width_mm / 25.4) / drawn_in
}

## --------------------------------------------------------------- palette ----
## Carried over from v22 so the two figure generations stay consistent.
ink   <- "#17233B"   # text, axes, neutral structure
plum  <- "#7B2D5F"   # PlumSPECTRA / final model
teal  <- "#238C86"   # residual CNN / positive increment
gold  <- "#D89B35"   # global PLSR / weak lower bound
muted <- "#6B7280"   # secondary text
## New in v26.
below <- "#C0552F"   # negative result, CI crossing zero, deployment failure
rule  <- "#D4D9DE"   # hairlines, grid, "not evaluated"
paper <- "#FFFFFF"

## Semantic lock. Do not reuse these for a different concept in another panel.
pal_regime <- c(
  "Same-session outer folds" = teal,
  "Held-batch audit"         = gold,
  "Unseen cultivar
(LOCO PLSR)"  = below
)
## Model colours are taken verbatim from model_cols in
## audit_scripts/render_v22_integrated_figures.R so that a v26 figure and a
## surviving v22 figure never disagree about what colour a model is.
pal_model <- c(
  "Global PLSR"            = gold,
  "Cultivar-aware PLSR"    = "#4679A8",   # their `blue`
  "Nested RBF-SVR"         = "#76528B",   # their `purple`
  "Residual CNN"           = teal,
  "PlumSPECTRA"            = plum,
  "No-neural B50 ensemble" = "#536878",
  "Cultivar-mean null"     = "#202020",
  "Fixed-gate sensitivity" = "#E67835",   # their `orange`
  "Batch-mean null"        = muted
)

## Also from render_v22_integrated_figures.R. Per the design rules, cultivar
## colour is used only where cultivar identity is the variable being read;
## v26 uses continuous or diverging scales in Figures 2e, 5c and 6b instead, so
## this palette is carried for compatibility rather than used.
cultivar_cols <- c(
  "L313" = "#264653", "LA191" = "#2A9D8F", "CHL" = "#54A4A8", "FTL" = "#8AB17D",
  "FWHH" = "#E9C46A", "FRL" = "#F4A261", "KLD" = "#E76F51", "L31" = "#B65B80",
  "NL"   = "#8C6BB1", "QCL"  = "#6C7FD8", "WD"  = "#4C9ED9", "WJ"  = "#65B5C0",
  "WW"   = "#3A7D6E", "WX"   = "#91B949", "ZSKLD" = "#C58A32"
)

## ----------------------------------------------------------------- theme ----
theme_classic2 <- function(base_size = minimum_source_text_pt) {
  base_size <- max(base_size, minimum_source_text_pt)
  theme_classic(base_size = base_size, base_family = font_family) +
    theme(
      axis.line        = element_line(colour = ink, linewidth = 0.55),
      axis.ticks       = element_line(colour = ink, linewidth = 0.50),
      axis.title       = element_text(colour = ink, face = "bold"),
      axis.text        = element_text(colour = ink),
      plot.title       = element_blank(),      # no panel titles, ever
      plot.subtitle    = element_blank(),
      plot.tag         = element_text(colour = ink, face = "bold",
                                      size = rel(1.35), hjust = 0, vjust = 1),
      plot.tag.position = c(0, 1),
      strip.background = element_blank(),
      strip.text       = element_text(colour = ink, face = "bold"),
      legend.title     = element_text(face = "bold", colour = ink),
      legend.text      = element_text(colour = ink),
      legend.key.height = unit(9, "pt"),
      legend.key.width  = unit(14, "pt"),
      legend.background = element_blank(),
      plot.background   = element_rect(fill = paper, colour = NA),
      panel.background  = element_rect(fill = paper, colour = NA),
      plot.margin       = margin(6, 6, 4, 4)
    )
}

## Backward-compatible alias used by the imported v26 plotting scripts.
theme_plum <- theme_classic2

theme_blank_panel <- function(base_size = minimum_source_text_pt) {
  theme_void(base_family = font_family) +
    theme(
      plot.tag = element_text(colour = ink, face = "bold",
                              size = rel(1.35), hjust = 0, vjust = 1),
      plot.tag.position = c(0, 1),
      plot.background = element_rect(fill = paper, colour = NA),
      plot.margin     = margin(6, 6, 4, 4),
      legend.position = "none"
    )
}

theme_set(theme_classic2())

## ------------------------------------------------------------- save/QA ----
save_figure <- function(plot, stem, out_dir, height,
                        width = standard_width, dpi = standard_dpi) {
  if (height > max_page_height) {
    warning(sprintf(
      "%s is %.2f in tall; the page budget is %.1f in. Recompose rather than shrink.",
      stem, height, max_page_height), call. = FALSE)
  }
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  for (ext in c("pdf", "png")) {
    ggsave(file.path(out_dir, paste0(stem, ".", ext)), plot,
           width = width, height = height, dpi = dpi,
           device = if (ext == "pdf") cairo_pdf else "png",
           bg = paper)
  }
  message(sprintf(
    "wrote %s  (%.2f x %.2f in; at %d mm the %.1f pt data floor prints %.2f pt, the %.1f pt note floor %.2f pt)",
    stem, width, height, repro_width_mm,
    pt_data, printed_pt(pt_data, drawn_in = width),
    pt_note, printed_pt(pt_note, drawn_in = width)))
  invisible(file.path(out_dir, paste0(stem, c(".pdf", ".png"))))
}

## Small helpers used by both figures ------------------------------------------
pct <- function(x, digits = 1) sprintf(paste0("%+.", digits, "f%%"), x)

trait_levels <- c("SRF", "RD", "PFD", "MFF", "F6", "LS", "LW", "PRW", "AF")

## LOCO target -> manuscript trait code
loco_trait_map <- c(
  skin_break_force_g_mean            = "SRF",
  skin_break_displacement_raw_mean   = "RD",
  skin_break_drop_g_mean             = "PFD",
  flesh_force_mean_g_mean            = "MFF",
  force_at_6_rawpos_g_mean           = "F6",
  loading_stiffness_g_per_rawpos_mean = "LS",
  loading_work_g_rawpos_mean         = "LW",
  post_break_work_g_rawpos_mean      = "PRW",
  adhesive_force_g_mean              = "AF"
)

## Registry cultivar name -> figure code, matching the published Figure 1B
cultivar_code_map <- c(
  "Konglongdan"        = "KLD",
  "Weidi"              = "WD",
  "Weiwang"            = "WW",
  "Qingcuili"          = "QCL",
  "Weijin"             = "WJ",
  "A181"               = "LA191",
  "3.13"               = "L313",
  "Cuihongli"          = "CHL",
  "Furongli"           = "FRL",
  "Zaoshu Konglongdan" = "ZSKLD",
  "L31"                = "L31",
  "Weixin"             = "WX",
  "Fengtangli"         = "FTL",
  "Naili"              = "NL",
  "Fengwei Huanghou"   = "FWHH"
)
