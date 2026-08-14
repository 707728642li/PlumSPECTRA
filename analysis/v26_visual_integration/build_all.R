## Rebuild every v26 figure in order and print the validation lines.
##   Rscript R/build_all.R [audit_package_root] [out_dir] [source_data_dir]
##
## Run prep/prepare_figure_data.py first if source_data/ is empty or the audit
## package has changed; Figures 2, 4 and 5 read from it.

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]]), mustWork = TRUE))
} else {
  normalizePath(".", mustWork = TRUE)
}
root <- if (length(args) >= 1) args[[1]] else
  "./review_package/HR_EXTERNAL_AUDIT_PACKAGE_V25_FINAL_20260810"
out  <- if (length(args) >= 2) args[[2]] else
  file.path(dirname(dirname(script_dir)), "results", "v26_claudecode_integration", "figures_integrated")
src  <- if (length(args) >= 3) args[[3]] else
  file.path(dirname(dirname(script_dir)), "results", "v26_claudecode_integration", "figure_data")

for (i in 1:6) {
  f <- file.path(script_dir, sprintf("plot_figure%d_v26.R", i))
  message("\n=== ", f, " ===")
  system2(file.path(R.home("bin"), "Rscript"), c(shQuote(f), shQuote(root),
                                                 shQuote(out), shQuote(file.path(script_dir, "plum_figstyle.R")),
                                                 shQuote(src)))
}
message("\nall six figures written to ", normalizePath(out, mustWork = FALSE))
