# ==========================
# 1. Clear session & setup
# ==========================
rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Libraries
library(readr)
library(dplyr)
library(ggplot2)
library(patchwork)
library(stringr)
library(purrr)

# ==========================
# 2. Metric functions
# ==========================
tp <- function(lhat, ltrue) sum(lhat %in% ltrue)
fp <- function(lhat, ltrue) sum(!(lhat %in% ltrue))
fn <- function(lhat, ltrue) sum(!(ltrue %in% lhat))

precision <- function(lhat, ltrue) {
  t <- tp(lhat, ltrue); f <- fp(lhat, ltrue)
  if (t == 0) return(0)
  t / (t + f)
}

recall <- function(lhat, ltrue) {
  t <- tp(lhat, ltrue); f <- fn(lhat, ltrue)
  if (t == 0) return(0)
  t / (t + f)
}

f1 <- function(lhat, ltrue) {
  if (any(is.na(lhat))) return(NA)
  p <- precision(lhat, ltrue)
  r <- recall(lhat, ltrue)
  if (p == 0 && r == 0) return(0)
  2 * p * r / (p + r)
}

parse_py_list <- function(s) {
  s %>%
    str_remove_all("\\[|\\]|'") %>%
    str_split(",\\s*") %>%
    map(~ trimws(.x))
}

# ==========================
# 3. Load data
# ==========================
load_results <- function(path_small, path_large) {
  small <- read_csv(path_small)
  big   <- read_csv(path_large)
  bind_rows(small, big)
}

res_identifiable_gauss <- load_results(
  "output_experiments_gaussian/final_results_identifiable_small.csv",
  "output_experiments_gaussian/final_results_identifiable_large.csv"
)

res_non_identifiable_gauss <- load_results(
  "output_experiments_gaussian/final_results_nonidentifiable_small.csv",
  "output_experiments_gaussian/final_results_nonidentifiable_large.csv"
)

# ==========================
# 4. Preprocess & compute F1
# ==========================
compute_f1 <- function(df) {
  df %>%
    mutate(
      adjustment_set_vec = parse_py_list(adjustment_set),
      true_parents_vec   = parse_py_list(true_parents),
      f1_score           = map2_dbl(adjustment_set_vec, true_parents_vec, f1)
    )
}

res_identifiable_gauss <- compute_f1(res_identifiable_gauss)
res_non_identifiable_gauss <- compute_f1(res_non_identifiable_gauss)

# ==========================
# 5. Plotting parameters
# ==========================
sorbonne_colors <- c(
  "PC"        = rgb(3, 40, 89, maxColorValue = 255),   # sorbonneblue
  "LocPC-CDE" = rgb(234, 67, 40, maxColorValue = 255)  # sorbonneorange
)

sorbonne_shapes <- c(
  "LocPC-CDE" = 17, "PC" = 15
)

method_levels <- c("PC", "LocPC-CDE")

sorbonne_shapes <- c(
  "LocPC-CDE" = 17, "PC" = 15
)

method_levels <- c("PC", "LocPC-CDE")

# ==========================
# 6. Summarisation helpers
# ==========================
summarise_metric <- function(df, value_col, ident_label, log_y = FALSE) {
  df %>%
    transmute(
      dag_size,
      method = recode(method,
                      "locpc" = "LocPC-CDE",
                      "pc" = "PC"),
      value = !!sym(value_col)
    ) %>%
    filter(method %in% c("PC", "LocPC-CDE")) %>%
    group_by(dag_size, method) %>%
    summarise(
      n = n(),
      mean_val = mean(value, na.rm = TRUE),
      se = sd(value, na.rm = TRUE)/sqrt(n),
      lower = pmax(0, mean_val - 1.96*se),
      upper = mean_val + 1.96*se,
      .groups = "drop"
    ) %>%
    mutate(method = factor(method, levels = method_levels), type = ident_label)
}

summarise_prop <- function(df, label) summarise_metric(df, "identifiability", label)
summarise_ci   <- function(df, label) summarise_metric(df, "nb_CI_tests", label, log_y = TRUE)
summarise_f1   <- function(df, label) summarise_metric(df, "f1_score", label)
summarise_prop_nonid <- function(df, label) {
  df %>%
    mutate(non_identifiable = !identifiability) %>%
    summarise_metric("non_identifiable", label)
}

# ==========================
# 7. Summarise results
# ==========================
prop_id_gauss <- summarise_prop(res_identifiable_gauss, "Identifiable")
prop_nonid_gauss <- summarise_prop_nonid(res_non_identifiable_gauss, "Non-identifiable")
ci_id_gauss <- summarise_ci(res_identifiable_gauss, "Identifiable")
ci_nonid_gauss <- summarise_ci(res_non_identifiable_gauss, "Non-identifiable")
f1_id_gauss <- summarise_f1(res_identifiable_gauss, "Identifiable")

# ==========================
# 8. Plotting helper
# ==========================
plot_metric <- function(df,y_col,y_label,log_y=FALSE,title="") {
  p <- ggplot(df,
              aes(x = as.factor(dag_size),
                  y = !!sym(y_col),
                  color = method,
                  group = method,
                  shape = method)) +
    geom_ribbon(aes(ymin = lower, ymax = upper, fill = method),
                alpha = 0.2, color = NA) +
    geom_line(size = .75, alpha = 0.8) +
    geom_point(size = 2, alpha = 0.8) +
    scale_color_manual(values = sorbonne_colors) +
    scale_fill_manual(values = sorbonne_colors) +
    scale_shape_manual(values = sorbonne_shapes) +
    labs(x = "DAG size",
         y = y_label,
         title = title,
         color = "Method",
         fill = "Method",
         shape = "Method") +
    theme_bw() +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  if(y_label=="TPR (%)") {
    p <- p + scale_y_continuous(labels = scales::percent_format(accuracy = 1),
                                limits = c(0,1))
  } else if(y_label=="F1 Score") {
    p <- p + ylim(0,1)
  }
  
  if(log_y) p <- p + scale_y_log10()
  p
}

# ==========================
# 9. Single combined plot for Gaussian SCM (all metrics)
# ==========================
p_gauss <- (
  plot_metric(ci_id_gauss, "mean_val", "# CI tests", log_y=TRUE, title = "Identifiable") |
    plot_metric(prop_id_gauss, "mean_val", "TPR (%)") |
    plot_metric(f1_id_gauss, "mean_val", "F1 Score")
) /
  (
    plot_metric(ci_nonid_gauss, "mean_val", "# CI tests", log_y=TRUE, title = "Non-identifiable") |
      plot_metric(prop_nonid_gauss, "mean_val", "TPR (%)")
  ) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

# ==========================
# 10. Save figure
# ==========================
ggsave("figures/gaussian_identifiable_vs_nonidentifiable_all_metrics.png", p_gauss, width = 8, height = 6, dpi = 800)