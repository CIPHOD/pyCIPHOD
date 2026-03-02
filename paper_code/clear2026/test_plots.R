# ==========================
# 1. Clear session & setup
# ==========================
rm(list = ls())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

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

precision <- function(lhat, ltrue) { t <- tp(lhat, ltrue); f <- fp(lhat, ltrue); if(t==0) 0 else t/(t+f) }
recall    <- function(lhat, ltrue) { t <- tp(lhat, ltrue); f <- fn(lhat, ltrue); if(t==0) 0 else t/(t+f) }
f1        <- function(lhat, ltrue) { if(any(is.na(lhat))) return(NA); p <- precision(lhat,ltrue); r <- recall(lhat,ltrue); if(p==0 && r==0) 0 else 2*p*r/(p+r) }

parse_py_list <- function(s) s %>% str_remove_all("\\[|\\]|'") %>% str_split(",\\s*") %>% map(~trimws(.x))

# ==========================
# 3. Load Gaussian small (identifiable only)
# ==========================
res_ident_gauss <- read_csv("~/output_experiments_gaussian/final_results_identifiable_small.csv")

# ==========================
# 4. Compute F1
# ==========================
res_ident_gauss <- res_ident_gauss %>%
  mutate(
    adjustment_set_vec = parse_py_list(adjustment_set),
    true_parents_vec   = parse_py_list(true_parents),
    f1_score           = map2_dbl(adjustment_set_vec, true_parents_vec, f1)
  )

# ==========================
# 5. Plotting parameters
# ==========================
sorbonne_colors <- c(
  "LocPC-CDE" = rgb(234, 67, 40, maxColorValue = 255),
  "LDECC"     = rgb(3, 40, 89, maxColorValue = 255),
  "PC"        = rgb(100, 190, 230, maxColorValue = 255),
  "CMB"       = rgb(50, 150, 50, maxColorValue = 255),
  "MBbyMB"    = rgb(180, 120, 200, maxColorValue = 255)
)
sorbonne_shapes <- c("LocPC-CDE"=17,"PC"=15,"LDECC"=16,"CMB"=18,"MBbyMB"=8)
method_levels <- c("PC","CMB","MBbyMB","LDECC","LocPC-CDE")

# ==========================
# 6. Summarise helpers
# ==========================
summarise_metric <- function(df, col, label) {
  df %>% transmute(
    dag_size,
    method = recode(method, "locpc"="LocPC-CDE","pc"="PC","ldecc"="LDECC","cmb"="CMB","MBbyMB"="MBbyMB"),
    value = !!sym(col)
  ) %>%
    group_by(dag_size, method) %>%
    summarise(
      n = n(),
      mean_val = mean(value, na.rm=TRUE),
      se = sd(value, na.rm=TRUE)/sqrt(n),
      lower = pmax(0, mean_val-1.96*se),
      upper = mean_val+1.96*se,
      .groups="drop"
    ) %>%
    mutate(method=factor(method,levels=method_levels), type=label)
}

summarise_prop <- function(df,label) summarise_metric(df,"identifiability",label)
summarise_f1   <- function(df,label) summarise_metric(df,"f1_score",label)
summarise_ci   <- function(df,label) summarise_metric(df,"nb_CI_tests",label)

# ==========================
# 7. Summarise results
# ==========================
prop_id <- summarise_prop(res_ident_gauss,"Identifiable")
ci_id   <- summarise_ci(res_ident_gauss,"Identifiable")
f1_id   <- summarise_f1(res_ident_gauss,"Identifiable")

# ==========================
# 8. Plot helpers
# ==========================
plot_metric <- function(df,y_col,y_label,log_y=FALSE,title="") {
  p <- ggplot(df,aes(x=as.factor(dag_size),y=!!sym(y_col),color=method,group=method,shape=method)) +
    geom_ribbon(aes(ymin=lower,ymax=upper,fill=method),alpha=0.2,color=NA) +
    geom_line(size=1) + geom_point(size=3) +
    scale_color_manual(values=sorbonne_colors) +
    scale_fill_manual(values=sorbonne_colors) +
    scale_shape_manual(values=sorbonne_shapes) +
    labs(x="DAG size",y=y_label,title=title,color="Method",fill="Method",shape="Method") +
    theme_bw() + theme(legend.position="bottom")
  if(!log_y) p <- p + ylim(0,if(y_col=="mean_val") 100 else 1)
  if(log_y) p <- p + scale_y_log10()
  p
}

# ==========================
# 9. Combine plots (test)
# ==========================
horizontal_separator <- ggplot() +
  geom_hline(yintercept=0,color="grey40",size=0.5,linetype="dashed") +
  theme_void() +
  theme(plot.margin=margin(0,0,0,0),panel.background=element_rect(fill="white",color=NA))

p_test <- (plot_metric(ci_id,"mean_val","# CI tests",log_y=TRUE,"LINEAR GAUSSIAN") |
             plot_metric(prop_id,"mean_val","TPR (%)") |
             plot_metric(f1_id,"mean_val","F1 Score")) +
  plot_layout(heights=c(1), guides="collect") & theme(legend.position="bottom")

print(p_test)

# ==========================
# 10. Save figure
# ==========================
ggsave("test_gauss_small_identifiable.png",p_test,width=6,height=4,dpi=800)