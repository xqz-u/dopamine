library(here)
library(scales)
library(mongolite)
library(tidyverse)
library(ggpubr)
library(glue)
library(zeallot)

source(here("config.R"))
source(here("utils.R"))


## TODO
## - functions that can do what's below on different collections
## - add info about environment and agent type to plot titles (not
## saved on mongo yet)
## (experiments)

perstep_metrics <- function(df, metrics, step_var) {
    step_multiplier <- max(df[df$current_schedule == "train", ]$steps)
    df %>%
        group_by(redundancy_nr) %>%
        drop_na({{metrics}}) %>%
        summarise(across(.cols = {{metrics}},
                         .fns = ~ .x / .data[[step_var]]),
                  steps = row_number() * step_multiplier) %>%
        ungroup()
}


## assumes metric_suffix is passed as as string
metric_rowwise_mean <- function(df, metric_suffix) {
    metric_suffix_sym <- as.name(metric_suffix)
    df %>%
        rowwise() %>%
        mutate("mean_{{metric_suffix_sym}}" :=
                   mean(c_across(contains({{metric_suffix}}))),
               "sd_{{metric_suffix_sym}}" :=
                   sd(c_across(contains({{metric_suffix}}))))
}


aggr_redundancies <- function(df, aggr_vars) {
    res <- df %>%
        pivot_wider(names_from = redundancy_nr,
                    values_from = !!aggr_vars,
                    names_glue = "exp_{.value}_{redundancy_nr}")
    for (v in aggr_vars)
        res <- res %>% metric_rowwise_mean(v)
    expanded_cols_prefix <- add_prefix("exp_", aggr_vars)
    res %>%
        pivot_longer(cols = starts_with(expanded_cols_prefix),
                     names_to = c(".value", "redundancy"),
                     names_pattern = "exp_(.*)_(\\d)")
}


plot_by_redundancy <- function(df, ymetric, ytitle, title, lsize=0.5) {
    nruns <- as.numeric(max(with(df, redundancy))) + 1
    title <- glue("{title} (#runs={nruns})")
    df %>%
        ggplot(aes_string(x = "steps",
                          y = ymetric,
                          color = "redundancy")) +
        geom_line(size = lsize, show.legend = FALSE) +
        theme_bw() +
        scale_x_continuous(labels = label_number(suffix = "k",
                                                 scale = 1e-3),
                          expand = c(0, 0)) +
        scale_y_continuous(expand = c(0, 0)) +
        labs(title = title, x = "Steps", y = ytitle) +
        theme(plot.title = element_text(hjust = 0.5))
}


## will add prefixes "mean_" and "sd_" to ymetric, which will then
## be queried in df; "mean_{ymetric}" is then plotted
plot_runs_mean_metric <- function(df, ymetric, ytitle, title) {
    c(m_metric, sd_metric) %<-% add_suffix(glue("_{ymetric}"),
                                              c("mean", "sd"))
    df %>%
        plot_by_redundancy(m_metric, ytitle, title, lsize = 0.8) +
        geom_ribbon(aes(ymin = .data[[m_metric]] - .data[[sd_metric]],
                        ymax = .data[[m_metric]] + .data[[sd_metric]],
                        fill = redundancy),
                    alpha = 0.2,
                    color = NA,
                    show.legend = FALSE)
}


plot_all <- function(df, ymetric, ytitle, title) {
    lapply(c(plot_by_redundancy, plot_runs_mean_metric),
           function(fn) fn(df, ymetric, ytitle, title))
}


summarise_redundancies <- function(df, metrics, step_var="steps") {
    df %>%
        perstep_metrics(metrics, step_var = step_var) %>%
        aggr_redundancies(metrics)
}


## env <- "cp"
env <- "ab"
coll_name <- glue("{env}_dqvmax_distr_shift")
coll <- mongo(collection = coll_name,
              db = "thesis_db",
              url = xxx_mongodb_uri)

alldata <- coll$find("{}") %>% as_tibble()


## ----------------------------- plots reward
reward_var <- "reward"
reward_summ <- summarise_redundancies(alldata, reward_var, "episodes")
reward_plots <- plot_all(reward_summ,
                         reward_var,
                         "Mean reward by episode",
                         "Evaluation")


## ----------------------------- plots q estimates
q_estims_var <- "q_estimates"
q_estims_summ <- summarise_redundancies(alldata, q_estims_var)
q_estims_plots <- plot_all(q_estims_summ,
                           q_estims_var,
                           "Q-estimates",
                           "Training estimates of Q values")


## ----------------------------- plot losses
losses_names <- c("qfunc_huber_loss", "vfunc_huber_loss")
losses_summ <- alldata %>%
    unnest(loss) %>%
    summarise_redundancies(losses_names)
losses_plots <- data.frame(losses_names,
                           add_suffix("-function Huber Loss",
                                      c("Q", "V")),
                           add_suffix("-function training Huber loss",
                                      c("Q", "V"))) %>%
    apply(1, function(df_row) {
        c(ymetric, ytitle, title) %<-% df_row
        plot_all(losses_summ, ymetric, ytitle, title)
    })



plots_redunds <- ggarrange(losses_plots[[1]][[1]],
                           losses_plots[[2]][[1]],
                           q_estims_plots[[1]],
                           reward_plots[[1]],
                           ncol = 2, nrow = 2)
## ggsave(glue("{images_path}/{env}_dqvmax_redunds.png"),
##        plot = plots_redunds,
##        width = 17.9, height = 12.1)


plots_means <- ggarrange(losses_plots[[1]][[2]],
                         losses_plots[[2]][[2]],
                         q_estims_plots[[2]],
                         reward_plots[[2]],
                         ncol = 2, nrow = 2)
## ggsave(glue("{images_path}/{env}_dqvmax_means.png"),
##        plot = plots_means,
##        width = 17.9, height = 12.1)





## for centered moving average
## install.packages("RcppRoll")
## library(RcppRoll)

## reward_summ %>%
##     group_by(redundancy) %>%
##     mutate(roll_reward = roll_mean(reward, 25, fill = NA)) %>%
##     ungroup() %>%
##     plot_by_redundancy("roll_reward", "pippo", "pippo")
