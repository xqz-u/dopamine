library(here)
library(mongolite)
library(tidyverse)
library(glue)

source(here("config.R"))

coll_name <- "ab_dqn_full_experience_%%"
## coll_name <- "ab_dqvmax_distr_shift"
coll <- mongo(collection = coll_name,
              db = "thesis_db",
              url = xxx_mongodb_uri)

alldata <- coll$find("{}")
alldata <- alldata %>% as_tibble()
colnames(alldata)
summary(alldata)

avg_losses <- alldata %>%
    ## NOTE only while redundancy_nr is not saved
    filter(!is.na(curr_redundancy)) %>%
    group_by(curr_redundancy) %>%
    select(loss, steps) %>%
    unnest(cols = loss) %>%
    summarise(across(.cols = ends_with("loss"),
                     .fns = ~ .x / steps,
                     .names = "avg_step_{.col}")) %>%
    mutate(iterations = row_number(),
           curr_redundancy = as.factor(curr_redundancy))
avg_losses

avg_losses %>%
    drop_na() %>%
    filter(curr_redundancy == 0) %>%
    ggplot(aes(x = iterations,
               y = avg_step_qfunc_huber_loss,
               colour = curr_redundancy)) +
    geom_line() +
    theme_bw()



avg_losses %>%
    drop_na() %>%
    filter(curr_redundancy == 0) %>%
    ggplot(aes(x=iterations,y=avg_step_qfunc_huber_loss)) +
  stat_smooth(method="loess", span=0.1, se=TRUE, alpha=0.3) +
  theme_bw()
