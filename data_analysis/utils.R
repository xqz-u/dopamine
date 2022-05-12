library(purrr)
library(glue)

add_suffix <- function(suffix, els)
    els %>% purrr:::map(~ glue("{.x}{suffix}")) %>% unlist()

add_prefix <- function(prefix, els)
    els %>% purrr:::map(~ glue("{prefix}{.x}")) %>% unlist()
