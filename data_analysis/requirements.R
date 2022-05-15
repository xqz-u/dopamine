#! /usr/bin/Rscript

if (!require(pacman)) {
    install.packages("pacman", repos = "https://cloud.r-project.org")
}

packages <- c("here",
              "glue",
              "viridis",
              "tidyverse",
              "mongolite",
              "ggpubr",
              "zeallot")

library(pacman)
pacman::p_load(char = packages)
