#! /usr/bin/Rscript

if (!require(pacman)) {
    install.packages("pacman", lib = "~/", repos = "https://cloud.r-project.org")
}

packages <- c("here",
              "glue",
              "viridis",
              "tidyverse",
              "mongolite",
              "ggpubr", # NOTE needs cmake and gcc-fortran (on Arch)
              "zeallot")

library(pacman)
pacman::p_load(char = packages)
