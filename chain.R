#!/usr/bin/Rscript

# This scripts creates a riverswim-style problem dataset, samples data from it,
# computes the posterior distribution, and then saves samples from this distribution
#
# WARNING: This problem is inspired by the classical riverswim problem, but
#          its specific parameters are quite different.
#
# The states in this problem are arranged as a chain with the index increasing
# from left to right.
#
# The rewards in this problem depend on state, action, and the next state and
# are assumed to be known.
#
# The prior assumptions in this model are:
#  - action a0 is known
#  - action a1 transitions only to 3 possible states (left, middle, and right)
#    and all transition probabilities are the same
#  - attempting to go over each end of the chain keeps the state unchanged
#  - rewards are assumed to be known
#
# See the domains README.md file for details about the files that are created


# remove anything in the namespace just in case this being
# run repeatedly interactively

rm(list = ls())

library(rcraam)
library(dplyr)
library(readr)
library(gtools)
library(reshape2)
loadNamespace("tidyr")
loadNamespace("reshape2")
loadNamespace("stringr")

## ----- Parameters --------
args <- commandArgs(trailingOnly=TRUE)
traj_number <- as.integer(args[1])
trial <- as.integer(args[2])
# data output (platform independent construction)
folder_input <- file.path('chain')
# riverswim problem specification
state.count <- 5                 # number of states in the riverswim problem
discount <- 0.9                   # discount rate

# transition samples
sample.seed <- 1994               # reproducibility
samples <- 15                     # number of transition samples per episode
episodes <- 1                     # number of episodes to sample from
bayes.samples <- 1000

# posterior samples
postsamples_train <- 900          # number of posterior training samples
postsamples_test <- 100           # number of posterior test samples
n_chains <- 4                     # number of MCMC chains
posterior.seeds <- c(2016, 5874, 12, 99)  # reproducibility, one seed per chain
stopifnot(n_chains == length(posterior.seeds))


## ----- Parameters --------

# weight on the risk measure used for evaluation
risk_weight_eval <- 0.5

# the algorithms can read and use these parameters
params <- new.env()
with(params, {
    confidence <- 0.01      # value at risk confidence (1.0 is the worst case)
    time_limit <- 10000    # time limit on computing the solution
    cat("Using confidence =", confidence, ", risk_weight =", risk_weight_eval, "\n")
})


## ------ Define domains ------

# list all domains that are being considered; each one should be
# in a separate directory that should have 3 files:
#     - true.csv.xz
#     - training.csv.xz    (posterior optimization samples)
#     - test.csv.xz            (posterior evaluation samples)
domains <- list(
    chain = "chain"
)

domains_paths <- "chain"

## ----- Define algorithms --------

# list of algorithms, each one implemented in a separate
# input file with a function:
# result = algorithm_main(mdpo, initial, discount), where the result is a list:
#        result$policy = computed policy
#        result$estimate = estimated return (whatever metric is optimized)
# and the parameters are:
#        mdpo: dataframe with idstatefrom, idaction, idstateto, idoutcome, probability, reward
#        initial: initial distribution, dataframe with idstate, probability (sums to 1)
#        discount: discount rate [0,1]
#
# It is a good practice for each algorithm to the risk parameters
# it is using. They may be reading the parameters from
# the global environment. That is convenient but fragile
algorithms_path <- "algorithms"

algorithms <- list(
    nominal = "nominal.R",
    bcr_l = "bcr_local.R",
    bcr_g = "bcr_global.R",
    norbu_r = "norbu_r.R",
    norbu_sr = "norbu_sr.R",
    norbuv_r = "norbuv_r.R"
)

# Determines which parameter are used to optimize the risk for all algorithms
risk_weights_optimize <-
    #c(0, 0.25, 0.5, 0.75, 1.0)
    c(0.75)

# construct paths to algorithms
algorithms_paths <- lapply(algorithms, function(a){file.path(algorithms_path, a)} )

## ------ Check domain availability ----

cat("Checking if domains are available ...\n")
if (dir.exists(domains_paths)) {
    cat("Domain:", names(domains), "available, using cached version.\n")
}


## ---- Helper Methods:    -------

#' Loads problem domain information
#'
#' @param dir_path Path to the directory with the required files
load_domain <- function(dir_path){
    cat("        Loading parameters ... \n")
    parameters <- read_csv(file.path(dir_path, "parameters.csv"), col_types = cols()) %>%
        na.fail()


    cat("        Loading true MDP ... ")
    true_mdp <- read_csv(file.path(dir_path, "true.csv.xz"), col_types =
                 cols(idstatefrom = "i", idaction = "i", idstateto = "i",
                 probability = "d", reward = "d")) %>% na.fail()

    sa.count <- select(true_mdp, idstatefrom, idaction) %>% unique() %>% nrow()
    cat(sa.count, "state-actions \n")

    cat("        Loading initial distribution ... ")
    initial <- read_csv(file.path(dir_path, "initial.csv.xz"), col_types =
                            cols(idstate = "i", probability = "d")) %>%
        na.fail()
    s.count <- select(initial, idstate) %>% unique() %>% nrow()
    cat(s.count, "states \n")


    list(
        parameters = parameters,
        discount = filter(parameters, parameter == "discount")$value[[1]],
        initial_dist = initial,
        true_mdp = true_mdp
    )
}

domain_spec <- load_domain("chain")

rewards.truth <- domain_spec$true_mdp %>% select(-probability)

#transitions <- read.csv("domains/chain/chain_traj.csv")
#print(transitions)

##  Uninformative Bayesian Posterior Sampling
#' Generate a sample MDP from dirichlet distribution
#' @param simulation Simulation results
#' @param rewards.df Rewards for each idstatefrom, idaction, idstateto
#' @param outcomes Number of outcomes to generate
mdpo_bayes <- function(simulation, rewards.df, outcomes){
  # prior contains a set of s_a_s' that accured in true mdp
  priors <- rewards.df %>% select(-reward) %>% unique()
  # compute sampled state and action counts
  # add a uniform sample of each state and action to work as the dirichlet prior
  sas_post_counts <- simulation %>%
    select(idstatefrom, idaction, idstateto) %>%
    rbind(priors) %>%
    group_by(idstatefrom, idaction, idstateto) %>%
    summarize(count = n())


  # construct dirichlet posteriors
  posteriors <- sas_post_counts %>%
    group_by(idstatefrom, idaction) %>%
    arrange(idstateto) %>%
    summarize(posterior = list(count), idstatesto = list(idstateto))


  # draw a dirichlet sample
  trans.prob <-
    mapply(function(idstatefrom, idaction, posterior, idstatesto){
      samples <- do.call(function(x) {rdirichlet(outcomes,x)}, list(posterior) )
      # make sure that the dimensions are named correctly
      dimnames(samples) <- list(seq(0, outcomes-1), idstatesto)
      reshape2::melt(samples, varnames=c('idoutcome', 'idstateto'),
                     value.name = "probability" ) %>%
        mutate(idstatefrom = idstatefrom, idaction = idaction)
    },
    posteriors$idstatefrom,
    posteriors$idaction,
    posteriors$posterior,
    posteriors$idstatesto,
    SIMPLIFY = FALSE)

  mdpo <- bind_rows(trans.prob) %>%
    full_join(rewards.df,
              by = c('idstatefrom', 'idaction','idstateto')) %>%
    na.fail()
  return(mdpo)
}

wd <- getwd()
file_path <- paste("chain/traj/traj_", as.character(traj_number), "_", as.character(trial), ".csv", sep="")

transitions <- read.csv(file_path)
mdpo <- mdpo_bayes(transitions, rewards.truth, bayes.samples)
# make sure that all probabilities sum to 1 (select all s,a,o with that do not sum to 1)
invalid <- mdpo %>% group_by(idstatefrom, idaction, idstateto, idoutcome) %>%
        summarize(sumprob = sum(probability), .groups = "keep") %>%
        filter(sumprob > 1 + 1e-6, sumprob < 1 - 1e-6)

stopifnot(nrow(invalid) == 0)

# split into test and training sets (idoutcome is 0-based)
mdpo_train <- mdpo %>% filter(idoutcome < postsamples_train)
mdpo_test <- mdpo %>% filter(idoutcome >= postsamples_train) %>%
      mutate(idoutcome = idoutcome - postsamples_train )

## ------- Save results in the directory ------

folder_output <- file.path('chain', 'posteriors', paste(as.character(traj_number), as.character(trial), sep="_"))

cat("Writing results to ", folder_output, " .... \n")
if(!dir.exists(folder_output)) dir.create(folder_output, recursive = TRUE)

write_csv(domain_spec$parameters, file.path(folder_output, 'parameters.csv'))
write_csv(domain_spec$true_mdp, file.path(folder_output, 'true.csv.xz'))
write_csv(domain_spec$initial, file.path(folder_output, 'initial.csv.xz'))
write_csv(mdpo_train, file.path(folder_output, 'training.csv.xz'))
write_csv(mdpo_test, file.path(folder_output, 'test.csv.xz'))