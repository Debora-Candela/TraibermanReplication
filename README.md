# Traiberman (2019) Replication Project

[![Build Status](https://github.com/Debora/TraibermanReplication.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Debora/TraibermanReplication.jl/actions/workflows/CI.yml?query=branch%3Amain)


## Project Description
This repository contains a partial computational replication of the structural labor model presented in Traiberman (2019). Because the original Danish administrative data are strictly confidential, this project translates the core Expectation-Maximization (EM) first-stage algorithm into **Julia** and validates its numerical stability and convergence using a synthetic Data Generating Process (DGP) build exactly to match all the First Stage code requirements

* **Paper:** "Occupations and Import Competition: Evidence from Denmark" by Sharon Traiberman (AER 2019). [DOI: 10.1257/aer.20161925](https://doi.org/10.1257/aer.20161925)
* **Original Replication Package:** [DOI: 10.3886/E231387V1](https://doi.org/10.3886/E231387V1)

## Course and Author
* **Course:** [Computational Economics](https://floswald.github.io/CompEcon/) (PhD program at Collegio Carlo Alberto) taught by [Prof. Florian Oswald](https://floswald.github.io/)
* **Author:** Debora Candela - Senior Allieva at CCA

## Computational Requirements & Data
* **Software:** Tested on Julia 1.12.4 (and Quarto 1.9.37 for the report)
* **Runtime & Memory:** The full pipeline (DGP + EM algorithm) takes only a few seconds to run on a standard laptop, i.e. it requires minimal RAM.
* **Data Access:** *No external data downloads are required.* Because the original Danish administrative data is strictly confidential, this package dynamically generates a synthetic 12,000 worker-year panel upon execution.

## Installation and Environment Setup

This project ensures full reproducibility by utilizing a localized Julia environment (`Project.toml` and `Manifest.toml`). 

To download the repository and start Julia, run the following commands in your standard terminal:

```bash
git clone https://github.com/Debora-Candela/TraibermanReplication.git
cd TraibermanReplication
julia 
```
Then, to replicate easily the outputs or test the environment, run the following commands in the Julia REPL in the package environment (pkg>) using `]`:

```julia
activate .
instantiate
```
Press backspace to return to the standard Julia prompt (julia>), then load the package:

```julia
using TraibermanReplication

TraibermanReplication.run()
TraibermanReplication.run_tests()
```

## Compile the Technical Report
The project includes a detailed Quarto report (report.qmd) containing the theoretical background, implementation details, and replication exhibits. To render the report to HTML, exit Julia and run the following command in your standard terminal:

```bash
quarto render report.qmd
```

## Repository Structure

The replication package is a modular Julia project with the following structure:

```text
├── src/
│   ├── DGP.jl                     # Generates the synthetic 12,000 worker-year panel and covariates
│   ├── FirstStage.jl              # Implements the core Expectation-Maximization (EM) algorithm
│   ├── Results.jl                 # Processes EM outputs to build the synthetic Table 4 analogue
│   └── TraibermanReplication.jl   # Main module orchestrating the end-to-end pipeline
├── test/
│   └── runtests.jl                # Unit testing suite asserting DGP integrity and First Stage mathematical bounds
├── report.qmd                     # Quarto source file containing the technical replication report
├── Project.toml                   # Julia environment dependencies
├── Manifest.toml                  # Exact package versions for strict reproducibility
└── README.md                      # This file
```

File Descriptions:

* **TraibermanReplication.jl**: The entry point of the package. It exports the run() function which sequentially triggers the data generation and the EM estimator;

* **DGP.jl**: Contains generate_full_dgp(), which simulates worker demographics, calculates structural human capital profiles, and pre-allocates the transition matrices (XPI, YPI, dex) required by the EM algorithm;

* **FirstStage.jl**: Contains estimate_first_stage(), the computational engine of the project. It alternates between Weighted Least Squares (WLS) estimation of wage/transition parameters and Bayesian updating of latent type probabilities until convergence;

* **Results.jl**: Contains table4_like(), a post-processing function that aggregates the raw EM outputs into structural income and comparative advantage exhibits;

* **runtests.jl**: Executes the full pipeline on a reduced parameter set to automatically verify data dimensions, probability distribution constraints and estimator stability.