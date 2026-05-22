using DataFrames
using Distributions
using Random
using Statistics

# Simulates a synthetic worker panel to validate the First-Stage Expectation-Maximization (EM) algorithm. 
# Since the original Traiberman (2019) Danish administrative data is strictly confidential, this Data Generating 
# Process (DGP) recreates the core structural features of the model: unobserved latent types (comparative advantage), 
# age-dependent human capital, and state-dependent occupational switching costs.

# Returns:
#        df: The main panel DataFrame.
#        XINC`: Pre-allocated covariates matrix for the wage regressions.
#        YPI, XPI, dex: Sliced transition arrays mapping the original MATLAB Cell Arrays to native Julia, 
 # drastically reducing memory allocation inside the EM loop.

function generate_full_dgp(N::Int=1000, T::Int=12, nOccs::Int=5; seed::Int=123)
    Random.seed!(seed)

    # Base variables
    NT = N * T # Tot number of obs
    y1 = 1996 # Set the starting year to 1996 to match the beginning of Traiberman's sample
    y2 = y1 + T - 1 # Last year in the dataset = 1996 + 12 - 1 = 2007, matching the end of Traiberman's sample

    # Pre-allocate empty vectors for each column of DataFrame
    pid = Vector{Int}(undef, NT)              # Will store the unique worker ID
    year = Vector{Int}(undef, NT)             # Will store the calendar year
    gCodeB = Vector{Int}(undef, NT)           # Will store the current occupation code
    next_gCodeB = Vector{Int}(undef, NT)      # Will store the destination occupation code
    age = Vector{Int}(undef, NT)              # Will store the worker's age
    true_type = Vector{Int}(undef, NT)        # Will store the unobserved latent type (1 or 2)
    LINC = Vector{Float64}(undef, NT)         # Will store the realized log-income

    # Base structural effects for wages and transitions
    # I create the "true" parameters of the economy that the EM algorithm will later try to estimate
    occ_wage_effect = range(-0.20, 0.20; length=nOccs) # Baseline wage premium for each of the occupations
    # I create the Comparative Advantage matrix: Type 1 is more productive in early occupations, 
    # while Type 2 is more productive in later occupations
    type_occ_effect = hcat(range(0.15, -0.15; length=nOccs),
                           range(-0.15, 0.15; length=nOccs))
    destination_effect = range(-0.30, 0.30; length=nOccs) # Baseline utility parameter governing the attractiveness of moving to a specific occupation

    # Simulating Worker Trajectories
    # loop over each individual worker to generate their specific career arc
    for i in 1:N
        worker_type = rand() < 0.55 ? 1 : 2 # permanent latent type to the worker (55% chance for Type 1) - Assumption 1
        initial_age = rand(23:45) # Draw a random starting age for the worker between 23 and 45
        current_occ = rand(1:nOccs) # randomly assign their starting occupation

        # For each worker, loop through time to generate their history
        for t_idx in 1:T
            row = (i - 1) * T + t_idx # compute the exact row index for the current observation
            current_year = y1 + t_idx - 1 # Compute the current calendar year
            current_age = initial_age + t_idx - 1 # Update the worker's age based on the time step

            # Fill in the pre-allocated vectors with the worker's current state
            pid[row] = i
            year[row] = current_year
            age[row] = current_age
            true_type[row] = worker_type
            gCodeB[row] = current_occ

            # Mincerian Wage Equation: 
            # Base income + returns to age + occupation fixed effect + comparative advantage
            log_income_mean = 5.0 +
                              0.035 * current_age +
                              occ_wage_effect[current_occ] +
                              type_occ_effect[current_occ, worker_type]
            # add a random Gaussian shock to generate the final observed log-income
            LINC[row] = log_income_mean + rand(Normal(0, 0.25))

            # Simulate the occupational choice for the next period (if not the last period)
            if t_idx < T
                utility = similar(collect(destination_effect), Float64) # Initialize empty vector
                # Calculate the specific utility for each possible destination occupation
                for dest in 1:nOccs
                    stay_bonus = dest == current_occ ? 1.25 : 0.0 # add a utility bonus for staying in the same occupation (switching cost friction)
                    type_match = worker_type == 1 ? -0.10 * dest : 0.10 * dest # add a preference parameter based on the worker's latent type
                    age_switch_penalty = dest == current_occ ? 0.0 : -0.01 * max(current_age - 35, 0) # Penalty for switching that increases if the worker is older than 35
                    
                    # Put all components together to get the final utility for this destination
                    utility[dest] = destination_effect[dest] + stay_bonus + type_match + age_switch_penalty
                end

                # Convert utilities into probabilities using a Multinomial Logit specification
                exp_utility = exp.(utility .- maximum(utility)) # subtract the maximum utility to prevent (Inf) when exponentiating
                transition_prob = exp_utility ./ sum(exp_utility)

                # Draw the next occupation using the calculated probability distribution
                next_occ = rand(Categorical(transition_prob))
            else
                # But if it's the terminal period, no transition occurs
                next_occ = current_occ
            end

            # Store the destination occupation and update the state for the next loop
            next_gCodeB[row] = next_occ
            current_occ = next_occ
        end
    end

    # final dataframe
    df = DataFrame(
        pid=pid,
        year=year,
        gCodeB=gCodeB,
        next_gCodeB=next_gCodeB,
        age=age,
        true_type=true_type,
        LINC=LINC,
    )

    # Constructing Estimator Inputs (Arrays)
    # Build the specific matrix structures required by the First-Stage EM Algorithm

    age_centered = age .- mean(age) # center the age variable and build the covariate matrix XINC (Intercept, Age, Age^2)
    XINC = hcat(ones(NT), age_centered ./ 10, (age_centered ./ 10) .^ 2)

    #Initialization of Julia equivalents for MATLAB Cell Arrays
    YPI = Array{Any, 2}(undef, nOccs, T)
    XPI = Array{Any, 2}(undef, nOccs, T)
    dex = Array{Any, 2}(undef, nOccs, T)

    # Fill the arrays by looping over all possible origin occupations and time periods
    for occ in 1:nOccs # occupations
        for t_idx in 1:T # time periods
            current_year = y1 + t_idx - 1

            if t_idx < T
                row_indices = findall((df.year .== current_year) .& (df.gCodeB .== occ)) # Find all row indices in the DataFrame matching the specific occupation-year
            else
                row_indices = Int[]
            end
            dex[occ, t_idx] = row_indices # store these indices in the 'dex' matrix for filtering in the EM loop (FirsStage.jl file)

            n_obs_t = length(row_indices) 

            destinations = df.next_gCodeB[row_indices] # Get the destination occupations chosen by the workers in this cell

            # Create a one-hot encoded matrix (Y) for the destinations
            Y = zeros(n_obs_t, nOccs)
            for (row_pos, dest) in enumerate(destinations)
                Y[row_pos, dest] = 1.0
            end
            YPI[occ, t_idx] = Y # and save it into YPI

            # Construct the XPI matrix containing covariates for the transition probabilities
            if n_obs_t == 0
                XPI[occ, t_idx] = zeros(0, 3)
            else
                XPI[occ, t_idx] = hcat(
                    ones(n_obs_t),
                    (df.age[row_indices] .- mean(age)) ./ 10,
                    fill(occ / nOccs, n_obs_t),
                )
            end
        end
    end

    return df, XINC, YPI, XPI, dex
end
