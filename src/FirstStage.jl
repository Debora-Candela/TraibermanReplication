using DataFrames
using LinearAlgebra
using Statistics

# This function executes the Expectation-Maximization (EM) algorithm to estimate the first stage of the Traiberman structural model

# This script (tries to be) a direct Julia translation of the original author's MATLAB script `emCode.m` 
# The structure is similar to it, with specific computational adaptations for Julia's coding
# Comments that starts with "%%" are the exact replication of author's comments


function estimate_first_stage(df::DataFrame, XINC::Matrix, YPI::Matrix, XPI::Matrix, dex::Matrix; maxIters::Int=125, tol::Float64=1e-5)
    
    println("--- Initializing EM Algorithm ---")

    # Define contatnt variables
    N = length(unique(df.pid))
    T = length(unique(df.year))
    NT = nrow(df)
    nOccs = maximum(df.gCodeB)
    nTypes = 2
    
    kappa = -0.5 * log(2 * pi)
    y1 = minimum(df.year)
    
    # %% Initialization of Programming Objects
    # MATLAB cell(nOccs,T) arrays were replaced by Julia Array{Any, 2} in the DGP step
    sigma = zeros(nOccs)
    BETA = zeros(nOccs, 4) 
    LLWAGE = zeros(NT * nTypes)
    epsilon = zeros(NT * nTypes)
    LLPI = zeros(NT, nTypes)
    L = zeros(N, nTypes)
    
    q1 = 0.4 .+ 0.2 .* rand(N) 
    PI = [mean(q1), 1 - mean(q1)]
    PIOLD = 1.0
    
    iter = 0
    
    # Creating the grouping index G for demeaning and MATLAB's repmat(G,2,1) is translated using vcat
    G = df.gCodeB .+ nOccs .* (df.year .- y1)
    G_stacked = vcat(G, G)
    max_G = nOccs * T
    
    println("Initial Guess of Pi Low: ", round(PI[1], digits=4))

    #Main EM Loop
    while iter < maxIters
        iter += 1
        println("\nStarting Iteration ", iter)
        
        # %% Getting the Probabilities
        PI = [mean(q1), 1 - mean(q1)]
        PIOLD = PI[1]
        PI_TYPE = hcat(q1, 1 .- q1) 
        
        # Step Maximization - Part 1: Running the LPMs for transitions
        # %% Running the LPMs (Linear Probability Models for transitions)
        println("Running Set of LPMs...")
        for type in 1:nTypes
            P = PI_TYPE[df.pid, type] 
            for occ in 1:nOccs
                for t_idx in 1:T
                    idx = dex[occ, t_idx]
                    if length(idx) > 0
                        X = XPI[occ, t_idx]
                        Y = YPI[occ, t_idx]
                        P_idx = P[idx]
                        
                        # MATLAB's bsxfun(@times, X, P) is replaced by Julia's broadcasting element .*
                        X_weighted = X .* P_idx
                        Y_weighted = Y .* P_idx
                        
                        # MATLAB uses the backslash operator \ for WLS but because synthetic data can generate 
                        # perfectly collinear occupation-year cells (it happened at the beginning of the implementation of the code), `\` throws a SingularException in Julia. 
                        # I use the Moore-Penrose pseudo-inverse (pinv) to guarantee stability.
                        beta_LPM = pinv(X' * X_weighted) * (X' * Y_weighted)
                        y_hat = X * beta_LPM
                        
                        LLPI[idx, type] .= vec(sum(y_hat .* Y, dims=2))
                    end
                end
            end
        end

        # For bounding probabilities, log(max(min(LLPI,1),.00001)) in MATLAB code is translated to Julia's `clamp` function
        LLPI = log.(clamp.(LLPI, 0.00001, 1.0))
        
        # Step Maximization - Part 2: Demeaning and Wage Regressions
        
        # %% Demeaning
        println("Running Mincer Regressions...")
        P_stacked = vcat(PI_TYPE[df.pid, 1], PI_TYPE[df.pid, 2])
        LINC_stacked = vcat(df.LINC, df.LINC)
        
        # Adding Type Dummy: 0 for Type 1, 1 for Type 2
        XINC2 = vcat(hcat(XINC, zeros(NT)), hcat(XINC, ones(NT)))
        
        # MATLAB relies heavily on accumarray(G, P.*LINC, [], @mean) command for grouped means
        # Since Julia's base library lacks a direct 1:1 accumarray, I explicitly implement the grouping and averaging via a loop
        mLINC = zeros(max_G)
        count_G = zeros(max_G)
        for i in 1:(2*NT)
            mLINC[G_stacked[i]] += P_stacked[i] * LINC_stacked[i]
            count_G[G_stacked[i]] += 1
        end
        mLINC .= mLINC ./ max.(count_G, 1.0)
        LINC2 = LINC_stacked .- 2 .* mLINC[G_stacked]
        
        for j in 1:4
            mXINC = zeros(max_G)
            for i in 1:(2*NT)
                mXINC[G_stacked[i]] += P_stacked[i] * XINC2[i, j]
            end
            mXINC .= mXINC ./ max.(count_G, 1.0)
            XINC2[:, j] .= XINC2[:, j] .- 2 .* mXINC[G_stacked]
        end
        
        # %% Running the Wage Regression
        gCodeB_stacked = vcat(df.gCodeB, df.gCodeB)
        for occ in 1:nOccs
            dexINC = gCodeB_stacked .== occ
            if sum(dexINC) > 0
                X_occ = XINC2[dexINC, :]
                Y_occ = LINC2[dexINC]
                P_occ = P_stacked[dexINC]
                
                # WLS for Wages and again I use pinv instead of \ 
                X_weighted = X_occ .* P_occ
                Y_weighted = Y_occ .* P_occ
                
                BETA[occ, :] = pinv(X_occ' * X_weighted) * (X_occ' * Y_weighted)
                
                epsilon[dexINC] = Y_occ .- X_occ * BETA[occ, :]
                
                # Weighted Variance (MATLAB's var(epsilon, P))
                sigma[occ] = sum(P_occ .* (epsilon[dexINC] .^ 2)) / sum(P_occ)
                
                LLWAGE[dexINC] = kappa .- 0.5 .* log(sigma[occ]) .- 0.5 .* (epsilon[dexINC] .^ 2) ./ sigma[occ]
            end
        end 
        
        # Step Expectation: Updating Likelihoods and Bayes' Rule
        LLWAGE_mat = reshape(LLWAGE, NT, 2)
        
        # Translating accumarray(pid, LLPI + LLWAGE) from MATLAB's code
        fill!(L, 0.0) # Reset L for the new iteration
        for i in 1:NT
            L[df.pid[i], 1] += LLPI[i, 1] + LLWAGE_mat[i, 1]
            L[df.pid[i], 2] += LLPI[i, 2] + LLWAGE_mat[i, 2]
        end
        
        # Bayes Rule
        q1 = PI[1] ./ (PI[1] .+ PI[2] .* exp.(L[:, 2] .- L[:, 1]))
        PI_new = mean(q1)
        
        # Break the loop early if we converged
        if abs(PIOLD - PI_new) < tol
            println("\nConvergence reached at iteration ", iter)
            break
        end
    end

    PI = [mean(q1), 1 - mean(q1)]

    println("--- EM Algorithm Finished ---")
    println("Final Guess of Pi Low: ", round(PI[1], digits=4))
    
    return q1, BETA, sigma
end
