module TraibermanReplication

# export the main execution function and the exhibit generation function so that can be called after loading the pkg
export run_first_stage_pipeline, table4_like, run_tests

# read the .jl files created in the replication package
include("DGP.jl") #data generation
include("FirstStage.jl") # EM estimation
include("Results.jl") # Table 4 of the paper


# this function puts together the data generation and the Expectation-Maximization algorithm for the first stage of the Traiberman model
function run(; N::Int=1000, T::Int=12, nOccs::Int=5, maxIters::Int=125)
    println("======================================================")
    println("  Starting Traiberman First Stage Pipeline     ")
    println("======================================================")
    
    # generate synthethic dataset
    println("\n>>> STEP 1: Generating Synthetic Data (DGP) <<<")
    df, XINC, YPI, XPI, dex = generate_full_dgp(N, T, nOccs)
    println("Data generated successfully. N = ", length(unique(df.pid)), ", T = ", length(unique(df.year)))
    
    # Go for the EM algorithm
    println("\n>>> STEP 2: Running EM Algorithm <<<")
    q1_est, BETA_est, sigma_est = estimate_first_stage(df, XINC, YPI, XPI, dex; maxIters=maxIters)
    
    # generate and display the structural exhibits
    println("\n>>> STEP 3: Generating Replication Exhibits <<<")
    raw_results = (q1=q1_est, beta=BETA_est, sigma=sigma_est, data=df)
    table4 = table4_like(raw_results)
    
    println("\n--- Table 4 Analogue (Structural Sorting) ---")
    display(table4)

    println("\n======================================================")
    println("  Pipeline Completed Successfully!                    ")
    println("======================================================")
    
    # give back the variables produced
    return (q1=q1_est, beta=BETA_est, sigma=sigma_est, data=df)
end

run_first_stage_pipeline() = run()

# this function executes the project's internal test suite
function run_tests()
    Base.run(`$(Base.julia_cmd()) --project=. -e "using Pkg; Pkg.test(\"TraibermanReplication\")"`)
end

end