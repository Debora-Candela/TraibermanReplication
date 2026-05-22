using TraibermanReplication
using Test

@testset "TraibermanReplication.jl" begin
    # Test on the Data Generating Process
    # test it on a small scale (N=20, T=4, nOccs=3) to make sure that the matrices created are correctly pre-allocated for the EM algorithm
    df, XINC, YPI, XPI, dex = TraibermanReplication.generate_full_dgp(20, 4, 3)

    # I verify that the wage covariate matrix correctly stacks all N*T (20*4=80) observations
    @test size(XINC) == (80, 3)

    # I ensure that the simulated occupational choices stay strictly within the defined bounds [1, nOccs]
    @test all(df.gCodeB .>= 1)
    @test all(df.gCodeB .<= 3)

    # I assert that every worker transition to exactly one occupation, so the rows must sum perfectly to 1
    @test all(row -> sum(row) == 1, eachrow(YPI[1, 1]))

    # I check that 'dex' filters the data perfectly: dex[2, 1] must contain workers from Occupation 2
    @test all(df.gCodeB[dex[2, 1]] .== 2)

    # I verify that in the final period T, no future transitions occur, so the array must be empty
    @test isempty(dex[1, 4])


    # Test on the First Stage EM estimation process
    # test a very short EM estimation (maxIters=2) on a small subset to verify bounds and numerical stability of the estimator
    q1, beta, sigma = TraibermanReplication.run(N=50, T=4, nOccs=3, maxIters=2)

    # I check that the algorithm outputs the correct dimensions for the latent type probabilities (N 50), the wage coefficients (nOccs x 4 covariates),
    # and the occupation-specific variances (nOccs 3)
    @test length(q1) == 50
    @test size(beta) == (3, 4)
    @test length(sigma) == 3

    # I check that the updated posterior probabilities of belonging to the latent type is always between 0 and 1
    @test all(0 .<= q1 .<= 1)

    # I assert that the WLS regressions do not produce NaN or Inf
    @test all(isfinite, beta)

    # I check that the estimated variances are strictly positive
    @test all(sigma .> 0)


    # Test on the Table 4 
    # giving the tested EM outputs into the results function, want to verify the final aggregation
    table4 = TraibermanReplication.table4_like((q1=q1, beta=beta, sigma=sigma, data=df))
    
    # I verify that the output has the structure 2-type x 5-column
    @test size(table4) == (2, 5)

    # I assert that the aggregate unconditional type shares (Mean Theta) remain valid probabilities
    @test all(table4.Mean_theta .>= 0)
    @test all(table4.Mean_theta .<= 1)

    # I ensure that the income weighting and correlation calculations do not produce missing values,
    # (so I am checking that the denominator in the weighted averages is never zero)
    @test all(isfinite, table4.Mean_income_relative)
    @test all(isfinite, table4.corr_CA_wage)
end
