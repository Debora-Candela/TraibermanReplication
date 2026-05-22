using DataFrames
using Statistics


# Build a Table 4-style exhibit from the synthetic first-stage results

# The original paper reports six empirical types. This package estimates two
# latent types, following the simplified first-stage implementation, so the
# output is gonna be a two-row analogue.

function table4_like(results)

    # Get the tuple results from the EM algorithm
    q1 = results.q1 # individual-level posterior probabilities
    beta = results.beta
    df = results.data

    # Map q1 to the full N*T panel using the worker IDs (pid) to create a matrix of posterior weights for both types
    type_weights = hcat(q1[df.pid], 1 .- q1[df.pid])
    income = exp.(df.LINC) # convert log-income into levels to compute averages
    mean_income = [sum(type_weights[:, k] .* income) / sum(type_weights[:, k]) for k in 1:2]
    relative_income = mean_income ./ maximum(mean_income) # Normalize incomes relative to the highest-earning type (following the paper's table 4 structure)
    type_share = [mean(q1), 1 - mean(q1)] # compute the unconditional aggregate share (Mean Theta) for each type

    # compute the unconditional mean income for each occupation to proxy its overall "quality"
    occ_income = [
        mean(income[df.gCodeB .== occ])
        for occ in 1:size(beta, 1)
    ]

    type2_wage_premium = beta[:, 4] # estimated wage premium for Type 2
    comparative_advantage = hcat(-type2_wage_premium, type2_wage_premium)
    
    # Correlation between a type's comparative advantage in an occupation and that occupation's average income
    # In the paper we have a test for sorting, cioè High types sort into high-paying jobs
    corr_ca_wage = [
        cor(comparative_advantage[:, k], occ_income)
        for k in 1:2
    ]

    # Define a median cutoff to separate the "Low" productivity type from the "High" type
    income_cutoff = median(relative_income)

    # assign the Synthetic L - Low - or Synthetic H - high - label based on the relative income cutoff
    classification = [
        rel_income < income_cutoff ? "Synthetic L" : "Synthetic H"
        for rel_income in relative_income
    ]

    return DataFrame(
        Type = 1:2,
        Classification = classification,
        Mean_theta = type_share,
        Mean_income_relative = relative_income,
        corr_CA_wage = corr_ca_wage,
    )
end
