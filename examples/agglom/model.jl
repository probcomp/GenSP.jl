using StatsBase: mean
using StatsFuns: loggamma
using CSV, DataFrames

dataset = map(Float64, CSV.read("data/galaxies.csv", DataFrame; header=false)[!, 1])

GAUSSIAN_HYPERS = GaussianHypers(0, 1 / 100, 1 / 2, 1 / 2)

@gen function dpmm(records, alpha)
    N = length(records)
    partition ~ crp(N, alpha)

    # We would like to write observe(gaussian_bag, generated_bag)
    # in each iteration of the loop below. To get around Gen's lack 
    # of inline conditioning, we instead compute a log probability 
    # and draw a dummy exponential sample, which we can later condition 
    # to equal 0.
    logprob = 0.0
    for table in partition
        generated_bag = Set(records[i] for i in table)
        logprob += Gen.logpdf(gaussian_bag, generated_bag, length(table), GAUSSIAN_HYPERS)
    end
    condition_this ~ exponential(exp(logprob))
end

# Observations:
constraints = choicemap(:condition_this => 0.0)

# Importance sampling from the prior
traces_prior, weights_prior, lml_prior = importance_sampling(dpmm, (dataset, 1.0), observations, 1000);

# Importance sampling with agglom
@gen function dpmm_proposal(dataset)
    partition ~ agglom(dataset)
    return partition
end
traces_agglom, weights_agglom, lml_agglom = importance_sampling(dpmm, (dataset, 1.0), observations, dpmm_proposal, (dataset,), 1000)