# A merge is either the symbol :stop or a pair of sets to merge,
# where the minimum value of the first set is smaller than that of
# the second set.
Merge = Union{Symbol, Tuple{Set{Int}, Set{Int}}}

function enumerate_merges(clustering)
    possible_merges = Merge[:stop]
    for ci in clustering
        for cj in clustering
            if (minimum(ci) < minimum(cj))
                push!(possible_merges, (ci, cj))
            end
        end
    end
    return possible_merges
end

struct RandomMerge <: Distribution{Merge} end
const random_merge = RandomMerge()

function score_merge(merge::Merge, data, alpha)
    if (merge == :stop)
        return 0.0
    end
    # Compute the log probability ratio of the proposed change.
    # There is a CRP term for this, and a term based on the likelihood.
    # For the likelihood: bag probability of the union - sum of bag probabilities of the individual clusters.
    # For the CRP: look at size(data) - size(union). That's the base number of customers.
    # Then compare adding size(union) to a single table, vs. two tables.
    data_cluster_i = Set(data[i] for i in (merge[1]))
    data_cluster_j = Set(data[i] for i in (merge[2]))
    data_union = union(data_cluster_i, data_cluster_j)
    likelihood_ratio = Gen.logpdf(gaussian_bag, data_union, length(data_union), GAUSSIAN_HYPERS) - Gen.logpdf(gaussian_bag, data_cluster_i, length(data_cluster_i), GAUSSIAN_HYPERS) - Gen.logpdf(gaussian_bag, data_cluster_j, length(data_cluster_j), GAUSSIAN_HYPERS)
    customers = length(data) - length(data_union)
    crp_union = (log(alpha) - log(customers + alpha)) + sum(log(i) - log(customers + i + alpha) for i in 1:length(data_union)-1; init=0)
    crp_separate = (log(alpha) - log(customers + alpha)) + sum(log(i) - log(customers + i + alpha) for i in 1:length(data_cluster_i)-1; init=0) + (log(alpha) - log(customers+length(data_cluster_i)+alpha)) + sum(log(i) - log(customers + length(data_cluster_i) + i + alpha) for i in 1:length(data_cluster_j)-1; init=0)
    crp_ratio = crp_union - crp_separate
    return likelihood_ratio + crp_ratio
end

function Gen.random(::RandomMerge, clustering::Set{Set{Int}}, data, alpha, predicate)
    possible_merges = [m for m in enumerate_merges(clustering) if predicate(m)]
    scores = [score_merge(m, data, alpha) for m in possible_merges]
    normalized_scores = scores .- logsumexp(scores)
    return possible_merges[categorical(exp.(normalized_scores))]
end

function Gen.logpdf(::RandomMerge, merge::Merge, clustering::Set{Set{Int}}, data, alpha, predicate)
    possible_merges = [m for m in enumerate_merges(clustering) if predicate(m)]
    scores = Dict(m => score_merge(m, data, alpha) for m in possible_merges)
    total = logsumexp(collect(values(scores)))
    return haskey(scores, merge) ? scores[merge] - total : -Inf
end
