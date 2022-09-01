using StatsBase

struct KnuthsSample <: Distribution{Set{Int}} end
const knuths_sample = KnuthsSample()

function Gen.random(::KnuthsSample, N, K)
    indices = Vector{Int}(undef, K)
    StatsBase.knuths_sample!(1:N, indices)
    return Set(indices)
end

function Gen.logpdf(::KnuthsSample, indices, N, K)
    # Probability of a particular combination `indices`
    # being sampled without replacement from a collection
    # of `N` elements. This is just (N choose K).
    return sum(log(i) for i in 1:K; init=0) - sum(log(i) for i in (N-K+1):N; init=0)
end