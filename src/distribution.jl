using Gen

abstract type ProxDistribution{T} <: Distribution{T} end

"""
    (val::T, weight) = random_weighted(dist::ProxDistribution{T}, args...)

Sample a random choice from the given distribution with the given arguments, and compute the log of
a weight whose reciprocal is an unbiased estimate of the reciprocal probability (density) of the value.
"""
function random_weighted end

"""
    weight = estimate_logpdf(dist::ProxDistribution{T}, value::T, args...)

Return the log of an unbiased estimate of the probability (density) of the value.
"""
function estimate_logpdf end

Gen.DistributionTrace{T, Dist}(val::T, args::Tuple, dist::Dist) where {T, Dist <: ProxDistribution} = Gen.DistributionTrace{T, Dist}(val, args, estimate_logpdf(dist, val, args...), dist)
@inline Gen.DistributionTrace(val::T, args::Tuple, dist::Dist) where {T, Dist <: ProxDistribution}  = Gen.DistributionTrace{T, Dist}(val, args, estimate_logpdf(dist, val, args...), dist)

@inline function Gen.simulate(dist::Dist, args::Tuple) where {T, Dist <: ProxDistribution{T}}
    val, weight = random_weighted(dist, args...)
    Gen.DistributionTrace{T, Dist}(val, args, weight, dist)
end

@inline function Gen.propose(dist::ProxDistribution, args::Tuple)
    val, score = random_weighted(dist, args...)
    (ValueChoiceMap(val), score, val)
end

@inline function Gen.assess(dist::ProxDistribution, args::Tuple, choices::ValueChoiceMap)
    weight = estimate_logpdf(dist, get_value(choices), args...)
    (weight, choices.val)
end

(dist::ProxDistribution)(args...) = first(random_weighted(dist, args...))

export random_weighted, estimate_logpdf, ProxDistribution
