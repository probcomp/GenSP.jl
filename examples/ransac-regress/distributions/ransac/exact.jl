using StatsBase
using Random
using Distributions
using Combinatorics

struct RANSACExact <: Distribution{Vector{Float64}} end
const ransac_exact = RANSACExact()

function Gen.random(::RANSACExact, xs, ys, K=3)

    N = length(xs)
    indices = Vector{Int}(undef, K)
    StatsBase.knuths_sample!(1:N, indices)
    
    b, m = hcat(ones(K), xs[indices]) \ ys[indices]
    return [b + randn(), m + randn()]

end


function Gen.logpdf(::RANSACExact, params, xs, ys, K=3)
    b, m = params
    logpdfs = Float64[]
    N = length(xs)

    for indices in combinations(1:N, K)
        solved_b, solved_m = hcat(ones(K), xs[indices]) \ ys[indices]
        push!(logpdfs, Distributions.logpdf(Normal(solved_b, 1), b) + Distributions.logpdf(Normal(solved_m, 1), m))
    end

    GenSP.logmeanexp(logpdfs)
end

