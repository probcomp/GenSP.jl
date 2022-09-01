struct RANSACUnsafe <: ProxDistribution{Vector{Float64}} end
const ransac_unsafe = RANSACUnsafe()

function GenProx.random_weighted(::RANSACUnsafe, xs, ys, K=3)
    N = length(xs)
    indices = Vector{Int}(undef, K)
    StatsBase.knuths_sample!(1:N, indices)
    
    b, m = hcat(ones(K), xs[indices]) \ ys[indices]
    randomized = [b + randn(), m + randn()]
    return randomized, logpdf(Normal(b, 1), randomized[1]) + logpdf(Normal(m, 1), randomized[2])
end

function GenProx.estimate_logpdf(::RANSACUnsafe, params, xs, ys, K=3)
    b, m = params

    N = length(xs)
    indices = Vector{Int}(undef, K)
    StatsBase.knuths_sample!(1:N, indices)
    
    solved_b, solved_m = hcat(ones(K), xs[indices]) \ ys[indices]

    Distributions.logpdf(Normal(solved_b, 1), b) + Distributions.logpdf(Normal(solved_m, 1), m)
end