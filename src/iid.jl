struct IIDSPDist{T} <: SPDistribution{Vector{T}}
    dist :: SPDistribution{T}
end

function random_weighted(d::IIDSPDist, n, args...)
    runs = [random_weighted(d.dist, args...) for _ in 1:n]
    vals   = [run[1] for run in runs]
    weight = sum([run[2] for run in runs])
    return vals, weight
end

function estimate_logpdf(d::IIDSPDist, vals, n, args...)
    return sum(estimate_logpdf(d.dist, val, args...) for val in vals)
end

