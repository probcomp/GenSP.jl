struct RenderPointExactUnsafe <: SPDistribution{Vector{Float64}} end

const render_point_exact_unsafe = RenderPointExactUnsafe()

function GenSP.random_weighted(::RenderPointExactUnsafe, X, X_tree, sigma)
    error("Not implemented")
end

function GenSP.estimate_logpdf(::RenderPointExactUnsafe, y, X, X_tree, sigma)
    m = size(X, 2)
    return logmeanexp([Gen.logpdf(Gen.broadcasted_normal, y, X[:, i], sigma) for i in 1:m])
end