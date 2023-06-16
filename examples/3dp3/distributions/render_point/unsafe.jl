struct RenderPointUnsafe <: ProxDistribution{Vector{Float64}} end

const render_point_unsafe = RenderPointUnsafe()

function GenProx.random_weighted(::RenderPointUnsafe, X, X_tree, sigma)
    error("Not implemented")
end

NUM_NEIGHBORS = 100

function GenProx.estimate_logpdf(::RenderPointUnsafe, y, X, X_tree, sigma)
    m = size(X, 2)
    # Nearest neighbors lookup
    nearest_indices, dists = NearestNeighbors.knn(X_tree, y, NUM_NEIGHBORS, true)
    # Categorical probabilities
    near_weights = -0.5 .* (dists .^ 2 / sigma^2)
    far_weight = last(near_weights)
    log_total = logsumexp(near_weights) #[near_weights..., log(m-NUM_NEIGHBORS) + far_weight])
    log_total = logsumexp(log_total, log(m-NUM_NEIGHBORS) + far_weight)

    weights = fill(Base.Math.exp_fast(far_weight - log_total), m)
    weights[nearest_indices] = Base.Math.exp_fast.(near_weights .- log_total)
    index = categorical(weights)
    # Weight is actual p(x, y) / q(x)
    return logpdf(broadcasted_normal, y, X[:, index], sigma) - log(m) - log(weights[index])
end