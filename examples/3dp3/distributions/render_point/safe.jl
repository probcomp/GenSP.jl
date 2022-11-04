@gen function render_point_model(X, X_tree, sigma)
    m = size(X, 2)
    index ~ uniform_discrete(1, m)
    point ~ broadcasted_normal(X[:, index], sigma)
end

NUM_NEIGHBORS = 100

@gen function render_point_proposal(target)
    y = target.constraints[:point]
    X, X_tree, sigma = target.args
    m = size(X, 2)

    # Nearest neighbors lookup
    nearest_indices, dists = NearestNeighbors.knn(X_tree, y, NUM_NEIGHBORS, true)

    # Categorical probabilities
    near_weights = -0.5 .* (dists .^ 2 / sigma^2)
    far_weight = last(near_weights)
    log_total = logsumexp([near_weights..., log(m-NUM_NEIGHBORS) + far_weight])

    weights = fill(far_weight - log_total, m)
    weights[nearest_indices] = near_weights .- log_total
    probs = exp.(weights)
    index ~ categorical(probs)
end

render_point_safe = Marginal{Vector{Float64}}(render_point_model, custom_importance(ChoiceMapDistribution(render_point_proposal), 1), :point)

