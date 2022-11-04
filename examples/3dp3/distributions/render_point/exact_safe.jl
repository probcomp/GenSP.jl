@gen function render_point_model(X, sigma)
    m = size(X, 2)
    index ~ uniform_discrete(1, m)
    point ~ broadcasted_normal(X[:, index], sigma)
end

render_point_exact = Marginal{Vector{Float64}}(
    render_point_model, enumeration(_ -> [:index]), :point)