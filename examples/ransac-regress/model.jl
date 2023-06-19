using GenSP
using Gen

include("distributions/knuths.jl")
include("distributions/ransac/exact.jl")
include("distributions/ransac/safe.jl")
include("distributions/ransac/unsafe.jl")

inlier_noise = 0.3

@gen function ransac_model(xs)
    params ~ mvnormal([0, 0], zeros(2, 2) + I)
    for (i, x) in enumerate(xs)
        exact_y = params[1] + params[2] * x
        {(:y, i)} ~ normal(exact_y, inlier_noise)
    end
end

function synthetic_data(size)
    step = 10.0 / (size - 1)
    xs = collect(-5.0:step:5.0)
    m, b = randn(), randn()
    ys = m .* xs .+ b .+ randn(size)
    return (xs, ys)
end

K = 3
dataset = synthetic_data(100)
xs, ys = dataset
obs_cm = choicemap([(:y, i) => ys[i] for i in 1:length(ys)]...)

function ransac_inference(num_particles, model_kind)
    ransac_dist = begin
        if model_kind == :exact
            ransac_exact
        elseif model_kind == :safe
            ransac_safe
        elseif model_kind == :unsafe
            ransac_unsafe
        else
            error("Unknown model kind: $model_kind")
        end
    end
    @gen function ransac_proposal(xs, ys, K)
        params ~ ransac_dist(xs, ys, K)
        return params
    end
    #_, wts, lml = importance_sampling(ransac_model, (xs,), obs_cm, ransac_proposal, (xs, ys, K), num_particles)
    #return (exp.(wts .+ log(num_particles))), lml
    return last(importance_sampling(ransac_model, (xs,), obs_cm, ransac_proposal, (xs, ys, K), num_particles))
end

ransac_settings = [(1, 100), (2, 50), (3, 50), (4, 50), (5, 50), (10, 30), (25, 20), (50, 20), (100, 2), (1000, 2), (10000, 2)];
ransac_model_config = Model("RANSAC", ransac_inference, ransac_settings)


