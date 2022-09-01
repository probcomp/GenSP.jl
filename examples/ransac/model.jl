using GenProx

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
