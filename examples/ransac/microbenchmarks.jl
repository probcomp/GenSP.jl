using Gen, GenProx, BenchmarkTools

include("../benchmark_utils.jl")
include("model.jl")


function synthetic_data(size)
    step = 10.0 / (size - 1)
    xs = collect(-5.0:step:5.0)
    m, b = randn(), randn()
    ys = m .* xs .+ b .+ randn(size)
    return (xs, ys)
end

Ks = [1, 2, 3, 5, 10]
datasets = [synthetic_data(N) for N in [10, 50, 100, 500]]
NUM_SAMPLES = 10000

for K in Ks
    for (xs, ys) in datasets
        println("K = $K, N = $(length(xs))")
        println("\texact")
        display(@benchmark Gen.logpdf(ransac_exact, [0.0, 0.0], $xs, $ys, $K))
        println("\tsafe")
        display(@benchmark GenProx.estimate_logpdf(ransac_safe, [0.0, 0.0], $xs, $ys, $K))
        println("\tunsafe")
        display(@benchmark GenProx.estimate_logpdf(ransac_unsafe, [0.0, 0.0], $xs, $ys, $K))

        estimates = [GenProx.estimate_logpdf(ransac_unsafe, [0.0, 0.0], xs, ys, K) for _ in 1:NUM_SAMPLES]
        println("\trelative variance: $(relative_variance(estimates))")
        println("\tabsolute variance: $(abs_variance(estimates))")
    end
end