using Gen, GenSP, BenchmarkTools

include("../utils.jl")
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
        display(@benchmark GenSP.estimate_logpdf(ransac_safe, [0.0, 0.0], $xs, $ys, $K))
        println("\tunsafe")
        display(@benchmark GenSP.estimate_logpdf(ransac_unsafe, [0.0, 0.0], $xs, $ys, $K))
    end
end