using LinearAlgebra

@gen function ransac_sample(xs, ys, K=3)
    N = length(xs)
    indices ~ knuths_sample(N, K)
    
    index_vec = collect(indices)
    b, m = hcat(ones(K), xs[index_vec]) \ ys[index_vec]
    params ~ mvnormal([b, m], zeros(2, 2) + I)
end

ransac_safe = Marginal{Vector{Float64}}(ransac_sample, importance(1), :params)